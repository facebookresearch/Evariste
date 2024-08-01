# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, Sequence, List, Dict, Union, Any, Set
from multiprocessing.context import SpawnProcess
from multiprocessing import synchronize
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from queue import Empty
import abc
import time

from evariste.utils import PicklingQueue
from evariste.backward.graph import (
    Theorem,
    Tactic,
    Token,
    ProofId,
    BackwardGoal,
    UnMaterializedTheorem,
)

logger = getLogger()


@dataclass
class TacticJob:
    theorem: Theorem
    tactic_tokens: List[Token]
    proof_id: ProofId


@dataclass
class TacticJobResult:
    tactic: Tactic
    children: Sequence[Theorem]
    # used in lean to store parsed_goal
    theorem: Optional[Theorem] = None
    duration: float = 0


@dataclass
class ModelExpansion:
    exp_duration: float
    gpu_duration: float
    error: Optional[str] = None
    log_critic: Optional[float] = None
    tactics: Optional[List[List[str]]] = None
    log_priors: Optional[List[float]] = None

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def __post_init__(self):
        is_none = [
            self.log_critic is None,
            self.tactics is None,
            self.log_priors is None,
        ]
        if self.is_error:
            assert all(is_none)
        else:
            assert not any(is_none)
            assert len(self.tactics) == len(self.log_priors) > 0
            assert type(self.log_critic) is float and self.log_critic <= 0
            assert type(self.log_priors) is list
            assert all(type(p) is float and p <= 0 for p in self.log_priors)


class EnvWorker(abc.ABC):
    """
    Represents the part of an environment that does the processing : apply_tactic / check_proof
    """

    def init(self, rank: Optional[int] = None) -> None:
        pass

    @abc.abstractmethod
    def apply_tactic(
        self,
        theorem: Theorem,
        tactic_tokens: Optional[List[Token]],
        tactic: Optional[Tactic] = None,
        keep_if_hyp: bool = False,
    ) -> TacticJobResult:
        pass

    def wait_till_ready(self) -> None:
        """For async workers"""
        pass

    def close(self):
        pass

    @abc.abstractmethod
    def materialize_theorem(self, th: UnMaterializedTheorem) -> Theorem:
        pass


class ExpanderEnv(abc.ABC):
    """Either a sync or async wrapper around an EnvWorker"""

    @abc.abstractmethod
    def process(self, task: TacticJob, batch_id: int) -> int:
        pass

    @abc.abstractmethod
    def get_all_ready(self) -> List[Tuple[ProofId, int, TacticJobResult]]:
        pass

    def finish_theorem(self, goal: Theorem):
        pass

    @abc.abstractmethod
    def materialize_goal(self, goal: BackwardGoal) -> BackwardGoal:
        pass

    @abc.abstractmethod
    def theorem_ready(self, batch_id: Any) -> Optional[Theorem]:
        pass

    def conditioning_labels(self) -> Optional[Set[str]]:
        return None

    def close(self):
        pass

    def get_stats(self) -> Dict[str, float]:
        return {}

    def filter_model_expansion(self, model_expansion: ModelExpansion) -> int:
        return 0


class SyncEnv(ExpanderEnv):
    """ SyncWrapper around a worker_func """

    def __init__(self, worker: EnvWorker):
        self.worker = worker
        self.worker.init()
        self.ready: Dict[Tuple[ProofId, int], TacticJobResult] = {}
        self.ready_theorem: Dict[int, Theorem] = {}
        self.query_id = 0
        self.goal_id = 0

    def process(self, task: TacticJob, batch_id: int) -> int:
        start = time.time()
        result = self.worker.apply_tactic(task.theorem, task.tactic_tokens)
        result.duration = time.time() - start
        self.ready[batch_id, self.query_id] = result

        self.query_id += 1
        return self.query_id - 1

    def get_all_ready(self) -> List[Tuple[ProofId, int, TacticJobResult]]:
        ready = [(pid, qid, res) for (pid, qid), res in self.ready.items()]
        self.ready.clear()
        return ready

    def materialize_goal(self, goal: BackwardGoal) -> BackwardGoal:
        if isinstance(goal.theorem, UnMaterializedTheorem):
            theorem = self.worker.materialize_theorem(goal.theorem)
            goal.theorem.batch_id = self.goal_id
        else:
            theorem = goal.theorem
            # mcts expects unmaterialized theorems.
            goal.theorem = UnMaterializedTheorem(label=goal.label)
            goal.theorem.batch_id = self.goal_id
        self.ready_theorem[self.goal_id] = theorem
        self.goal_id += 1
        return goal

    def theorem_ready(self, batch_id: Any) -> Optional[Theorem]:
        return self.ready_theorem.pop(batch_id, None)


@dataclass
class AsyncTask:
    batch_id: int
    query_id: int
    task: Union[TacticJob, UnMaterializedTheorem]


@dataclass
class AsyncResult:
    batch_id: int
    query_id: int
    result: Union[TacticJobResult, Theorem]


def async_worker(
    queue_in: PicklingQueue[AsyncTask],
    queue_out: PicklingQueue[AsyncResult],
    stop: synchronize.Event,
    worker: EnvWorker,
    rank: int,
):
    queue_in.cancel_join_thread()
    queue_out.cancel_join_thread()
    try:
        worker.init(rank)
        while not stop.is_set():
            worker.wait_till_ready()
            try:
                task: AsyncTask = queue_in.get(timeout=0.1)
            except Empty:
                continue
            job = task.task
            if isinstance(job, TacticJob):
                res: Union[TacticJobResult, Theorem] = worker.apply_tactic(
                    job.theorem, job.tactic_tokens
                )
            else:
                assert isinstance(job, UnMaterializedTheorem)
                res = worker.materialize_theorem(job)
            assert res is not None
            queue_out.put(AsyncResult(task.batch_id, task.query_id, result=res))

    except Exception as e:
        # Any unhandled exception in the worker should stop all other workers, and raise.
        stop.set()
        raise e
    finally:
        worker.close()


class AsyncEnv(ExpanderEnv):
    """ Allows multiple processes to communicate with a one/many EnvWorker """

    def __init__(
        self,
        input_queue: PicklingQueue[AsyncTask],
        output_queue: PicklingQueue[AsyncResult],
        stop: synchronize.Event,
        processes: List[SpawnProcess],
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop = stop
        self.processes = processes

        # requests are sent by batch
        self.query_id = 0
        self.goal_id = 0
        self.processing: Dict[int, List[Tuple[int, TacticJob]]] = defaultdict(list)
        self.processing_start: Dict[Tuple[int, int], float] = {}
        self.received: Dict[int, int] = defaultdict(int)
        self.received_goals: Dict[int, Theorem] = {}

        self.last_alive_check = time.time()

    def check_alive(self):
        if time.time() - self.last_alive_check > 10:
            for p in self.processes:
                if not p.is_alive():
                    raise RuntimeError("Worker process killed in AsyncEnv")
            self.last_alive_check = time.time()

    def process(self, task: TacticJob, batch_id: int):
        if self.stop.is_set():
            raise RuntimeError("Stop is set in AsyncEnv")
        self.check_alive()
        qid = self.query_id
        self.query_id += 1
        self.processing[batch_id].append((qid, task))
        self.processing_start[(batch_id, qid)] = time.time()
        self.input_queue.put(AsyncTask(batch_id=batch_id, query_id=qid, task=task))
        return qid

    def materialize_goal(self, goal: BackwardGoal) -> BackwardGoal:
        assert isinstance(goal.theorem, Theorem)
        if isinstance(goal.theorem, UnMaterializedTheorem):
            goal.theorem.batch_id = self.goal_id
            self.input_queue.put(
                AsyncTask(batch_id=self.goal_id, query_id=0, task=goal.theorem)
            )
        else:
            assert self.goal_id not in self.received_goals
            self.received_goals[self.goal_id] = goal.theorem
            # mcts expects unmaterialized theorems.
            goal.theorem = UnMaterializedTheorem(label=goal.label)
            goal.theorem.batch_id = self.goal_id
        self.goal_id += 1
        return goal

    def theorem_ready(self, batch_id: Any) -> Optional[Theorem]:
        self.check_alive()
        return self.received_goals.pop(batch_id, None)

    def get_all_ready(self) -> List[Tuple[ProofId, int, TacticJobResult]]:
        results = []
        try:
            while not self.stop.is_set():
                self.check_alive()
                result: AsyncResult = self.output_queue.get_nowait()
                assert result is not None, result
                if isinstance(result.result, TacticJobResult):
                    start = self.processing_start.pop(
                        (result.batch_id, result.query_id)
                    )
                    result.result.duration = time.time() - start
                    results.append((result.batch_id, result.query_id, result.result))
                    self.received[result.batch_id] += 1
                    if self.received[result.batch_id] == len(
                        self.processing[result.batch_id]
                    ):
                        del self.received[result.batch_id]
                        del self.processing[result.batch_id]
                elif isinstance(result.result, Theorem):
                    assert result.batch_id not in self.received_goals
                    self.received_goals[result.batch_id] = result.result
        except Empty:
            pass
        return results

    def status(self):
        pass
