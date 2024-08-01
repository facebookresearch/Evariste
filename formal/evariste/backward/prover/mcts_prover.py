# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Union, List, Set, Dict, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass
from queue import Empty
import time
import os
import sys
import cProfile
from torch import multiprocessing as mp
from multiprocessing import synchronize
from multiprocessing.context import SpawnProcess

from evariste.backward.env.core import BackwardGoal
from evariste.backward.env.lean.env import (
    DeclNotFound,
    LeanEvalCmdJob,
    LeanEvalCmdResult,
)
from evariste.backward.prover.mcts import MCTSHandler, MCTSResult
from evariste.backward.prover.simple_mcts import SimpleMCTSHandler
from evariste.backward.prover.core import ProofHandler, ProofResult, ProofHandlerDied
from evariste.backward.prover.prover_args import ProverKind, ProverParams

from evariste.backward.graph import ProofId, Theorem, UnMaterializedTheorem
from evariste.backward.env.core import EnvExpansion
from evariste.backward.env.lean.graph import (
    LeanTheorem,
    LeanTactic,
    LeanState,
)
from evariste.datasets import DatasetConf
from evariste.utils import (
    PicklingQueue,
    PROFILE,
    get_tail_logger,
    NoneLogger,
    my_gb_memory,
)
from evariste.logger import create_logger


class Stop:
    pass


@dataclass
class IsDone:
    done: bool


class StitchingDone:
    pass


MCTSInput = Union[
    BackwardGoal,
    Theorem,
    List[EnvExpansion],
    Stop,
    dict,
    DeclNotFound,
    LeanEvalCmdResult,
]
MCTSOutput = Union[
    IsDone,
    List[Theorem],
    MCTSResult,
    None,
    UnMaterializedTheorem,
    Tuple[LeanState, LeanTactic],
    StitchingDone,
    bool,
    LeanEvalCmdJob,
]


global_logger = create_logger(None)


class MCTSRunner:
    def __init__(self, tail_logger, prover_params, process_id, output):
        self.mcts: Optional[Union[MCTSHandler, SimpleMCTSHandler]] = None
        self.next_goal: Optional[BackwardGoal] = None
        self.tail_logger = tail_logger
        self.prover_params = prover_params
        self.process_id = process_id
        self.n_mcts = 0
        self.result = None
        self.output = output

    def reset(self):
        self.result = None
        self.mcts = None

    def do_proving(self, todo):
        """Advances proof search depending on parameter `todo`:
        :param todo:
        * at the beginning of a proof, expects a BackwardGoal (with UnmaterializedTheorem)
        * then, expects a Theorem (to replace the unmaterialized one in the backward goal)
        * during proving, only expects lists of :class:`EnvExpansion`

        Returns nothing, but may output stuff to the output queue, such as next theorems to expand, or the final MCTSResult
        """
        if isinstance(todo, BackwardGoal):
            self.tail_logger.info(f"Got backward goal")
            self.next_goal = todo
        elif isinstance(todo, Theorem):
            assert self.next_goal is not None
            self.tail_logger.info("Got materialized theorem")
            self.next_goal.theorem = todo

            assert self.mcts is None
            self.tail_logger.info(f"Next goal: {self.next_goal.name}")
            self.n_mcts += 1
            if self.prover_params.prover_kind is ProverKind.BackwardMCTS:
                self.mcts = MCTSHandler(
                    goal=self.next_goal,
                    prover_params=self.prover_params,
                    process_id=self.process_id,
                )
            elif self.prover_params.prover_kind is ProverKind.BackwardSimpleMCTS:
                self.mcts = SimpleMCTSHandler(
                    goal=self.next_goal,
                    prover_params=self.prover_params,
                    process_id=self.process_id,
                )
            else:
                raise RuntimeError(f"Unknown prover kind")
            assert self.mcts is not None
            self.next_goal = None
            self.tail_logger.info("Put FIRST to_expand")
            self.output.put(self.mcts.get_theorems_to_expand())
        elif isinstance(todo, list):
            # filter out old expansion results from failed mcts
            assert self.mcts is not None
            self.tail_logger.info("Got expansion result, expanding")
            self.mcts.expand_and_backup(todo)
            self.tail_logger.info(f"Is Done {self.mcts.done}")
            if not self.mcts.done:
                self.output.put(IsDone(False))
                self.tail_logger.info("Finding to expand")
                to_expand = self.mcts.get_theorems_to_expand()
                if isinstance(self.mcts, MCTSHandler):
                    assert len(to_expand) > 0 or (
                        self.mcts.done
                    ), f"Empty to expand {self.mcts.goal.name} {self.mcts.done}"
                self.tail_logger.info(
                    f"Putting to expand {len(to_expand)} {self.mcts.goal.name}"
                )
                before_put = time.time()
                self.output.put(to_expand)
                self.tail_logger.info(f"Put to expand {time.time() - before_put}")
            else:
                self.tail_logger.info("getting result")
                self.result = self.mcts.result()
                self.tail_logger.info("putting is done")
                return True  # proving is done until new proof
        return False  # proving continues


def one_mcts_handler(
    process_id: int,
    prover_params: ProverParams,
    input_queue: PicklingQueue[MCTSInput],
    output_queue: PicklingQueue[MCTSOutput],
    stop_signal: synchronize.Event,
    death_signal: synchronize.Event,
):
    output_queue.cancel_join_thread()

    # sys.setrecursionlimit(200)  # for debugging purposes

    profile_dir = prover_params.dump_path / "mcts_profile"
    tail_log_dir = prover_params.dump_path / "mcts_logs"

    os.makedirs(profile_dir, exist_ok=True)
    os.makedirs(tail_log_dir, exist_ok=True)

    logger = (
        get_tail_logger(tail_log_dir / f"one_mcts.{process_id}.log", loc_info=False)
        if not prover_params.quiet
        else NoneLogger()
    )
    pr = None
    if PROFILE:
        pr = cProfile.Profile()
        pr.enable()

    runner = MCTSRunner(logger, prover_params, process_id, output_queue)

    last_profile = time.time()
    try:
        while not stop_signal.is_set() and not death_signal.is_set():

            if pr is not None:
                pr.disable()
            try:
                todo = input_queue.get(timeout=0.1)  # From MCTSProofHandler
            except Empty:
                continue
            except Exception as e:
                print("PROBLEM HEHE", file=sys.stderr)
                print(e, file=sys.stderr)
                # those are due to cancel_join_thread or SIGINT
                # Something went wrong, stop immediately
                break
            if pr is not None:
                pr.enable()

            if isinstance(todo, Stop):
                runner.mcts = None
                output_queue.put(None)
            else:
                proving_finished = runner.do_proving(todo)
                if proving_finished:
                    output_queue.put(IsDone(True))
                    logger.info("putting result")
                    output_queue.put(runner.result)
                    runner.reset()

            if time.time() - last_profile > 60 and runner.mcts is not None:
                if pr is not None:
                    pr.dump_stats(profile_dir / f"{process_id:>2}.profile")
                    pr.enable()
                mem = my_gb_memory()
                if isinstance(runner.mcts, MCTSHandler):
                    n_nodes = len(runner.mcts.mcts.nodes)
                elif isinstance(runner.mcts, SimpleMCTSHandler):
                    n_nodes = len(runner.mcts.node_id_to_th)
                else:
                    n_nodes = 0
                logger.info(
                    f"[Mem] process_id: {process_id:>2} -- "
                    f"{mem:.3f}GB -- MCTS ID: {runner.n_mcts}"
                    f"MCTS nodes: {n_nodes}"
                )

                last_profile = time.time()
    except KeyboardInterrupt:
        pass  # handled outside
    except Exception:
        death_signal.set()
        raise
    finally:
        stop_signal.set()


class MCTSProofHandlerGenerator:
    """A process pool for :class:`MCTSProofHandler`. If we plan on working on `n_simultaneous_proofs`, we start this number of processes and associated communication queues.
    All these processes share the same stop / death signals.
    """

    def __init__(
        self, prover_params: ProverParams,
    ):
        n_processes = prover_params.n_simultaneous_proofs
        ctx = mp.get_context("spawn")
        self.inputs: List[PicklingQueue[MCTSInput]] = [
            PicklingQueue(ctx.Queue()) for _ in range(n_processes)
        ]
        for input_queue in self.inputs:
            input_queue.cancel_join_thread()
        self.results: List[PicklingQueue[MCTSOutput]] = [
            PicklingQueue(ctx.Queue()) for _ in range(n_processes)
        ]
        self.stop_signal, self.death_signal = ctx.Event(), ctx.Event()
        self.stop_signal.clear()
        self.death_signal.clear()
        self.processes = [
            ctx.Process(
                name=f"mcts_{i}",
                target=one_mcts_handler,
                args=(
                    i,
                    prover_params,
                    self.inputs[i],
                    self.results[i],
                    self.stop_signal,
                    self.death_signal,
                ),
            )
            for i in range(n_processes)
        ]
        for x in self.processes:
            x.start()
        self.available: Set[int] = set(range(n_processes))
        self.initialized = True

    def __call__(self, goal: BackwardGoal):
        if self.available:
            process_id = self.available.pop()
        else:
            print("Setting proof handlers stop signal", flush=True)
            self.stop_signal.set()
            raise RuntimeError(
                "attempted to create a new proof handler when none are available"
            )
        return MCTSProofHandler(
            goal,
            self.processes[process_id],
            self.inputs[process_id],
            self.results[process_id],
            self.stop_signal,
            self.death_signal,
            process_id,
            self.available,
        )

    def close(self):
        print("Closing MCTSProofHandlerGenerator ...")
        if self.stop_signal is not None:
            self.stop_signal.set()
        for x in self.processes:
            x.join()
        print("Closed MCTSProofHandlerGenerator")


class MCTSProofHandler(ProofHandler):
    """Handles communications with the process running MCTS code.
    In addition to the usual :class:`ProofHandler` methods, we add :meth:`check_alive` to make sure the underlying process is still running.
    
    We use :class:`PicklingQueue` for input/outputs to avoid exceptions in the background during pickling which lead to difficult to debug hangs.
    """

    def __init__(
        self,
        goal: BackwardGoal,
        process: SpawnProcess,
        input_q: PicklingQueue[MCTSInput],
        results: PicklingQueue[MCTSOutput],
        stop_signal: synchronize.Event,
        death_signal: synchronize.Event,
        process_id: int,
        available: Set[int],
    ):
        super().__init__(goal)
        self.process = process
        self.input_q = input_q
        self.results = results
        self.stop_signal = stop_signal
        self.death_signal = death_signal
        self.available = available
        self.process_id = process_id
        self.input_q.put(goal)
        self.last_alive_check = time.time()

        self.stitching_root: Union[None, UnMaterializedTheorem, LeanTheorem] = None

        # stats
        self.stats: Dict[str, int] = defaultdict(int)

    def send_materialized(self, th: Theorem):
        self.input_q.put(th)

    def check_alive(self):
        if time.time() - self.last_alive_check < 10:
            return
        self.last_alive_check = time.time()
        if not self.process.is_alive():
            raise ProofHandlerDied(f"Process id {self.process_id} is dead!")
        is_stopped = self.stop_signal.is_set()
        is_dead = self.death_signal.is_set()
        if is_stopped or is_dead:
            raise ProofHandlerDied(f"is_stopped={is_stopped} is_dead={is_dead}")

    def get_theorems_to_expand(self) -> Optional[List[Theorem]]:
        assert self.stats["got_theorems_to_expand"] == self.stats["send_env_expansions"]
        # self.stats["get_theorems_to_expand"] += 1
        self.check_alive()
        try:
            goals = self.results.get_nowait()
            assert isinstance(goals, list), goals
            assert all(isinstance(x, Theorem) for x in goals), goals
            self.stats["got_theorems_to_expand"] += 1
            self.stats["total_goals"] += len(goals)
            self.stats["last_goals"] = len(goals)
            return goals
        except Empty:
            return None

    def send_env_expansions(self, tactics: List[EnvExpansion]) -> None:
        self.stats["send_env_expansions"] += 1
        self.stats["total_expansions"] += len(tactics)
        assert self.stats["got_theorems_to_expand"] == self.stats["send_env_expansions"]
        self.check_alive()
        self.input_q.put(tactics)

    def get_done(self) -> bool:
        self.check_alive()
        try:
            is_done = self.results.get_nowait()
            assert isinstance(is_done, IsDone), is_done
            self.done = is_done.done
            return True
        except Empty:
            return False

    def get_result(self) -> ProofResult:
        self.check_alive()
        res = self.results.get(block=True)
        assert isinstance(res, MCTSResult)
        return res

    def stop(self) -> None:
        self.input_q.put(Stop())
        assert self.results.get() is None

    def close(self):
        self.available.add(self.process_id)

    def status(self) -> Dict[str, Any]:
        status = {"process_id": self.process_id, "done": self.done, **self.stats}
        return status
