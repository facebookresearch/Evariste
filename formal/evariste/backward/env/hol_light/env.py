# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from multiprocessing.context import SpawnProcess
from typing import Optional, List
import multiprocessing as mp
from pathlib import Path
import os

from evariste.logger import create_logger, _MixinLoggerFactory
from evariste.datasets import HolLightDatasetConf
from evariste.backward.env.core import EnvGen, BackwardEnv
from evariste.backward.graph import Theorem, UnMaterializedTheorem
from evariste.backward.env.hol_light.graph import HLTactic, HLTheorem
from evariste.backward.env.worker import EnvWorker, SyncEnv, AsyncEnv
from evariste.backward.env.worker import async_worker, TacticJobResult
from evariste.envs.hl.api import HOLLightAPI, HOLLightException, HOLLightGoal
from evariste.envs.hl.utils import get_dag_and_th
from evariste.metrics import timeit, log_timeit
from evariste.utils import PicklingQueue


class HLEnvWorker(EnvWorker, _MixinLoggerFactory("debug")):
    def __init__(
        self, dataset: HolLightDatasetConf, dump_path: Optional[str] = None,
    ):
        self.dataset = dataset
        self.env = None
        self._dump_path = dump_path
        _, self.label_to_th, self.label_to_num_steps = get_dag_and_th(
            self.dataset.data_dir
        )
        self.has_num_steps = self.dataset.goals_with_num_steps

    def init(self, rank: Optional[int] = None):
        SLURM_ARRAY_JOB_ID = os.environ.get("SLURM_ARRAY_JOB_ID", None)
        SLURM_ARRAY_TASK_ID = os.environ.get("SLURM_ARRAY_TASK_ID", None)
        SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", None)

        if self._dump_path is not None:
            if SLURM_ARRAY_JOB_ID is not None and SLURM_ARRAY_TASK_ID is not None:
                slurm_array_suffix = f"-{SLURM_ARRAY_JOB_ID}-{SLURM_ARRAY_TASK_ID}"
            else:
                slurm_array_suffix = ""
            slurm_suffix = f"-{SLURM_JOB_ID}" if SLURM_JOB_ID is not None else ""
            rank_suffix = f"-{rank}" if rank is not None else ""
            log_filename = f"hol{slurm_array_suffix}{slurm_suffix}{rank_suffix}.log"
            log_filepath = os.path.join(os.path.join(self._dump_path, log_filename))
            logger = create_logger(filepath=log_filepath)
        else:
            logger = create_logger(None)
        self.set_logger(logger)
        self.log("Building HOL-Light API...", "info")
        self.env = HOLLightAPI(
            checkpoint_path=self.dataset.checkpoint_path,
            timeout=self.dataset.timeout,
            logger=self.logger,
            rank=rank,
        )

    def wait_till_ready(self) -> None:
        self.env.wait_till_ready()

    @log_timeit
    def close(self):
        if self.env is not None:
            self.env.close()

    @staticmethod
    def api_to_graph(
        goal: HOLLightGoal, train_label: Optional[str] = None
    ) -> HLTheorem:
        """Turn the output from HOLLightAPI into HLTheorem"""
        return HLTheorem(
            conclusion=" ".join(goal.concl_tokens),
            hyps=[(name, " ".join(hyp)) for name, hyp in goal.hyps_tokens],
            state=goal.raw.state,
            train_label=train_label,
        )

    @staticmethod
    def graph_to_api(theorem: HLTheorem) -> dict:
        if theorem.state is not None:
            return {"state": theorem.state}
        return {
            "concl_tokens": theorem.conclusion.split(),
            "hyps_tokens": [(name, hyp.split()) for name, hyp in theorem.hyps],
        }

    @timeit
    def apply_tactic(
        self, theorem: HLTheorem, tactic_tokens: List[str]
    ) -> TacticJobResult:
        tactic = HLTactic.from_tokens(tactic_tokens)
        if not tactic.is_valid:
            return TacticJobResult(tactic, [])
        try:
            sample = self.env.bwd_apply_tactic(
                tactic_tokens=tactic._tactic.split(),
                **HLEnvWorker.graph_to_api(theorem),
            )
            normalized_sample = sample.normalize()  # normalize systypes
            subgoals = [
                HLEnvWorker.api_to_graph(sg, theorem.train_label)
                for sg in normalized_sample.subgoals
            ]
            return TacticJobResult(tactic, children=subgoals)
        except HOLLightException as e:
            tactic.is_valid = False
            tactic.error_msg = f"{type(e).__name__}: {str(e)}"
            self.env.log(tactic.error_msg)
            return TacticJobResult(tactic, children=[])

    def materialize_theorem(self, th: UnMaterializedTheorem) -> Theorem:
        try:
            theorem = self.label_to_th[th.label]
        except KeyError:  # maybe it is a miniF2F label?
            from evariste.benchmark.miniF2F.hl_miniF2F import (
                get_miniF2F_goals_from_repo,
            )

            self.env.warning(
                f"The theorem {th.label} is not in (train|valid|test) split - check in miniF2F"
            )
            assert (
                not self.has_num_steps
            ), "Only the train|valid|test splits have num steps"

            self.label_to_th.update(
                {
                    goal.name: goal.theorem  # type: ignore ## fix get_miniF2F_goals_from_repo
                    for goal in get_miniF2F_goals_from_repo(
                        proving_mode="bwd", splitted=False
                    )
                }
            )
            theorem = self.label_to_th[th.label]
        # TODO Rescucitate (just put in HLTheorem)
        # num_steps = None if not self.has_num_steps else self.label_to_num_steps[th.label]
        return theorem


class HLEnvGenerator(EnvGen):
    def __init__(
        self, dataset: HolLightDatasetConf, dump_path: Path, debug: bool = False,
    ):
        self.dataset = dataset
        self.debug = debug
        self.worker = HLEnvWorker(dataset=dataset, dump_path=dump_path)
        self.worker_procs: List[SpawnProcess] = []
        self.n_envs = dataset.n_envs

        if not debug:
            self._ctx = mp.get_context("spawn")
            self.to_apply = PicklingQueue(self._ctx.Queue())
            self.results = PicklingQueue(self._ctx.Queue())
            self.stop = self._ctx.Event()
            for rank in range(self.n_envs):
                self.worker_procs.append(
                    self._ctx.Process(
                        name=f"hol_env_worker_{rank}",
                        target=async_worker,
                        args=(
                            self.to_apply,
                            self.results,
                            self.stop,
                            self.worker,
                            rank,
                        ),
                    )
                )
                self.worker_procs[-1].start()

    def __call__(self):
        if self.debug:
            return BackwardEnv(SyncEnv(worker=self.worker))
        else:
            assert len(self.worker_procs) > 0
            return BackwardEnv(
                AsyncEnv(self.to_apply, self.results, self.stop, self.worker_procs)
            )

    def close(self):
        if self.debug:
            return
        self.stop.set()
        for worker in self.worker_procs:
            worker.join()
