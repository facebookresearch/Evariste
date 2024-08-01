# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path
from typing import Optional, List
import os

from numpy import isin

from evariste.datasets.isabelle import IsabelleDatasetConf
from evariste.logger import create_logger, _MixinLoggerFactory
from evariste.envs.isabelle.controller import IsabelleController
from evariste.metrics import log_timeit, timeit
from evariste.backward.env.isabelle.graph import IsabelleTheorem, IsabelleTactic
from evariste.backward.env.worker import EnvWorker, TacticJobResult
from evariste.backward.graph import (
    MalformedTactic,
    Theorem,
    Tactic,
    Token,
    ProofId,
    BackwardGoal,
    UnMaterializedTheorem,
)


class IsabelleEnvWorker(EnvWorker):
    def __init__(self, dataset: IsabelleDatasetConf, dump_path: Optional[str] = None):
        self.dataset = dataset
        self._dump_path = dump_path

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
        self.log("Build Isabelle controller...")
        self.env = IsabelleController(
            dataset_path=self.dataset.data_dir,
            step_timeout_in_seconds=self.dataset.step_timeout_in_seconds,
            sledgehammer_timeout_in_seconds=self.dataset.sledgehammer_timeout_in_seconds,
            logger=self.logger,
            rank=rank,
        )

    def set_logger(self, logger):
        self.logger = logger

    def log(self, msg: str):
        self.logger.info(msg)

    @timeit
    def apply_tactic(
        self,
        theorem: IsabelleTheorem,
        tactic_tokens: Optional[List[Token]],
        tactic: Optional[IsabelleTactic] = None,
    ) -> TacticJobResult:
        # Either supply the full tactic or the tokens of it
        assert (tactic_tokens is None) != (tactic is None)
        if tactic is None:
            assert tactic_tokens is not None
            tactic = IsabelleTactic.from_tokens(tactic_tokens)

        assert isinstance(tactic, IsabelleTactic)

        if not tactic.is_valid:
            return TacticJobResult(tactic, children=[])
        assert self.env is not None
        children: List[IsabelleTheorem] = self.env.apply_step(
            theorem=theorem, tactic=tactic
        )
        return TacticJobResult(tactic=tactic, children=children)

    def wait_till_ready(self) -> None:
        self.env.wait_till_ready()

    @log_timeit
    def close(self):
        if self.env is not None:
            self.env.close()

    def materialize_theorem(self, th: UnMaterializedTheorem) -> Theorem:
        # IsabelleTheorem
        raise NotImplementedError
