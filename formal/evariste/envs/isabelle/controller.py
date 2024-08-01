# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from ast import Is
from typing import List, Union, Optional
from threading import Thread, Event

from evariste.logger import create_logger
from evariste.metrics import log_timeit, timeit
from evariste.backward.graph import (
    Theorem,
    Tactic,
    Token,
    ProofId,
    BackwardGoal,
    UnMaterializedTheorem,
)
from evariste.backward.env.worker import EnvWorker
from evariste.backward.env.isabelle.graph import IsabelleTactic, IsabelleTheorem


class IsabelleController:
    """
    Isabelle controller that holds an environment and handles the interactions with the scala server.
    """

    def __init__(
        self,
        dataset_path: str,
        step_timeout_in_seconds: float = 0.2,
        sledgehammer_timeout_in_seconds: float = 35,
        logger=None,
        rank: Optional[int] = None,
    ):
        logger = logger if logger is not None else create_logger(None)
        self.set_logger(logger)

        self.dataset_path = dataset_path
        self.step_timeout_in_seconds = step_timeout_in_seconds
        self.sledgehammer_timeout_in_seconds = sledgehammer_timeout_in_seconds
        self.rank = rank
        self.env_ready = Event()
        self.make_env()

    def set_logger(self, logger):
        self.logger = logger

    def log(self, msg: str):
        self.logger.info(msg)

    @timeit
    def wait_till_ready(self):
        self.env_ready.wait()

    @timeit
    def _make_env(
        self, isabelle_path: str, working_directory: str, theory_file_path: str
    ) -> None:
        if not hasattr(self, "_env"):
            self.log("Starting Isabelle environment")
        else:
            self.log("Restarting Isabelle environment")

        try:
            if hasattr(self, "_env"):
                self._env.kill()
            self._env: IsabelleMessenger = IsabelleMessenger(
                isabelle_path=isabelle_path,
                working_directory=working_directory,
                theory_file_path=theory_file_path,
                dataset_path=self.dataset_path,
                step_timeout_in_seconds=self.step_timeout_in_seconds,
                sledgehammer_timeout_in_seconds=self.sledgehammer_timeout_in_seconds,
                logger=self.logger,
            )
        except Exception as e:
            self.log("Failed to start Isabelle environment")
            self.log(f"Error: {e}")
            raise Exception

        self.log(f"Isabelle environment started: PID#{self._env.rank}")
        self.env_ready.set()

    @timeit
    def make_env(self) -> None:
        self.env_ready.clear()
        Thread(
            target=IsabelleController._make_env, name="_make_env", args=(self,)
        ).start()

    @timeit
    def _send(self, cmd: str) -> str:
        self.env_ready.wait()
        self.log(f"Sending raw command to Isabelle messenger: {cmd}")
        s = self._env.send(cmd)
        self.log(f"Receiving raw response from Isabelle messenger: {s}")
        return s

    @timeit
    def _apply_step(self, step: str, state_name: str,) -> str:
        """
        Apply a step to a state with a given state_name
        """

        raise NotImplementedError

    @timeit
    def apply_step(
        self, theorem: IsabelleTheorem, tactic: IsabelleTactic
    ) -> List[IsabelleTheorem]:
        raise NotImplementedError

    @timeit
    def materialize_theorem(self, th: UnMaterializedTheorem) -> Theorem:
        label_of_the_unmaterialized_theorem = th.label
        self.log(f"Materializing theorem {label_of_the_unmaterialized_theorem}")
        raise NotImplementedError

    @log_timeit
    def close(self):
        self.log("Isabelle controller env self destructing...")
        if hasattr(self, "_env"):
            self.log("Isabelle messenger env self destructing...")
            self._env.kill()


class IsabelleMessenger:
    def __init__(
        self,
        isabelle_path: str,
        working_directory: str,
        theory_file_path: str,
        dataset_path: str,
        step_timeout_in_seconds: float,
        sledgehammer_timeout_in_seconds: float,
        logger=None,
    ):
        self.isabelle_path = isabelle_path
        self.working_directory = working_directory
        self.theory_file_path = theory_file_path
        self.dataset_path = dataset_path
        self.step_timeout_in_seconds = step_timeout_in_seconds
        self.sledgehammer_timeout_in_seconds = sledgehammer_timeout_in_seconds
        self.logger = logger if logger is not None else create_logger(None)
        self.rank = None

    def set_rank(self):
        raise NotImplementedError

    def kill(self):
        raise NotImplementedError

    def send(self, msg: str):
        raise NotImplementedError
