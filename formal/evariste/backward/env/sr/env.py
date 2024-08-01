# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from multiprocessing.context import SpawnProcess
from typing import Optional, List
from logging import getLogger
import os
import re
import traceback
import multiprocessing as mp

from params import ConfStore
from evariste.datasets import SRDatasetConf
from evariste.model.data.dictionary import Dictionary
from evariste.backward.env.core import EnvGen, BackwardEnv
from evariste.backward.graph import UnMaterializedTheorem
from evariste.backward.env.sr.graph import SRTactic, SRTheorem
from evariste.backward.env.worker import (
    EnvWorker,
    SyncEnv,
    AsyncEnv,
    TacticJobResult,
    async_worker,
    AsyncTask,
    AsyncResult,
)
from evariste.utils import PicklingQueue
from evariste.backward.graph import Tactic, Theorem, Token
from evariste.envs.eq.env import EquationEnv
from evariste.envs.eq.graph import EqMatchException, Node
from evariste.envs.eq.rules import TRule
from evariste.envs.sr.env import SREnv, SREnvArgs, Transition, XYValues


logger = getLogger()


def apply_bwd_tactic(
    sr_env: SREnv, theorem: SRTheorem, tactic: SRTactic
) -> Optional[List[SRTheorem]]:
    """
    Apply a transformation tactic to a node.
    """
    eq_env: EquationEnv = sr_env.eq_env
    assert tactic.is_valid, "invalid tactic"
    try:
        assert isinstance(tactic.rule, TRule), "Rule type issue"
        applied = eq_env.apply_t_rule(
            eq=theorem.curr_node,
            rule=tactic.rule,
            fwd=True,
            prefix_pos=tactic.prefix_pos,
            to_fill=tactic.to_fill,
        )
        next_eq: Node = applied["eq"]
        true_xy: XYValues = theorem.true_xy
        curr_xy: XYValues = XYValues(true_xy.x, sr_env.evaluate_at(next_eq, true_xy.x))
        assert len(applied["hyps"]) == 0, "Hyps is not None"
        if sr_env.is_identical(curr_xy, true_xy):
            return []
        else:
            return [SRTheorem(next_eq, true_xy, curr_xy, theorem.target_node)]
    except (
        EqMatchException,
        ZeroDivisionError,
        KeyError,
        IndexError,
        RecursionError,
    ) as e:
        tactic.is_valid = False
        tactic.error_msg = f"{type(e)}:{e}"
        return None
    except AssertionError:
        tactic.is_valid = False
        tactic.error_msg = traceback.format_exc()
        return None


class SREnvWorker(EnvWorker):
    def __init__(self, dataset: SRDatasetConf, sr_env: Optional[SREnv] = None):
        from evariste.model.data.envs.sr import SRDataEnvironment

        self.dataset: SRDatasetConf = dataset
        self.sr_env: Optional[SREnv] = sr_env
        self.sr_data_env: Optional[SRDataEnvironment] = None
        self.rank: Optional[int] = None

    def init(self, rank: Optional[int] = None) -> None:
        if self.sr_data_env is None:
            from evariste.trainer.args import ConfStore
            from evariste.model.data.envs.sr import SRDataEnvironment

            params = ConfStore["default_cfg"]
            params.tasks = "sr_bwd_backtrack_walk_seq2seq"
            params.sr.dataset = self.dataset
            params.check_and_mutate_args(avoid={type(params.slurm_conf)})

            self.sr_data_env = SRDataEnvironment(Dictionary.create_empty(), params)
            self.sr_env = self.sr_data_env.sr_env
        self.rank = rank

    def apply_tactic(
        self,
        theorem: Theorem,
        tactic_tokens: Optional[List[Token]],
        tactic: Optional[Tactic] = None,
        keep_if_hyp: bool = False,
    ) -> TacticJobResult:
        assert isinstance(theorem, SRTheorem)
        # tactic
        assert (tactic_tokens is None) != (tactic is None)
        if tactic is None:
            assert tactic_tokens is not None
            try:
                tactic = SRTactic.from_tokens(tactic_tokens)
            except RecursionError:
                tactic = SRTactic.from_error(
                    error_msg=f"RecursionError when parsing tactic from tokens",
                    tokens=tactic_tokens,
                )
        assert isinstance(tactic, SRTactic)

        if not tactic.is_valid:
            return TacticJobResult(tactic, children=[])

        assert self.sr_env is not None
        result = apply_bwd_tactic(self.sr_env, theorem, tactic)

        # invalid tactic
        if result is None:
            assert not tactic.is_valid
            return TacticJobResult(tactic, children=[])

        assert tactic.is_valid
        assert len(result) <= 1, len(result)

        return TacticJobResult(tactic, children=result)

    def materialize_theorem(self, th: UnMaterializedTheorem) -> Theorem:
        assert self.sr_data_env is not None
        if re.match(r"sr_bwd_backtrack_walk_seq2seq_SAMPLE_\d+", th.label):
            seed = hash((self.rank, os.environ.get("SLURM_JOB_ID", None))) % (2 ** 32)
            task = "sr_bwd_backtrack_walk_seq2seq"
            theorem, _ = self.sr_data_env.get_theorem(task, seed=seed)
            return theorem
        return self.sr_data_env.label_to_eq[th.label]


class SREnvGenerator(EnvGen):
    def __init__(self, dataset: SRDatasetConf, n_async_envs: int):
        self.dataset = dataset
        self.n_async_envs = n_async_envs

        self.worker = SREnvWorker(self.dataset)
        self.worker_procs: List[SpawnProcess] = []

        if self.n_async_envs > 0:
            self._ctx = mp.get_context("spawn")
            self.to_apply: PicklingQueue[AsyncTask] = PicklingQueue(self._ctx.Queue())
            self.results: PicklingQueue[AsyncResult] = PicklingQueue(self._ctx.Queue())
            self.stop = self._ctx.Event()
            for rank in range(self.n_async_envs):
                self.worker_procs.append(
                    self._ctx.Process(
                        name=f"eq_env_worker_{rank}",
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
        if self.n_async_envs == 0:
            return BackwardEnv(SyncEnv(worker=self.worker))
        else:
            assert len(self.worker_procs) > 0
            return BackwardEnv(
                AsyncEnv(self.to_apply, self.results, self.stop, self.worker_procs)
            )

    def close(self):
        logger.info("Closing SREnvGenerator ...")
        if self.n_async_envs > 0:
            self.stop.set()
            for worker in self.worker_procs:
                worker.join()
        logger.info("Closed SREnvGenerator")


if __name__ == "__main__":

    # python -m evariste.backward.env.sr.env

    import evariste.datasets

    sr_args = SREnvArgs(
        eq_env=ConfStore["sr_eq_env_default"], max_backward_steps=100, max_n_points=100
    )
    env = SREnv.build(sr_args)

    def test_apply_bwd_tactic():
        expr = env.eq_env.generate_expr(n_ops=10)
        true_xy: XYValues = env.sample_dataset(expr)
        step: Transition = env.sample_transition(expr)

        curr_y = env.evaluate_at(step.eq, true_xy.x)
        curr_xy = XYValues(x=true_xy.x, y=curr_y)
        theorem = SRTheorem(node=step.eq, true_xy=true_xy, curr_xy=curr_xy)
        rule: TRule = step.rule
        tactic = SRTactic(rule.name, prefix_pos=step.prefix_pos, to_fill=step.tgt_vars,)
        next_theorem = apply_bwd_tactic(env, theorem, tactic)
        assert (
            isinstance(next_theorem, list) and len(next_theorem) == 0
        ), "Expected empty list"

        step_before: Transition = env.sample_transition(step.eq)
        if step_before is None:
            return
        curr_y = env.evaluate_at(step_before.eq, true_xy.x)
        curr_xy = XYValues(x=true_xy.x, y=curr_y)
        theorem = SRTheorem(node=step_before.eq, true_xy=true_xy, curr_xy=curr_xy)
        rule: TRule = step_before.rule
        tactic = SRTactic(
            rule.name, prefix_pos=step_before.prefix_pos, to_fill=step_before.tgt_vars,
        )
        next_theorem = apply_bwd_tactic(env, theorem, tactic)
        ##following test can fail because the perfect accuracy can be achieved without reaching target equation
        assert (
            isinstance(next_theorem, list) and len(next_theorem) == 1
        ), "Expected list with 1 theorem, got {}".format(next_theorem)

    for _ in range(10):
        test_apply_bwd_tactic()
        print("test {} passed".format(_))
