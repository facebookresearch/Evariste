# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from multiprocessing.context import SpawnProcess
from typing import Optional, List, Tuple
from logging import getLogger
import os
import traceback
import multiprocessing as mp

from evariste.datasets import EquationsDatasetConf
from evariste.envs.eq.env import EquationEnv
from evariste.envs.eq.graph import EqMatchException, Node, NodeSet
from evariste.envs.eq.rules import eval_assert, TRule, ARule, Rule
from evariste.envs.eq.rules_lib import ALL_A_RULES, ALL_RULES
from evariste.backward.env.core import EnvGen, BackwardEnv
from evariste.backward.env.equations.graph import (
    EQTactic,
    EQTheorem,
    EQRuleTactic,
    EQSimpTactic,
    EQNormNumTactic,
)
from evariste.backward.graph import UnMaterializedTheorem, Theorem, Token, Tactic
from evariste.backward.env.worker import (
    EnvWorker,
    SyncEnv,
    AsyncEnv,
    TacticJobResult,
    async_worker,
    AsyncTask,
    AsyncResult,
)
from evariste.model.data.dictionary import Dictionary
from evariste.utils import PicklingQueue


logger = getLogger()


# if a subgoal is False, optionally make the corresponding tactic invalid
# with `SKIP_ALWAYS_FALSE`. this results in faster proving by cutting branches,
# but this may results in failures if the initial theorem hypotheses are
# False (which can happen at generation time)
SKIP_ALWAYS_FALSE = True


class EqGenRuleEnv:
    def __init__(
        self, rule_env: str, env: EquationEnv, rules: Optional[List[Rule]] = None
    ):
        self.rule_env = rule_env
        self.env = env

        if rules is None:
            self.rules = [
                rule
                for rule in ALL_RULES[self.rule_env]
                if rule.get_unary_ops().issubset(self.env.unary_ops)
                and rule.get_binary_ops().issubset(self.env.binary_ops)
            ]
        else:
            self.rules = rules
        self.rules_t = [r for r in self.rules if isinstance(r, TRule)]
        self.rules_a = [r for r in self.rules if isinstance(r, ARule)]
        self.rules_t_counts = {
            (rule.name, fwd): 0 for rule in self.rules_t for fwd in [True, False]
        }
        self.rules_a_counts = {rule.name: 0 for rule in self.rules_a}
        self.rules_a_hyps = [rule for rule in self.rules_a if len(rule.hyps) > 0]
        self.rules_a_no_hyps = [rule for rule in self.rules_a if len(rule.hyps) == 0]

        # for lean simplification:
        self.simp_rules: List[TRule] = [
            r  # type: ignore
            for r in self.rules
            if (
                r.lean_rule is not None and r.lean_rule.is_simp and isinstance(r, TRule)
            )
        ]

        # for graph operations that are specifically not simplification:
        self.not_simp_rules = [
            r
            for r in self.rules
            if (r.lean_rule is not None and not r.lean_rule.is_simp)
        ]
        assert all(r in self.rules_t or r in self.rules_a for r in self.not_simp_rules)


class AutoError(Exception):
    pass


def simplify_node(
    node: Node,
    env: EquationEnv,
    a_rules: List[ARule],
    simp_rules: List[TRule],
    available_hyps: NodeSet,
) -> Tuple[Node, List[TRule], NodeSet]:
    changed = True
    assert (
        len(simp_rules) > 0
    ), "no simp rules. probably invalid file in LEAN_THEOREMS_PATHS"
    # Apply simp rules repeatedly until we hit a true node or no simp rule can be applied
    rules: List[TRule] = []
    hyps = NodeSet()
    last_node = node
    while changed:
        changed = False
        for rule in simp_rules:
            matches = rule.eligible(last_node, fwd=True, fast=True)
            if len(matches) == 0:
                continue
            applied = env.apply_t_rule(
                eq=last_node, rule=rule, fwd=True, prefix_pos=matches[0][0], to_fill={},
            )
            if len(applied["to_fill"]) > 0 and not rule.arithmetic:
                # we don't want simp to make shit up
                continue
            if applied["eq"].eq(last_node):
                continue

            to_add_hyps = []
            not_ok = False
            for h in applied["hyps"]:
                if h not in available_hyps:
                    if eval_assert(h, a_rules, env.vtype):
                        continue
                    not_ok = True
                    break
                to_add_hyps.append(h)
            if not_ok:
                continue
            for h in to_add_hyps:
                hyps.add(h)
            rules.append(rule)
            last_node = applied["eq"]
            changed = True
    return last_node, rules, hyps


def apply_bwd_tactic(
    eq_env: EquationEnv,
    theorem: EQTheorem,
    tactic: EQTactic,
    keep_if_hyp: bool,
    rule_env: str,
    a_rules: Optional[List[ARule]] = None,
    simp_rules: Optional[List[TRule]] = None,
) -> List[EQTheorem]:
    """
    Apply a transformation or an assertion tactic to a node.
    """
    assert tactic.is_valid, tactic
    try:
        if isinstance(tactic, EQRuleTactic):
            if tactic.rule_type == "t":
                assert isinstance(tactic.rule, TRule)
                applied = eq_env.apply_t_rule(
                    eq=theorem.eq_node,
                    rule=tactic.rule,
                    fwd=tactic.fwd,
                    prefix_pos=tactic.prefix_pos,
                    to_fill=tactic.to_fill,
                )
                new_theorem = EQTheorem(applied["eq"], theorem.eq_hyps)
                hyps = [EQTheorem(hyp, theorem.eq_hyps) for hyp in applied["hyps"]]
                subgoals = [new_theorem, *hyps]
            else:
                assert tactic.rule_type == "a"
                assert isinstance(tactic.rule, ARule)
                applied = eq_env.apply_a_rule(
                    eq=theorem.eq_node, rule=tactic.rule, to_fill=tactic.to_fill
                )
                subgoals = [EQTheorem(hyp, theorem.eq_hyps) for hyp in applied["hyps"]]
        elif isinstance(tactic, EQSimpTactic):
            assert a_rules is not None and simp_rules is not None
            assert tactic.target is not None
            available_hyps = NodeSet(theorem.eq_hyps + tactic.hyps)
            simplified, _rules, used_hyps = simplify_node(
                tactic.target, eq_env, a_rules, simp_rules, available_hyps
            )
            if not simplified.eq(theorem.eq_node):
                raise AutoError(
                    f"Simp {tactic.target.infix()} = {simplified.infix()} != {theorem.eq_node.infix()}"
                )
            subgoals = [
                EQTheorem(n, theorem.eq_hyps) for n in [tactic.target, *list(used_hyps)]
            ]
        elif isinstance(tactic, EQNormNumTactic):
            assert tactic.target is not None
            normnumified = tactic.target._norm_numify(eq_env.vtype)
            normnumified2 = theorem.eq_node._norm_numify(eq_env.vtype)
            if not normnumified.eq(normnumified2):
                raise AutoError(
                    f"Normnum {tactic.target.infix()} = {normnumified.infix()} != {theorem.eq_node.infix()}"
                )
            subgoals = [EQTheorem(tactic.target, theorem.eq_hyps)]
        else:
            raise NotImplementedError(f"{type(tactic)}:{tactic}")

        children, all_valid = [], True
        for child in subgoals:
            try:
                if not child.eq_node.is_valid(is_lean="lean" in rule_env):
                    tactic.is_valid = False
                    tactic.error_msg = f"Invalid child: {child.eq_node.prefix()}"
                    all_valid = False
                    break
                known_truth = eval_assert(child.eq_node, ALL_A_RULES[rule_env])
            except RecursionError:
                tactic.is_valid = False
                tactic.error_msg = f"RecursionError for eval_assert in apply tactic"
                all_valid = False
                break
            if not keep_if_hyp and child.conc_in_hyp() or known_truth is True:
                continue
            if SKIP_ALWAYS_FALSE and known_truth is False:
                tactic.is_valid = False
                tactic.error_msg = f"Subgoal {child.eq_node} is always False."
                all_valid = False
                break
            assert known_truth is None
            children.append(child)
        assert all_valid == tactic.is_valid
        if not tactic.is_valid:
            return []
        return children

    except (
        EqMatchException,
        ZeroDivisionError,
        KeyError,
        IndexError,
        RecursionError,
        AutoError,
    ) as e:
        tactic.is_valid = False
        tactic.error_msg = f"{type(e)}:{e}"
        return []
    except AssertionError:
        tactic.is_valid = False
        tactic.error_msg = traceback.format_exc()
        return []


class EQEnvWorker(EnvWorker):
    def __init__(
        self, dataset: EquationsDatasetConf, eq_env: Optional[EquationEnv] = None
    ):
        from evariste.model.data.envs.equations import EquationsEnvironment

        self.dataset: EquationsDatasetConf = dataset
        self.eq_env: Optional[EquationEnv] = eq_env
        self.eq_data_env: Optional[EquationsEnvironment] = None
        self.rank: Optional[int] = None
        self.rule_env = None
        if self.eq_env is not None:
            self.rule_env = EqGenRuleEnv(self.dataset.rule_env, self.eq_env)

    def init(self, rank: Optional[int] = None) -> None:
        if self.eq_data_env is None:
            from evariste.trainer.args import ConfStore
            from evariste.model.data.envs.equations import EquationsEnvironment

            params = ConfStore["default_cfg"]
            params.tasks = "eq_bwd_graph_seq2seq"
            params.eq.dataset = self.dataset
            params.check_and_mutate_args(avoid={type(params.slurm_conf)})

            self.eq_data_env = EquationsEnvironment(Dictionary.create_empty(), params)
            self.eq_env = self.eq_data_env.eq_env
        if self.rule_env is None:
            assert self.eq_env is not None
            self.rule_env = EqGenRuleEnv(self.dataset.rule_env, self.eq_env)
        self.rank = rank

    def apply_tactic(
        self,
        theorem: Theorem,
        tactic_tokens: Optional[List[Token]],
        tactic: Optional[Tactic] = None,
        keep_if_hyp: bool = False,
    ) -> TacticJobResult:
        assert isinstance(
            theorem, EQTheorem
        )  # liskov substitution principle is violated?
        # tactic
        assert (tactic_tokens is None) != (tactic is None)
        if tactic is None:
            assert tactic_tokens is not None
            try:
                tactic = EQTactic.from_tokens(tactic_tokens)
            except RecursionError:
                tactic = EQTactic.from_error(
                    error_msg=f"RecursionError when parsing tactic from tokens",
                    tokens=tactic_tokens,
                )
        assert isinstance(tactic, EQTactic)

        if not tactic.is_valid:
            return TacticJobResult(tactic, children=[])
        assert self.eq_env is not None and self.rule_env is not None
        children = apply_bwd_tactic(
            self.eq_env,
            theorem,
            tactic,
            keep_if_hyp,
            self.dataset.rule_env,
            self.rule_env.rules_a,
            self.rule_env.simp_rules,
        )
        return TacticJobResult(tactic, children=children)

    def materialize_theorem(self, th: UnMaterializedTheorem) -> Theorem:
        assert self.eq_data_env is not None
        if th.label in ["eq_bwd_rwalk_seq2seq", "eq_bwd_graph_seq2seq"]:
            seed = hash((self.rank, os.environ.get("SLURM_JOB_ID", None))) % (2 ** 32)
            theorem, _ = self.eq_data_env.get_theorem(th.label, seed=seed)
            return theorem
        return self.eq_data_env.label_to_eq[th.label]


class EQEnvGenerator(EnvGen):
    def __init__(self, dataset: EquationsDatasetConf, n_async_envs: int):
        self.dataset = dataset
        self.n_async_envs = n_async_envs

        self.worker = EQEnvWorker(self.dataset)
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
        logger.info("Closing EQEnvGenerator ...")
        if self.n_async_envs > 0:
            self.stop.set()
            for worker in self.worker_procs:
                worker.join()
        logger.info("Closed EQEnvGenerator")
