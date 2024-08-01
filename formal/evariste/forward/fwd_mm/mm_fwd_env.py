# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import itertools
from typing import Optional, Dict, List, Tuple, Any

from evariste.envs.mm.env import MetamathEnv
from evariste.envs.mm.assertion import Assertion
from evariste.forward.common import ForwardGraph, ForwardTactic, GenerationError
from evariste.forward.common import FwdEnvOutput
from evariste.forward.core.generation_errors import (
    InvalidLabel,
    ForbiddenLabel,
    NodeInGraph,
    MissingHyp,
    WrongSubsKeys,
    SyntacticGenError,
    DisjointGenError,
    GoalDisjointError,
    ChildrenMismatch,
)
from evariste.forward.core.maybe import Maybe, Fail, Ok
from evariste.forward.env_specifics.prover_env_specifics import (
    Statement,
    ChildrenIds,
    ForwardEnv,
    AsyncForwardEnv,
)
from evariste.forward.fwd_mm.mm_helpers import (
    get_mand_vars,
    MMForwardTactic,
    MMEnvInfo,
)
from evariste.forward.fwd_mm.mm_fwd_tasks import MMFwdFormat
from evariste.syntax.parser import Parser, ParseError


class MMForwardEnv(ForwardEnv):
    def __init__(
        self,
        mm_env: MetamathEnv,
        use_f_hyps: bool,
        parser: Optional[Parser],
        check_syntactic: bool = True,
    ):
        self.use_f_hyps = use_f_hyps
        self.mm_env = mm_env
        self.active_vars = set.union(*(frame.v for frame in self.mm_env.fs.frames))

        self.debug = False

        self.use_f_hyps = use_f_hyps
        self.use_ptrs = False
        self.embseq = False

        self.check_syntactic = check_syntactic
        self.parser = parser
        if self.check_syntactic:
            assert self.parser is not None

    def apply_tactic(
        self, graph: ForwardGraph, tactic: MMForwardTactic
    ) -> Tuple[Statement, ChildrenIds, MMEnvInfo]:
        nodes = graph.nodes
        token, subst, maybe_children_ids = (
            tactic.label,
            tactic.substitutions,
            tactic.children_ids,
        )
        # $f hypothesis or assertion
        if token not in self.mm_env.labels:
            if not self.use_f_hyps:
                raise InvalidLabel(f"Invalid token: {token}")
            elif token not in self.mm_env.fs.frames[0].f_labels:
                raise InvalidLabel(f"Invalid token: {token}")
        if graph.forbidden and token in graph.forbidden:
            raise ForbiddenLabel(f"Forbidden label: {token}")
        # $f hypothesis
        children_ids = []
        new_disjoints = set()
        if self.use_f_hyps and (token in self.mm_env.fs.frames[0].f_labels):
            if len(subst) > 0:
                raise WrongSubsKeys(
                    f"Unexpected substitutions generated for {token}: {subst}"
                )
            new_node_str = " ".join(self.mm_env.fs.frames[0].f_labels[token])
            if new_node_str in nodes:
                raise NodeInGraph(f"{new_node_str} already in graph")

        # $a or $p assertion
        else:
            label_type, _ = self.mm_env.labels[token]
            new_node_str, children_ids = self._create_new_node(nodes, token, subst)

            # duplicate node
            if new_node_str in nodes:
                raise NodeInGraph(f"{new_node_str} already in graph")

            _, assertion = self.mm_env.labels[token]
            for x, y in assertion.mand_disj:
                x_vars = set(subst[x].split()) & self.active_vars
                y_vars = set(subst[y].split()) & self.active_vars
                for sub_x, sub_y in itertools.product(x_vars, y_vars):
                    if sub_x == sub_y:
                        raise DisjointGenError(
                            f"Disjoint not respected for label '{token}'"
                        )
                    new_disj = (min(sub_x, sub_y), max(sub_x, sub_y))
                    new_disjoints.add(new_disj)

            if graph.fwd_goal.mand_disj is not None:
                fwd_goal = graph.fwd_goal
                goal_mand_vars = get_mand_vars(fwd_goal, self.active_vars)
                goal_mand_disj = fwd_goal.mand_disj
                assert goal_mand_disj is not None
                mand_disj = {
                    (x, y)
                    for x, y in new_disjoints
                    if x in goal_mand_vars and y in goal_mand_vars
                }
                if not mand_disj.issubset(goal_mand_disj):
                    raise GoalDisjointError(
                        f"Goal mand disjoints not respected: received {mand_disj},"
                        f" expected: {goal_mand_disj}"
                    )

            # we check only once everything is ok
            if self.check_syntactic:
                assert isinstance(assertion, Assertion)
                f_hyps = assertion.f_hyps
                for var_type, var_name in f_hyps:
                    expression = [var_type] + subst[var_name].split()
                    try:
                        _ = self.parser.parse(expression)
                    except ParseError:
                        raise SyntacticGenError(
                            f"syntactic error: {' '.join(expression)}"
                        )
        if maybe_children_ids is not None and children_ids != maybe_children_ids:
            raise ChildrenMismatch(
                msg=f"predicted={maybe_children_ids}, subst gave={children_ids}"
            )
        return new_node_str, children_ids, MMEnvInfo(new_disjoints=new_disjoints)

    def _create_new_node(
        self, nodes: List[str], token: str, subst: Dict[str, str]
    ) -> Tuple[str, List[int]]:
        """
        TODO: update docstring
        Take as input the current graph, the predicted token and
        the children of the new node, and retrieve the associated
        substitutions to apply to create the new node. Return None
        if we fail to find substitutions that match hypotheses.
        """
        no_syntactic = not self.use_f_hyps
        mm_env = self.mm_env

        # retrieve assertion, hypotheses, and variables to substitute
        _, assertion = mm_env.labels[token]
        f_hyps = list(assertion["f_hyps"])
        e_hyps = list(assertion["e_hyps"])
        var_names = set([var_name for _, var_name in assertion["f_hyps"]])

        # check that variables to substitute are all provided
        if set(subst.keys()) != var_names:
            raise WrongSubsKeys(
                f'Different set of substitutions. Expected: "{var_names}" but found "{set(subst.keys())}"'
            )

        # build hypotheses
        f_hyps = [" ".join(subst.get(tok, tok) for tok in hyp) for hyp in f_hyps]
        e_hyps = [" ".join(subst.get(tok, tok) for tok in hyp) for hyp in e_hyps]

        available_children = set(nodes)

        children_ids = []
        if no_syntactic:
            for hyp in e_hyps:
                if hyp not in available_children:
                    raise MissingHyp(f'$e hypothesis "{hyp}" not found in graph!')
                children_ids.append(nodes.index(hyp))
        else:
            for hyp in f_hyps:
                if hyp not in available_children:
                    raise MissingHyp(f'$f hypothesis "{hyp}" not found in graph!')
                children_ids.append(nodes.index(hyp))
            for hyp in e_hyps:
                if hyp not in available_children:
                    raise MissingHyp(f'$e hypothesis "{hyp}" not found in graph!')
                children_ids.append(nodes.index(hyp))
        # create new node
        new_node = [subst.get(tok, tok) for tok in assertion["tokens"]]

        return " ".join(new_node), children_ids

    @classmethod
    def from_trainer_args(cls, params: Any, mm_env: MetamathEnv) -> "MMForwardEnv":
        from evariste.trainer.args import TrainerArgs  # cyclic imports
        from evariste.syntax.parser import get_parser

        assert isinstance(params, TrainerArgs)

        fmt = MMFwdFormat.from_trainer_args(params)
        parser = get_parser(params.mm.dataset.parser)
        return MMForwardEnv(mm_env=mm_env, use_f_hyps=False, parser=parser)


class DebugMMAsyncFwdEnv(AsyncForwardEnv):
    """
    Fake async env to test async fwd env pipeline
    """

    def __init__(self, fwd_env: MMForwardEnv):
        self.fwd_env = fwd_env

        self.waiting = []

    def submit_tactic(
        self, tactic_id: int, fwd_tactic: ForwardTactic, graph: ForwardGraph
    ):
        self.waiting.append((tactic_id, fwd_tactic, graph))

    def ready_statements(self) -> List[Tuple[int, Maybe[FwdEnvOutput]]]:
        import random

        to_pop = set()
        for i in range(len(self.waiting)):
            if random.random() > 0.5:
                to_pop.add(i)

        results = []
        for i in to_pop:
            tactic_id, fwd_tactic, graph = self.waiting[i]
            try:
                outputs = self.fwd_env.apply_tactic(graph, fwd_tactic)
            except GenerationError as err:
                result = Fail(err)
            else:
                result = Ok(outputs)
            results.append((tactic_id, result))
        self.waiting = [w for i, w in enumerate(self.waiting) if i not in to_pop]
        return results

    def close(self):
        pass
