# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from typing import Tuple, List

from evariste.envs.eq.env import EquationEnv
from evariste.envs.eq.rules import eval_assert
from evariste.forward.common import (
    ForwardGraph,
    ForwardTactic,
    FwdEnvOutput,
    GenerationError,
)
from evariste.envs.eq.rules_lib import ALL_A_RULES
from evariste.forward.core.generation_errors import (
    NodeInGraph,
    MissingHyp,
    InvalidTactic,
)
from evariste.forward.core.maybe import Maybe, Fail, Ok
from evariste.forward.common import FwdEnvOutput, SimpleFwdEnvOutput
from evariste.forward.env_specifics.prover_env_specifics import AsyncForwardEnv

from evariste.backward.env.equations import EQTactic, EQTheorem
from evariste.backward.env.equations.env import apply_bwd_tactic


EqEnvOutput = SimpleFwdEnvOutput[EQTheorem, EQTactic]


@dataclass
class EqForwardTactic(ForwardTactic):
    next_node: EQTheorem
    bwd_tactic: EQTactic


class EqForwardEnv(AsyncForwardEnv):
    def __init__(self, eq_env: EquationEnv, rule_env: str):
        self.eq_env = eq_env
        self.rule_env = rule_env
        assert rule_env in ["default", "lean_real", "lean_nat", "lean_int"]

        self.to_process: List[Tuple[int, ForwardTactic, ForwardGraph]] = []

    def submit_tactic(
        self, tactic_id: int, fwd_tactic: ForwardTactic, graph: ForwardGraph
    ):
        self.to_process.append((tactic_id, fwd_tactic, graph))

    def ready_statements(self) -> List[Tuple[int, Maybe[FwdEnvOutput]]]:
        outs: List[Tuple[int, Maybe[FwdEnvOutput]]] = []
        for tid, fwd_tac, graph in self.to_process:
            try:
                env_output = self._apply_tactic(graph, fwd_tac)
            except GenerationError as err:
                result: Maybe[FwdEnvOutput] = Fail(fail=err)
            else:
                result = Ok(ok=env_output)
            outs.append((tid, result))

        self.to_process = []
        return outs

    def close(self):
        pass

    def _apply_tactic(
        self, graph: ForwardGraph, fwd_tactic: ForwardTactic
    ) -> FwdEnvOutput:

        assert isinstance(fwd_tactic, EqForwardTactic)
        next_node: EQTheorem = fwd_tactic.next_node
        tactic: EQTactic = fwd_tactic.bwd_tactic

        # sanity check
        assert isinstance(next_node, EQTheorem)
        assert isinstance(tactic, EQTactic)

        # apply tactic to next goal
        _subgoals = apply_bwd_tactic(
            self.eq_env,
            theorem=next_node,
            tactic=tactic,
            keep_if_hyp=True,  # linked with hyp in graph at the end of this function
            rule_env=self.rule_env,
        )

        # invalid tactic
        if not tactic.is_valid:
            assert tactic.is_valid is False
            assert tactic.error_msg is not None
            raise InvalidTactic(tactic.error_msg)

        # remove subgoals that are trivially true
        subgoals = []
        for sg in _subgoals:
            if eval_assert(sg.eq_node, ALL_A_RULES[self.rule_env]) is True:
                continue
            subgoals.append(sg)

        # check that the node is not already in the graph
        assert graph.generated_thms is not None
        thm_to_id = {s: i for i, s in enumerate(graph.global_hyps_and_generated)}
        if next_node in thm_to_id:
            raise NodeInGraph(next_node.eq_node.prefix())

        # check that subgoals are in the graph
        children_ids = []
        for sg in subgoals:
            assert isinstance(sg, EQTheorem)
            if sg not in thm_to_id:
                raise MissingHyp(sg.eq_node.prefix())
            children_ids.append(thm_to_id[sg])

        return EqEnvOutput(
            generated=next_node, tactic=tactic, children_ids=children_ids
        )
