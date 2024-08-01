# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List

from evariste.backward.env.equations.graph import EQTheorem, EQRuleTactic
from evariste.forward.common import ForwardGraph
from evariste.forward.env_specifics.prover_env_specifics import FwdTokenizer
from evariste.forward.fwd_eq.eq_fwd_env import EqForwardTactic
from evariste.forward.training.helpers import tokenize_fwd_graph, detokenize_command
from evariste.model.data.dictionary import (
    EOS_WORD,
    B_GOAL_WORD,
    E_GOAL_WORD,
    B_NODE_WORD,
    E_NODE_WORD,
)


class EqFwdTokenizer(FwdTokenizer):
    def __init__(self, is_generation: bool = False):
        self.is_generation = is_generation

    def tokenize_graph(self, graph: ForwardGraph) -> List[str]:
        return tokenize_fwd_graph(
            goal=graph.fwd_goal.thm,
            graph=graph.global_hyps_and_generated,
            include_goal=not self.is_generation,
            tokenize_thm_fn=tokenize_eq_thm,
            tokenize_goal_fn=tokenize_eq_thm,
        )

    def detokenize_command(
        self, command: List[str], graph: ForwardGraph
    ) -> EqForwardTactic:
        """
        Retrieve the next node and the associated tactic.
        """
        next_node, tactic = detokenize_command(
            command,
            tactic_cls=EQRuleTactic,
            theorem_cls=EQTheorem,
            next_node_first=True,
            last_token=EOS_WORD,
            parse_thm_fn=_make_parse_eq_fn(graph=graph),
        )

        return EqForwardTactic(next_node=next_node, bwd_tactic=tactic)

    @staticmethod
    def tokenize_command(fwd_tactic: EqForwardTactic) -> List[str]:
        return [
            EOS_WORD,
            B_NODE_WORD,
            *fwd_tactic.next_node.eq_node.prefix_tokens(),
            E_NODE_WORD,
            *fwd_tactic.bwd_tactic.tokenize(),
            EOS_WORD,
        ]


def tokenize_eq_thm(thm: EQTheorem) -> List[str]:
    # We don't tokenize hyps
    return thm.eq_node.prefix().split()


def _make_parse_eq_fn(graph: ForwardGraph):
    # As we don't tokenize hyps we add them from the global_hyps in the graph
    def parse_eq_thm(cmd: List[str]) -> EQTheorem:
        thm_without_hyps = EQTheorem.from_tokens([B_GOAL_WORD, *cmd, E_GOAL_WORD])
        hyps = graph.fwd_goal.global_hyps
        assert hyps is not None
        eq_hyps = [hyp.eq_node for hyp in hyps]
        thm = EQTheorem(node=thm_without_hyps.eq_node, hyps=eq_hyps)
        return thm

    return parse_eq_thm


def tokenize_graph(goal: str, graph: List[str], is_generation: bool):
    if is_generation:
        assert goal == ""
        tokens = []
    else:
        assert goal != ""
        tokens = [B_GOAL_WORD, *goal.split(), E_GOAL_WORD]
    for node in graph:
        tokens.append(B_NODE_WORD)
        tokens.extend(node.split())
        tokens.append(E_NODE_WORD)
    return tokens
