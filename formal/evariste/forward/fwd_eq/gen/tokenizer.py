# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List
from dataclasses import dataclass

from params import Params

from evariste.backward.env.equations.graph import EQTheorem
from evariste.envs.eq.generation import GraphTrueNode
from evariste.model.data.dictionary import EOS_WORD

from evariste.forward.common import ForwardGraph, ForwardTactic
from evariste.forward.env_specifics.prover_env_specifics import FwdTokenizer
from evariste.forward.fwd_eq.gen.graph import EqGenForwardGraph
from evariste.forward.fwd_eq.gen.tactics import EqGenForwardTactic


class EqGenFwdTokenizer(FwdTokenizer):
    def tokenize_graph(self, graph: ForwardGraph) -> List[str]:
        assert isinstance(graph, EqGenForwardGraph)
        return (
            [EOS_WORD]
            + sum(
                [
                    EQTheorem(gn.node, hyps=[]).tokenize()
                    for gn in graph.nodes
                    if not isinstance(gn, GraphTrueNode)
                ],
                [],
            )
            + [EOS_WORD]
        )

    def detokenize_command(
        self, command: List[str], graph: ForwardGraph
    ) -> ForwardTactic:
        assert isinstance(graph, EqGenForwardGraph)
        return EqGenForwardTactic.detokenize(command, graph)

    def tokenize_command(
        self, fwd_tactic: EqGenForwardTactic, graph: EqGenForwardGraph
    ) -> List[str]:
        assert isinstance(fwd_tactic, EqGenForwardTactic)
        return fwd_tactic.tokenize(graph)
