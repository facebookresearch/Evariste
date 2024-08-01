# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List

from evariste.backward.graph import Token
from evariste.backward.env.hol_light.graph import HLTactic, HLTheorem
from evariste.forward.env_specifics.prover_env_specifics import FwdTokenizer
from evariste.forward.training.helpers import detokenize_command
from evariste.model.data.dictionary import (
    EOS_WORD,
    B_NODE_WORD,
    E_NODE_WORD,
    B_GOAL_WORD,
    E_GOAL_WORD,
)
from evariste.forward.fwd_hl.common import HLForwardGraph
from evariste.forward.fwd_hl.hl_fwd_env import HLForwardTactic


# TODO: most of this code is common with eq_fwd_tokenizer


class HLFwdTokenizer(FwdTokenizer):
    def __init__(self, is_generation: bool = False):
        self.is_generation = is_generation

    def tokenize_graph(self, graph: HLForwardGraph) -> List[Token]:
        fwd_goal_tokens = graph.fwd_goal.thm.tokenize()
        if self.is_generation:
            assert fwd_goal_tokens == [B_GOAL_WORD, E_GOAL_WORD]
            tokens = []
        else:
            assert len(fwd_goal_tokens) > 2
            tokens = fwd_goal_tokens
        for thm in graph.generated_thms:
            tokens.append(B_NODE_WORD)
            tokens.extend(thm.tokenize()[1:-1])  # remove B/E_GOAL_WORD
            tokens.append(E_NODE_WORD)
        return [EOS_WORD, *tokens, EOS_WORD]

    def detokenize_command(
        self, command: List[str], graph: HLForwardGraph
    ) -> HLForwardTactic:
        """
        Retrieve the next node and the associated tactic.
        """

        next_node, tactic = detokenize_command(
            command, tactic_cls=HLTactic, theorem_cls=HLTheorem
        )
        return HLForwardTactic(next_node=next_node, bwd_tactic=tactic)
