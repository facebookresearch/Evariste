# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Dict


from evariste.forward.common import GenerationHistory
from evariste.forward.env_specifics.fwd_env_helper import FwdEnvHelper
from evariste.forward.env_specifics.fwd_goal_factory import ForwardGoalFactory
from evariste.forward.fwd_hl.hl_fwd_goal_factory import HLForwardGoalFactory
from evariste.forward.fwd_hl.hl_prover_env_specifics import hl_env_specifics
from evariste.forward.env_specifics.generation_annotator import GenerationAnnotator
from evariste.forward.fwd_hl.hl_fwd_env import HLForwardTactic
from evariste.forward.env_specifics.prover_env_specifics import ProverEnvSpecifics
from evariste.model.data.dictionary import Dictionary
from evariste.backward.env.hol_light.graph import (
    HLProofNode,
    HLTheorem,
)
from evariste.trainer.args import TrainerArgs


class HLFwdEnvHelper(FwdEnvHelper):
    def __init__(self, params: TrainerArgs, dico: Dictionary):
        self.params = params
        self.dico = dico

    def get_env_name(self) -> str:
        return "hl"

    def get_prover_env_specifics(self) -> ProverEnvSpecifics:
        return hl_env_specifics(self.params)

    def get_goal_factory(self) -> ForwardGoalFactory:
        return HLForwardGoalFactory(self.params)

    def get_annotator(self) -> GenerationAnnotator:
        pass


def history_to_hl_proof_nodes(gen: GenerationHistory) -> List[HLProofNode]:
    nodes: List[HLProofNode] = []
    th_to_node: Dict[HLTheorem, HLProofNode] = {}
    for step in gen.forward_steps():
        assert isinstance(step.tactic, HLForwardTactic)
        bwd_tactic = step.tactic.bwd_tactic
        children = [th_to_node[nodes[cix].theorem] for cix in step.children]
        hl_thm: HLTheorem = HLTheorem.from_tokens(step.statement.split())
        node = HLProofNode(theorem=hl_thm, tactic=bwd_tactic, children=children)
        th_to_node[node.theorem] = node
        nodes.append(node)
    return nodes
