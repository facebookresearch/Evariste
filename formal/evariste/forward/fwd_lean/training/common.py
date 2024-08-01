# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field
from typing import List, Dict, Any, Set

from evariste.backward.env.lean.graph import LeanTheorem, LeanTactic
from evariste.forward.common import ProofNode
from evariste.forward.training.graph_sampler import GraphTrainingSample


LeanProofNode = ProofNode[LeanTheorem, LeanTactic]
LeanGraphTrainingSample = GraphTrainingSample[LeanProofNode]


@dataclass
class TacticAndChildren:
    # find better name
    tactic: LeanTactic
    children: List["LeanMetaProofNode"]


@dataclass
class LeanMetaProofNode:
    """
    Meta dag for proof

    Meta dag since it can contains multiple possible dags for a proof
    (one node can have multiple tactics, each pointig to a different list
    of subgoals / children
    """

    thm: LeanTheorem
    tactics_and_children: List["TacticAndChildren"]
    uncompleted: bool = False  # we can allow some nodes that don't have solving
    # tactics to increase the number of samples
    _cache: Dict[str, Any] = field(default_factory=lambda: {})

    def max_depth(self):
        """Max depth for all possible sub dags"""
        all_children = [
            child for tac in self.tactics_and_children for child in tac.children
        ]
        if len(all_children) == 0:
            return 0
        else:
            return 1 + max([c.max_depth() for c in all_children])

    def n_possible_dags(self):
        """Counts the number of possible proof dags"""
        n_possible = 0
        assert self.tactics_and_children
        for tac in self.tactics_and_children:
            tac_possible = 1
            for child in tac.children:
                tac_possible *= child.n_possible_dags()
            n_possible += tac_possible
        return n_possible

    def child_theorems(self) -> Set[LeanTheorem]:
        """Returns all the child theorem (including itself) for a node.
        All these possible theorems are not forced to belong to a same proof dag
        """
        key = "child_theorems"
        if key in self._cache:
            return self._cache[key]
        thms: Set[LeanTheorem] = {self.thm}
        all_children = [
            child for tac in self.tactics_and_children for child in tac.children
        ]
        for child in all_children:
            thms.update(child.child_theorems())
        self._cache[key] = thms
        return thms


def is_complete(node: LeanMetaProofNode, allow_global_hyps: bool = False) -> bool:
    """
    Check that a node as at least of possible proof
    TODO: add caching if too slow
    """
    if node.uncompleted:  # considered as global hyp
        assert allow_global_hyps
        return True

    if len(node.tactics_and_children) == 0:
        return False
    return any(
        all(
            is_complete(child, allow_global_hyps=allow_global_hyps)
            for child in tactic.children
        )
        for tactic in node.tactics_and_children
    )  # ok if one of the tactic is ok


class CycleDetected(ValueError):
    pass


def detect_cycles(root: LeanMetaProofNode):
    """For the moment we consider as cyclic a graph where one of the possible dag is
    cyclic. So we throw all the graph even if some proof dag could be extracted from it.

    We should design a algorithm that remove tactics (if possible) until reaching a dag.

    Note: for the moment I detected only 2 cycles within Lean proofs (after removing
    tactics where goal is within subgoals)
    """

    path: Set[LeanTheorem] = set()

    def _detect_cycle(node: LeanMetaProofNode):
        if node.thm in path:
            raise CycleDetected
        assert node.thm not in path
        path.add(node.thm)
        for tac in node.tactics_and_children:
            for child in tac.children:
                _detect_cycle(child)
        path.remove(node.thm)

    has_cycle = False
    try:
        _detect_cycle(root)
    except CycleDetected:
        has_cycle = True
    return has_cycle
