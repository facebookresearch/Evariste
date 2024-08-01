# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Set, Dict, Optional

from evariste.backward.env.core import EnvExpansion

from evariste.backward.env.core import BackwardGoal
from evariste.backward.graph import Proof, Theorem, Tactic, UnMaterializedTheorem
from evariste.backward.prover.core import ProofHandler, ProofResult


class HasCycle(Exception):
    pass


MAX_GREEDY_DEPTH = 2000


class TooDeep(Exception):
    pass


class ProofTreeNode:
    def __init__(self, id: int):
        self.children: List[ProofTreeNode] = []
        self.id = id
        self.tactic: Optional[Tactic] = None

    def add_child(self, child: "ProofTreeNode"):
        self.children.append(child)


class GreedyProofTreeHandler(ProofHandler):
    """
    ProofHandler used for handling a Greedy proof strategy.
    """

    def __init__(self, goal: BackwardGoal):
        super().__init__(goal)
        self.th_to_node: Dict[Theorem, ProofTreeNode] = dict()
        self.th_to_node[goal.theorem] = ProofTreeNode(0)
        self.id_to_th: List[Theorem] = [goal.theorem]
        self.leaves: List[Theorem] = [goal.theorem]
        self.root = goal.theorem
        self.fail = False

    def is_ready(self):
        if isinstance(self.root, UnMaterializedTheorem):
            return False
        return True

    def send_materialized(self, th: Theorem):
        self.th_to_node.pop(self.root)
        self.th_to_node[th] = ProofTreeNode(0)
        self.root = th
        self.id_to_th = [th]
        self.leaves = [th]

    def get_theorems_to_expand(self) -> Optional[List[Theorem]]:
        if not self.is_ready():
            return None
        return self.leaves

    def send_env_expansions(self, tactics: List[EnvExpansion]) -> None:
        if any(exp.is_error for exp in tactics):
            self.fail = True
        else:
            for exp in tactics:
                assert exp.tactics is not None
                assert len(exp.tactics) == 1, len(exp.tactics)
                if not exp.tactics[0].is_valid:
                    self.fail = True

            old_leaves = self.get_theorems_to_expand()
            assert old_leaves is not None and len(old_leaves) == len(tactics)
            self.leaves = []

            new_dest = []
            for src, exp in zip(old_leaves, tactics):
                assert exp.tactics is not None and exp.child_for_tac is not None
                assert len(exp.tactics) == 1 and len(exp.child_for_tac) == 1
                self.set_tactic(src, exp.tactics[0])
                for child in exp.child_for_tac[0]:
                    self.add_edge(src, child)
                    new_dest.append(child)
            try:
                self.check_cycle_and_depth()
            except (HasCycle, TooDeep):
                self.fail = True
        self.done = self.is_done() or self.fail

    def add_edge(self, src: Theorem, dest: Theorem):
        src_node = self.th_to_node[src]
        if dest not in self.th_to_node:
            dest_node = ProofTreeNode(len(self.th_to_node))
            self.id_to_th.append(dest)
            self.th_to_node[dest] = dest_node
            self.leaves.append(dest)
        else:
            dest_node = self.th_to_node[dest]
        src_node.add_child(dest_node)

    def set_tactic(self, src: Theorem, tactic: Tactic):
        src_node = self.th_to_node[src]
        assert src_node.tactic is None, "setting tactic multiple times"
        src_node.tactic = tactic

    def check_cycle_and_depth(self):
        being_seen: Set[int] = set()

        def walk(node: "ProofTreeNode", cur_depth: int):
            if node.id in being_seen:
                raise HasCycle
            if cur_depth > MAX_GREEDY_DEPTH:
                raise TooDeep
            being_seen.add(node.id)
            for child in node.children:
                walk(child, cur_depth + 1)
            being_seen.remove(node.id)

        walk(self.th_to_node[self.root], 0)

    def is_done(self):
        return len(self.leaves) == 0

    def get_result(self) -> ProofResult:
        if self.done and not self.fail:
            return ProofResult(proof=self.get_proof(), goal=self.goal, exception=None)
        else:
            return ProofResult(proof=None, goal=self.goal, exception=None)

    def get_proof(self) -> Proof:
        def build_proof(cur_id):
            th = self.id_to_th[cur_id]
            node = self.th_to_node[th]
            return th, node.tactic, [build_proof(c.id) for c in node.children]

        return build_proof(0)
