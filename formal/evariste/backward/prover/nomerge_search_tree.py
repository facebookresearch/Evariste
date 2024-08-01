# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Mapping, Set, Dict, Tuple, Any

from evariste.backward.graph import Theorem, Tactic, Proof
from evariste.backward.prover.bfs_proof_graph import ProofTheoremNode, ProofTacticNode
from evariste.metrics import StatsCollection, Timer

ProofSize = int


@dataclass
class NoMergeSearchTreeStats(StatsCollection):
    n_nodes: int = -1
    n_thms: int = -1
    n_failed: int = -1
    n_solved: int = -1
    n_expanded: int = -1
    time_in_propagate_solved: Timer = field(default_factory=lambda: Timer())
    time_in_propagate_failed: Timer = field(default_factory=lambda: Timer())
    time_in_get_proof: Timer = field(default_factory=lambda: Timer())


class NoMergeSearchTree:
    def __init__(self, root_thm: Theorem):

        self.nodes: List[ProofTheoremNode] = []
        self.th_to_node_ids: Mapping[Theorem, Set[int]] = defaultdict(set)
        self.parents: Dict[int, int] = {}
        self._ancestors: Dict[int, Set[int]] = {}

        # init graph
        self.root: ProofTheoremNode = self._create_node(root_thm)

        self.solved: Set[int] = set([])
        self.failed: Set[int] = set([])
        self.n_expandable_descendants: Dict[int, int] = {}

        self.stats = NoMergeSearchTreeStats()

    def is_failed_node(self, node: ProofTheoremNode) -> bool:
        return self.is_failed(node.id)

    def is_solved_node(self, node: ProofTheoremNode) -> bool:
        return self.is_solved(node.id)

    def is_failed(self, node_id: int) -> bool:
        return node_id in self.failed

    def is_solved(self, node_id: int) -> bool:
        return node_id in self.solved

    def expand_node(
        self,
        node_id: int,
        tactics: List[Tactic],
        child_for_tac: List[List[Theorem]],
        priors: List[float],
    ):
        node = self.nodes[node_id]
        assert not node.expanded
        assert node.thm in self.th_to_node_ids

        ancestors = self._get_thm_ancestors(node_id)
        assert node.thm not in ancestors
        ancestors.add(node.thm)

        tactic_nodes = []
        for tactic, children, prior in zip(tactics, child_for_tac, priors):
            assert 0 <= prior <= 1
            if tactic.is_error():
                # print("error")
                continue
            if any(sg in ancestors for sg in children):  # cycle
                # print("cycle")
                continue
            tactic_node = ProofTacticNode(
                tactic=tactic,
                prior=prior,
                sub_goals=[self._create_node(sg) for sg in children],
            )
            # print("tactic_node", tactic_node)
            for child in tactic_node.sub_goals:
                assert child.id not in self.parents
                assert child.thm not in ancestors

                self.parents[child.id] = node.id
                # print("parent", child.id, "->", self.parents[child.id])

            tactic_nodes.append(tactic_node)

        if len(tactic_nodes) == 0:
            # print("all failed!")
            return self.set_expansion_failure(
                node_id=node_id, error="all tactics failed"
            )

        node.set_tactics(tactic_nodes)

        self._update_state(node)

        return node

    def set_expansion_failure(self, node_id: int, error: str) -> ProofTheoremNode:
        node = self.nodes[node_id]
        assert not node.expanded
        node.set_failure(error)
        assert node.is_failed_leaf
        self._update_state(node)
        return node

    def _create_node(self, thm: Theorem) -> ProofTheoremNode:
        assert isinstance(thm, Theorem)
        cur_id = len(self.nodes)
        node = ProofTheoremNode(id=cur_id, thm=thm, tactics=None)
        self.nodes.append(node)
        assert node.id not in self.th_to_node_ids[thm]
        self.th_to_node_ids[thm].add(node.id)
        return node

    def _get_thm_ancestors(self, node_id: int) -> Set[Theorem]:
        if node_id == 0:
            return set()

        if node_id not in self._ancestors:
            ancestors = []
            parent_id = self.parents.get(node_id, -1)
            while parent_id > -1:
                ancestors.append(parent_id)
                parent_id = self.parents.get(parent_id, -1)
            assert ancestors[-1] == 0
            ancestors_set = set(ancestors)
            assert len(ancestors_set) == len(ancestors)
            self._ancestors[node_id] = set(ancestors)

        thms = [self.nodes[aid].thm for aid in self._ancestors[node_id]]

        thm_to_ids = defaultdict(list)
        for aid in self._ancestors[node_id]:
            node__ = self.nodes[aid]
            thm_to_ids[node__.thm].append(node__.id)

        unique_thms = set(thms)
        # print(
        #     f"ancestors of {node_id}: {self._ancestors[node_id]} unique:{len(unique_thms)}"
        # )
        assert len(unique_thms) == len(thms), (len(unique_thms), len(thms), thm_to_ids)
        return unique_thms

    def _update_state(self, node: ProofTheoremNode):
        with self.stats.time_in_propagate_solved.timeit():
            self._propagate_solved(node)
        with self.stats.time_in_propagate_failed.timeit():
            self._propagate_failed(node)
        self._propagate_n_expandable_descendant(node)

    def _propagate_solved(self, node: ProofTheoremNode):
        assert node.expanded
        node_id = node.id
        if not self._is_solved_given_solved(node):
            return
        was_in_solved = node_id in self.solved
        if node.is_solved_leaf:
            assert not was_in_solved, node.id
        self.solved.add(node_id)
        if not was_in_solved and node_id != 0:
            parent = self.nodes[self.parents[node_id]]
            self._propagate_solved(parent)

    def _propagate_failed(self, node: ProofTheoremNode):
        assert node.expanded
        node_id = node.id
        if not self._is_failed_given_failed(node):
            return
        was_in_failed = node_id in self.failed
        if node.is_failed_leaf:
            assert not was_in_failed, node.id
        self.failed.add(node_id)
        if not was_in_failed and node_id != 0:
            parent = self.nodes[self.parents[node_id]]
            self._propagate_failed(parent)

    def _propagate_n_expandable_descendant(self, node: ProofTheoremNode):
        assert node.expanded
        node_id = node.id
        assert node_id not in self.n_expandable_descendants
        self.n_expandable_descendants[node_id] = (
            len(node.tactics) if node.tactics is not None else 0
        )

        # For the moment we don't propagate n_expandable_descendants

        # n_exp_desc_before = self.n_expandable_descendants.get(node_id, -1)
        # print("n_exp_desc_before", n_exp_desc_before, "node_id", node_id)
        # n_exp_desc_now = self._n_expandable_descendants_given_n_expandable_descendants(
        #     node
        # )
        # if node.is_failed_leaf or node.is_solved_leaf:
        #     assert n_exp_desc_now != n_exp_desc_before, node_id
        #
        # if n_exp_desc_now != n_exp_desc_before:
        #     self.n_expandable_descendants[node_id] = n_exp_desc_now
        #     if node_id != 0:
        #         parent = self.nodes[self.parents[node_id]]
        #         self._propagate_n_expandable_descendant(parent)

    def _is_solved_given_solved(self, node: ProofTheoremNode) -> bool:
        if node.is_failed_leaf_or_not_expanded:
            return False
        if node.is_solved_leaf:
            return True
        assert node.tactics is not None  # mypy
        return any(
            all(sg.id in self.solved for sg in t.sub_goals) for t in node.tactics
        )

    def _is_failed_given_failed(self, node: ProofTheoremNode) -> bool:
        if node.is_failed_leaf:
            return True
        assert not node.is_failed_leaf_or_not_expanded
        assert node.tactics is not None  # mypy
        return all(self.is_failed_tactic(t) for t in node.tactics)

    def is_failed_tactic(self, tactic: ProofTacticNode) -> bool:
        return any(self.is_failed_node(sg) for sg in tactic.sub_goals)

    def is_solving_tactic(self, tactic: ProofTacticNode) -> bool:
        return all(self.is_solved_node(sg) for sg in tactic.sub_goals)

    def _n_expandable_descendants_given_n_expandable_descendants(
        self, node: ProofTheoremNode
    ) -> int:
        node_id = node.id
        if node_id in self.failed or node_id in self.solved:
            return 0
        assert not node.is_failed_leaf_or_not_expanded
        assert node.tactics is not None  # mypy
        return sum(
            sum(self.n_expandable_descendants[sg.id] for sg in t.sub_goals)
            for t in node.tactics
        )

    def get_root_proof_and_proof_size(self) -> Tuple[Proof, ProofSize]:
        def _build_proof(node: ProofTheoremNode) -> Tuple[Proof, ProofSize]:
            assert self.is_solved_node(node)
            tacs = node.get_tactics()
            solving = [t for t in tacs if self.is_solving_tactic(t)]
            assert len(solving) >= 1
            selected_tac = solving[0]
            child_proofs: List[Proof] = []
            proof_size: int = 1
            for sg in selected_tac.sub_goals:
                child_proof, child_psize = _build_proof(sg)
                child_proofs.append(child_proof)
                proof_size += child_psize

            return (node.thm, selected_tac.tactic, child_proofs), proof_size

        with self.stats.time_in_get_proof.timeit():
            return _build_proof(self.root)

    def update_stats(self):
        self.stats.n_nodes = len(self.nodes)
        self.stats.n_thms = len({n.thm for n in self.nodes})
        self.stats.n_solved = len(self.solved)
        self.stats.n_failed = len(self.failed)
        self.stats.n_expanded = len([n for n in self.nodes if n.expanded])
