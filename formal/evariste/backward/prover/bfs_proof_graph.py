# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Tuple, Set, Dict, Iterator

import numpy as np

from evariste.backward.graph import Theorem, Proof, Tactic


@dataclass
class ProofTheoremNode:
    id: int
    thm: Theorem
    tactics: Optional[List["ProofTacticNode"]] = None
    error: Optional[str] = None

    def set_tactics(self, tactics: List["ProofTacticNode"]):
        assert self.tactics is None
        assert len(tactics) > 0
        self.tactics = tactics

    def get_tactics(self) -> List["ProofTacticNode"]:
        assert self.tactics is not None
        return self.tactics

    def set_failure(self, error: str):
        assert self.error is None
        self.error = error

    @property
    def is_failed_leaf_or_not_expanded(self) -> bool:
        return self.tactics is None

    @property
    def is_failed_leaf(self):
        return self.error is not None

    @property
    def is_solved_leaf(self):
        return _is_proved_given_proven(node=self, proven=set())

    @property
    def expanded(self):
        return self.is_failed_leaf or self.tactics is not None


@dataclass
class ProofTacticNode:
    tactic: Tactic
    sub_goals: List[ProofTheoremNode]
    prior: float = -1.0


class ProofGraph:
    def __init__(self, root_thm: Theorem):

        # state updated automatically when setting an expansion
        self.th_to_node: Dict[Theorem, ProofTheoremNode] = {}
        self.parents: Dict[Theorem, Set[Theorem]] = defaultdict(set)
        self.proved_leafs: Set[Theorem] = set()
        self.failed_leafs: Set[Theorem] = set()
        self.leafs: Set[Theorem] = set()

        # state updated manually
        # (protected not to be accessed when not up to date)
        self._proved: Set[Theorem] = set()
        self._failed: Set[Theorem] = set()
        self._leafs_worth_to_expand: Set[Theorem] = set()
        self._min_proof_sizes: Dict[Theorem, int] = {}
        self._best_tactic_ids: Dict[Theorem, int] = {}

        # init graph
        self.root = self._get_or_create_node(root_thm)
        self.is_up_to_date: bool = False

    def set_expansion(
        self, thm: Theorem, tactics: List[Tactic], child_for_tac: List[List[Theorem]],
    ) -> ProofTheoremNode:
        self.is_up_to_date = False

        assert thm in self.th_to_node
        assert thm in self.leafs

        node = self.th_to_node[thm]

        tactic_nodes = []
        for tactic, children in zip(tactics, child_for_tac):
            if tactic.is_error():
                continue
            tactic_node = ProofTacticNode(
                tactic=tactic,
                sub_goals=[self._get_or_create_node(sg) for sg in children],
            )
            for sg in children:
                self.parents[sg].add(thm)

            tactic_nodes.append(tactic_node)

        if len(tactic_nodes) == 0:
            return self.set_expansion_failure(node.thm, error="all tactics failed")

        node.set_tactics(tactic_nodes)
        assert node.tactics is not None
        if node.is_solved_leaf:
            self.proved_leafs.add(node.thm)
        self.leafs.remove(node.thm)

        return node

    def set_expansion_failure(self, thm: Theorem, error: str) -> ProofTheoremNode:
        self.is_up_to_date = False
        assert thm in self.th_to_node
        node = self.th_to_node[thm]
        node.set_failure(error)
        assert node.is_failed_leaf
        self.failed_leafs.add(thm)
        self.leafs.remove(thm)
        return node

    def update_state(self):
        self._proved = find_proved(
            proved_leafs=self.proved_leafs,
            parents=self.parents,
            th_to_node=self.th_to_node,
        )

        self._failed = find_failed(
            failed_leafs=self.failed_leafs,
            parents=self.parents,
            th_to_node=self.th_to_node,
        )

        self._leafs_worth_to_expand = find_leafs_worth_to_expand(
            root_th=self.root.thm,
            failed=self._failed,
            leafs=self.leafs,
            th_to_node=self.th_to_node,
        )

        self._min_proof_sizes, self._best_tactic_ids = find_min_proofs(
            proved_leafs=self.proved_leafs,
            parents=self.parents,
            th_to_node=self.th_to_node,
        )

        assert (
            self._proved
            == set(self._min_proof_sizes.keys())
            == set(self._best_tactic_ids.keys())
        ), (
            self._min_proof_sizes.keys(),
            self._proved,
        )
        self.is_up_to_date = True

    def is_proved(self, thm: Theorem) -> bool:
        assert self.is_up_to_date
        return thm in self._proved

    def is_failed(self, thm: Theorem) -> bool:
        assert self.is_up_to_date
        return thm in self._failed

    def is_leaf_worth_to_expand(self, thm: Theorem) -> bool:
        assert self.is_up_to_date
        assert thm in self.leafs
        return thm in self._leafs_worth_to_expand

    def proof(self, thm: Theorem) -> Proof:
        assert self.is_up_to_date

        path: Set[Theorem] = set()

        proved_cache: Dict[Theorem, Proof] = {}

        def _get_proof(this_thm: Theorem) -> Proof:

            assert (
                this_thm not in path
            ), f"{this_thm}\n{len(path)}\n{self._min_proof_sizes[this_thm]}"

            node = self.th_to_node[this_thm]

            assert this_thm in self._proved

            if this_thm in proved_cache:
                return proved_cache[this_thm]

            path.add(this_thm)
            assert this_thm in self._min_proof_sizes, this_thm
            assert this_thm in self._best_tactic_ids, this_thm
            assert node.tactics is not None
            best_tactic_node = node.tactics[self._best_tactic_ids[this_thm]]
            proof = (
                this_thm,
                best_tactic_node.tactic,
                [_get_proof(sg.thm) for sg in best_tactic_node.sub_goals],
            )
            path.remove(this_thm)
            proved_cache[this_thm] = proof
            return proof

        return _get_proof(thm)

    def proof_size(self, thm: Theorem) -> int:
        assert self.is_up_to_date
        assert thm in self._proved
        assert thm in self._min_proof_sizes
        return self._min_proof_sizes[thm]

    def _get_or_create_node(self, thm: Theorem) -> ProofTheoremNode:
        if thm not in self.th_to_node:
            cur_id = len(self.th_to_node)
            node = ProofTheoremNode(id=cur_id, thm=thm, tactics=None)
            self.th_to_node[thm] = node
            self.leafs.add(thm)
        return self.th_to_node[thm]

    def get_graph_stats(self) -> Dict[str, float]:
        assert self.is_up_to_date
        return {
            "n_nodes": len(self.th_to_node),
            "n_proved": len(self._proved),
            "n_failed": len(self._failed),
            "n_leafs_proved": len(self.proved_leafs),
            "n_leafs_failed": len(self.failed_leafs),
            "n_leafs": len(self.leafs),
            "n_leafs_worth_to_expand": len(self._leafs_worth_to_expand),
            "avg_n_parents_by_thm": float(
                np.mean([len(v) for v in self.parents.values()])
            )
            if self.parents
            else 0.0,
        }


def find_proved(
    proved_leafs: Set[Theorem],
    parents: Dict[Theorem, Set[Theorem]],
    th_to_node: Dict[Theorem, ProofTheoremNode],
) -> Set[Theorem]:
    proved: Set[Theorem] = set()
    newly_proved: Set[Theorem] = set(proved_leafs)  # copy

    while newly_proved:
        th = newly_proved.pop()
        proved.add(th)
        for parent_th in parents[th]:
            if parent_th in proved or parent_th in newly_proved:
                continue
            parent = th_to_node[parent_th]
            if _is_proved_given_proven(node=parent, proven=proved):
                newly_proved.add(parent_th)

    return proved


def find_failed(
    failed_leafs: Set[Theorem],
    parents: Dict[Theorem, Set[Theorem]],
    th_to_node: Dict[Theorem, ProofTheoremNode],
) -> Set[Theorem]:
    failed: Set[Theorem] = set()
    newly_failed: Set[Theorem] = set(failed_leafs)  # copy

    while newly_failed:
        th = newly_failed.pop()
        failed.add(th)
        for parent_th in parents[th]:
            if parent_th in failed or parent_th in newly_failed:
                continue
            parent = th_to_node[parent_th]
            if _is_failed_given_failed(node=parent, failed=failed):
                newly_failed.add(parent_th)

    return failed


def find_min_proofs(
    proved_leafs: Set[Theorem],
    parents: Dict[Theorem, Set[Theorem]],
    th_to_node: Dict[Theorem, ProofTheoremNode],
) -> Tuple[Dict[Theorem, float], Dict[Theorem, int]]:
    # float because of math.inf
    # not using defaultdict with math.inf as default
    min_proof_sizes: Dict[Theorem, float] = {}
    best_tactic_ids: Dict[Theorem, int] = {}

    to_process: Set[Theorem] = set()
    for thm in proved_leafs:
        node = th_to_node[thm]
        result = _best_tactic_id_and_proof_size(node, min_proof_sizes)
        assert result is not None
        best_tactic_id, proof_size = result
        assert proof_size == 1
        min_proof_sizes[thm] = proof_size
        best_tactic_ids[thm] = best_tactic_id
        to_process.add(thm)

    while to_process:
        thm = to_process.pop()
        for parent_thm in parents[thm]:
            parent_node = th_to_node[parent_thm]
            result = _best_tactic_id_and_proof_size(parent_node, min_proof_sizes)
            if result is None:
                continue
            best_tactic_id, proof_size = result
            if proof_size < min_proof_sizes.get(parent_thm, math.inf):
                min_proof_sizes[parent_thm] = proof_size
                best_tactic_ids[parent_thm] = best_tactic_id
                to_process.add(parent_thm)

    return min_proof_sizes, best_tactic_ids


def find_leafs_worth_to_expand(
    root_th: Theorem,
    failed: Set[Theorem],
    leafs: Set[Theorem],
    th_to_node: Dict[Theorem, ProofTheoremNode],
) -> Set[Theorem]:
    """ Find subset of not expanded nodes that are worth to expand
    (that can belong to the proof of the root)"""
    worth: Set[Theorem] = set()
    visited: Set[Theorem] = set()

    if root_th in failed:
        return worth

    queue: List[Theorem] = [root_th]
    visited.add(root_th)

    while queue:
        thm = queue.pop(0)
        assert thm not in failed
        assert thm in visited
        if thm in leafs:
            worth.add(thm)
            continue
        node = th_to_node[thm]
        one_tactic_ok: bool = False
        assert node.tactics is not None
        for tactic in node.tactics:
            if _is_dead_tactic(tactic, failed):
                continue
            one_tactic_ok = True
            for sg in tactic.sub_goals:
                if sg.thm not in visited:
                    visited.add(sg.thm)
                    queue.append(sg.thm)
        assert one_tactic_ok  # we are sure that there is at least one path,
        # if not the root was supposed to be failed

    assert worth.issubset(leafs)
    return worth


def _is_proved_given_proven(node: ProofTheoremNode, proven: Set[Theorem]):
    if node.is_failed_leaf:
        return False
    assert node.expanded
    assert node.tactics is not None  # mypy
    return any(all(sg.thm in proven for sg in t.sub_goals) for t in node.tactics)


def _is_failed_given_failed(node: ProofTheoremNode, failed: Set[Theorem]):
    if node.is_failed_leaf:
        return True
    assert not node.is_failed_leaf_or_not_expanded
    assert node.tactics is not None  # mypy
    return all(any(sg.thm in failed for sg in t.sub_goals) for t in node.tactics)


def _best_tactic_id_and_proof_size(
    node: ProofTheoremNode, min_proof_sizes: Dict[Theorem, float]
) -> Optional[Tuple[int, int]]:
    assert node.tactics is not None  # mypy
    sizes: List[float] = [
        1.0 + sum(min_proof_sizes.get(n.thm, math.inf) for n in tac.sub_goals)
        for tac in node.tactics
    ]

    if min(sizes) == math.inf:
        return None
    else:
        best_tactic_id = int(np.argmin(sizes))
        return best_tactic_id, int(sizes[best_tactic_id])


def _is_dead_tactic(tactic: ProofTacticNode, failed: Set[Theorem]) -> bool:
    return any(sg.thm in failed for sg in tactic.sub_goals)


############################################
# deprecated helpers
############################################


def bfs(root: ProofTheoremNode) -> Iterator[ProofTheoremNode]:
    visited: Set[Theorem] = {root.thm}
    queue: List[ProofTheoremNode] = [root]
    while queue:
        node = queue.pop(0)
        yield node

        if node.is_failed_leaf_or_not_expanded:
            continue

        assert node.tactics is not None  # mypy
        for tactic_node in node.tactics:
            for children in tactic_node.sub_goals:
                if children.thm in visited:
                    continue
                visited.add(children.thm)
                queue.append(children)


def compute_graph_structure(root: ProofTheoremNode):
    th_to_node = {}
    failed_leafs = set()
    not_expanded = set()
    proved_leafs = set()
    parents = defaultdict(set)

    for node in bfs(root):
        th_to_node[node.thm] = node

        if node.is_failed_leaf:
            failed_leafs.add(node.thm)
            continue
        if not node.expanded:
            not_expanded.add(node.thm)
            continue

        assert not node.is_failed_leaf_or_not_expanded

        if node.is_solved_leaf:
            proved_leafs.add(node.thm)
        assert node.tactics is not None  # mypy
        for tactic_node in node.tactics:
            for children in tactic_node.sub_goals:
                parents[children.thm].add(node.thm)
