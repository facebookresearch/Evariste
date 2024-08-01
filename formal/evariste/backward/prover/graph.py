# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import (
    Optional,
    Tuple,
    List,
    Set,
    Dict,
    Deque,
    Callable,
    Sequence,
    TypeVar,
    Generic,
)
from collections import deque, defaultdict
import math
import numpy as np
import functools
import heapq
from dataclasses import field, dataclass
from evariste.backward.env.lean.graph import LeanTactic, LeanTheorem

from evariste.backward.env.core import SPECIAL_TACTIC_TOKENS
from evariste.backward.graph import Tactic, Theorem, Proof
from evariste.backward.prover.args import STYPES
from evariste.utils import wrap_timer


class History:
    def __init__(self, t: int, n_tactics: int):
        self.n_tactics: int = n_tactics
        self.exists_since: int = t
        self.killed_node: Optional[int] = None  # when did this node become invalid
        self.killed_tactic: Dict[int, int] = {}  # which tactic was killed when
        self.updates: List[Tuple[int, int, float]] = []  # (timestamp, tactic, value)

    def kill(self, t: int):
        assert self.killed_node is None, "Node killed more than once!"
        self.killed_node = t

    def kill_tactic(self, t: int, tactic_id: int):
        assert tactic_id not in self.killed_tactic, "Tactic killed more than once!"
        self.killed_tactic[tactic_id] = t

    def update(self, t: int, tactic_id: int, value: float):
        self.updates.append((t, tactic_id, value))

    # def state(self, t: int):
    #     q: List[List[float]] = [[] for _ in range(self.n_tactics)]
    #     for ts, tactic_id, value in self.updates:
    #         if ts > t:
    #             break
    #         q[tactic_id].append(value)
    #     counts = [len(qq) for qq in q]
    #     q_mean = [math.exp(logmeanexp(qq)) for qq in q]
    #     return {
    #         "exists": self.exists_since <= t,
    #         "dead": self.killed_node is None or t < self.killed_node,
    #         "killed_tactics": {
    #             tactic_id
    #             for tactic_id, t_kill in self.killed_tactic.items()
    #             if t_kill <= t
    #         },
    #         "q": q_mean,
    #         "counts": counts,
    #     }

    def __str__(self):
        s_kill = (
            "alive" if self.killed_node is None else f"killed at t={self.killed_node}"
        )
        s = (
            f"Created at t={self.exists_since}, {self.n_tactics} tactics, "
            f"{s_kill}, {len(self.killed_tactic)} tactics killed, "
            f"{len(self.updates)} updates"
        )
        return s


class Node:
    def __init__(
        self,
        theorem: Theorem,
        tactics: List[Tactic],
        children_for_tactic: List[List[Theorem]],
        time_created: int,
    ):
        self.stypes = STYPES
        if not isinstance(theorem, LeanTheorem):
            self.stypes = [x for x in STYPES if x != "time"]
        self.theorem = theorem
        self.children_for_tactic = children_for_tactic
        self.tactics: List[Tactic] = tactics
        assert len(self.tactics) == len(self.children_for_tactic)
        self.n_tactics = len(self.tactics)
        self.killed_tactics: Set[int] = set()
        self.killed_mask: np.ndarray = np.full(self.n_tactics, 0, dtype=np.bool_)
        self.tactic_is_expandable = np.full(self.n_tactics, 1, dtype=np.bool_)
        self.solving_tactics: Set[int] = set()

        # check if the node is either invalid, solved or standard
        solving = [
            tac.is_valid and len(children) == 0
            for tac, children in zip(tactics, children_for_tactic)
        ]
        assert self.n_tactics == 0 or all(solving) or not any(solving)

        if self.n_tactics > 0 and all(solving):  # solved
            assert all(tac.is_valid for tac in tactics)
            self.solved = True  # updated through backprop
            # if solved is True we won't go further so this is definitely a leaf
            self.is_solved_leaf = True
            self.solving_tactics = set(range(self.n_tactics))
        else:
            self.solved = False
            self.is_solved_leaf = False

        self.history = History(time_created, self.n_tactics)

        # Attributes related to proof
        self.in_proof: bool = False
        # the size of the minimal proof to prove this particular node
        self.my_minproof_size: Dict[str, float] = {t: math.inf for t in self.stypes}
        # the size of minimal proof to prove this particular node, that includes a given tactic
        self.my_minproof_size_for_tactic: Dict[str, np.ndarray] = {
            t: np.full(self.n_tactics, math.inf, dtype=np.float64) for t in self.stypes
        }
        # the tactics that lead to smallest proof for this particular node
        self.my_minproof_tactics: Dict[str, List[int]] = {t: [] for t in self.stypes}
        self.in_minproof: Dict[str, bool] = {t: False for t in self.stypes}

        # kill "fake" tactics
        for tid, tactic in enumerate(tactics):
            if not tactic.is_valid:
                assert not self.is_solved_leaf
                assert tactic.error_msg in SPECIAL_TACTIC_TOKENS
                self.kill_tactic(time_created, tid)
        assert self.n_tactics == 0 or not self.all_tactics_killed

    def reset_minproof_stats(self) -> None:
        self.in_proof = False
        self.my_minproof_size = {t: math.inf for t in self.stypes}
        self.my_minproof_size_for_tactic = {
            t: np.full(self.n_tactics, math.inf, dtype=np.float64) for t in self.stypes
        }
        self.my_minproof_tactics = {t: [] for t in self.stypes}
        self.in_minproof = {t: False for t in self.stypes}

    @property
    def all_tactics_killed(self) -> bool:
        return len(self.killed_tactics) == self.n_tactics

    def is_terminal(self) -> bool:
        return (
            self.is_solved_leaf
            or len(self.children_for_tactic) == 0
            or self.all_tactics_killed
        )

    def is_bad(self) -> bool:
        return self.is_terminal() and not self.is_solved_leaf

    def kill_tactic(self, t: int, tactic_id: int) -> bool:
        assert 0 <= tactic_id < self.n_tactics, f"{tactic_id}, {self.n_tactics}"
        if tactic_id in self.killed_tactics:
            assert self.tactic_is_expandable[tactic_id] == 0
            return False  # already killed
        self.history.kill_tactic(t, tactic_id)
        self.killed_tactics.add(tactic_id)
        self.killed_mask[tactic_id] = 1
        # If all of our tactics are invalid, we become an invalid node.
        # In that case, invalidate parent tactic
        if self.all_tactics_killed:
            self.history.kill(t)
            return True
        return False


@dataclass(order=True)
class PrioritizedNode:
    priority: float
    node: Node = field(compare=False)
    tid: int


N = TypeVar("N", bound=Node)


class GraphAction:
    pass


class AddNodes(GraphAction):
    def __init__(self, theorems: List[Theorem]):
        self.theorems = theorems

    def __eq__(self, other):
        assert isinstance(other, GraphAction)
        if isinstance(other, AddNodes):
            return self.theorems == other.theorems
        return False


class KillTactic(GraphAction):
    def __init__(self, theorem: Theorem, tid: int):
        self.theorem = theorem
        self.tid = tid

    def __eq__(self, other):
        assert isinstance(other, GraphAction)
        if isinstance(other, KillTactic):
            return self.theorem == other.theorem and self.tid == other.tid
        return False


class Graph(Generic[N]):
    """ 
    Represents a dynamic **rooted** graph, maintaining history and minimum *proofs*.
    """

    def __init__(self, root: Theorem):
        self.root = root
        self.stypes = STYPES
        if not isinstance(root, LeanTheorem):
            self.stypes = [x for x in STYPES if x != "time"]
        self.nodes: Dict[Theorem, N] = {}
        self.history: List[GraphAction] = []

        # This is stored here instead of in the nodes because it has to be maintained globally
        # This contains theorems that have not necessarily been materialized as Nodes.
        self.ancestors: Dict[Theorem, Set[Tuple[Optional[Theorem], int]]] = defaultdict(
            set
        )
        self.ancestors[self.root].add((None, 0))  # root has no parent initially

        # Whereas ancestors has elements removed over time. Permanent_ancestors is strictly increasing
        self.permanent_ancestors: Dict[
            Theorem, Set[Tuple[Optional[Theorem], int]]
        ] = defaultdict(set)
        self.permanent_ancestors[self.root].add(
            (None, 0)
        )  # root has no parent initially
        self.unexplored_th: Set[Theorem] = {self.root}

        self.minproof_size: Dict[str, Optional[float]] = {
            stype: None for stype in self.stypes
        }
        self.init_minproof_size: Optional[
            Dict[str, float]
        ] = None  # minproof_size the first time the root was solved

    def reset_minproof_stats(self):
        self.minproof_size = {stype: None for stype in self.stypes}
        for node in self.nodes.values():
            node.reset_minproof_stats()

    @property
    def is_proved(self):
        """ A graph is proved if its root is solved"""
        return self.root in self.nodes and self.nodes[self.root].solved

    @property
    def dead_root(self):
        """ A graph is dead if all the tactics at its root have been killed"""
        return self.root in self.nodes and self.nodes[self.root].all_tactics_killed

    @staticmethod
    def build_from_history(
        root: Theorem, nodes: Dict[Theorem, N], history: List[GraphAction]
    ):
        """ Used for tests, re-apply a full history """
        graph: Graph[N] = Graph(root)
        for action in history:
            if isinstance(action, AddNodes):
                new_nodes = [
                    Node(
                        theorem=th,
                        tactics=nodes[th].tactics,
                        children_for_tactic=nodes[th].children_for_tactic,
                        time_created=timestep,
                    )
                    for timestep, th in enumerate(action.theorems)
                ]
                graph.add_nodes(new_nodes, add_to_history=True)  # type: ignore
            elif isinstance(action, KillTactic):
                graph.kill_tactic(
                    graph.nodes[action.theorem], action.tid, 0, add_to_history=True
                )
            else:
                raise RuntimeError("unknown action")
        return graph

    def add_nodes(
        self, new_nodes: Sequence[N], timestep: int = 0, add_to_history: bool = False
    ):
        """
        Add a set of nodes. If a node is bad, kill all tactics reaching it.

        After adding all nodes and killing all tactics, recompute the "solved" status.
        """
        if add_to_history:
            self.history.append(AddNodes([node.theorem for node in new_nodes]))
        to_check_solved_nodes: List[N] = []
        newly_solved_nodes = []
        for node in new_nodes:
            th = node.theorem
            assert th in self.permanent_ancestors and th in self.ancestors
            assert th not in self.nodes
            self.nodes[th] = node
            if node.is_bad():  # node is invalid
                for parent_th, tac_id in list(self.ancestors[th]):
                    if parent_th is not None:  # root
                        self.kill_tactic(
                            self.nodes[parent_th],
                            tac_id,
                            timestep,
                            add_to_history=False,
                        )
            elif node.solved:
                newly_solved_nodes.append(node)
            else:
                bad_tac_ids = set()
                for tac_id, children in enumerate(node.children_for_tactic):
                    for child in children:
                        self.permanent_ancestors[child].add((th, tac_id))
                        self.ancestors[child].add((th, tac_id))
                        if child in self.nodes and self.nodes[child].is_bad():
                            bad_tac_ids.add(tac_id)
                for bad_tac_id in bad_tac_ids:
                    self.kill_tactic(node, bad_tac_id, timestep, add_to_history=False)
                to_check_solved_nodes.append(node)
        self.propagate_and_check_solved(
            newly_solved_nodes, to_check_solved_nodes,
        )

    @wrap_timer()
    def kill_tactic(
        self, node: N, tactic_id: int, timestep: int = 0, add_to_history: bool = False
    ):
        """ 
            Kill a tactic. If this was the node's last valid tactic, propagate.
        """
        if add_to_history:
            self.history.append(KillTactic(node.theorem, tactic_id))
        to_kill = deque([(node, tactic_id)])
        while to_kill:
            cur_node, cur_tactic_id = to_kill.pop()
            if cur_tactic_id in cur_node.killed_tactics:
                continue
            # remove all ancestors corresponding to this tactic
            cur_th = cur_node.theorem
            for child in cur_node.children_for_tactic[cur_tactic_id]:
                if (cur_th, cur_tactic_id) in self.ancestors[child]:
                    self.ancestors[child].remove((cur_th, cur_tactic_id))
                    # This is done approximately, we could go down the leaves to check
                    # if this node was a dominator of any unexplored child.
                    # In practice, we only care about maintaining ancestors[],
                    # unexplored_th will be rebuilt if needed
                    if len(self.ancestors[child]) == 0:
                        if child not in self.nodes and child in self.unexplored_th:
                            self.unexplored_th.remove(child)

            # if killing this tactic leads to cur_node.killed_tactics == cur_node.tactics
            # kill all tactics leading to the node since it has become invalid
            if cur_node.kill_tactic(timestep, cur_tactic_id):
                for maybe_th, next_tac_id in self.ancestors[cur_node.theorem]:
                    if maybe_th is not None:
                        to_kill.appendleft((self.nodes[maybe_th], next_tac_id))

    def find_unexplored(self, ignore_solved: bool) -> None:
        """ Starting from the root and only using valid tactics, find unexpanded nodes"""
        self.unexplored_th = set()
        if self.root not in self.nodes:
            self.unexplored_th = {self.root}
        else:
            to_explore = deque([self.nodes[self.root]])
            seen: Set[N] = set()
            while to_explore:
                cur = to_explore.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                # same stop conditions as find_leaves_aux
                if ignore_solved and cur.solved:
                    continue

                for i, children in enumerate(cur.children_for_tactic):
                    if i in cur.killed_tactics:
                        continue
                    for child in children:
                        if child in self.nodes:
                            to_explore.appendleft(self.nodes[child])
                        else:
                            self.unexplored_th.add(child)

    def do_propagate_expandable(self) -> None:
        """ Starting from an up-to-date set of reachable unexplored leaves, 
        propagate backward an "expandable" tactic filter to only keep tactics leading to unexplored theorems.
        """
        for th, node in self.nodes.items():
            node.tactic_is_expandable = np.full(node.n_tactics, 0, dtype=np.bool_)
        to_propagate: List[Theorem] = []
        for th in self.unexplored_th:
            for ancestor_th, tac_id in self.ancestors[th]:
                if ancestor_th is None:
                    continue
                ancestor_node = self.nodes[ancestor_th]
                if (
                    tac_id in ancestor_node.killed_tactics
                ):  # check : I don't think this can happen as we deal with ancestors not permanent_ancestors
                    continue
                ancestor_node.tactic_is_expandable[tac_id] = 1
                to_propagate.append(ancestor_th)

        to_propagate_q = deque(to_propagate)
        seen: Set[Theorem] = set()
        while to_propagate_q:
            curr_th = to_propagate_q.pop()
            if curr_th in seen:
                continue
            seen.add(curr_th)
            for ancestor_th, tac_id in self.ancestors[curr_th]:
                if ancestor_th is None:
                    continue
                node = self.nodes[ancestor_th]
                # check : I don't think this can happen as we deal with ancestors not permanent_ancestors
                if tac_id in node.killed_tactics:
                    continue
                node.tactic_is_expandable[tac_id] = 1
                to_propagate_q.appendleft(ancestor_th)

    def final_expandable_check(self):
        for node in self.nodes.values():
            assert (
                node.tactic_is_expandable[list(node.killed_tactics)] == 0
            ).all(), "killed not included in tactic is expandable"

    @wrap_timer()
    def find_unexplored_and_propagate_expandable(self, ignore_solved: bool) -> None:
        """ First :func:`find_unexplored`, then :func:`propagate_expandable`"""
        self.find_unexplored(ignore_solved)
        self.do_propagate_expandable()

        good_propagate = (
            self.unexplored_th == {self.root}
            or sum(self.nodes[self.root].tactic_is_expandable) > 0
            or len(self.unexplored_th) == 0
        )
        if not good_propagate:
            raise RuntimeError(
                f"GOOD PROPAGATE FAILURE {self.unexplored_th == {self.root}} // "
                f"{sum(self.nodes[self.root].tactic_is_expandable)} // "
                f"len(unexplored_th)={len(self.unexplored_th)}"
            )
        self.final_expandable_check()

    def propagate_and_check_solved(
        self, newly_solved_nodes: Sequence[N], to_check_solved_nodes: Sequence[N],
    ) -> None:
        """ Given a set of newly solved nodes, back-propagate solved status"""
        newly_solved_nodes = list(newly_solved_nodes)
        to_check_solved = [
            (node, tac_id)
            for node in to_check_solved_nodes
            for tac_id in range(node.n_tactics)
        ]

        while len(newly_solved_nodes) + len(to_check_solved) > 0:

            # for each newly solved node, check whether their parents are solved
            if len(newly_solved_nodes) > 0:
                node = newly_solved_nodes.pop()
                assert node.solved
                for parent, tac_id in self.permanent_ancestors[node.theorem]:
                    if parent is not None:
                        to_check_solved.append((self.nodes[parent], tac_id))

            # check whether a given tactic solves a node. the node might be in the
            # middle of the graph, but it could already be solved if all of its
            # children were solved before.
            else:
                # NOTE: a solving tactic can be killed!
                node, tac_id = to_check_solved.pop()
                if not node.tactics[tac_id].is_valid:
                    continue
                if all(
                    self.nodes[child].solved if child in self.nodes else False
                    for child in node.children_for_tactic[tac_id]
                ):
                    node.solving_tactics.add(tac_id)
                    if not node.solved:
                        node.solved = True
                        newly_solved_nodes.append(node)

    @wrap_timer(verbose=1)
    def check_solved_ok(self) -> None:
        """
        Sanity check of solved status : a :attr:`solved` should be true if any of the tactics have all its children solved.
        Also check for missing ancestors
        """
        assert all(
            node.solved == (len(node.solving_tactics) > 0)
            for node in self.nodes.values()
        )
        # first check for missing ancestors (which implies wrong solved)
        for node_th, node in self.nodes.items():
            for tac_id, children in enumerate(node.children_for_tactic):
                for child in children:
                    assert (node_th, tac_id) in self.ancestors[
                        child
                    ] or tac_id in node.killed_tactics, (
                        f"MISSING ANCESTOR {node_th} for {child}"
                    )

        # then verify that given good ancestors, there were no errors.
        for node in self.nodes.values():
            should_be_solved = any(
                tactic.is_valid
                and all(
                    child in self.nodes and self.nodes[child].solved
                    for child in children
                )
                for tactic, children in zip(node.tactics, node.children_for_tactic)
            )
            if node.solved != should_be_solved:
                raise RuntimeError(
                    f"Node {node.theorem} should have solved={should_be_solved} "
                    f"but ={node.solved}"
                )

    @wrap_timer(verbose=1)
    def get_inproof_nodes(self) -> None:
        """
        Find all nodes that belong to at least one proof of the root node.
        """
        if not self.nodes[self.root].solved:
            return
        seen: Set[Theorem] = set()
        to_visit: Deque[Theorem] = deque([self.root])
        while to_visit:
            cur = to_visit.pop()  # pop right
            if cur in seen or cur not in self.nodes:
                continue
            node = self.nodes[cur]
            assert len(node.solving_tactics) > 0
            node.in_proof = True
            for tid in node.solving_tactics:
                to_visit.extendleft(node.children_for_tactic[tid])
            seen.add(cur)

    @wrap_timer()
    def get_minproof(self, theorem: Theorem, stype: str) -> Proof:
        """
        Extract one minimal proof for a node in a minimal proof.
        """
        assert stype in STYPES
        assert theorem in self.nodes
        node = self.nodes[theorem]
        assert node.solved and node.in_proof and node.in_minproof[stype]
        assert len(node.my_minproof_tactics[stype]) > 0
        tid = node.my_minproof_tactics[stype][0]
        return (
            node.theorem,
            node.tactics[tid],
            [
                self.get_minproof(child, stype)
                for child in node.children_for_tactic[tid]
            ],
        )

    @wrap_timer(verbose=1)
    def get_node_proof_sizes_and_depths(self) -> None:
        ops: Dict[str, Callable[[List[float]], float]] = {
            "depth": functools.partial(max, default=0),  # type: ignore
            "size": sum,
            "time": sum,
        }
        for stype in self.stypes:
            # Find my min proof size for each node
            op = ops[stype]
            to_process: List[PrioritizedNode] = []
            for node in self.nodes.values():
                if not node.is_solved_leaf:
                    continue
                for tid in node.solving_tactics:
                    priority = 1.0
                    if stype == "time":
                        tac = node.tactics[tid]
                        assert isinstance(tac, LeanTactic)
                        assert tac.duration is not None
                        priority = tac.duration
                    to_process.append(
                        PrioritizedNode(priority=priority, node=node, tid=tid)
                    )

            heapq.heapify(to_process)
            while len(to_process) > 0:
                p_node = heapq.heappop(to_process)
                v, node, tid = p_node.priority, p_node.node, p_node.tid  # type: ignore

                if node.my_minproof_size_for_tactic[stype][tid] == math.inf:
                    node.my_minproof_size_for_tactic[stype][tid] = v
                    if v <= node.my_minproof_size[stype]:
                        assert tid not in node.my_minproof_tactics[stype]
                        node.my_minproof_tactics[stype].append(tid)

                if node.my_minproof_size[stype] < math.inf:
                    assert v >= node.my_minproof_size[stype]
                    continue

                node.my_minproof_size[stype] = v

                for parent_th, tid in self.permanent_ancestors[node.theorem]:
                    if parent_th is None:  # root
                        continue
                    parent = self.nodes[parent_th]
                    added_v = 1.0
                    if stype == "time":
                        parent_tac = parent.tactics[tid]
                        assert isinstance(parent_tac, LeanTactic)
                        assert parent_tac.duration is not None
                        added_v = parent_tac.duration
                    new_v = added_v + op(
                        [
                            math.inf
                            if child not in self.nodes
                            else self.nodes[child].my_minproof_size[stype]
                            for child in parent.children_for_tactic[tid]
                        ]
                    )
                    if new_v < math.inf:
                        heapq.heappush(
                            to_process,
                            PrioritizedNode(priority=new_v, node=parent, tid=tid),
                        )

        for stype in self.stypes:
            # Find the minimal proofs
            if not self.is_proved:
                return
            assert self.nodes[self.root].my_minproof_size[stype] < math.inf
            self.minproof_size[stype] = self.nodes[self.root].my_minproof_size[stype]
            visit_minimal: Deque[Theorem] = deque([self.root])
            seen: Set[Theorem] = set()
            while visit_minimal:
                cur = visit_minimal.pop()
                if cur in seen:
                    continue  # not because of cycle, but because we consider all minimal proofs
                seen.add(cur)
                node = self.nodes[cur]
                node.in_minproof[stype] = True
                assert node.in_proof
                assert len(node.my_minproof_tactics[stype]) > 0
                # add child
                for tid in node.my_minproof_tactics[stype]:
                    for c in node.children_for_tactic[tid]:
                        visit_minimal.appendleft(c)
