# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Set, Dict, Optional, Union, Tuple
from dataclasses import dataclass, field
from random import sample

from evariste.backward.env.core import EnvExpansion, BackwardGoal
from evariste.backward.graph import Proof, Theorem, Tactic
from evariste.backward.prover.core import ProofHandler, ProofHandlerFailure, ProofResult


class HasCycle(Exception):
    pass


class ProofTreeNode:
    def __init__(self, id: int):
        self.children: List[ProofTreeNode] = []
        self.id: int = id
        self.tactic: Optional[Tactic] = None
        self.proven: bool = False

    def add_child(self, child: "ProofTreeNode"):
        self.children.append(child)


class ProofTree:
    """a simplified proof tree. could probably be merged with SimuTree or some other tree we have"""

    def __init__(self, root: Theorem):
        self.th_to_node: Dict[Theorem, ProofTreeNode] = dict()
        self.th_to_node[root] = ProofTreeNode(0)
        self.id_to_th: List[Theorem] = [root]
        self.leaves: Set[Theorem] = {root}
        self.root = root

        self.n_proven = 0

    def add_edge(self, src: Theorem, dest: Theorem) -> bool:
        src_node = self.th_to_node[src]
        dest_added: bool = False
        if dest not in self.th_to_node:
            dest_added = True
            dest_node = ProofTreeNode(len(self.th_to_node))
            self.id_to_th.append(dest)
            self.th_to_node[dest] = dest_node
            self.leaves.add(dest)

        else:
            dest_node = self.th_to_node[dest]

        if src in self.leaves:
            self.leaves.remove(src)

        src_node.add_child(dest_node)

        return dest_added

    def __len__(self):
        return len(self.id_to_th)

    def add_subgoals(
        self, thm: Theorem, tactic: Tactic, subgoals: List[Theorem]
    ) -> List[Theorem]:
        self.set_tactic(thm, tactic=tactic)
        list_added = []
        for s in subgoals:
            added = self.add_edge(src=thm, dest=s)
            if added:
                list_added.append(s)
        return list_added

    def remove(self, thm: Theorem, added: List[Theorem]):
        node = self.th_to_node[thm]

        children_to_remove: List[ProofTreeNode] = [self.th_to_node[t] for t in added]

        node.tactic = None
        node.proven = False

        to_remove = []
        for c in children_to_remove:
            t = self.id_to_th[c.id]

            if t != thm:
                if t in self.leaves:
                    self.leaves.remove(t)
                self.th_to_node.pop(t, None)
                to_remove.append(c.id)
                del c

        for idx in sorted(to_remove, reverse=True):
            if idx < len(self.id_to_th):
                del self.id_to_th[idx]

        node.children = []
        self.leaves.add(thm)

    def set_tactic(self, src: Theorem, tactic: Tactic):
        src_node = self.th_to_node[src]
        assert src_node.tactic is None, "setting tactic multiple times"
        src_node.tactic = tactic

    def set_proved(self, src: Theorem):
        src_node = self.th_to_node[src]
        assert src_node.proven is False, "proved multiple times"
        src_node.proven = True

        # If it's proved it's not a leave anymore
        if src in self.leaves:
            self.leaves.remove(src)

    def check_cycle(self, src: Theorem = None):
        being_seen: Set[int] = set()

        def walk(node: ProofTreeNode):
            if node.id in being_seen:
                raise HasCycle
            being_seen.add(node.id)
            for child in node.children:
                walk(child)
            being_seen.remove(node.id)

        if src:
            walk(self.th_to_node[src])
        else:
            walk(self.th_to_node[self.root])

    def propagate_proven(self) -> Tuple[bool, int]:
        seen = set()
        seen_proven = set()

        # if self.th_to_node[self.root].proven:
        #    return True, self.n_proven

        def walk(src_node: ProofTreeNode) -> bool:
            if src_node.proven or self.id_to_th[src_node.id] in seen_proven:
                seen_proven.add(src_node)

                return True
            elif src_node in seen:
                return False
            proved = False
            seen.add(src_node)

            if self.id_to_th[src_node.id] not in seen_proven:
                children = src_node.children

                if children:
                    proved = all(walk(c) for c in children)

                    if proved:
                        seen_proven.add(self.id_to_th[src_node.id])

            return src_node.proven or proved

        proved = walk(self.th_to_node[self.root])
        self.n_proven = len(seen_proven)

        return proved, self.n_proven

    def depth(self):
        def walk(src_node: ProofTreeNode):
            children = src_node.children
            return max([walk(c) for c in children], default=0) + 1

        return walk(self.th_to_node[self.root])

    def is_done(self):
        return len(self.leaves) == 0

    def get_proof(self) -> Proof:
        def build_proof(cur_id):
            th = self.id_to_th[cur_id]
            node = self.th_to_node[th]
            return th, node.tactic, [build_proof(c.id) for c in node.children]

        return build_proof(0)


@dataclass
class ImprovedGreedyGoal(BackwardGoal):
    hypothesis: Set[Theorem] = field(default_factory=set)
    max_steps: int = 5


@dataclass
class ProofResultsWithProofHandler(ProofResult):
    proofhandler: ProofHandler


class ImprovedGreedy(ProofHandler):
    """
    ProofHandler used for handling a Greedy proof strategy.
    """

    def __init__(self, goal: Union[ImprovedGreedyGoal, BackwardGoal]):
        super().__init__(goal)

        if not isinstance(goal, ImprovedGreedyGoal):
            goal = ImprovedGreedyGoal(theorem=goal.theorem, name=goal.name)

        self.proof_tree = ProofTree(goal.theorem)
        self.steps = 0
        self.max_steps = goal.max_steps

        # todo: pass arg from goal
        # todo: find a good value of 3
        self.max_to_expand = 5

        self.additional_hyps: Set[Theorem] = set()
        self.n_used_add_hyps = 0

        self.done_byfwd = False
        if goal.theorem in self.additional_hyps:
            print("Already proved!")
            self.proof_tree.leaves = set()
            self.proof_tree.th_to_node[self.goal.theorem].proven = True
            self.done = True
            self.done_byfwd = True

        self._to_expand = []
        self._to_expand_called = False
        self.fail = None

    def to_expand_by_forward(self) -> Optional[List[Theorem]]:
        return list(self.proof_tree.leaves)

    def add_hyps(self, hyps: Set[Theorem]):
        for h in hyps:
            if h in list(self.proof_tree.leaves):
                self.proof_tree.th_to_node[h].proven = True
                self.proof_tree.leaves.remove(h)
                self.n_used_add_hyps += 1
        self.additional_hyps |= hyps

    def restart(self):
        self.steps = 0

    def get_theorems_to_expand(self) -> List[Theorem]:
        if len(self.proof_tree.leaves) < self.max_to_expand:
            toexpand = list(self.proof_tree.leaves)
        else:
            # We still use list(set) because set support in random.sample will be depracted in
            # python 3.9
            toexpand = sample(list(self.proof_tree.leaves), k=self.max_to_expand)

        self._to_expand = toexpand
        self._to_expand_called = True

        return self._to_expand

    def send_env_expansions(self, tactics: List[EnvExpansion]) -> None:
        self.steps += 1

        # We store old leaves
        assert (
            self._to_expand_called
        ), "to_expand() should have been called before apply()"
        goals = self._to_expand
        self._to_expand_called = False

        # For each leave, results
        for goal, results in zip(goals, tactics):

            # If this leave gives an error we retry itnext time
            if results.is_error:
                continue

            # We keep only valid tactics, we also want the subgoals they create and their score
            valid_tactics = [
                (tactic, results.priors[k], results.child_for_tac[k])
                for k, tactic in enumerate(results.tactics)
                if tactic.is_valid
            ]

            # We sort the tactic from the best to the worst according to the prior
            valid_tactics.sort(key=lambda x: x[1], reverse=True)

            # Track if we succeded in expanding the current goal
            goal_done = False

            # First we check if there is a tactic that complete the goal
            # We check it here because the tactic that works is not necessarly the one
            # with the higher prior
            for t, _, subgoals in valid_tactics:
                # If the tactic is valid and has no subgoal it means the goal trivial and we won
                if not subgoals:
                    # We mark the goal as proved and we store the tactic that worked
                    self.proof_tree.set_proved(goal)
                    self.proof_tree.set_tactic(goal, t)

                    # We note that we succeded
                    goal_done = True
                    # No need to check other options
                    break

            # If no tactic terminate the proof of that goal we try each tactic in descending order of their prior
            if not goal_done:
                for t, _, subgoals in valid_tactics:
                    # We add all the subgoals to the goal
                    # We keep track of the subgoals that we added just now (because they were not already in the tree)
                    # We use that to be able to remove them if we detect a cycle, without removing node that
                    # were there before
                    added = self.proof_tree.add_subgoals(goal, t, subgoals)

                    # We check for cycles
                    try:
                        # We check for cycle begining from the current goal
                        # It should reduce a little bit the amount of computations to do
                        self.proof_tree.check_cycle(goal)
                    # If there is a cycle we remove the nodes we added and all the connections to
                    # rollback the tree to its state before we added that tactic
                    except HasCycle:
                        self.proof_tree.remove(goal, added)
                    else:
                        # If there is no cycle that goal has been expanded

                        # We check which one of these subgoals are in additional hyps so we can
                        # mark them as proven and remove them from the leaves
                        for s in subgoals:
                            if s in self.additional_hyps:
                                print("HEY: Helped by forward!")
                                self.proof_tree.th_to_node[s].proven = True
                                self.n_used_add_hyps += 1
                                if s in self.proof_tree.leaves:
                                    self.proof_tree.leaves.remove(s)

                        break

        # If we have reach the maximum number of steps, we stop
        if not self.is_done():
            if self.steps >= self.max_steps:
                self.fail = ProofHandlerFailure.TERMINATED_WITHOUT_PROVING
            if len(self.proof_tree.leaves) >= 20:
                self.fail = ProofHandlerFailure.TOO_BIG

        # Check if we are done (ie if there are no leaves left to expand)
        self.done = self.is_done() or self.fail

    def is_done(self):
        return len(self.proof_tree.leaves) == 0

    def get_result(self) -> ProofResult:
        return ProofResultsWithProofHandler(
            proof=self.get_proof() if not self.fail else None,
            goal=self.goal,
            proofhandler=self,
            exception=None,
        )

    def get_proof(self) -> Proof:
        return self.proof_tree.get_proof()

    def __str__(self) -> str:
        proven, n_proven = self.proof_tree.propagate_proven()
        return (
            f"Thm: {self.goal.theorem.train_label}, "
            f"IsProven?: {proven}, proven nodes: {n_proven}, "
            f"tot nodes: {len(self.proof_tree)}, "
            f"n_leaves: {len(self.proof_tree.leaves)}, "
            f"depth: {self.proof_tree.depth()}, "
            f"steps: {self.steps} / {self.max_steps}, "
            f"done by fwd ?: {self.done_byfwd}, "
            f"node proven by fwd: {self.n_used_add_hyps}, "
        )
