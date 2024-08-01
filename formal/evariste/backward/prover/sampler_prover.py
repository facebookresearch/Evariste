# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Set, Dict, Optional, Union, Tuple

from evariste.backward.env.core import EnvExpansion
from evariste.backward.env.metamath import MMTheorem
from evariste.backward.prover.igreedy_prover import (
    ImprovedGreedyGoal,
    ProofResultsWithProofHandler,
)

from evariste.backward.prover.prover import ProofResult
from evariste.backward.env.core import BackwardGoal
from evariste.backward.graph import Proof, Theorem, Tactic
from evariste.backward.prover.prover import ProofHandler
from evariste.backward.prover.prover_args import ProofStatus
from copy import copy


class HasCycle(Exception):
    pass


class HyperTreeNode:
    def __init__(self, id: int):
        self.id: int = id
        self.tactics: List[TacticAndSubGoals] = []

        self.proved_in_branch: Optional[TacticAndSubGoals] = None
        self._proved: bool = False

    def is_proved(self) -> bool:
        return self._proved or (self.proved_in_branch is not None)


class TacticAndSubGoals:
    def __init__(self, tactic: Tactic, subgoals_node: List[HyperTreeNode]):
        self.children: List[HyperTreeNode] = subgoals_node
        self.tactic: Tactic = tactic


class HyperTree:
    """
    Store a meta proof tree with for each thm multiple possible tactics and subgoals
    """

    def __init__(self, root: Theorem):
        self.th_to_node: Dict[Theorem, HyperTreeNode] = dict()
        self.th_to_node[root] = HyperTreeNode(0)
        self.id_to_th: List[Theorem] = [root]
        self.leaves: Set[Theorem] = {root}
        self.root = root

        self.proved: Set[Theorem] = set()
        self.n_proven = 0

    def __len__(self):
        return len(self.th_to_node)

    def add_node(self, thm: Theorem) -> HyperTreeNode:
        if thm in self.th_to_node:
            return self.th_to_node[thm]

        node = HyperTreeNode(len(self.th_to_node))
        self.id_to_th.append(thm)
        self.th_to_node[thm] = node

        return node

    def expand_with_tactic(self, src: Theorem, tactic: Tactic, subgoals: List[Theorem]):
        src_node = self.th_to_node[src]

        for s in subgoals:
            self.add_node(s)

        subgoals_nodes = [self.th_to_node[s] for s in subgoals]

        tactic_branch = TacticAndSubGoals(tactic, subgoals_nodes)

        src_node.tactics.append(tactic_branch)
        if not subgoals_nodes:
            src_node.proved_in_branch = tactic_branch

    def propagate_proved(self, src: Theorem) -> bool:
        seen_proved: Set[HyperTreeNode] = {self.th_to_node[thm] for thm in self.proved}

        def walk(src_node: HyperTreeNode, seen: Set[HyperTreeNode]):
            if src_node in seen_proved or src_node.is_proved():
                seen_proved.add(src_node)
                return True
            if src_node in seen:
                return False

            seen.add(src_node)

            tactic_branches = src_node.tactics
            _seen = copy(seen)
            for tands in tactic_branches:
                subgoals = tands.children

                proved = all(walk(s, seen) for s in subgoals)

                if proved:
                    src_node.proved_in_branch = tands
                    seen_proved.add(src_node)
                    return True

                seen = _seen

            return False

        p = walk(self.th_to_node[src], seen=set())

        for s in seen_proved:
            self.proved.add(self.id_to_th[s.id])

        return p

    def get_proof(self) -> Proof:
        def build_proof(cur_id):
            th = self.id_to_th[cur_id]
            node = self.th_to_node[th]

            if node.proved_in_branch:
                tacticsubogals = node.proved_in_branch
                return (
                    th,
                    tacticsubogals.tactic,
                    [build_proof(c.id) for c in tacticsubogals.children],
                )
            else:
                return (
                    th,
                    None,
                    None,
                )

        return build_proof(0)


class GreedySamplerProver(ProofHandler):
    def __init__(self, goal: Union[ImprovedGreedyGoal, BackwardGoal]):
        super().__init__(goal)

        if not isinstance(goal, ImprovedGreedyGoal):
            goal = ImprovedGreedyGoal(theorem=goal.theorem, name=goal.name)

        self.hyper_tree = HyperTree(goal.theorem)
        # todo: generalize
        self.hyper_tree.proved = {
            MMTheorem(conclusion=concl, hyps=goal.theorem.hyps)
            for concl in goal.hypothesis
        }

        for thm in self.hyper_tree.proved:
            node = self.hyper_tree.add_node(thm)
            node._proved = True

        # When we add a thm to be expanded we also store the number of times we want it to be expanded
        self.n_expansion: Dict[Theorem, int] = {goal.theorem: 10}
        self.done_byfwd = False

        if goal.theorem in self.hyper_tree.proved:
            print("Already proved!")
            self.hyper_tree.th_to_node[self.goal.theorem]._proved = True
            self.n_expansion: Dict[Theorem, int] = {goal.theorem: -1}
            self.done = True
            self.done_byfwd = True

        # If we get a tactic applied on a goal it has already been applied we will not do it again
        self.used_tactic_goal: Set[Tuple[Theorem, Tactic]] = set()

        # Since the expansion is now kind of destructive we need to store what it did
        self._last_to_expand: List[Theorem] = []
        self._to_expand_called: bool = False

    def get_theorems_to_expand(self) -> Optional[List[Theorem]]:
        """
        Look at self.n_expansion and send to the expander every Theorem in it and reduce the number of expansions to
        do for each.
        @return: A list of theorem to expand
        """

        to_expand = []

        # get theorem to expand and reduce their number of left expansions
        for k, v in self.n_expansion.items():
            node = self.hyper_tree.th_to_node[k]
            if v > 0:
                # If the node has been proved somehow we set it to -1 so it can't be expanded in the future
                if node.is_proved():
                    self.n_expansion[k] = -1
                else:
                    self.n_expansion[k] = min(0, v - 1)
                    to_expand.append(k)

        # We called to_expand, thus we store its output and that it has been called
        self._last_to_expand = to_expand
        self._to_expand_called = True

        return to_expand

    def send_env_expansions(self, tactics: List[EnvExpansion]) -> None:

        # To work properly we need to know what we sent to the expander
        # If we did not call to expand then nothing was sent
        assert self._to_expand_called, "to_expand() was not called"
        goals = self._last_to_expand
        # We use this expansion step and we say that for next time we will need another expansion
        self._to_expand_called = False

        # For each leave, results
        for goal, results in zip(goals, tactics):
            if results.is_error:
                self.fail = True
                continue

            # We keep only valid tactics, we also want the subgoals they create and their score
            valid_tactics = [
                (tactic, results.priors[k], results.child_for_tac[k])
                for k, tactic in enumerate(results.tactics)
                if tactic.is_valid
            ]

            # We sort the tactic from the best to the worst according to the prior
            valid_tactics.sort(key=lambda x: x[1], reverse=True)

            # We here take only the first tactic output
            if valid_tactics:
                t, _, subgoals = valid_tactics[0]

                # If we already used that tactic on that goal we dont need to put it again in the tree
                if not (goal, t) in self.used_tactic_goal:
                    # we memorize that we applied that tactic on that goal
                    self.used_tactic_goal.add((goal, t))
                    # We expand that goal with this tactic
                    # It add a tactic and its children to that node
                    self.hyper_tree.expand_with_tactic(goal, t, subgoals)

                # We set for expansion all the new goals
                for s in subgoals:
                    if s not in self.n_expansion:
                        self.n_expansion[s] = 3
                    elif self.n_expansion[s] >= 0:
                        self.n_expansion[s] += 1

        # We walk trough the hyper tree to propagate proved results
        # If there exist a path that make the root proved then we got a proof
        proven = self.hyper_tree.propagate_proved(self.hyper_tree.root)
        # If we did all the expansion we wanted
        if all(v <= 0 for _, v in self.n_expansion.items()):
            self.fail = True
        self.done = proven or self.fail

    def get_proof(self):
        return self.hyper_tree.get_proof()

    def get_result(self) -> ProofResult:
        return ProofResultsWithProofHandler(
            proof=self.get_proof() if not self.fail else None,
            goal=self.goal,
            proofhandler=self,
        )

    def __str__(self) -> str:
        return (
            f"Thm: {self.goal.theorem.train_label}, "
            f"IsProven?: {self.done and not self.fail}, "
            f"proven nodes: {len(self.hyper_tree.proved)}, "
            f"tot nodes: {len(self.hyper_tree)}, "
            f"done by fwd ?: {self.done_byfwd}, "
        )
