# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import re
from collections import defaultdict
from pprint import pprint
from typing import List, Dict, Set, Tuple

from numpy.random.mtrand import RandomState
import numpy as np

from evariste.backward.env.lean.graph import LeanContext, LeanTheorem, LeanTactic
from evariste.forward.fwd_lean.training.common import (
    LeanMetaProofNode,
    is_complete,
    detect_cycles,
    LeanProofNode,
    TacticAndChildren,
)
from evariste.forward.common import ProofNode
from evariste.forward.training.helpers import count_unique_theorems
from evariste.metrics import Avg


class MissingSubGoal(Exception):
    pass


class SubGoalEqualGoal(Exception):
    pass


def nodes_from_steps_v2(
    steps: List,
    stats: Dict,
    context: LeanContext,
    allow_uncompleted_proofs: bool = False,
) -> List[LeanMetaProofNode]:
    theorem_to_node: Dict[LeanTheorem, LeanMetaProofNode] = {}
    theorem_to_tactics_and_sub_goals = defaultdict(list)

    if len(steps) == 0:
        return []

    for step in steps:
        stats["n_steps"] += 1
        goal_pp = str(step["goal_pp"])
        try:
            check_is_normalized(goal_pp)
        except AssertionError:
            stats["n_steps_not_normalised"] += 1
        if " goals" in goal_pp:
            stats["n_steps_bad_with_many_goals"] += 1
            raise RuntimeError("Issue in parse_goal: ' goals' in goal_pp")
        thm = LeanTheorem(conclusion=goal_pp, context=context, state=None)
        if thm not in theorem_to_node:
            node = LeanMetaProofNode(thm=thm, tactics_and_children=[])
            theorem_to_node[thm] = node

        tactic = LeanTactic(step["tactic"])
        result = step["res"]
        subgoals = [
            LeanTheorem(conclusion=node["full_pp"], context=context, state=None)
            for node in result["nodes"]
        ]
        theorem_to_tactics_and_sub_goals[thm].append((tactic, subgoals))

    # populate children
    for goal in theorem_to_node.keys():
        stats["n_nodes"] += 1
        succeed_tacs: List[TacticAndChildren] = []
        assert len(theorem_to_tactics_and_sub_goals[goal]) >= 1
        for tactic, subgoals in theorem_to_tactics_and_sub_goals[goal]:
            stats["n_tactics"] += 1
            try:
                children = []
                for sg in subgoals:
                    if sg == goal:
                        stats["n_tactics_bad_sub_goal_equal_goal"] += 1
                        raise SubGoalEqualGoal
                    elif sg not in theorem_to_node and not allow_uncompleted_proofs:
                        stats["n_tactics_bad_missing_sub_goal"] += 1
                        if sg.conclusion.startswith("case"):
                            stats["n_tactics_bad_missing_sub_goal_with_case"] += 1
                        raise MissingSubGoal
                    elif sg not in theorem_to_node:
                        stats["n_global_hyp"] += 1
                        # we create a global hyp
                        hyp = LeanMetaProofNode(
                            thm=sg, uncompleted=True, tactics_and_children=[]
                        )
                        children.append(hyp)
                    else:
                        if sg.conclusion.startswith("case"):
                            stats["n_sg_good_with_case"] += 1
                        children.append(theorem_to_node[sg])
                succeed_tacs.append(TacticAndChildren(tactic=tactic, children=children))
                stats["n_tactics_ok"] += 1
            except (MissingSubGoal, SubGoalEqualGoal):
                stats["n_tactics_bad"] += 1
                continue

        if len(succeed_tacs) == 0:
            stats["n_nodes_with_no_tactic"] += 1

        node = theorem_to_node[goal]
        node.tactics_and_children = succeed_tacs

    # detect all nodes that need to be removed:
    # 1. because they have cycles
    # 2. because the don't have a complete proof
    bad_nodes: Set[LeanTheorem] = set()
    for key, node in list(theorem_to_node.items()):
        if detect_cycles(node):
            stats["n_nodes_with_cycle"] += 1
            bad_nodes.add(key)
            theorem_to_node.pop(key)
            continue

        if not is_complete(node, allow_global_hyps=allow_uncompleted_proofs):
            stats["n_nodes_with_no_complete_proof"] += 1
            bad_nodes.add(key)
            theorem_to_node.pop(key)
            continue

        stats["n_nodes_ok"] += 1

    # remove all bad nodes from children
    for node in theorem_to_node.values():
        to_pop = set()
        for i, tac in enumerate(node.tactics_and_children):
            for child in tac.children:
                if child.thm in bad_nodes:
                    to_pop.add(i)
                    break
        assert len(to_pop) < len(
            node.tactics_and_children
        ), f"{len(to_pop)}, {len(node.tactics_and_children)}"  # if not should already be
        # detected as bad_node
        node.tactics_and_children = [
            tac for i, tac in enumerate(node.tactics_and_children) if i not in to_pop
        ]
        assert len(node.tactics_and_children) > 0
        assert is_complete(node, allow_global_hyps=allow_uncompleted_proofs)
        assert not detect_cycles(node)

    return list(theorem_to_node.values())


def nodes_from_steps_v3(
    steps: List,
    stats: Dict,
    context: LeanContext,
    allow_uncompleted_proofs: bool = False,
    use_reparsed_children: bool = False,
    use_pp_as_fp: bool = True,
    use_parsed_pp_as_conclusion: bool = False,
) -> List[LeanMetaProofNode]:
    theorem_to_node: Dict[LeanTheorem, LeanMetaProofNode] = {}
    theorem_to_tactics_and_sub_goals = defaultdict(list)

    if len(steps) == 0:
        return []

    for step in steps:
        stats["n_steps"] += 1
        # this is the goal that the transformer will be taught to output
        original_pp = step["req"]["reparsed"]["full_pp"]
        original_fingerprint = step["req"]["reparsed"]["fingerprint"]
        # this goal will be parsed by parse_goal_and_apply_tactic, hence giving
        # a fingerprint potentially different than the one in step["req"]["reparsed"]
        # we will use this fingerprint to create the training dataset goal (since it
        # will be the one we obtain at training time
        parse_goal_and_apply_tactic_output = step["res"]
        parsed_pp = parse_goal_and_apply_tactic_output["parsed_goal"]["full_pp"]
        parsed_fingerprint = parse_goal_and_apply_tactic_output["parsed_goal"][
            "fingerprint"
        ]

        if parsed_fingerprint != original_fingerprint:
            stats["n_steps_with_parsing_change_fingerprint"] += 1

        if parsed_pp != original_pp:
            stats["n_steps_with_parsing_change_full_pp"] += 1

        try:
            check_is_normalized(original_pp)
        except AssertionError:
            stats["n_steps_not_normalised"] += 1
        if " goals" in original_pp:
            stats["n_steps_bad_with_many_goals"] += 1
            raise RuntimeError("Issue in parse_goal: ' goals' in goal_pp")

        thm = LeanTheorem(
            conclusion=original_pp
            if not use_parsed_pp_as_conclusion
            else parsed_pp,  # will be used as transformer output
            fingerprint=parsed_pp
            if use_pp_as_fp
            else parsed_fingerprint,  # will be used to match with children
            context=context,
            state=None,
        )
        if thm not in theorem_to_node:
            node = LeanMetaProofNode(thm=thm, tactics_and_children=[])
            theorem_to_node[thm] = node

        tactic = LeanTactic(step["req"]["goal"]["tactic"])

        if use_reparsed_children:
            if not step["reparsed_nodes_success"]:
                stats["n_steps_with_failed_chilren_reparsing"] += 1
                continue
            children = step["reparsed_nodes"]
        else:
            children = parse_goal_and_apply_tactic_output["nodes"]

        if use_reparsed_children:
            assert step["reparsed_nodes_success"]
            assert len(children) == len(parse_goal_and_apply_tactic_output["nodes"])

        subgoals = [
            LeanTheorem(
                conclusion=node["full_pp"],
                fingerprint=node["full_pp"] if use_pp_as_fp else node["fingerprint"],
                context=context,
                state=None,
            )
            for node in children
        ]
        theorem_to_tactics_and_sub_goals[thm].append((tactic, subgoals))

    # populate children
    for goal in theorem_to_node.keys():
        stats["n_nodes"] += 1
        succeed_tacs: List[TacticAndChildren] = []
        if use_reparsed_children and len(theorem_to_tactics_and_sub_goals[goal]) == 0:
            continue
        assert len(theorem_to_tactics_and_sub_goals[goal]) >= 1
        for tactic, subgoals in theorem_to_tactics_and_sub_goals[goal]:
            stats["n_tactics"] += 1
            try:
                children = []
                if len(subgoals) == 0:
                    stats["n_tactics_with_no_subgoal"] += 1
                for sg in subgoals:
                    if sg == goal:
                        stats["n_tactics_bad_sub_goal_equal_goal"] += 1
                        raise SubGoalEqualGoal
                    elif sg not in theorem_to_node and not allow_uncompleted_proofs:
                        stats["n_tactics_bad_missing_sub_goal"] += 1
                        if sg.conclusion.startswith("case"):
                            stats["n_tactics_bad_missing_sub_goal_with_case"] += 1
                        raise MissingSubGoal
                    elif sg not in theorem_to_node:
                        stats["n_global_hyp"] += 1
                        # we create a global hyp
                        hyp = LeanMetaProofNode(
                            thm=sg, uncompleted=True, tactics_and_children=[]
                        )
                        children.append(hyp)
                    else:
                        if sg.conclusion.startswith("case"):
                            stats["n_sg_good_with_case"] += 1
                        children.append(theorem_to_node[sg])
                succeed_tacs.append(TacticAndChildren(tactic=tactic, children=children))
                stats["n_tactics_ok"] += 1
            except (MissingSubGoal, SubGoalEqualGoal):
                stats["n_tactics_bad"] += 1
                continue

        if len(succeed_tacs) == 0:
            stats["n_nodes_with_no_tactic"] += 1

        node = theorem_to_node[goal]
        node.tactics_and_children = succeed_tacs

    # detect all nodes that need to be removed:
    # 1. because they have cycles
    # 2. because the don't have a complete proof
    bad_nodes: Set[LeanTheorem] = set()
    for key, node in list(theorem_to_node.items()):
        if detect_cycles(node):
            stats["n_nodes_with_cycle"] += 1
            bad_nodes.add(key)
            theorem_to_node.pop(key)
            continue

        if not is_complete(node, allow_global_hyps=allow_uncompleted_proofs):
            stats["n_nodes_with_no_complete_proof"] += 1
            bad_nodes.add(key)
            theorem_to_node.pop(key)
            continue

        stats["n_nodes_ok"] += 1

    # remove all bad nodes from children
    for node in theorem_to_node.values():
        to_pop = set()
        for i, tac in enumerate(node.tactics_and_children):
            for child in tac.children:
                if child.thm in bad_nodes:
                    to_pop.add(i)
                    break
        assert len(to_pop) < len(
            node.tactics_and_children
        ), f"{len(to_pop)}, {len(node.tactics_and_children)}"  # if not should already be
        # detected as bad_node
        node.tactics_and_children = [
            tac for i, tac in enumerate(node.tactics_and_children) if i not in to_pop
        ]
        assert len(node.tactics_and_children) > 0
        assert is_complete(node, allow_global_hyps=allow_uncompleted_proofs)
        assert not detect_cycles(node)

    return list(theorem_to_node.values())


def update_stats(stats: Dict, all_nodes: List[LeanMetaProofNode]):
    assert len(all_nodes) > 0
    avg_n_tactics = Avg()
    avg_n_children_by_tactic = Avg()
    for node in all_nodes:
        avg_n_tactics.act(len(node.tactics_and_children))
        for tac in node.tactics_and_children:
            avg_n_children_by_tactic.act(len(tac.children))

    stats = dict(stats)
    stats["avg_n_children_by_tactic"] = avg_n_children_by_tactic.stats_and_reset()
    stats["avg_n_tactics"] = avg_n_tactics.stats_and_reset()
    return stats


def sample_simple_proof(
    root: LeanMetaProofNode, rng: RandomState, allow_global_hyps: bool = False
) -> LeanProofNode:
    """Here node is a meta-dag, since it represents multiples proof dags,
    so we build a simple proof from it"""

    built: Dict[LeanTheorem, LeanProofNode] = {}

    def sample(node: LeanMetaProofNode) -> LeanProofNode:
        if node.thm in built:
            return built[node.thm]

        if not allow_global_hyps:
            assert len(node.tactics_and_children) > 0

        if allow_global_hyps and len(node.tactics_and_children) == 0:
            hyp = ProofNode.create_hyp(node.thm)
            built[hyp.theorem] = hyp
            return hyp

        # we sample given number of theorems in each children
        def _weight(tac_: TacticAndChildren) -> int:
            if not tac_.children:
                return 1
            return 1 + len(set.union(*[c.child_theorems() for c in tac_.children]))

        p = np.array([_weight(tac) for tac in node.tactics_and_children])
        p = p / p.sum()

        tac: TacticAndChildren = rng.choice(node.tactics_and_children, p=p)
        tactic = tac.tactic
        children = list(tac.children)
        children = [sample(c) for c in children]
        assert node.thm not in built  # no cycle

        simple_node = ProofNode(theorem=node.thm, tactic=tactic, children=children)
        built[simple_node.theorem] = simple_node
        return simple_node

    return sample(root)


def extract_longest_subproof(root: LeanMetaProofNode) -> LeanProofNode:
    _built: Dict[LeanTheorem, Tuple[LeanProofNode, Set[LeanTheorem]]] = {}

    def longuest(node: LeanMetaProofNode) -> Tuple[LeanProofNode, Set[LeanTheorem]]:
        if node.thm in _built:
            return _built[node.thm]

        n_max = -1
        chosen_children = None
        chosen_tactic = None
        chosen_child_thms = None
        assert len(node.tactics_and_children) > 0

        all_sizes = []

        for tactic_and_children in node.tactics_and_children:
            children = []
            theorems = set([])
            for child in tactic_and_children.children:
                built_child, child_thms = longuest(child)
                children.append(built_child)
                theorems.update(child_thms)

            proof_size = len(theorems)

            if proof_size > n_max:
                n_max = proof_size
                chosen_children = children
                chosen_tactic = tactic_and_children.tactic
                chosen_child_thms = theorems
            all_sizes.append(proof_size)

        assert len(all_sizes) == len(node.tactics_and_children)

        assert n_max >= 0

        proof_node = ProofNode(
            theorem=node.thm, tactic=chosen_tactic, children=chosen_children
        )
        child_thms = chosen_child_thms.union({node.thm})
        _built[node.thm] = (proof_node, child_thms)
        return proof_node, child_thms

    this_proof, these_child_thms = longuest(root)
    assert count_unique_theorems(this_proof) == len(these_child_thms)
    return this_proof


def check_is_normalized(full_pp: str) -> None:
    assert "\t" not in full_pp
    normalised = "\n".join(line.strip() for line in full_pp.split("\n"))
    normalised2 = re.sub(r" *\n *", r"\n", full_pp.strip())
    if full_pp != normalised:
        if "\n  " not in full_pp:
            raise ValueError(f"{full_pp!r}")
    assert full_pp == normalised, f"{full_pp!r} != {normalised!r}"
    assert full_pp == normalised2, f"{full_pp!r} != {normalised2!r}"
