# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Tuple, Dict

from evariste.backward.env.metamath import MMTheorem, MMTactic
from evariste.backward.graph import Theorem, Tactic, Proof
from evariste.backward.prover.bfs_proof_graph import (
    ProofTheoremNode,
    ProofTacticNode,
    ProofGraph,
)
from evariste.backward.prover.nomerge_search_tree import NoMergeSearchTree
from evariste.testing.common import MyTheorem, MyTactic


def my_thm(conclusion: str) -> Theorem:
    return MyTheorem(conclusion=conclusion)


def my_tac() -> Tactic:
    return MyTactic(unique_str="a tactic")


def my_node(id_: int) -> ProofTheoremNode:
    return ProofTheoremNode(thm=my_thm(str(id_)), id=id_)


def create_thms(n: int) -> List[Theorem]:
    return [my_thm(f"{i}") for i in range(n)]


def _proof_size(proof: Proof) -> int:
    assert isinstance(proof, tuple)  # mypy
    _, _, children = proof
    return 1 + sum(_proof_size(c) for c in children)


def set_children(
    node_id: int, sub_goals_groups: List[List[Theorem]], tree: NoMergeSearchTree
):
    tactics = []
    priors = []
    curr_prior = 0.5
    counter = len(tree.nodes)
    if len(sub_goals_groups) == 0:
        tree.set_expansion_failure(node_id, "no tacs")
        return
    assert len(sub_goals_groups) > 0
    for group in sub_goals_groups:
        assert isinstance(group, list)
        for sg in group:
            assert isinstance(sg, Theorem)
            assert int(sg.conclusion) == counter
            counter += 1
        tactics.append(my_tac())
        priors.append(curr_prior)
        curr_prior *= 0.5

    tree.expand_node(
        node_id=node_id, tactics=tactics, child_for_tac=sub_goals_groups, priors=priors
    )


def test_tree_0():
    """
     0
    (not expanded)
    """
    thms = create_thms(1)
    tree = NoMergeSearchTree(thms[0])
    assert not tree.root.expanded


def test_tree_1():
    """
     0
     |
     no_sg
    """
    thms = create_thms(1)
    tree = NoMergeSearchTree(thms[0])
    set_children(0, [[]], tree)
    assert 0 in tree.solved
    assert 0 not in tree.failed


def test_tree_2():
    """
     0
     |
     X
    """
    thms = create_thms(1)
    tree = NoMergeSearchTree(thms[0])
    tree.set_expansion_failure(0, "hoho")
    assert 0 not in tree.solved
    assert 0 in tree.failed
    assert tree.n_expandable_descendants[0] == 0


def test_tree_3():
    """
     0
     |
     1
    """
    thms = create_thms(2)
    tree = NoMergeSearchTree(thms[0])
    set_children(0, [[thms[1]]], tree)
    assert tree.solved == set()
    assert tree.failed == set()


def test_tree_4():
    """
     0
     |      \
     1       2
             \
              3
    """

    thms = create_thms(4)
    tree = NoMergeSearchTree(thms[0])
    set_children(0, [[thms[1]], [thms[2]]], tree)
    set_children(2, [[thms[3]]], tree)

    assert tree.solved == set()
    assert tree.failed == set()


def test_tree_5():
    """
     0
     |      \
     1       2
             \
              3
               \
                failure
    """

    thms = create_thms(4)
    tree = NoMergeSearchTree(thms[0])
    set_children(0, [[thms[1]], [thms[2]]], tree)
    set_children(2, [[thms[3]]], tree)
    set_children(3, [], tree)  # cycle
    assert tree.solved == set()
    assert tree.failed == {2, 3}


def create_tree(n_thms: int, edges: Dict[int, List[List[int]]]) -> NoMergeSearchTree:
    thms = create_thms(n_thms)
    tree = NoMergeSearchTree(thms[0])
    for key, values in edges.items():
        set_children(key, [[thms[i] for i in children] for children in values], tree)
    return tree


def test_tree_6():
    """
     0
     |      \\
     1       2
             \
                3
               \\   \\
                2   no_sg
    """

    tree = create_tree(5, {0: [[1], [2]], 2: [[3]], 3: [[4], []]})
    assert tree.solved == {0, 2, 3}
    assert tree.failed == set()


def test_tree_7():
    """
        0
     |      \
     1      (1,   2)
                  \
                   3
    """

    tree = create_tree(5, {0: [[1], [2, 3]], 3: [[4]]})
    assert tree.solved == set()
    assert tree.failed == set()


def test_tree_8():
    """
        0
     |        \
     1       (2,   3)
     |             \
    no_sg           4
                    \
                    no_sg
    """

    tree = create_tree(5, {0: [[1], [2, 3]], 3: [[4]], 4: [[]], 1: [[]]})
    assert tree.solved == {0, 1, 4, 3}
    assert tree.failed == set()
