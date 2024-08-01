# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List

from evariste.backward.env.metamath import MMTheorem, MMTactic
from evariste.backward.graph import Theorem, Tactic, Proof
from evariste.backward.prover.bfs_proof_graph import (
    ProofTheoremNode,
    ProofTacticNode,
    ProofGraph,
)


def my_thm(conclusion: str) -> Theorem:
    return MMTheorem(conclusion=conclusion, hyps=[])


def my_tac() -> Tactic:
    return MMTactic(label="a tactic", subs={})


def my_node(id_: int) -> ProofTheoremNode:
    return ProofTheoremNode(thm=my_thm(str(id_)), id=id_)


def create_thms(n: int) -> List[Theorem]:
    return [my_thm(f"{i}") for i in range(n)]


def _proof_size(proof: Proof) -> int:
    assert isinstance(proof, tuple)  # mypy
    _, _, children = proof
    return 1 + sum(_proof_size(c) for c in children)


def set_children(
    thm: Theorem, sub_goals_groups: List[List[Theorem]], graph: ProofGraph
):
    tactics = []
    assert len(sub_goals_groups) > 0
    for group in sub_goals_groups:
        assert isinstance(group, list)
        for sg in group:
            assert isinstance(sg, Theorem)
        tactics.append(my_tac())

    graph.set_expansion(thm=thm, tactics=tactics, child_for_tac=sub_goals_groups)


def test_tree_0():
    """
     0
    (not expanded)
    """
    thms = create_thms(1)
    graph = ProofGraph(thms[0])
    graph.update_state()
    assert not graph.is_proved(thms[0])
    assert graph._leafs_worth_to_expand == {thms[0]}


def test_tree_1():
    """
     0
     |
     no_sg
    """
    thms = create_thms(1)
    graph = ProofGraph(thms[0])
    set_children(thms[0], [[]], graph)
    graph.update_state()
    assert graph.is_proved(thms[0])
    proof_size = graph.proof_size(thms[0])
    assert proof_size == 1
    assert graph._leafs_worth_to_expand == set()


def test_tree_3():
    """
     0
     |      \\
     no_sg   1
    """
    thms = create_thms(2)
    graph = ProofGraph(thms[0])
    set_children(thms[0], [[], [thms[1]]], graph)
    graph.update_state()
    assert graph.root.expanded
    assert graph.is_proved(thms[0])
    proof_size = graph.proof_size(thms[0])
    assert proof_size == 1
    assert graph._leafs_worth_to_expand == {thms[1]}


def test_tree_4():
    """
     0
     |      \
     1       2
             \
              3
    """

    thms = create_thms(4)
    graph = ProofGraph(thms[0])
    set_children(thms[0], [[thms[1]], [thms[2]]], graph)
    set_children(thms[2], [[thms[3]]], graph)
    graph.update_state()
    assert not graph.is_proved(thms[0])
    assert graph._leafs_worth_to_expand == {thms[1], thms[3]}


def test_tree_5():
    """
     0
     |      \
     1       2
             \\
              3
               \\
                2
    """

    thms = create_thms(4)
    graph = ProofGraph(thms[0])
    set_children(thms[0], [[thms[1]], [thms[2]]], graph)
    set_children(thms[2], [[thms[3]]], graph)
    set_children(thms[3], [[thms[2]]], graph)
    graph.update_state()
    assert not graph.is_proved(thms[0])
    assert graph._leafs_worth_to_expand == {thms[1]}


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

    thms = create_thms(4)
    graph = ProofGraph(thms[0])
    set_children(thms[0], [[thms[1]], [thms[2]]], graph)
    set_children(thms[2], [[thms[3]]], graph)
    set_children(thms[3], [[thms[2]], []], graph)
    graph.update_state()
    assert graph.is_proved(thms[0])
    proof_size = graph.proof_size(thms[0])
    assert proof_size == 3
    assert graph._leafs_worth_to_expand == {thms[1]}


def test_tree_7():
    """
        0
     |      \
     1      (1,   2)
                  \
                   3
    """

    thms = create_thms(4)
    graph = ProofGraph(thms[0])
    set_children(thms[0], [[thms[1]], [thms[1], thms[2]]], graph)
    set_children(thms[2], [[thms[3]]], graph)
    graph.update_state()
    assert not graph.is_proved(thms[0])
    assert graph._leafs_worth_to_expand == {thms[1], thms[3]}


def test_tree_8():
    """
        0
     |        \
     1       (1,   2)
     |             \
    no_sg           3
                    \
                    no_sg
    """

    thms = create_thms(4)
    graph = ProofGraph(thms[0])
    set_children(thms[0], [[thms[1]], [thms[1], thms[2]]], graph)
    set_children(thms[2], [[thms[3]]], graph)
    set_children(thms[3], [[]], graph)
    set_children(thms[1], [[]], graph)
    graph.update_state()
    assert graph.is_proved(thms[0])
    proof_size = graph.proof_size(thms[0])
    assert proof_size == 2
    assert graph._leafs_worth_to_expand == set()


def test_tree_9():
    """
        0
     |        \
     1       (3,      2)
             |  |     |
            2  no_sg  3

    """

    thms = create_thms(4)
    graph = ProofGraph(thms[0])
    set_children(thms[0], [[thms[1]], [thms[3], thms[2]]], graph)
    set_children(thms[2], [[thms[3]]], graph)
    set_children(thms[3], [[]], graph)
    graph.update_state()
    assert graph.is_proved(thms[0])
    proof_size = graph.proof_size(thms[0])
    assert proof_size == 4
    for thm in graph._proved:
        proof = graph.proof(thm)
        assert _proof_size(proof) == graph.proof_size(thm)
    assert graph._leafs_worth_to_expand == {thms[1]}


def test_tree_10():
    """
        0
     |        \
     1       (3,   2)
             /      \
            no_sg    3

    """

    thms = create_thms(4)
    graph = ProofGraph(thms[0])
    set_children(thms[0], [[thms[1]], [thms[3], thms[2]]], graph)
    set_children(thms[2], [[thms[3]]], graph)
    set_children(thms[3], [[]], graph)
    graph.update_state()
    assert graph.is_proved(thms[0])
    proof_size = graph.proof_size(thms[0])
    assert proof_size == 4
    assert graph._leafs_worth_to_expand == {thms[1]}
    for thm in graph._proved:
        proof = graph.proof(thm)
        assert _proof_size(proof) == graph.proof_size(thm)


def test_tree_11():
    """
               0
     |         |           |
     1       (3,   2)    (4, 5)
              |           |   |
              X           x   6

    """

    thms = create_thms(7)
    graph = ProofGraph(thms[0])
    set_children(thms[0], [[thms[1]], [thms[3], thms[2]], [thms[4], thms[5]]], graph)
    graph.set_expansion_failure(thms[3], "failure")
    graph.set_expansion_failure(thms[4], "failure")
    set_children(thms[5], [[thms[6]]], graph)
    graph.update_state()
    assert not graph.is_failed(thms[0])
    assert graph.is_failed(thms[3])
    assert graph._leafs_worth_to_expand == {thms[1]}
    for thm in graph._proved:
        proof = graph.proof(thm)
        assert _proof_size(proof) == graph.proof_size(thm)


def test_tree_12():
    """
                  0
            |           |       |
        (1,   2)     (3, 4)     6
         x    |       |  x
              4       2

    """

    thms = create_thms(7)
    graph = ProofGraph(thms[0])
    set_children(thms[0], [[thms[1], thms[2]], [thms[3], thms[4]], [thms[6]]], graph)
    graph.set_expansion_failure(thms[1], "failure")
    set_children(thms[2], [[thms[4]]], graph)
    set_children(thms[3], [[thms[2]]], graph)
    graph.set_expansion_failure(thms[4], "failure")
    graph.update_state()
    assert not graph.is_failed(thms[0])
    assert graph._leafs_worth_to_expand == {thms[6]}
    for thm in graph._proved:
        proof = graph.proof(thm)
        assert _proof_size(proof) == graph.proof_size(thm)


def test_tree_13():
    """
                0
                |
           (2,   3,   1)
            |    |    ok
            1    1



    """

    thms = create_thms(4)
    graph = ProofGraph(thms[0])
    set_children(thms[0], [[thms[2], thms[3], thms[1]]], graph)
    set_children(thms[3], [[thms[1]]], graph)
    set_children(thms[2], [[thms[1]]], graph)
    set_children(thms[1], [[]], graph)
    graph.update_state()
    assert graph.is_proved(thms[0])
    assert graph._leafs_worth_to_expand == set()
    assert graph._proved == {thms[0], thms[1], thms[2], thms[3]}
    for thm in graph._proved:
        proof = graph.proof(thm)
        assert _proof_size(proof) == graph.proof_size(thm)
