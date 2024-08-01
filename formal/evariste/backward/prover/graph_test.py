# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from evariste.backward.prover.graph import Node, Graph, KillTactic
from evariste.backward.graph import Tactic, Theorem
from evariste.backward.prover.mcts import MCTSNode
from typing import List, Dict
import math
import time
import pickle
import pytest
import numpy as np

from evariste.testing.common import MyTheorem, MyTactic


def create_thms(n: int) -> List[Theorem]:
    return [MyTheorem(conclusion=f"{i}") for i in range(n)]


def _node(theorem: Theorem, children_for_tactic: List[List[Theorem]]):
    tactics: List[Tactic] = [MyTactic("") for _ in range(len(children_for_tactic))]
    return Node(theorem, tactics, children_for_tactic, 0)


# In the following tests, x = unvalid tactic, ø = solving tactic i.e no child
def test_tree_0():
    r"""
    First expansion, we expand the root.
             0     0      0      0
    0   ->   |     |     / \     |
             x     ø    1   ø    1
    """
    thms = create_thms(2)
    graph = Graph(thms[0])
    graph.add_nodes([_node(thms[0], [])])
    assert graph.nodes[thms[0]].is_bad()
    assert not graph.is_proved

    graph = Graph(thms[0])
    graph.add_nodes([_node(thms[0], [[]])])
    assert graph.nodes[thms[0]].solved
    assert graph.is_proved

    graph = Graph(thms[0])
    try:
        graph.add_nodes([_node(thms[0], [[], [1]])])
        assert False, "You cannot have a solving tactic + non solving one for leaf"
    except:
        assert True

    graph = Graph(thms[0])
    graph.add_nodes([_node(thms[0], [[1]])])
    assert not graph.nodes[thms[0]].solved
    assert not graph.is_proved


def test_tree_1():
    r"""
    Check solved
         0         0        0
        / \       / \      / \
       1  2,3    1  2,3   1  2,3
                    |        | |
                    ø        ø ø
    """
    t = create_thms(4)
    graph = Graph(t[0])
    graph.add_nodes([_node(t[0], [[t[1]], [t[2], t[3]]])])
    assert not graph.is_proved

    graph.add_nodes([_node(t[2], [[]])])
    assert not graph.is_proved

    graph.add_nodes([_node(t[3], [[]])])
    assert graph.is_proved

    graph.check_solved_ok()


def test_tree_2():
    r"""
    Check solved with cycles
    3 points to 2
          0              0
        /  \           /  \
      1,2 - 3   ->   1,2 - 3
      |     |        | |   |
      x     4        x ø   4
    not proved        proved
    """
    t = create_thms(5)
    graph = Graph(t[0])
    graph.add_nodes([_node(t[0], [[t[1], t[2]], [t[3]]])])
    graph.add_nodes([_node(t[1], [])])
    graph.add_nodes([_node(t[3], [[t[2]], [t[4]]])])
    assert not graph.is_proved

    graph.add_nodes([_node(t[2], [[]])])
    assert graph.is_proved
    graph.check_solved_ok()


def test_tree_3():
    r"""
    Check kill tactic + expandable when adding bad nodes, wo/ cycle
          0                0           0
        /  \             /  \         / \
      1,2   3          1,2   3      1,2  3
                       |            |    |
                       x            x    x
    """
    t = create_thms(4)
    graph = Graph(t[0])
    graph.add_nodes([_node(t[0], [[t[1], t[2]], [t[3]]])])
    assert not graph.is_proved
    assert len(graph.nodes[t[0]].killed_tactics) == 0
    graph.find_unexplored_and_propagate_expandable(ignore_solved=False)
    assert graph.unexplored_th == {t[1], t[2], t[3]}

    # we expand 1 as a dead node
    graph.add_nodes([_node(t[1], [])])
    graph.find_unexplored_and_propagate_expandable(ignore_solved=False)
    assert graph.unexplored_th == {t[3]}

    # we expand 3 as a dead node
    graph.add_nodes([_node(t[3], [])])
    graph.find_unexplored_and_propagate_expandable(ignore_solved=False)
    assert len(graph.unexplored_th) == 0
    assert graph.dead_root


def test_tree_3_1():
    r"""
    Check kill tactic + expandable when adding bad nodes, wo/ cycle
             0
        /    |    \
       1     2     3
       |     |     |
       9    4 5    6
       |    | |    |
       10   7 x    x
            |
            8
    """
    t = create_thms(11)
    graph = Graph(t[0])
    graph.add_nodes([_node(t[0], [[t[1]], [t[2]], [t[3]]])])
    graph.add_nodes([_node(t[1], [[t[9]]])])
    graph.add_nodes([_node(t[2], [[t[4], t[5]]])])
    graph.add_nodes([_node(t[3], [[t[6]]])])
    graph.add_nodes([_node(t[4], [[t[7]]])])
    graph.add_nodes([_node(t[5], [])])
    graph.add_nodes([_node(t[6], [])])
    graph.add_nodes([_node(t[7], [[t[8]]])])
    graph.add_nodes([_node(t[9], [[t[10]]])])

    graph.find_unexplored_and_propagate_expandable(ignore_solved=False)

    assert graph.unexplored_th == {t[10]}
    assert list(graph.nodes[graph.root].tactic_is_expandable) == [1, 0, 0]


def test_tree_3_2():
    r"""
    Check kill tactic + expandable when adding bad nodes, wo/ cycle
             0
        /    |    \
       1     2     3
       |     |  /  |
       9    4 5    6
       |    | |    |
       10   7 x    x
            |
            8
    """
    t = create_thms(11)
    graph = Graph(t[0])
    graph.add_nodes([_node(t[0], [[t[1]], [t[2]], [t[3]]])])
    graph.add_nodes([_node(t[1], [[t[9]]])])
    graph.add_nodes([_node(t[2], [[t[4], t[5]]])])
    graph.add_nodes([_node(t[3], [[t[6]], [t[4]]])])
    graph.add_nodes([_node(t[4], [[t[7]]])])
    graph.add_nodes([_node(t[5], [])])
    graph.add_nodes([_node(t[6], [])])
    graph.add_nodes([_node(t[7], [[t[8]]])])
    graph.add_nodes([_node(t[9], [[t[10]]])])

    graph.find_unexplored_and_propagate_expandable(ignore_solved=False)

    assert graph.unexplored_th == {t[10], t[8]}
    assert list(graph.nodes[graph.root].tactic_is_expandable) == [1, 0, 1]


def test_tree_3_3():
    r"""
    Check kill tactic + expandable when adding bad nodes, w/ cycle 3 -> 4 -> 1 -> 3
             0
        /    |    \
       1 ----2---> 3
       |  \  |  /  |
       9    4 5    6
       |    | |    |
       10   7 x    x
            |
            8
    """
    t = create_thms(11)
    graph = Graph(t[0])
    graph.add_nodes([_node(t[0], [[t[1]], [t[2]], [t[3]]])])
    graph.add_nodes([_node(t[1], [[t[9]], [t[3]]])])
    graph.add_nodes([_node(t[2], [[t[4], t[5]]])])
    graph.add_nodes([_node(t[3], [[t[6]], [t[4]]])])
    graph.add_nodes([_node(t[4], [[t[7]], [t[1]]])])
    graph.add_nodes([_node(t[5], [])])
    graph.add_nodes([_node(t[6], [])])
    graph.add_nodes([_node(t[7], [[t[8]]])])
    graph.add_nodes([_node(t[9], [[t[10]]])])

    graph.find_unexplored_and_propagate_expandable(ignore_solved=False)

    assert graph.unexplored_th == {t[10], t[8]}
    assert list(graph.nodes[graph.root].tactic_is_expandable) == [1, 0, 1]
    assert list(graph.nodes[t[1]].tactic_is_expandable) == [1, 1]


def test_tree_3_4():
    r"""
    Check kill tactic + expandable when adding bad nodes, w/ cycle 3 -> 4 -> 1 -> 3
             0
        /    |    \
       1 ----2---> 3
       |  \  |  /  |
       9    4 5    6
       |    | |    |
       10   7 x    x
       |    |
       ø    8
    """
    t = create_thms(11)
    graph = Graph(t[0])
    graph.add_nodes([_node(t[0], [[t[1]], [t[2]], [t[3]]])])
    graph.add_nodes([_node(t[1], [[t[9]], [t[3]]])])
    graph.add_nodes([_node(t[2], [[t[4], t[5]]])])
    graph.add_nodes([_node(t[3], [[t[6]], [t[4]]])])
    graph.add_nodes([_node(t[4], [[t[7]], [t[1]]])])
    graph.add_nodes([_node(t[5], [])])
    graph.add_nodes([_node(t[6], [])])
    graph.add_nodes([_node(t[7], [[t[8]]])])
    graph.add_nodes([_node(t[9], [[t[10]]])])
    graph.add_nodes([_node(t[10], [[]])])

    graph.find_unexplored_and_propagate_expandable(ignore_solved=False)
    assert graph.unexplored_th == {t[8]}

    assert list(graph.nodes[graph.root].tactic_is_expandable) == [1, 0, 1]
    assert list(graph.nodes[t[1]].tactic_is_expandable) == [0, 1]


def test_tree_4():
    r"""
    Check kill tactic + expandable when adding bad nodes, w/ cycle
    versions here: one -> 3 points to 2 // two -> 3 points to {1,2}
          0               0
        /  \             /  \
      1,2 - 3          1,2 - 3
                       |
                       x
    """
    # 3 only points to 2
    # 1 is not dead yet
    t = create_thms(4)
    graph = Graph(t[0])
    graph.add_nodes([_node(t[0], [[t[1], t[2]], [t[3]]])])
    graph.add_nodes([_node(t[3], [[t[2]]])])
    assert not graph.is_proved
    assert len(graph.nodes[t[0]].killed_tactics) == 0
    graph.find_unexplored_and_propagate_expandable(ignore_solved=False)
    assert graph.unexplored_th == {t[1], t[2]}
    # we expand 1 as a dead node
    graph.add_nodes([_node(t[1], [])])
    graph.find_unexplored_and_propagate_expandable(ignore_solved=False)
    assert graph.unexplored_th == {t[2]}

    # 3 only points to {1,2}
    graph = Graph(t[0])
    graph.add_nodes([_node(t[0], [[t[1], t[2]], [t[3]]])])
    graph.add_nodes([_node(t[3], [[t[1], t[2]]])])
    graph.add_nodes([_node(t[1], [])])
    graph.find_unexplored_and_propagate_expandable(ignore_solved=False)
    assert len(graph.unexplored_th) == 0
    assert graph.dead_root


def test_tree_5():
    r"""
    Check expandabe with solved nodes, wo cycles.
          0
        /  \
      1,2   3
            |
            ø
    """
    # 3 only points to 2
    # 1 is not dead yet
    t = create_thms(4)
    graph = Graph(t[0])
    graph.add_nodes([_node(t[0], [[t[1], t[2]], [t[3]]])])
    graph.find_unexplored_and_propagate_expandable(ignore_solved=True)
    assert graph.unexplored_th == {t[1], t[2], t[3]}
    # we expand 3 as a solved node
    graph.add_nodes([_node(t[3], [[]])])
    assert graph.is_proved
    graph.find_unexplored_and_propagate_expandable(ignore_solved=False)
    assert graph.unexplored_th == {t[1], t[2]}
    graph.find_unexplored_and_propagate_expandable(ignore_solved=True)
    assert len(graph.unexplored_th) == 0


def test_tree_6():
    r"""
    Check expandabe with solved nodes.
    3 points toward 1 only
          0
        /  \
      1,2 - 3
        |   |
        ø   4
    """
    # 3 only points to 2
    # 1 is not dead yet
    t = create_thms(5)
    graph = Graph(t[0])
    graph.add_nodes([_node(t[0], [[t[1], t[2]], [t[3]]])])
    graph.add_nodes([_node(t[3], [[t[2]], [t[4]]])])
    graph.find_unexplored_and_propagate_expandable(ignore_solved=True)
    assert graph.unexplored_th == {t[1], t[2], t[4]}
    # we expand 2 as a solved node
    graph.add_nodes([_node(t[2], [[]])])
    assert graph.is_proved
    graph.find_unexplored_and_propagate_expandable(ignore_solved=False)
    assert graph.unexplored_th == {t[1], t[4]}, graph.unexplored_th
    graph.find_unexplored_and_propagate_expandable(ignore_solved=True)
    assert len(graph.unexplored_th) == 0


def test_tree_7():
    r"""
    Check computation of minproof.
    No cycle.
             0
        /    |    \
       1     2     3
       |     |     |
       ø    4 5    6
            | |
            7 ø
            |
            8
            |
            ø
    """
    t = create_thms(9)
    graph = Graph(t[0])
    graph.add_nodes([_node(t[0], [[t[1]], [t[2]], [t[3]]])])
    graph.add_nodes([_node(t[1], [[]])])
    graph.add_nodes([_node(t[2], [[t[4], t[5]]])])
    graph.add_nodes([_node(t[3], [[t[6]]])])
    graph.add_nodes([_node(t[4], [[t[7]]])])
    graph.add_nodes([_node(t[5], [[]])])
    graph.add_nodes([_node(t[7], [[t[8]]])])
    graph.add_nodes([_node(t[8], [[]])])

    graph.get_inproof_nodes()
    graph.get_node_proof_sizes_and_depths()
    assert graph.nodes[t[0]].my_minproof_size["size"] == 2
    assert graph.nodes[t[0]].my_minproof_size["depth"] == 2
    assert graph.nodes[t[0]].my_minproof_tactics["size"] == [0]
    assert graph.nodes[t[0]].my_minproof_tactics["depth"] == [0]

    assert graph.nodes[t[1]].my_minproof_size["depth"] == 1
    assert graph.nodes[t[1]].my_minproof_size["size"] == 1

    assert graph.nodes[t[2]].my_minproof_size["size"] == 5
    assert graph.nodes[t[2]].my_minproof_size["depth"] == 4

    assert graph.nodes[t[3]].my_minproof_size["size"] == math.inf
    assert graph.nodes[t[3]].my_minproof_size["depth"] == math.inf
    assert graph.nodes[t[3]].my_minproof_tactics["size"] == []
    assert graph.nodes[t[3]].my_minproof_tactics["depth"] == []

    for stype in ["size", "depth"]:
        for i in [0, 1]:
            assert graph.nodes[t[i]].in_minproof[stype]
        for i in range(2, 9):
            if i == 6:
                continue
            assert not graph.nodes[t[i]].in_minproof[stype]


def test_tree_8():
    r"""
    Check computation of depth minproof w\ cycle.
    5 points toward 2.
        0
        |
        1
      /   \
     2    3,4
     |  \ | |
     6    5 ø
     |
     ø
    """
    t = create_thms(9)
    graph = Graph(t[0])
    graph.add_nodes([_node(t[0], [[t[1]]])])
    graph.add_nodes([_node(t[1], [[t[2]], [t[3], t[4]]])])
    graph.add_nodes([_node(t[2], [[t[6]]])])
    graph.add_nodes([_node(t[3], [[t[5]]])])
    graph.add_nodes([_node(t[4], [[]])])
    graph.add_nodes([_node(t[5], [[t[2]]])])
    graph.add_nodes([_node(t[6], [[]])])

    graph.get_inproof_nodes()
    graph.get_node_proof_sizes_and_depths()
    assert graph.nodes[t[0]].my_minproof_size["depth"] == 4
    assert graph.nodes[t[0]].my_minproof_tactics["depth"] == [0]

    assert graph.nodes[t[3]].my_minproof_size["depth"] == 4

    for stype in ["size", "depth"]:
        for i in [0, 1, 2, 6]:
            assert graph.nodes[t[i]].in_minproof[stype]
        for i in [3, 4, 5]:
            assert not graph.nodes[t[i]].in_minproof[stype]


def test_tree_8_1():
    r"""
    Check computation of depth minproof w\ cycle.
    5 points toward 2.
        0
        |
        1
      /   \
     2    3,4
     |  \ | |
     6 - 5  ø
     |
     7
     |
     ø
    """
    t = create_thms(9)
    graph = Graph(t[0])
    graph.add_nodes([_node(t[0], [[t[1]]])])
    graph.add_nodes([_node(t[1], [[t[2]], [t[3], t[4]]])])
    graph.add_nodes([_node(t[2], [[t[6]]])])
    graph.add_nodes([_node(t[3], [[t[5]]])])
    graph.add_nodes([_node(t[4], [[]])])
    graph.add_nodes([_node(t[5], [[t[2]]])])
    graph.add_nodes([_node(t[6], [[t[5]], [t[7]]])])
    graph.add_nodes([_node(t[7], [[]])])

    graph.get_inproof_nodes()
    graph.get_node_proof_sizes_and_depths()
    assert graph.nodes[t[0]].my_minproof_size["depth"] == 5
    assert graph.nodes[t[0]].my_minproof_tactics["depth"] == [0]

    assert graph.nodes[t[3]].my_minproof_size["depth"] == 5

    for stype in ["size", "depth"]:
        for i in [0, 1, 2, 6]:
            assert graph.nodes[t[i]].in_minproof[stype]
        for i in [3, 4, 5]:
            assert not graph.nodes[t[i]].in_minproof[stype]


def test_tree_9_0():
    r"""
    Check computation of size minproof w\ diamond.
    2 points toward 3.
           0
        /    \
    1,2,3     4
    | | |     |
    ø ø ø     ø
    """
    t = create_thms(9)
    graph = Graph(t[0])
    graph.add_nodes([_node(t[0], [[t[1], t[2], t[3]], [t[4]]])])
    graph.add_nodes([_node(t[1], [[]])])
    graph.add_nodes([_node(t[2], [[]])])
    graph.add_nodes([_node(t[3], [[]])])
    graph.add_nodes([_node(t[4], [[]])])

    graph.get_inproof_nodes()
    graph.get_node_proof_sizes_and_depths()
    assert graph.nodes[t[0]].my_minproof_size["size"] == 2
    assert list(graph.nodes[t[0]].my_minproof_size_for_tactic["size"]) == [4, 2]


def test_tree_9_1():
    r"""
    Check computation of size minproof w\ diamond.
    2 points toward 3.
           0
        /    \
       4    1,2,3
       |    | | |
       ø    ø ø ø
    """
    t = create_thms(9)
    graph = Graph(t[0])
    graph.add_nodes([_node(t[0], [[t[4]], [t[1], t[2], t[3]]])])
    graph.add_nodes([_node(t[1], [[]])])
    graph.add_nodes([_node(t[2], [[]])])
    graph.add_nodes([_node(t[3], [[]])])
    graph.add_nodes([_node(t[4], [[]])])

    graph.get_inproof_nodes()
    graph.get_node_proof_sizes_and_depths()
    assert graph.nodes[t[0]].my_minproof_size["size"] == 2
    assert list(graph.nodes[t[0]].my_minproof_size_for_tactic["size"]) == [2, 4]


def test_tree_9_2():
    r"""
    Check computation of size minproof w\ diamond.
    2 points toward 3.
           0
        /    \
    1,2,3     4
    | | |     |
    ø ø ø     5
              |
              ø
    """
    t = create_thms(9)
    graph = Graph(t[0])
    graph.add_nodes([_node(t[0], [[t[1], t[2], t[3]], [t[4]]])])
    graph.add_nodes([_node(t[1], [[]])])
    graph.add_nodes([_node(t[2], [[]])])
    graph.add_nodes([_node(t[3], [[]])])
    graph.add_nodes([_node(t[4], [[t[5]]])])
    graph.add_nodes([_node(t[5], [[]])])

    graph.get_inproof_nodes()
    graph.get_node_proof_sizes_and_depths()
    assert graph.nodes[t[0]].my_minproof_size["size"] == 3
    assert list(graph.nodes[t[0]].my_minproof_size_for_tactic["size"]) == [4, 3]


def test_tree_10():
    r"""
    Check computation of size minproof w\ diamond.
    2 points toward 3.
        0
      /   \
    1      2
    |  /   |
    3    4,5,6
    |    | | |
    7    ø ø ø
    |
    ø
    """
    t = create_thms(9)
    graph = Graph(t[0])
    graph.add_nodes([_node(t[0], [[t[1]], [t[2]]])])
    graph.add_nodes([_node(t[1], [[t[3]]])])
    graph.add_nodes([_node(t[2], [[t[3]], [t[4], t[5], t[6]]])])
    graph.add_nodes([_node(t[3], [[t[7]]])])
    graph.add_nodes([_node(t[4], [[]])])
    graph.add_nodes([_node(t[5], [[]])])
    graph.add_nodes([_node(t[6], [[]])])
    graph.add_nodes([_node(t[7], [[]])])

    graph.get_inproof_nodes()
    graph.get_node_proof_sizes_and_depths()
    assert graph.nodes[t[0]].my_minproof_size["size"] == 4
    assert graph.nodes[t[0]].my_minproof_tactics["size"] == [0, 1]
    assert graph.nodes[t[2]].my_minproof_size["size"] == 3
    assert graph.nodes[t[2]].my_minproof_tactics["size"] == [0]

    for i in [0, 1, 2, 3, 7]:
        assert graph.nodes[t[i]].in_minproof["size"]
    for i in [4, 5, 6]:
        assert not graph.nodes[t[i]].in_minproof["size"]


def test_tree_11():
    r"""
    Check minproof, w/ cycle 3 -> 4 -> 1 -> 3
             0
        /    |    \
       1 ----2---> 3
       |  \  |  /  |
       9    4 5    6
       |    | |    |
       ø    7 ø    x
            |
            8
            |
            10
            |
            ø
    """
    t = create_thms(11)
    graph = Graph(t[0])
    graph.add_nodes([_node(t[0], [[t[1]], [t[2]], [t[3]]])])
    graph.add_nodes([_node(t[1], [[t[9]], [t[3]]])])
    graph.add_nodes([_node(t[2], [[t[4], t[5]]])])
    graph.add_nodes([_node(t[3], [[t[6]], [t[4]]])])
    graph.add_nodes([_node(t[4], [[t[7]], [t[1]]])])
    graph.add_nodes([_node(t[5], [[]])])
    graph.add_nodes([_node(t[6], [])])
    graph.add_nodes([_node(t[7], [[t[8]]])])
    graph.add_nodes([_node(t[8], [[t[10]]])])
    graph.add_nodes([_node(t[9], [[]])])
    graph.add_nodes([_node(t[10], [[]])])

    graph.get_inproof_nodes()
    graph.get_node_proof_sizes_and_depths()

    assert graph.minproof_size["size"] == 3
    assert graph.minproof_size["depth"] == 3
    assert graph.nodes[t[0]].my_minproof_size["size"] == 3
    assert graph.nodes[t[0]].my_minproof_tactics["size"] == [0]
    assert graph.nodes[t[0]].my_minproof_size_for_tactic["size"][1] == 6
    assert graph.nodes[t[1]].my_minproof_size_for_tactic["size"][1] == 5

    for i in [0, 1, 9]:
        assert graph.nodes[t[i]].in_minproof["size"]
    for i in [2, 3, 4, 5, 6, 7, 8, 10]:
        assert not graph.nodes[t[i]].in_minproof["size"]


def test_tree_12():
    r"""
    Check solved with real cycles
           0
       /      \
    1 ,  2     3
    |   / \\  /
    4  5    6
       |
       ø
    Proved only with cycle 6 -> 2
    """
    t = create_thms(7)
    graph = Graph(t[0])
    graph.add_nodes([_node(t[0], [[t[1], t[2]], [t[3]]])])
    graph.add_nodes([_node(t[1], [[t[4]]])])
    graph.add_nodes([_node(t[2], [[t[5]], [t[6]]])])
    graph.add_nodes([_node(t[3], [[t[6]]])])
    graph.add_nodes([_node(t[5], [[]])])

    assert not graph.is_proved

    graph.add_nodes([_node(t[6], [[t[2]]])])
    assert graph.is_proved
    graph.check_solved_ok()


def nodes_are_equal(nodes1: Dict[Theorem, Node], nodes2: Dict[Theorem, MCTSNode]):
    want_to_test = [
        "children_for_tactic",
        "killed_tactics",
        "killed_mask",
        "solving_tactics",
        "solved",
        "is_solved_leaf",
        "in_proof",
        "my_minproof_size",
        "in_minproof",
    ]
    assert set(nodes1.keys()) == set(nodes2.keys())
    for th in nodes1.keys():
        node1 = nodes1[th]
        node2 = nodes2[th]
        for attr_name in want_to_test:
            v1 = getattr(node1, attr_name)
            v2 = getattr(node2, attr_name)
            if isinstance(v2, np.ndarray):
                assert (v2 == v1).all()
            else:
                assert (v2, attr_name) == (v1, attr_name)


def reload_mcts(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    root = data["mcts_root"]
    nodes = data["mcts_nodes"]
    permanent_ancestors = data["permanent_ancestors"]
    ancestors = data["ancestors"]
    history = data["history"]
    return root, nodes, permanent_ancestors, ancestors, history


data = [
    {
        "label": "identities_((tanh(x0)_**_2)_==_((cosh((2_*_x0))_+_(-1))_*_(1|(1_+_cosh((2_*_x0))))))__qWzmjjDmza.pkl",
        "is_proved": True,
        "n_nodes": 514,
        "history_len": 108,
        "kill_tactic_cycles": 0,
        "time_build": (0.01, 0.1),
        "time_get_inproof": (5e-4, 1e-3),
        "time_get_node_proof_sizes_and_depths": (5e-3, 2e-2),
    },
    {
        "label": "identities_(sin(((2_*_PI)_+_x0))_==_sin(x0))__OtaEfuycly.pkl",
        "is_proved": False,
        "n_nodes": 1004,
        "history_len": 944,
        "kill_tactic_cycles": 800,
        "time_build": (0.01, 0.15),
        "time_get_inproof": (1e-6, 1.2e-5),
        "time_get_node_proof_sizes_and_depths": (1e-4, 1.2e-3),
    },
    {
        "label": "identities_((tanh(x0)_+_tanh(x1))_==_(sinh((x0_+_x1))_*_(1|(cosh(x0)_*_cosh(x1)))))__QcNdTPCWfD.pkl",
        "is_proved": True,
        "n_nodes": 440,
        "history_len": 114,
        "kill_tactic_cycles": 2,
        "time_build": (0.01, 0.04),
        "time_get_inproof": (6e-4, 1.2e-3),
        "time_get_node_proof_sizes_and_depths": (5e-3, 2e-2),
    },
]
folder = "PATH/data/tests/graph_test/"


@pytest.mark.slow
def test_tree_slow_1():
    # Load equations graph and
    # Check that if we rebuild graph we find back the same statistics
    for dp in data:
        label = dp["label"]
        root, nodes, permanent_ancestors, ancestors, history = reload_mcts(
            folder + label
        )
        re_graph = Graph.build_from_history(root, nodes, history)
        assert len(re_graph.nodes) == dp["n_nodes"]
        assert len(re_graph.history) == dp["history_len"]
        assert (
            sum([isinstance(action, KillTactic) for action in history])
            == dp["kill_tactic_cycles"]
        )
        assert re_graph.is_proved == dp["is_proved"]
        assert permanent_ancestors == re_graph.permanent_ancestors
        assert ancestors == re_graph.ancestors
        assert history == re_graph.history

        re_graph.get_inproof_nodes()
        re_graph.get_node_proof_sizes_and_depths()

        nodes_are_equal(re_graph.nodes, nodes)


@pytest.mark.slow
def test_tree_slow_2():
    # Check that if graph algorithm timings
    for dp in data:
        label = dp["label"]
        root, nodes, _, _, history = reload_mcts(folder + label)
        start = time.time()
        re_graph = Graph.build_from_history(root, nodes, history)
        diff = time.time() - start
        assert dp["time_build"][0] <= diff <= dp["time_build"][1]

        start = time.time()
        re_graph.get_inproof_nodes()
        diff = time.time() - start
        assert dp["time_get_inproof"][0] <= diff <= dp["time_get_inproof"][1]
        start = time.time()
        re_graph.get_node_proof_sizes_and_depths()
        diff = time.time() - start
        assert (
            dp["time_get_node_proof_sizes_and_depths"][0]
            <= diff
            <= dp["time_get_node_proof_sizes_and_depths"][1]
        ), (dp["label"], diff)
