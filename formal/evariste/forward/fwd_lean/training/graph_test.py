# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from evariste.forward.fwd_lean.training.common import (
    detect_cycles,
    TacticAndChildren,
    LeanMetaProofNode,
)


def test_detect_cycles():
    root = LeanMetaProofNode(thm="root", tactics_and_children=[])
    has_cycle = detect_cycles(root)
    assert not has_cycle


def test_detect_cycles_simple():
    root = LeanMetaProofNode(thm="root", tactics_and_children=[])
    tac = TacticAndChildren(tactic="tactic 0", children=[root])
    root.tactics_and_children.append(tac)

    has_cycle = detect_cycles(root)
    assert has_cycle


def test_detect_cycles_1():
    root = LeanMetaProofNode(thm="root", tactics_and_children=[])
    other1 = LeanMetaProofNode(thm="other 1", tactics_and_children=[])
    other2 = LeanMetaProofNode(thm="other 2", tactics_and_children=[])
    other3 = LeanMetaProofNode(thm="other 3", tactics_and_children=[])
    tac1 = TacticAndChildren(tactic="tactic 1", children=[other1, other2])
    tac2 = TacticAndChildren(tactic="tactic 2", children=[other1, other3])
    tac3 = TacticAndChildren(tactic="tactic 3", children=[other2, root])

    root.tactics_and_children.append(tac1)
    root.tactics_and_children.append(tac2)
    other3.tactics_and_children.append(tac3)

    has_cycle = detect_cycles(root)
    assert has_cycle


def test_detect_cycles_2():
    root = LeanMetaProofNode(thm="root", tactics_and_children=[])
    other1 = LeanMetaProofNode(thm="other 1", tactics_and_children=[])
    other2 = LeanMetaProofNode(thm="other 2", tactics_and_children=[])
    other3 = LeanMetaProofNode(thm="other 3", tactics_and_children=[])
    tac1 = TacticAndChildren(tactic="tactic 1", children=[other1, other2])
    tac2 = TacticAndChildren(tactic="tactic 2", children=[other1, other3])
    tac3 = TacticAndChildren(tactic="tactic 3", children=[other2, other1])

    root.tactics_and_children.append(tac1)
    root.tactics_and_children.append(tac2)
    other3.tactics_and_children.append(tac3)

    has_cycle = detect_cycles(root)
    assert not has_cycle


def test_n_possible_dag():
    root = LeanMetaProofNode(thm="root", tactics_and_children=[])
    other1 = LeanMetaProofNode(thm="other 1", tactics_and_children=[])
    other2 = LeanMetaProofNode(thm="other 2", tactics_and_children=[])
    other3 = LeanMetaProofNode(thm="other 3", tactics_and_children=[])
    tac1 = TacticAndChildren(tactic="tactic 1", children=[other1, other2])
    tac2 = TacticAndChildren(tactic="tactic 3", children=[other3, other1])
    finish_tac = TacticAndChildren(tactic="finish tactic", children=[])

    root.tactics_and_children.append(tac1)
    other1.tactics_and_children.append(finish_tac)
    other2.tactics_and_children.append(tac2)
    other3.tactics_and_children.append(finish_tac)

    assert not detect_cycles(root)
    assert root.n_possible_dags() == 1


def test_n_possible_dag_2():
    finish_tac = TacticAndChildren(tactic="finish tactic", children=[])
    simple1 = LeanMetaProofNode(thm="simple1", tactics_and_children=[])
    simple1.tactics_and_children.append(finish_tac)

    simple2 = LeanMetaProofNode(thm="simple2", tactics_and_children=[])
    simple2.tactics_and_children.append(finish_tac)

    tac1 = TacticAndChildren(tactic="tactic 1", children=[simple1, simple2])
    tac2 = TacticAndChildren(tactic="tactic 2", children=[simple1, simple2])
    double1 = LeanMetaProofNode(thm="double1", tactics_and_children=[tac1, tac2])
    double2 = LeanMetaProofNode(thm="double2", tactics_and_children=[tac1, tac2])

    tac3 = TacticAndChildren(tactic="tactic 3", children=[double1])
    tac4 = TacticAndChildren(tactic="tactic 4", children=[double2])

    root = LeanMetaProofNode(thm="root", tactics_and_children=[])
    root.tactics_and_children.append(tac3)
    root.tactics_and_children.append(tac4)

    assert not detect_cycles(root)
    assert root.n_possible_dags() == 4
