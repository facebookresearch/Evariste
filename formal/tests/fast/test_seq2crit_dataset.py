# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Optional
import unittest

from evariste.envs.mm.utils import Node_a_p


class SpoofNode:
    def __init__(self, statement: List[str], children: Optional[List["SpoofNode"]]):
        if children is not None:
            self.children = children
        self.statement = statement

    def set_nodes_and_depth(self):
        Node_a_p.set_nodes_and_depth(self)


class TestComputeCritVal(unittest.TestCase):
    def test_set_steps_and_depth(self):
        tree = SpoofNode(
            ["|-"],
            [  # first tactic
                SpoofNode(
                    ["|-"], [SpoofNode(["|-"], None)]  # second tactic  # terminal
                ),
                SpoofNode(
                    ["|-"], [SpoofNode(["|-"], None)]  # third tactic  # terminal
                ),
                SpoofNode(
                    ["|-"],
                    [  # fourth tactic
                        SpoofNode(
                            ["class"],
                            [  # fifth tactic
                                SpoofNode(["|-"], None)  # maximal depth = 3
                            ],
                        ),
                        SpoofNode(["class"], None),  # terminal
                    ],
                ),
            ],
        )
        tree.set_nodes_and_depth()
        self.assertEqual(tree.nodes["syntactic"], 5, "Wrong number of nodes")
        self.assertEqual(tree.depth["syntactic"], 3, "Wrong depth")
        self.assertEqual(tree.nodes["no_syntactic"], 4, "Wrong number of nodes")
        self.assertEqual(tree.depth["no_syntactic"], 2, "Wrong depth")


if __name__ == "__main__":
    unittest.main()
