# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple, List, Dict
import re
import math
import numexpr as ne
from evariste.envs.eq.rules import TRule
from evariste.envs.eq.graph import Node, A, B, C, ZERO, ONE
from evariste.utils import MyTimeoutError
from params import Params, ConfStore
from evariste.envs.eq.rules import TRule
from evariste.envs.eq.graph import Node, ZERO, infix_for_numexpr  # , INode, VNode
from evariste.envs.sr.rules import RULES, INIT_RULE
from evariste.envs.eq.graph import (
    Node,
    CNode,
    VNode,
    INode,
    UNode,
    BNode,
    PNode,
    U_OPS,
    C_OPS,
    infix_for_numexpr,
    Min,
    Max,
    GCD,
    LCM,
)
from evariste.envs.eq.utils import simplify_equation


simplification_rules = [
    TRule(A + B - A, B),
    TRule(A + B + C - A, B + C),
    TRule(A + B - C - A, B - C),
    TRule(A - B - C - A, -B - C),
    TRule(A + (B - A), B),
    TRule(A / A, ONE),
    TRule(A - A, ZERO),
]


def simplify_via_rules(eq):
    curr_eq = eq
    while True:
        n_simplif = 0
        for rule in simplification_rules:
            matches = rule.eligible(curr_eq, fwd=True)
            if len(matches) == 0:
                continue
            else:
                positions = [pos for pos, _ in matches]
                assert sorted(positions) == positions
                for pos in positions[::-1]:
                    curr_eq = rule.apply(curr_eq, fwd=True, prefix_pos=pos)["eq"]
                n_simplif += 1
        if n_simplif == 0:
            break
    curr_eq = simplify_equation(curr_eq)
    # if curr_eq.is_valid():
    #    return curr_eq
    # else:
    #    raise NotValidExpression("Not valid. Before simplif: {}, after: {}".format(eq, curr_eq))
    return curr_eq


class NotValidExpression(Exception):
    pass


if __name__ == "__main__":

    # python -m evariste.envs.sr.simplification

    import evariste.datasets

    def test_simplify():

        x = VNode("x0")
        y = VNode("x1")

        tests: List[Tuple[Node, Node]] = [
            (x + 7 - 2 - x, INode(5)),
            # (x-y-y-x, -y-y),
            ((x + 1) ** 2 / (x + 1) ** 2, ONE),
        ]
        for x, y in tests:
            simplified_x = simplify_via_rules(x)
            assert simplified_x.eq(y), "Expected {}, got {}".format(y, simplified_x)
        print("Passed tests")

    test_simplify()
