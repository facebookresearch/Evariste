# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple, List, Dict

from evariste.envs.eq.graph import Node, A, B, ZERO
from evariste.envs.eq.graph import exp, ln, sqrt, sin, cos, tan, sinh, cosh, tanh
from evariste.envs.eq.rules import TRule


_UNARY_RULES: List[Tuple[Node, Node]] = [
    (A, -A),
    (A, exp(A)),
    (A, ln(A)),
    (A, A ** 2),
    (A, A.inv()),
    (A, sqrt(A)),
    (A, sin(A)),
    (A, cos(A)),
    (A, tan(A)),
    (A, sinh(A)),
    (A, cosh(A)),
    (A, tanh(A)),
    (A, abs(A)),
]

_BINARY_RULES: List[Tuple[Node, Node]] = [
    (A, A + B),
    (A, A - B),
    (A, A * B),
    (A, A / B),
]

_SPECIAL_RULES: List[Tuple[Node, Node]] = [
    (ZERO, A),
]


def t_rule(left: Node, right: Node, rule_type: str) -> TRule:
    return TRule(left, right, rule_type=rule_type)


UNARY_RULES = [t_rule(left, right, "unary") for left, right in _UNARY_RULES]
BINARY_RULES = [t_rule(left, right, "binary") for left, right in _BINARY_RULES]
SPECIAL_RULES = [t_rule(left, right, "special") for left, right in _SPECIAL_RULES]
RULES: List[TRule] = UNARY_RULES + BINARY_RULES + SPECIAL_RULES

INIT_RULE = SPECIAL_RULES[0]


# check no duplicated labels
assert len(RULES) == len({rule.name for rule in RULES})

NAME_TO_RULE: Dict[str, TRule] = {rule.name: rule for rule in RULES}
