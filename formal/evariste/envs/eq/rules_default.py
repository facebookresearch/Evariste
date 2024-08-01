# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple, Union, List

from evariste.envs.eq.graph import Node
from evariste.envs.eq.graph import ZERO, ONE, PI
from evariste.envs.eq.graph import a, b, c, d, minus_a
from evariste.envs.eq.graph import A, B, C, D
from evariste.envs.eq.graph import Max, exp, ln, sqrt
from evariste.envs.eq.graph import sin, cos, tan, asin, acos, atan
from evariste.envs.eq.graph import sinh, cosh, tanh, asinh, acosh, atanh

from evariste.envs.eq.rules import (
    TRule,
    ARule,
    test_duplicated_rules,
    test_valid_transform_rules_e_numeric,
    test_valid_transform_rules_c_numeric,
    test_valid_assert_rules_numeric,
    test_eval_assert,
)


T_RULE_TYPE = Union[Tuple[Node, Node], Tuple[Node, Node, List[Node]]]
A_RULE_TYPE = Union[Node, Tuple[Node, List[Node]]]


#
# basic rules
#
RULES_BASIC_T_E: List[T_RULE_TYPE] = [
    (A, B, [A == B]),
    (ONE ** 2, ONE),
    (ZERO ** 2, ZERO),
    (ONE.inv(), ONE),
    (sqrt(ZERO), ZERO),
    (sqrt(ONE), ONE),
    (A + B, B + A),
    (A * B, B * A),
    ((A + B) + C, A + (B + C)),
    ((A * B) * C, A * (B * C)),
    ((A + B) * C, (A * C) + (B * C)),
    (A + 0, A),
    (A * 0, ZERO),
    (A * 1, A),
    (A - A, ZERO),
    (A / A, ONE, [A != ZERO]),
    (A - B, A + (-B)),
    (-(A * B), (-A) * B),
    ((-A) * (-B), A * B),
    (-(-A), A),
    ((-1) * A, -A),
    (-(A + B), -A - B),
    (-(A - B), -A + B),
    (A ** 2, A * A),
    ((-A) ** 2, A ** 2),
    ((A * B) ** 2, (A ** 2) * (B ** 2)),
    ((A.inv()).inv(), A, [A != ZERO]),
    (A.inv(), ONE / A, [A != ZERO]),
    ((A / B).inv(), B / A, [A != ZERO, B != ZERO]),
    (A / B, A * B.inv(), [B != ZERO]),
    ((A * B).inv(), (A.inv()) * (B.inv()), [A != ZERO, B != ZERO]),
    (ONE / (A * B), (ONE / A) * (ONE / B), [A != ZERO, B != ZERO]),
    ((-A) / (-B), A / B, [A != ZERO, B != ZERO]),
    ((A.inv()) ** 2, (A ** 2).inv(), [A != ZERO]),
    (sqrt(abs(A)) ** 2, abs(A)),
    (sqrt(A ** 2), abs(A)),
    (sqrt(A * B), sqrt(A) * sqrt(B), [A >= 0, B >= 0]),
    (sqrt(A.inv()), sqrt(A).inv(), [A > 0]),
    (abs(-A), abs(A)),
]

RULES_BASIC_T_C: List[T_RULE_TYPE] = [
    (A == B, B == A),
    (A > 0, A >= 0, [A != ZERO]),
    (A < 0, A <= 0, [A != ZERO]),
    (A - B == C, A == B + C),
    (A / B == C, A == B * C, [B != ZERO]),
    (A + C == B + C, A == B),
    (A * C == B * C, A == B, [C != ZERO]),
    (A / C == B / C, A == B, [C != ZERO]),
    (A * C <= B * C, A <= B, [C > 0]),
    (A * C < B * C, A < B, [C > 0]),
    (-B <= -A, A <= B),
    (-B < -A, A < B),
    (A != B, B != A),
    (A - B != C, A != B + C),
    (A / B != C, A != B * C, [B != ZERO]),
    (A + C != B + C, A != B),
    (A * C != B * C, A != B, [C != ZERO]),
    (A * B != ZERO, A != ZERO, [B != ZERO]),
    (A / C != B / C, A != B, [C != ZERO]),
    # inv
    (A.inv() < B.inv(), A > B, [A > 0, B > 0]),
    (A.inv() < B.inv(), B < A, [A < 0, B < 0]),
    (A.inv() <= B.inv(), A >= B, [A > 0, B > 0]),
    (A.inv() <= B.inv(), B <= A, [A < 0, B < 0]),
    # pow2
    (A ** 2 == B ** 2, A == B, [A * B >= 0]),
    (A ** 2 != B ** 2, A != B, [A * B >= 0]),
    (A ** 2 < B ** 2, B > A, [A >= 0, B >= 0]),
    (A ** 2 <= B ** 2, B >= A, [A >= 0, B >= 0]),
    (A ** 2 < B ** 2, B < A, [A <= 0, B <= 0]),
    (A ** 2 <= B ** 2, B <= A, [A <= 0, B <= 0]),
    # abs
    (abs(A + B) == abs(A) + abs(B), A * B >= 0),
    (abs(A + B) != abs(A) + abs(B), A * B < 0),
    # sqrt
    (sqrt(A) < sqrt(B), B > A, [A >= 0, B >= 0]),
    (sqrt(A) <= sqrt(B), B >= A, [A >= 0, B >= 0]),
    (sqrt(A) == sqrt(B), A == B, [A >= 0, B >= 0]),
    (sqrt(A) != sqrt(B), A != B, [A >= 0, B >= 0]),
]

RULES_BASIC_A: List[A_RULE_TYPE] = [
    (A == A),
    (A != B, [A < B]),
    (A != B, [B < A]),
    (A <= A),
    (A == B, [B == A]),
    (A != B, [B != A]),
    (A == B, [A <= B, A >= B]),
    (A >= 0, [A > 0]),
    (A <= 0, [A < 0]),
    (A > 0, [A >= 0, A != ZERO]),
    (A < 0, [A <= 0, A != ZERO]),
    # NOTE: sans les "ou" manque le complémentaire de celle là et de toutes celles qui ont deux hypothèses
    (A - B == C, [A == B + C]),
    (A - B != C, [A != B + C]),
    (A / B == C, [A == B * C, B != ZERO]),
    (A / B != C, [A != B * C, B != ZERO]),
    (A - B <= C, [A <= B + C]),
    (A / B <= C, [A <= B * C, B > ZERO]),
    (A - B < C, [A < B + C]),
    (A / B < C, [A < B * C, B > ZERO]),
    (A <= B, [A < B]),
    (B <= A, [B < A]),
    (A <= B, [A == B]),
    (B <= A, [A == B]),
    (A < A + B, [B > 0]),
    (B <= 0, [A >= A + B]),
    (A <= A + B, [B >= 0]),
    (B < 0, [A > A + B]),
    (A < C, [A < B, B <= C]),
    (A < C, [A <= B, B < C]),
    (A <= C, [A <= B, B <= C]),
    (A + C == B + C, [A == B]),
    (A + C != B + C, [A != B]),
    (A * C == B * C, [A == B]),
    (A != B, [A * C != B * C]),
    (A != ZERO, [A * B != ZERO]),
    (A * B != ZERO, [A != ZERO, B != ZERO]),
    (A / C == B / C, [A == B, C != ZERO]),
    (A / C != B / C, [A != B, C != ZERO]),
    (A + C < B + D, [A <= B, C < D]),
    (A + C <= B + D, [A <= B, C <= D]),
    (A * C <= B * C, [C >= 0, A <= B]),
    (A * B <= 0, [A >= 0, B <= 0]),  # a bit redondant with above
    (A * B >= 0, [A >= 0, B >= 0]),  # a bit redondant with above
    (A * B >= 0, [A <= 0, B <= 0]),  # a bit redondant with above
    (C < 0, [A * C > B * C, A <= B]),
    (A * C < B * C, [C > 0, A < B]),
    (A * B < 0, [A > 0, B < 0]),  # a bit redondant with above
    (A * B > 0, [A > 0, B > 0]),  # a bit redondant with above
    (A * B > 0, [A < 0, B < 0]),  # a bit redondant with above
    (C <= 0, [A * C >= B * C, A < B]),  # TODO voir si on peut le déduire autrement
    (-B <= -A, [A <= B]),
    (-B < -A, [A < B]),
    # inv
    (A.inv() > 0, [A > 0]),
    (A.inv() < 0, [A < 0]),
    (A.inv() != ZERO, [A != ZERO]),
    (B <= 0, [A.inv() >= B.inv(), A != ZERO, B != ZERO, A > B]),
    (B < 0, [A.inv() > B.inv(), A != ZERO, B != ZERO, A >= B]),
    (A.inv() < B.inv(), [B > 0, A > B]),
    (A.inv() < B.inv(), [A < 0, B < A]),
    (A.inv() <= B.inv(), [B > 0, A >= B]),
    (A.inv() <= B.inv(), [A < 0, B <= A]),
    # pow2
    (A ** 2 > 0, [A != ZERO]),
    (A ** 2 >= 0),
    (A ** 2 == B ** 2, [A == B]),
    (A != B, [A ** 2 != B ** 2]),
    (A ** 2 < B ** 2, [A >= 0, B > A]),
    (A ** 2 < B ** 2, [A <= 0, B < A]),
    (A ** 2 <= B ** 2, [A >= 0, B >= A]),
    (A ** 2 <= B ** 2, [A <= 0, B <= A]),
    (A < 0, [A ** 2 >= B ** 2, B > A]),
    (A < 0, [A ** 2 > B ** 2, B >= A]),
    (A > 0, [A ** 2 >= B ** 2, B < A]),
    (A > 0, [A ** 2 > B ** 2, B <= A]),
    # abs
    (abs(A) > 0, [A != ZERO]),
    (A == ZERO, [abs(A) <= 0]),
    (abs(A) >= 0),
    (abs(A) >= A),
    (abs(A) == A, [A >= 0]),
    (A < 0, [abs(A) != A]),
    (abs(A + B) <= abs(A) + abs(B)),
    (abs(A + B) == abs(A) + abs(B), [A * B >= 0]),
    (A * B < 0, [abs(A + B) != abs(A) + abs(B)]),
    (abs(A) <= B, [A <= B, -A <= B]),
    (abs(A) < B, [A < B, -A < B]),
    # sqrt
    (sqrt(A) > 0, [A > 0]),
    (sqrt(A) >= 0, [A >= 0]),
    (sqrt(A) < sqrt(B), [A >= 0, B > A]),
    (sqrt(A) <= sqrt(B), [A >= 0, B >= A]),
    (sqrt(A) == sqrt(B), [A >= 0, A == B]),
    (A != B, [A >= 0, B >= 0, sqrt(A) != sqrt(B)]),
    # max
    (A < Max(B, C), [A < C, B < C]),
    (A <= Max(B, C), [A <= C, B <= C]),
    (Max(A, B) >= A),
    (Max(A, B) >= B),
    (A + B <= 2 * Max(A, B)),
    (Max(A, B) <= Max(abs(A), abs(B))),
    (abs(Max(A, B)) <= Max(abs(A), abs(B))),
    # TODO: Introduire min et autres rules.
]

#
# exp / log rules
#
RULES_EXP_T_E: List[T_RULE_TYPE] = [
    (exp(ln(A)), A, [A > 0]),
    (ln(exp(A)), A),
    (exp(ZERO), ONE),
    (ln(ONE), ZERO),
    (exp(A + B), exp(A) * exp(B)),
    (ln(A * B), ln(A) + ln(B), [A > 0, B > 0]),
    (exp(-A), exp(A).inv()),
    (ln(A.inv()), -ln(A), [A > 0]),
    (ln(sqrt(A)), ln(A) / 2, [A > 0]),
    (ln(A ** 2), 2 * ln(abs(A)), [A != ZERO]),
]

RULES_EXP_T_C: List[T_RULE_TYPE] = [
    (exp(A) == exp(B), A == B),
    (exp(A) != exp(B), A != B),
    (exp(A) <= exp(B), A <= B),
    (exp(A) < exp(B), A < B),
    (ln(A) == ln(B), A == B, [A > 0, B > 0]),
    (ln(A) != ln(B), A != B, [A > 0, B > 0]),
    (ln(A) <= ln(B), A <= B, [A > 0, B > 0]),
    (ln(A) < ln(B), A < B, [A > 0, B > 0]),
]

RULES_EXP_A: List[A_RULE_TYPE] = [
    (exp(A) > 0),
    (exp(A) >= 0),
    (exp(A) != ZERO),
    (exp(A) < exp(B), [A < B]),
    (exp(A) <= exp(B), [A <= B]),
    (exp(A) == exp(B), [A == B]),
    (exp(A) != exp(B), [A != B]),
    (ln(A) < ln(B), [A > 0, A < B]),
    (ln(A) <= ln(B), [A > 0, A <= B]),
    (ln(A) == ln(B), [A > 0, A == B]),
    (ln(A) != ln(B), [A > 0, B > 0, A != B]),
]

#
# trigo rules
#
RULES_TRIGO_T_E: List[T_RULE_TYPE] = [
    (sin(ZERO), ZERO),
    (cos(ZERO), ONE),
    (sin(PI / 2), ONE),
    (cos(PI / 2), ZERO),
    (sin(-A), -sin(A)),
    (cos(-A), cos(A)),
    (tan(A), sin(A) / cos(A), [cos(A) != ZERO]),
    (sin(A + B), sin(B) * cos(A) + sin(A) * cos(B)),
    (cos(A + B), cos(A) * cos(B) - sin(A) * sin(B)),
    (sin(asin(A)), A, [abs(A) <= 1]),
    (cos(acos(A)), A, [abs(A) <= 1]),
    (tan(atan(A)), A),
    (asin(sin(A)), A, [abs(A) <= PI / 2]),
    (acos(cos(A)), A, [A >= 0, A <= PI]),
    (atan(tan(A)), A, [abs(A) < PI / 2]),
]

RULES_TRIGO_A: List[A_RULE_TYPE] = [
    (abs(cos(A)) <= 1),
    (abs(sin(A)) <= 1),
    (abs(sin(A)) <= abs(A)),
    (sin(A) == sin(B), [A == B]),
    (A != B, [sin(A) != sin(B)]),
    (cos(A) == cos(B), [A == B]),
    (A != B, [cos(A) != cos(B)]),
    (tan(A) == tan(B), [A == B, cos(A) != ZERO]),
    (A != B, [tan(A) != tan(B), cos(A) != ZERO, cos(B) != ZERO]),
]

#
# hyperbolic rules
#
RULES_HYPER_T_E: List[T_RULE_TYPE] = [
    (sinh(A), (exp(A) - exp(-A)) / 2),
    (cosh(A), (exp(A) + exp(-A)) / 2),
    (tanh(A), sinh(A) / cosh(A)),
    (sinh(asinh(A)), A),
    (cosh(acosh(A)), A, [A >= 1]),
    (tanh(atanh(A)), A, [abs(A) < 1]),
    (asinh(sinh(A)), A),
    (acosh(cosh(A)), A, [A >= 0]),
    (atanh(tanh(A)), A),
    ##########################
    # (cosh(A) + sinh(A), exp(A)),
    # (cosh(A), cosh(-A)),
    # (sinh(-A), -sinh(A)),
    # (tanh(A), sinh(A) / cosh(A)),
    # (cosh(A) ** 2 - sinh(A) ** 2, ONE),
    # (sinh(A + B), sinh(A) * cosh(B) + sinh(B) * cosh(A)),
    # (cosh(A + B), cosh(A) * cosh(B) + sinh(B) * sinh(A)),
    # (sinh(ZERO), ZERO),
    # (cosh(ZERO), ONE),
]

RULES_HYPER_T_C: List[T_RULE_TYPE] = [
    (cosh(A) == cosh(B), A == B, [A * B >= 0]),
    (cosh(A) != cosh(B), A != B, [A * B >= 0]),
    (sinh(A) != sinh(B), A != B),
    (tanh(A) == tanh(B), A == B),
    (tanh(A) != tanh(B), A != B),
]

RULES_HYPER_A: List[A_RULE_TYPE] = [
    (cosh(A) != ZERO),
    (cosh(A) >= 1),
    (cosh(A) > 1, [A != ZERO]),
]


# ######################
#
# Register default rules
#
# ######################

# arithmetic rules
RULES_ARITH = [
    TRule.create_arith(a + b, c, name="addition"),
    TRule.create_arith(a * b, c, name="multiplication"),
    TRule.create_arith(a / b, c / d, name="fraction"),
    TRule.create_arith(-a, minus_a, name="negation"),
]


def t_rule(x: T_RULE_TYPE, rule_type: str) -> TRule:
    assert 2 <= len(x) <= 3
    left, right, hyps = x if len(x) == 3 else (*x, [])
    return TRule(left, right, hyps, rule_type=rule_type)


def a_rule(x: A_RULE_TYPE, rule_type: str) -> ARule:
    if isinstance(x, Node):
        node = x
        hyps: List[Node] = []
    else:
        node, hyps = x
    return ARule(node, hyps, rule_type=rule_type)


RULES_T_E: List[TRule] = [
    *[t_rule(x, "basic") for x in RULES_BASIC_T_E],
    *[t_rule(x, "exp") for x in RULES_EXP_T_E],
    *[t_rule(x, "trigo") for x in RULES_TRIGO_T_E],
    *[t_rule(x, "hyper") for x in RULES_HYPER_T_E],
    *RULES_ARITH,
]
RULES_T_C: List[TRule] = [
    *[t_rule(x, "basic") for x in RULES_BASIC_T_C],
    *[t_rule(x, "exp") for x in RULES_EXP_T_C],
    *[t_rule(x, "hyper") for x in RULES_HYPER_T_C],
]
RULES_A: List[ARule] = [
    *[a_rule(x, "basic") for x in RULES_BASIC_A],
    *[a_rule(x, "exp") for x in RULES_EXP_A],
    *[a_rule(x, "trigo") for x in RULES_TRIGO_A],
    # *[ARule(*x, rule_type="hyper") for x in RULES_HYPER_A],
]
RULES_T: List[TRule] = RULES_T_E + RULES_T_C
RULES: List[Union[ARule, TRule]] = [*RULES_T, *RULES_A]
assert len(RULES) == len(set(rule.name for rule in RULES))


if __name__ == "__main__":
    test_duplicated_rules(RULES)
    test_valid_transform_rules_e_numeric(RULES_T_E)
    test_valid_transform_rules_c_numeric(RULES_T_C)
    test_valid_assert_rules_numeric(RULES_A)
    test_eval_assert(RULES_A)
