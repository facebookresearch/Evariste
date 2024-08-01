# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple, List, Set

from evariste.envs.eq.graph import VNode, INode, Node
from evariste.envs.eq.rules import (
    ZERO,
    ONE,
    PI,
    eval_numeric,
    get_test_substs,
)
from evariste.envs.eq.graph import exp, ln, sqrt
from evariste.envs.eq.graph import sin, cos, tan, asin, acos  # , atan
from evariste.envs.eq.graph import sinh, cosh, tanh, asinh, acosh, atanh


x = VNode("x0")
y = VNode("x1")
z = VNode("x2")


IDENTITIES_BASIC = [
    [(x + y) * (x - y) == x ** 2 - y ** 2],
    [x ** 2 + 2 * x * y + y ** 2 == (x + y) ** 2],
    [x ** 2 - 2 * x * y + y ** 2 == (x - y) ** 2],
    [(x + y) ** 2 - 4 * x * y == (x - y) ** 2],
    [x ** 3 - y ** 3 == (x - y) * (x ** 2 + x * y + y ** 2)],
    [x ** 3 + 3 * x ** 2 * y + 3 * x * y ** 2 + y ** 3 == (x + y) ** 3],
    [x ** 3 - 3 * x ** 2 * y + 3 * x * y ** 2 - y ** 3 == (x - y) ** 3],
    [x ** 2 + y ** 2 + z ** 2 + 2 * (x * y + y * z + x * z) == (x + y + z) ** 2],
    [
        x ** 3
        + y ** 3
        + z ** 3
        + 3 * ((x ** 2) * y + (y ** 2) * z + (x ** 2) * z)
        + 3 * (x * y ** 2 + y * z ** 2 + x * z ** 2)
        + 6 * x * y * z
        == (x + y + z) ** 3
    ],
    [
        x ** 3 + y ** 3 + z ** 3 - 3 * x * y * z
        == (x + y + z) * (x ** 2 + y ** 2 + z ** 2 - (x * y + y * z + x * z))
    ],
    [
        x ** 4 + y ** 4
        == (x ** 2 - sqrt(INode(2)) * x * y + y ** 2)
        * (x ** 2 + sqrt(INode(2)) * x * y + y ** 2)
    ],
    [
        x ** 4 + x ** 2 * y ** 2 + y ** 4
        == (x ** 2 - x * y + y ** 2) * (x ** 2 + x * y + y ** 2)
    ],
    [
        (x + y) ** 4
        == x ** 4 + 4 * x * y ** 3 + 6 * x ** 2 * y ** 2 + 4 * x ** 3 * y + y ** 4
    ],
    [(x ** 2 - y ** 2) / (x - y) == x + y, [x != y]],
    [(x ** 2 - y ** 2) / (x + y) == x - y, [x != -y]],
    [(x ** 3 - y ** 3) / (x - y) == x ** 2 + x * y + y ** 2, [x != y]],
    [(x ** 3 - y ** 3) / (x ** 2 + x * y + y ** 2) == (x - y), [x * y != ZERO]],
]

IDENTITIES_EXP = [
    [(ln(x) - ln(x / y)) / (ln(y)) == ONE, [x > 0, y > 0, y != ONE]],
    [(-ln(x) + ln(x / y)) / (ln(y)) == -ONE, [x > 0, y > 0, y != ONE]],
    [-(-ln(x) + ln(x / y)) / (ln(y)) == ONE, [x > 0, y > 0, y != ONE]],
    [-ln(x) + ln(x * y) == ln(y), [x > 0, y > 0]],
    [-ln(x) + ln(y * x) == ln(y), [x > 0, y > 0]],
    [-ln(x) + ln(x / y) == -ln(y), [x > 0, y > 0]],
    [-ln(x) + ln(y) + ln(x / y) == ZERO, [x > 0, y > 0]],
    [-ln(x) - ln(y) + ln(x * y) == ZERO, [x > 0, y > 0]],
    [ln(x) + ln(y) == ln(x * y), [x > 0, y > 0]],
    [ln(x) + ln(y / x) == ln(y), [x > 0, y > 0]],
    [ln(x) - ln(x / y) == ln(y), [x > 0, y > 0]],
    [ln(x) - ln(y) == ln(x / y), [x > 0, y > 0]],
    [-ln(x / y) + ln(x / y) == ZERO, [x > 0, y > 0]],
    [2 * ln(sqrt(x)) == ln(x), [x > 0]],
    [ln(sqrt(x)) == ln(x) / 2, [x > 0]],
    # Can be simplified if we include PNode("e") to the constants.
    [2 * ln(sqrt(x)) / ln(x) == ONE, [x > 0, ln(x) != ZERO]],
    [ln(sqrt(x)) / ln(x) == ONE / 2, [x > 0, ln(x) != ZERO]],
    [ln(exp(x)) == x, [x > 0]],
    [exp(-x) * exp(-y) * exp(x + y) == ONE],
    [exp(-x) * exp(x + y) == exp(y)],
    [exp(-x) * exp(y + x) == exp(y)],
    [exp(-x) * exp(x - y) == exp(-y)],
    [exp(-x) * exp(y) * exp(x - y) == ONE],
    [exp(x) * exp(y) == exp(x + y)],
    [exp(x) * exp(-y) == exp(x - y)],
    [exp(x) * exp(y - x) == exp(y)],
    [exp(x) / exp(y) == exp(x - y)],
    [sqrt(exp(x)) == exp(x / 2)],
    # [exp(x * ln(y)) == y ** x],
    # [ln(x ** y) == y * ln(x)],
    # [ln(x ** y) /(y * ln(x)) == 1],
    # [ln(x ** y) /y == ln(x)],
    # [ln(x ** y) /ln(x) == y],
]

IDENTITIES_TRIGO = [
    [cos(2 * PI + x) == cos(x)],
    [sin(2 * PI + x) == sin(x)],
    [cos(PI + x) == -cos(x)],
    [sin(PI + x) == -sin(x)],
    [sin(PI - x) == sin(x)],
    [cos(PI - x) == -cos(x)],
    [sin(PI / 2 - x) == cos(x)],
    [cos(PI / 2 - x) == sin(x)],
    [sin(PI / 2 + x) == cos(x)],
    [cos(PI / 2 + x) == -sin(x)],
    # [cos(PI) == -ONE],
    # [sin(PI) == ZERO],
    [cos(PI / 4) == sin(PI / 4)],
    [cos(PI / 3) == sin(PI / 6)],
    [cos(PI / 6) == sin(PI / 3)],
    [cos(PI / 4) == sqrt(INode(2)) / 2, [cos(PI / 4) > 0]],
    # [cos(PI / 3) == ONE / (INode(2)).inv()],
    # [cos(PI / 6) == sqrt(INode(3)) / 2],
    # [cos(PI / 12) == (sqrt(INode(2)) + sqrt(INode(6))) * (INode(4)).inv()],
    [cos(x) ** 2 + sin(x) ** 2 == ONE],
    [cos(x + y) == cos(x) * cos(y) - sin(x) * sin(y)],
    [sin(x + y) == sin(x) * cos(y) + sin(y) * cos(x)],
    [sin(x + y) - sin(x - y) == 2 * sin(y) * cos(x)],
    [sin(x - y) + sin(x + y) == 2 * sin(x) * cos(y)],
    [cos(x - y) + cos(x + y) == 2 * cos(x) * cos(y)],
    [cos(x - y) - cos(x + y) == 2 * sin(x) * sin(y)],
    [2 * cos((x + y) / 2) * cos((x - y) / 2) == cos(x) + cos(y)],
    [2 * sin((x + y) / 2) * sin((x - y) / 2) == cos(y) - cos(x)],
    [2 * sin((x + y) / 2) * cos((x - y) / 2) == sin(x) + sin(y)],
    [2 * cos((x + y) / 2) * sin((x - y) / 2) == sin(x) - sin(y)],
    [cos(x) * tan(x) == sin(x), [cos(x) != ZERO]],
    [sin(-x) == -sin(x)],
    [cos(-x) == cos(x)],
    [tan(-x) == -tan(x), [cos(x) != ZERO]],
    # sin(asin(x)) == x,
    # cos(acos(x)) == x,
    # tan(atan(x)) == x,
    [sin(x - y) == sin(x) * cos(y) - cos(x) * sin(y)],
    [cos(x - y) == cos(x) * cos(y) + sin(x) * sin(y)],
    [
        tan(x + y) * (1 - tan(x) * tan(y)) == (tan(x) + tan(y)),
        [cos(x) != ZERO, cos(y) != ZERO, cos(x + y) != ZERO],
    ],
    [
        tan(x - y) * (1 + tan(x) * tan(y)) == (tan(x) - tan(y)),
        [cos(x) != ZERO, cos(y) != ZERO, cos(x - y) != ZERO],
    ],
    [sin(2 * x) == 2 * sin(x) * cos(x)],
    [cos(2 * x) == cos(x) ** 2 - sin(x) ** 2],
    [cos(2 * x) == 2 * cos(x) ** 2 - 1],
    [cos(2 * x) == 1 - 2 * sin(x) ** 2],
    [
        tan(2 * x) * (1 - tan(x) ** 2) == 2 * tan(x),
        [cos(x) != ZERO, cos(2 * x) != ZERO],
    ],
    [sin(3 * x) == 3 * sin(x) - 4 * sin(x) ** 3],
    [cos(3 * x) == 4 * cos(x) ** 3 - 3 * cos(x)],
    [
        tan(3 * x) * (1 - 3 * tan(x) ** 2) == (3 * tan(x) - tan(x) ** 3),
        [cos(x) != ZERO, cos(3 * x) != ZERO],
    ],
    [sin(4 * x) == cos(x) * (4 * sin(x) - 8 * sin(x) * sin(x) ** 2)],
    [cos(4 * x) == 8 * (cos(x) ** 2) ** 2 - 8 * cos(x) ** 2 + 1],
    [
        tan(4 * x) * (1 - 6 * tan(x) ** 2 + (tan(x) ** 2) ** 2)
        == (4 * tan(x) - 4 * tan(x) ** 3),
        [cos(x) != ZERO, cos(4 * x) != ZERO],  # To check
    ],
    [tan(x / 2) == sin(x) / (1 + cos(x)), [cos(x / 2) != ZERO]],
    [tan(x / 2) == (1 - cos(x)) / sin(x), [cos(x / 2) != ZERO, sin(x) != ZERO]],
    # asin(x) + acos(x) == PNode("PI") / 2,  # TODO: check if one can make it
    # acos(sqrt(1 - x**2)) == abs(asin(x))
    [asin(x) == acos(sqrt(1 - (x ** 2))), [x >= 0, x <= 1]],
    [sin(acos(x)) == sqrt(1 - (x ** 2)), [x >= 0, x <= 1]],
    [cos(asin(x)) == sqrt(1 - (x ** 2)), [x >= 0, x <= 1]],
    [1 - tan(x) ** 2 == cos(2 * x) / cos(x) ** 2, [cos(x) != ZERO]],
    [1 - 3 * tan(x) ** 2 == cos(3 * x) / (cos(x) ** 3), [cos(x) != ZERO]],
    [INode(2) / tan(x) == sin(2 * x) / sin(x) ** 2, [sin(x) != ZERO, cos(x) != ZERO]],
    [sin(x) ** 2 == (1 - cos(2 * x)) / 2],
    [cos(x) ** 2 == (1 + cos(2 * x)) / 2],
    [tan(x) ** 2 == (1 - cos(2 * x)) / (1 + cos(2 * x)), [cos(x) != ZERO]],
    [sin(x) ** 3 == (3 * sin(x) - sin(3 * x)) / 4],
    [cos(x) ** 3 == (3 * cos(x) + cos(3 * x)) / 4],
    [
        tan(x) ** 3 == (3 * sin(x) - sin(3 * x)) / (3 * cos(x) + cos(3 * x)),
        [cos(x) != ZERO],
    ],
    [sin(x) ** 4 == (3 - 4 * cos(2 * x) + cos(4 * x)) / 8],
    [cos(x) ** 4 == (3 + 4 * cos(2 * x) + cos(4 * x)) / 8],
    [
        tan(x) ** 4 * (3 + 4 * cos(2 * x) + cos(4 * x))
        == (3 - 4 * cos(2 * x) + cos(4 * x)),
        [cos(x) != ZERO],
    ],
    [sqrt(1 - cos(x) ** 2) == abs(sin(x))],
    [sqrt(1 - sin(x) ** 2) == abs(cos(x))],
    [sin(x + y) * sin(x - y) == sin(x) ** 2 - sin(y) ** 2],
    [sin(x + y) * sin(y - x) == cos(x) ** 2 - cos(y) ** 2],
    [cos(x + y) * cos(x - y) == cos(x) ** 2 - sin(y) ** 2],
    [sqrt((1 - cos(x)) / 2) == abs(sin(x / 2))],
    [sqrt((1 + cos(x)) / 2) == abs(cos(x / 2))],
    [sin(x) == 2 * sin(x / 2) * cos(x / 2)],
    [cos(x) == cos(x / 2) ** 2 - sin(x / 2) ** 2],
    [
        tan(x) == 2 * tan(x / 2) / (1 - tan(x / 2) ** 2),
        [cos(x) != ZERO, cos(x / 2) != ZERO],
    ],
    [
        tan(x) + tan(y) == sin(x + y) / (cos(x) * cos(y)),
        [cos(x) != ZERO, cos(y) != ZERO],
    ],
    [
        tan(x) - tan(y) == sin(x - y) / (cos(x) * cos(y)),
        [cos(x) != ZERO, cos(y) != ZERO],
    ],
    # [ sin(x) * cot(x) == cos(x)],
    # [ tan(x) * cot(x) == 1],
    # [ cot(-x) == -cot(x)],
    # [ (cot(x) * cot(y) - 1) / (cot(x) + cot(y)) == cot(x + y)],
    # [ 2 * cot(x) / (cot(x) ** 2 - 1) == tan(2 * x)],
    # [ 2 / (cot(x) - tan(x)) == tan(2 * x)],
    # [ # 1 + tan(x) ** 2 == sec(x) ** 2],
    # [ # 1 + cot(x) ** 2 == csc(x) ** 2],
    # [ (cot(x) * cot(y) - 1) / (cot(x) + cot(y)) == cot(x + y)],
    # [ (cot(x) * cot(y) + 1) / (cot(y) - cot(x)) == cot(x - y)],
    # [ csc(x) == 1/sin(x)],
    # [ sec(x) == 1/cos(x)],
    # [ csc(-x) == -csc(x)],
    # [ sec(-x) == sec(x)],
    # [ cot(-x) == -cot(x)],
    # [ cot(x) ** 2 - 1 == cos(2 * x) / sin(x) ** 2],
    # [ 2 * cot(x) == sin(2 * x) / sin(x) ** 2],
    # [ 1 + tan(x) ** 2 == sec(x) ** 2],
    # [ 1 + cot(x) ** 2 == csc(x) ** 2],
    # [ sqrt(sec(x) ** 2 - 1) == abs(tan(x))],
    # [ 2 * cot(x / 2) / (csc(x / 2) ** 2 - 2) == tan(x)],
    # [ (1 + cos(x)) / sin(x) == cot(x / 2)],
    # [ sin(x) / (1 - cos(x)) == cot(x / 2)],
    # [ cot(x) - 2 * cot(2 * x) == tan(x)],
]

IDENTITIES_HYPER = [
    [sinh(x) == (exp(x) - exp(-x)) / 2],
    [cosh(x) == (exp(x) + exp(-x)) / 2],
    [tanh(x) == sinh(x) / cosh(x)],
    [cosh(x) * tanh(x) == sinh(x)],
    [x == sinh(asinh(x))],
    [x == cosh(acosh(x)), [x >= 1]],
    [x == tanh(atanh(x)), [abs(x) < 1]],
    [sinh(-x) == -sinh(x)],
    [cosh(-x) == cosh(x)],
    [tanh(-x) == -tanh(x)],
    [cosh(x) ** 2 - sinh(x) ** 2 == ONE],
    [sinh(x) == (1 - exp(-2 * x)) / (2 * exp(-x))],
    [cosh(x) == (1 + exp(-2 * x)) / (2 * exp(-x))],
    [sinh(x) == (exp(2 * x) - 1) / (2 * exp(x))],
    [cosh(x) == (exp(2 * x) + 1) / (2 * exp(x))],
    [tanh(x) == (exp(2 * x) - 1) / (exp(2 * x) + 1)],
    [tanh(x) == (exp(x) - exp(-x)) / (exp(x) + exp(-x))],
    [cosh(x) + sinh(x) == exp(x)],
    [cosh(x) - sinh(x) == exp(-x)],
    [sinh(x + y) - sinh(x - y) == 2 * sinh(y) * cosh(x)],
    [sinh(x - y) + sinh(x + y) == 2 * sinh(x) * cosh(y)],
    [cosh(x + y) - cosh(x - y) == 2 * sinh(x) * sinh(y)],
    [cosh(x - y) + cosh(x + y) == 2 * cosh(x) * cosh(y)],
    [cosh(x) + cosh(y) == 2 * cosh((x + y) / 2) * cosh((x - y) / 2)],
    [cosh(x) - cosh(y) == 2 * sinh((x + y) / 2) * sinh((x - y) / 2)],
    [sinh(x) + sinh(y) == 2 * sinh((x + y) / 2) * cosh((x - y) / 2)],
    [sinh(x) - sinh(y) == 2 * cosh((x + y) / 2) * sinh((x - y) / 2)],
    [sinh(x + y) == sinh(x) * cosh(y) + cosh(x) * sinh(y)],
    [cosh(x + y) == cosh(x) * cosh(y) + sinh(x) * sinh(y)],
    [sinh(x - y) == sinh(x) * cosh(y) - cosh(x) * sinh(y)],
    [cosh(x - y) == cosh(x) * cosh(y) - sinh(x) * sinh(y)],
    [tanh(x + y) == (tanh(x) + tanh(y)) / (1 + tanh(x) * tanh(y))],
    [tanh(x - y) == (tanh(x) - tanh(y)) / (1 - tanh(x) * tanh(y))],
    [sinh(2 * x) == 2 * sinh(x) * cosh(x)],
    [cosh(2 * x) == cosh(x) ** 2 + sinh(x) ** 2],
    [cosh(2 * x) == 2 * cosh(x) ** 2 - 1],
    [cosh(2 * x) == 1 + 2 * sinh(x) ** 2],
    [tanh(2 * x) == 2 * tanh(x) / (1 + tanh(x) ** 2)],
    [sinh(3 * x) == 3 * sinh(x) + 4 * sinh(x) ** 3],
    [cosh(3 * x) == 4 * cosh(x) ** 3 - 3 * cosh(x)],
    [tanh(3 * x) == (tanh(x) ** 3 + 3 * tanh(x)) / (1 + 3 * tanh(x) ** 2)],
    [sinh(x / 2) == sinh(x) / sqrt(2 * (1 + cosh(x)))],
    [cosh(x / 2) == sqrt((1 + cosh(x)) / 2)],
    [cosh(x / 2) == sqrt((cosh(x) + 1) / 2)],
    [tanh(x / 2) == sinh(x) / (1 + cosh(x))],
    [tanh(x / 2) == (cosh(x) - 1) / sinh(x), [x != ZERO]],
    [2 * x * cosh(ln(x)) == x ** 2 + 1, [x > 0]],
    [2 * x * sinh(ln(x)) == x ** 2 - 1, [x > 0]],
    [abs(sinh(x / 2)) == sqrt((cosh(x) - 1) / 2)],
    [cosh(x) == cosh(x / 2) ** 2 + sinh(x / 2) ** 2],
    [sinh(x) == 2 * sinh(x / 2) * cosh(x / 2)],
    [tanh(x) == 2 * tanh(x / 2) / (1 + tanh(x / 2) ** 2)],
    [sinh(x) ** 2 == (cosh(2 * x) - 1) / 2],
    [cosh(x) ** 2 == (1 + cosh(2 * x)) / 2],
    [tanh(x) ** 2 == (cosh(2 * x) - 1) / (1 + cosh(2 * x))],
    [sinh(x) ** 3 == (sinh(3 * x) - 3 * sinh(x)) / 4],
    [cosh(x) ** 3 == (3 * cosh(x) + cosh(3 * x)) / 4],
    [tanh(x) ** 3 == (sinh(3 * x) - 3 * sinh(x)) / (3 * cosh(x) + cosh(3 * x))],
    [cosh(x) - sinh(x) == (sinh(x) + cosh(x)).inv()],
    [tanh(x) + tanh(y) == sinh(x + y) / (cosh(x) * cosh(y))],
    [tanh(x) - tanh(y) == sinh(x - y) / (cosh(x) * cosh(y))],
    [sinh(x + y) * sinh(x - y) == sinh(x) ** 2 - sinh(y) ** 2],
    [cosh(x + y) * cosh(y - x) == sinh(x) ** 2 + cosh(y) ** 2],
    [sqrt(cosh(x) ** 2 - 1) == abs(sinh(x))],
    [sqrt(1 + sinh(x) ** 2) == cosh(x)],
    [asinh(x) == ln(x + sqrt(x ** 2 + 1))],
    [acosh(x) == ln(x + sqrt(x ** 2 - 1)), [x >= 1]],
    [asinh(x) == acosh(sqrt(x ** 2 + 1)), [x >= 0]],
    [asinh(x) == atanh(x / sqrt(x ** 2 + 1))],
    [acosh(x) == asinh(sqrt(x ** 2 - 1)), [x >= 1]],
    [acosh(x) == atanh(sqrt(x ** 2 - 1) / x), [x >= 1]],
    [atanh(x) == asinh(x / sqrt(1 - x ** 2)), [abs(x) < 1]],
    [atanh(x) == acosh(sqrt(1 - x ** 2).inv()), [x >= 0, x < 1]],
    [asinh(x) + asinh(y) == asinh(x * sqrt(1 + y ** 2) + y * sqrt(1 + x ** 2))],
    [asinh(x) - asinh(y) == asinh(x * sqrt(1 + y ** 2) - y * sqrt(1 + x ** 2))],
    [
        acosh(x) + acosh(y) == acosh(x * y + sqrt((x ** 2 - 1) * (y ** 2 - 1))),
        [x >= 1, y >= 1],
    ],
    [
        acosh(x) - acosh(y) == acosh(x * y - sqrt((x ** 2 - 1) * (y ** 2 - 1))),
        [y >= 1, y <= x],
    ],
    [atanh(x) + atanh(y) == atanh((x + y) / (1 + x * y)), [abs(y) < 1, abs(x) < 1]],
    [atanh(x) - atanh(y) == atanh((x - y) / (1 - x * y)), [abs(y) < 1, abs(x) < 1]],
    # sinh(x) * coth(x) == cosh(x),
    # tanh(x) * coth(x) == 1,
    # sinh(x) / (cosh(x) - 1) == coth(x / 2),
    # (cosh(x) + 1) / sinh(x) == coth(x / 2),
    # 2 * coth(x / 2) / (2 + csch(x / 2) ** 2) == tanh(x),
    # 2 * coth(2 * x) - coth(x) == tanh(x),
    # sqrt(1 - sech(x) ** 2) == abs(tanh(x)),
    # (coth(x) * coth(y) + 1) / (coth(x) + coth(y)) = coth(x + y)
    # (1 - coth(x) * coth(y)) / (coth(x) - coth(y)) = coth(x - y)
    # 1 - tanh(x) ** 2 = sech(x) ** 2
    # coth(x) ** 2 - 1 = csch(x) ** 2
    # coth(-x) = -coth(x)
    # csch(-x) = -csch(x)
    # sech(-x) = sech(x)
    # 1 / sinh(x) = csch(x)
    # 1 / cosh(x) = sech(x)
    # 1 / tanh(x) = coth(x)
    # (coth(x) * coth(y) + 1) / (coth(x) + coth(y)) = coth(x + y)
    # (coth(x) * coth(y) - 1) / (coth(y) - coth(x)) = coth(x - y)
]

CALCULUS_BASIC = [
    [x + x == INode(2) * x],
    [x + x + x == INode(3) * x],
    [x + x + x + x == INode(4) * x],
    [INode(3) * x + INode(2) * x == INode(5) * x],
    [INode(1) * INode(2).inv() + INode(1) * INode(2).inv() == INode(1)],
    [INode(4) * INode(2) == INode(8)],
    [INode(4) * (INode(3) * x + INode(2)) == INode(12) * x + INode(8)],
]
# TODO: add linear combinaisons


if __name__ == "__main__":

    IDENTITIES = IDENTITIES_BASIC + IDENTITIES_EXP + IDENTITIES_TRIGO + IDENTITIES_HYPER
    CALCULUS = CALCULUS_BASIC  # + CALCULUS_EXP + etc.  # TODO: finish
    # IDENTITIES = IDENTITIES + CALCULUS

    n_tests = 0
    found: Set[Tuple[str, ...]] = set()

    for identity in IDENTITIES:  # TODO: add CALCULUS
        assert type(identity) is list and 1 <= len(identity) <= 2
        eq: Node = identity[0]
        hyps: List[Node] = identity[1] if len(identity) == 2 else []
        assert isinstance(eq, Node)
        assert isinstance(hyps, list)
        assert all(isinstance(hyp, Node) for hyp in hyps)
        assert eq.is_comp()
        lhs, rhs = eq.children

        # check unique
        k = tuple(sorted([str(lhs), str(rhs)]))
        if k in found:
            raise Exception(f"{eq} provided multiple times!")
        found.add(k)

        # numerical check
        c_one_none, c_two_none, c_true, c_false, c_n = 0, 0, 0, 0, 0
        for subst in get_test_substs(eq.get_vars()):
            # check hypotheses
            if not all(eval_numeric(hyp, subst) is True for hyp in hyps):
                continue
            c_n += 1
            n_tests += 1
            # eval
            v0 = lhs.evaluate(subst)
            v1 = rhs.evaluate(subst)
            if v0 is None and v1 is None:
                c_two_none += 1
            elif v0 is None or v1 is None:
                c_one_none += 1
            else:
                if eq.value == "<":
                    valid = v0 < v1 - 1e-6
                elif eq.value == "<=":
                    valid = v0 < v1 - 1e-6 or abs(v0 - v1) <= 1e-10
                elif eq.value == "==":
                    valid = abs(v0 - v1) <= 1e-10
                else:
                    assert eq.value == "!="
                    valid = abs(v0 - v1) > 1e-6
                if valid:
                    c_true += 1
                else:
                    print(lhs, rhs, subst, v0, v1)
                    c_false += 1
        if c_n == 0:
            raise ValueError(f"No valid test found for {identity}")
        if c_true == 0 or c_false > 0 or c_one_none > 0 or c_two_none > 0:
            raise ValueError(
                f"Identity {identity} seems wrong: "
                f"one_none={c_one_none} two_none={c_two_none} "
                f"true={c_true} false={c_false}"
            )

    print(f"TESTS OK for {len(IDENTITIES)} identities ({n_tests} tests)")
