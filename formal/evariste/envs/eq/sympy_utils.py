# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List
import re
import sympy as sp

from evariste.envs.eq.graph import Node, VNode


SYMPY_OPERATORS = {
    # Elementary functions
    sp.Add: "add",
    sp.Mul: "mul",
    sp.Mod: "mod",
    sp.Pow: "pow",
    # comparisons
    sp.LessThan: "<=",
    sp.StrictLessThan: "<",
    sp.Equality: "==",
    sp.Unequality: "!=",
    # Misc
    sp.Abs: "abs",
    sp.sign: "sign",
    sp.Heaviside: "step",
    # Exp functions
    sp.exp: "exp",
    sp.log: "ln",
    # Trigonometric Functions
    sp.sin: "sin",
    sp.cos: "cos",
    sp.tan: "tan",
    # Trigonometric Inverses
    sp.asin: "arcsin",
    sp.acos: "arccos",
    sp.atan: "arctan",
}


def sympy_to_prefix(expr) -> List[str]:
    """
    Convert a SymPy expression to a prefix one.
    """
    if isinstance(expr, sp.Symbol):
        return [str(expr)]
    elif isinstance(expr, sp.Integer):
        sign = "+"
        if expr < 0:
            sign = "-"
            expr = -expr
        return [sign, *str(expr)]
    elif isinstance(expr, (sp.Float, sp.Rational)) and expr < 0:
        return ["neg"] + sympy_to_prefix(-expr)
    elif isinstance(expr, sp.Float):
        return [str(expr)]
    elif isinstance(expr, sp.Rational):
        return ["div", "+", *str(expr.p), "+", *str(expr.q)]
    # reverse comparisons, eq only does less than
    elif isinstance(expr, sp.StrictGreaterThan):
        return sympy_to_prefix(sp.StrictLessThan(expr.rhs, expr.lhs))
    elif isinstance(expr, sp.GreaterThan):
        return sympy_to_prefix(sp.LessThan(expr.rhs, expr.lhs))
    elif expr == sp.EulerGamma:
        return ["euler_gamma"]
    elif expr == sp.E:
        return ["e"]
    elif expr == sp.pi:
        return ["PI"]

    # TODO: warning when unknown operator?
    op = SYMPY_OPERATORS.get(type(expr), str(type(expr)))

    # TODO: take care of - that are badly dealt with by sympy, e.g. x**2 -2x is turned into x**2 + (-2)*x
    if op == "pow":
        if (
            isinstance(expr.args[1], sp.Rational)
            and expr.args[1].p == 1
            and expr.args[1].q == 2
        ):
            return ["sqrt"] + sympy_to_prefix(expr.args[0])
        elif str(expr.args[1]) == "2":
            return ["pow2"] + sympy_to_prefix(expr.args[0])
        elif str(expr.args[1]) == "-1":
            return ["inv"] + sympy_to_prefix(expr.args[0])
        else:
            return (
                ["**"] + sympy_to_prefix(expr.args[0]) + sympy_to_prefix(expr.args[1])
            )

    # parse children
    n_args = len(expr.args)
    parse_list: List[str] = []
    for i in range(n_args):
        if i == 0 or i < n_args - 1:  # to handle a + b + c (does it happen?)
            parse_list.append(op)
        parse_list += sympy_to_prefix(expr.args[i])
    return parse_list


def is_digit_token(s: str) -> bool:
    return re.match(r"[+-]?\d+", s) is not None


def simplify_from_infix(infix: str) -> Node:
    sp_eq = sp.parse_expr(infix)
    if sp_eq.has(sp.oo, -sp.oo, -sp.zoo, sp.zoo, sp.I, sp.Symbol("j"), sp.nan):
        raise SympyException(f"{infix} {sp_eq}")
    prefix = sympy_to_prefix(sp_eq)
    new_prefix = []
    while len(prefix) > 0:
        tok = prefix.pop(0)
        if tok == "PI":
            new_prefix.append("PI")
        elif not is_digit_token(tok):
            new_prefix.append(tok)
        else:
            if tok[0] not in ["+", "-"]:
                new_prefix.append("+")
            new_prefix.extend(list(tok))
    return Node.from_prefix_tokens(new_prefix)


def simplify_sp(eq: Node) -> Node:
    """
    Fast simplification with SymPy (i.e. does not call sp.simplify).
    """
    infix = eq.infix()
    return simplify_from_infix(infix)


class SympyException(Exception):
    pass


if __name__ == "__main__":

    # python -m evariste.envs.eq.sympy_utils

    def test_sympy_simplify():

        print("===== RUNNING SYMPY SIMPLIFY TESTS ...")

        x = VNode("x0")

        eq_tests = [
            (x + 10) + (4 + 3 * x) * 0,
            (x + 10) ** 2 - 20 * x - 100,
        ]
        for eq in eq_tests:
            sp_before = sp.parse_expr(eq.infix())
            simplified = simplify_sp(eq)
            sp_after = sp.parse_expr(simplified.infix(), evaluate=False)
            assert sp.simplify(sp_after - sp_before) == 0, "Problem simplification"
            print(f"{eq.infix():<80} -> {simplified.infix()}")

        tests = [
            ((4 + 3 * x) + (x + 10) - (4 + 3 * x), 10 + x),
        ]
        for eq, y in tests:
            z = simplify_sp(eq)
            assert y.eq(z), (y, z)
            print(f"{eq.infix():<80} -> {y.infix()}")

        print("OK")

    test_sympy_simplify()
