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
from evariste.envs.eq.rules import TEST_VALUES


def get_divs(a: int) -> List[int]:
    """
    Get the divisors of `a`.
    """
    res = []
    d = 1
    while d * d <= a:
        if a % d == 0:
            res.append(d)
        d += 1
    if len(res) == 0:
        return [1]
    return res


def gcd(a: int, b: int) -> int:
    """
    Get the GCD of a and b.
    """
    if b == 0:
        return 1
    while a % b != 0:
        c = a % b
        a = b
        b = c
    return b


ZERO = INode(0)
ONE = INode(1)


def simplify_equation(init_eq: Node) -> Node:
    """
    Simplify an equation.
    """
    eq = init_eq

    if len(eq.children) > 0:
        eq = Node(eq.type, eq.value, [simplify_equation(c) for c in eq.children])

    if eq.is_comp():
        return eq

    if eq.value == "add":
        assert len(eq.children) == 2
        if eq.children[0].eq(ZERO):
            eq = eq.children[1]
        elif eq.children[1].eq(ZERO):
            eq = eq.children[0]
        elif eq.children[1].eq(-eq.children[0]) or eq.children[0].eq(-eq.children[1]):
            return ZERO
    if eq.value == "sub":
        assert len(eq.children) == 2
        if eq.children[0].eq(ZERO):
            eq = eq.children[1].neg()
        elif eq.children[1].eq(ZERO):
            eq = eq.children[0]
        elif eq.children[1].eq(eq.children[0]):
            return ZERO
    if eq.value == "exp":
        if eq.children[0].value == "ln":
            eq = eq.children[0].children[0]
        elif eq.children[0].eq(ZERO):
            return ONE
    if eq.value == "ln":
        if eq.children[0].value == "exp":
            eq = eq.children[0].children[0]
        elif eq.children[0].eq(ONE):
            return ZERO
    if eq.value == "pow2":
        if eq.children[0].value == "sqrt":
            eq = eq.children[0].children[0]
        elif eq.children[0].eq(ZERO):
            return ZERO
        elif eq.children[0].eq(ONE):
            return ONE
        elif eq.children[0].value == "neg":
            eq = eq.children[0].children[0] ** 2
    if eq.value == "sqrt":
        if eq.children[0].value == "pow2":
            eq = UNode("abs", eq.children[0].children[0])
        elif eq.children[0].eq(ZERO):
            return ZERO
        elif eq.children[0].eq(ONE):
            return ONE
    if eq.value == "inv":
        if eq.children[0].eq(ONE):
            return ONE
        elif eq.children[0].value == "inv":
            eq = eq.children[0].children[0]
    if eq.value == "neg":
        if eq.children[0].eq(ZERO):
            return ZERO
        elif eq.children[0].value == "neg":
            eq = eq.children[0].children[0]
    if eq.value == "abs":
        if eq.children[0].value == "neg":
            eq = abs(eq.children[0].children[0])
    if eq.value == "mul":
        assert len(eq.children) == 2
        if eq.children[0].eq(ZERO):
            return ZERO
        elif eq.children[1].eq(ZERO):
            return ZERO
        elif eq.children[0].eq(ONE):
            eq = eq.children[1]
        elif eq.children[1].eq(ONE):
            eq = eq.children[0]
        elif eq.children[0].eq(eq.children[1].inv()) or eq.children[1].eq(
            eq.children[0].inv()
        ):
            return ONE
    if eq.value == "div":
        assert len(eq.children) == 2
        if eq.children[0].eq(eq.children[1]):
            return ONE
        elif eq.children[0].eq(ONE):
            eq = eq.children[1].inv()
        elif eq.children[1].eq(ONE):
            eq = eq.children[0]

    if len(eq.children) == 0:
        return eq

    if eq.has_vars():
        return eq

    try:
        e = eq.evaluate()
    except (AttributeError, TypeError, OverflowError):
        print(f"uh-oh {init_eq}")
        return eq
    if e is None:
        return eq
    elif type(e) is int:
        return INode(int(e))
    elif type(e) is float:
        if e.is_integer():
            return INode(int(e))
        else:
            return eq
    else:
        raise Exception(f"Unknown type {e}")


def infix_to_postfix(infix: str) -> List[str]:

    # use ~ to represent the substraction operator
    infix = infix.replace(" ", "")
    infix = re.sub(r"(?<=[a-zA-Z0-9₀-₉_)])-", "~", infix)
    infix = re.sub(r"⁻¹-", "⁻¹~", infix)
    infix = re.sub(rf"\*\*-(?P<arg>[a-zA-Z0-9₀-₉_()])", rf"**(-\g<arg>)", infix)

    # define the operators and their precedence and associativity
    operators: Dict[str, Tuple[int, int]] = {
        "**": (6, 1),  # precedence 4, associativity 1 (right)
        "⁻¹": (6, 0),  # precedence 4, associativity 0 (left) for Lean
        "-": (5, 0),
        "*": (3, 0),  # precedence 4, associativity 0 (left)
        "/": (3, 0),
        "%": (3, 0),
        "+": (2, 0),
        "~": (2, 0),  # sub operator
        "==": (1, 0),
        "!=": (1, 0),
        "<=": (1, 0),
        ">=": (1, 0),
        "<": (1, 0),
        ">": (1, 0),
        "∣": (1, 0),
    }
    for i in range(2, 11):  # consecutive negative operators
        operators["-" * i] = (5, 0)
    for op in U_OPS:
        assert op not in operators
        operators[op] = (10, 0)
    binary_ops = {"min", "max", "gcd", "lcm"}
    for op in binary_ops:
        assert op not in operators
        operators[op] = (10, 0)

    # tokenize
    tokens = [x for x in re.split(r"(-[0-9]+|[a-zA-Z0-9.]+|\(|\)|-|⁻¹)", infix) if x]

    # handle consecutive negative operators by first merging them
    tokens = re.sub(r"(?<=-) (?=- )", "", " ".join(tokens)).split()

    # define operator and output stacks
    op_stack: List[str] = list()
    out_stack: List[str] = list()

    # iterate through input tokens
    for token in tokens:
        if (
            len(token) == 1
            and token.isalpha()
            or re.match(r"-?[0-9]+", token)
            or token == "PI"
            or token == "0.5"
            or token == "-1"
        ):
            out_stack.append(token)
        elif token in operators:
            if len(op_stack) > 0:  # check if there are items in the operator stack
                # peek at the top of the stack and make sure that the
                # associativity and precedence rules are as follows
                while (
                    len(op_stack) > 0
                    and op_stack[-1] != "("
                    and (
                        (
                            operators[token][1] == 0
                            and operators[token][0] <= operators[op_stack[-1]][0]
                        )
                        or (
                            operators[token][1] == 1
                            and operators[token][0] < operators[op_stack[-1]][0]
                        )
                    )
                ):
                    # if the rules follow, push the top of the
                    # operator stack to the output stack
                    out_stack.append(op_stack.pop())
            # push the token to the operator stack
            op_stack.append(token)
        elif token == "(":
            op_stack.append(token)
        elif token == ")":
            # keep pushing the top of the operator stack to the output stack
            while op_stack[-1] != "(":
                out_stack.append(op_stack.pop())
                if len(op_stack) == 0:
                    raise Exception("Invalid parentheses")
            assert op_stack.pop() == "("
            # condition below is not frequent, but allows to deal with expressions
            # with multiple variables, such as: max cos(x) (y + 2)
            if len(op_stack) > 0 and op_stack[-1] in U_OPS:
                out_stack.append(op_stack.pop())
        elif set(token) == {"-"}:
            op_stack.append(token)
        else:
            raise RuntimeError(f"Unknown token: {token}")

    # append the operator stack onto the output stack
    while len(op_stack) > 0:
        out_stack.append(op_stack.pop())

    # sanity check
    assert all(tok != "(" and tok != ")" for tok in out_stack)

    return out_stack


def get_open_pos(s: str, j: int) -> int:
    """
    Given a closing parenthesis, return the position of the opening one.
    """
    assert s[j] == ")"
    n_closed = 1
    i = j - 1
    while i >= 0 and n_closed > 0:
        if s[i] == "(":
            n_closed -= 1
        elif s[i] == ")":
            n_closed += 1
        i -= 1
    i += 1
    assert s[i] == "(" and n_closed == 0
    return i


def get_close_pos(s: str, i: int) -> int:
    """
    Given an opening parenthesis, return the position of the closing one.
    """
    assert s[i] == "("
    n_open = 1
    j = i + 1
    while j < len(s) and n_open > 0:
        if s[j] == ")":
            n_open -= 1
        elif s[j] == "(":
            n_open += 1
        j += 1
    j -= 1
    assert s[j] == ")" and n_open == 0
    return j


def get_arg_lean_op(tokens: List[str]) -> Tuple[List[str], List[str]]:
    if tokens[0] != "(" and tokens[0] not in U_OPS:
        return ["(", tokens[0], ")"], tokens[1:]
    i = 0
    n_open = 0
    n_close = 0
    while (n_open == 0 or n_open > n_close) and i < len(tokens):
        if tokens[i] == "(":
            n_open += 1
        elif tokens[i] == ")":
            n_close += 1
        assert n_open >= n_close
        i += 1
    arg_toks = tokens[:i]
    remaining = tokens[i:]
    assert n_open == n_close
    assert arg_toks[-1] == ")"
    return arg_toks, remaining


def clean_lean_op(infix: str) -> str:

    # performs replacements of the form: (A + 1).gcd (B + 2) -> gcd (A + 1) (B + 2)
    for op in ["gcd", "lcm"]:
        pattern = rf"([a-zA-Z0-9₀-₉_⁻¹.'⁻¹]+).{op} "
        infix = re.sub(pattern, rf"{op} \1 ", infix)
        while True:
            match = re.search(rf"\)\.{op}", infix)
            if match is None:
                break
            j = match.span()[0]
            i = get_open_pos(infix, j)
            assert infix[i] == "("
            assert infix[j] == ")"
            infix = f"{infix[:i]}{op} {infix[i : j + 1]}{infix[j + 5 :]}"

    tokens = [x for x in re.split(r"(-[0-9]+|[a-zA-Z0-9.]+|\(|\)|-|⁻¹)", infix) if x]
    tokens = [x.strip() for x in tokens if x.strip()]
    j = 0
    while j < len(tokens):
        if tokens[j] in ["min", "max", "gcd", "lcm"]:
            arg1, toks = get_arg_lean_op(tokens[j + 1 :])
            arg2, toks = get_arg_lean_op(toks)
            tokens[j:] = [tokens[j], "(", *arg1, *arg2, ")", *toks]
        j += 1
    return "".join(tokens)


def clean_lean_infix(s: str) -> str:

    # UTF-8 symbols
    s = s.replace("^", "**")
    s = s.replace("≤", "<=")
    s = s.replace("≥", ">=")
    s = s.replace("≠", "!=")
    s = re.sub(r"(?<![<>=!])=(?![<>=!])", "==", s)

    # PI
    s = re.sub(rf"(?<![a-zA-Z0-9₀-₉_])(pi|π)(?![a-zA-Z0-9₀-₉_])", "PI", s)

    # function names
    for src, tgt in [
        ("real.", ""),
        ("nat.", ""),
        ("log", "ln"),
        ("arcsin", "asin"),
        ("arccos", "acos"),
        ("arctan", "atan"),
        ("sinh", "sinh"),
        ("cosh", "cosh"),
        ("tanh", "tanh"),
        ("arsinh", "asinh"),
        ("acosh", "acosh"),  # not in lean
        ("atanh", "atanh"),  # not in lean
    ]:
        s = s.replace(src, tgt)

    # add missing parentheses to unary operators (e.g. "cos x" -> cos(x))
    for op in set(U_OPS) - {"neg", "inv"}:
        s = re.sub(rf"{op} ([a-zA-Z0-9₀-₉_]+(⁻¹)?)", rf"{op}(\1)", s)

    # handle Lean operators and operators with multiple variables
    s = clean_lean_op(s)

    return s


def infix_to_node(infix: str) -> Node:
    try:
        return _infix_to_node(infix)
    except Exception:
        print(f"infix_to_node exception on: {infix}")
        raise


def _infix_to_node(infix: str) -> Node:

    init_infix = infix

    # normalize lean infix notations
    infix = clean_lean_infix(infix)

    # convert to postfix
    postfix: List[str] = infix_to_postfix(infix)

    stack: List[Node] = []

    i = 0
    while i < len(postfix):
        token = postfix[i]
        if len(token) == 1 and token.isalpha():
            stack.append(VNode(token))
        elif re.fullmatch(r"-?[0-9]+", token):
            stack.append(INode(int(token)))
        elif token == "0.5":
            stack.append(0.5)  # type: ignore
        elif token == "PI":
            stack.append(PNode(token))
        elif token == "-":
            a = stack.pop()
            stack.append(-a)
        elif token == "+":
            b = stack.pop()
            a = stack.pop()
            stack.append(a + b)
        elif token == "~":
            b = stack.pop()
            a = stack.pop()
            stack.append(a - b)
        elif token == "*":
            b = stack.pop()
            a = stack.pop()
            stack.append(a * b)
        elif token == "/":
            b = stack.pop()
            a = stack.pop()
            stack.append(a / b)
        elif token == "%":
            b = stack.pop()
            a = stack.pop()
            stack.append(a % b)
        elif token == "**":
            b = stack.pop()
            a = stack.pop()
            assert isinstance(b, Node) or isinstance(b, float) and b == 0.5
            stack.append(a ** b)
        elif token in U_OPS:
            a = stack.pop()
            stack.append(UNode(token, a))
        elif token == "⁻¹":
            a = stack.pop()
            stack.append(a ** INode(-1))
        elif token in C_OPS or token in {">=", ">"}:
            b = stack.pop()
            a = stack.pop()
            if token == "==":
                stack.append(a == b)
            elif token == "!=":
                stack.append(a != b)
            elif token == "<=":
                stack.append(a <= b)
            elif token == ">=":
                stack.append(a >= b)
            elif token == "<":
                stack.append(a < b)
            elif token == ">":
                stack.append(a > b)
            elif token == "∣":
                stack.append(CNode("∣", a, b))
            else:
                raise Exception(f"Unexpected comparison token: {token}")
        elif token in ["min", "max", "gcd", "lcm"]:
            b = stack.pop()
            a = stack.pop()
            stack.append(BNode(token, a, b))
        elif set(token) == {"-"}:
            a = stack.pop()
            for _ in range(len(token)):
                a = -a
            stack.append(a)
        else:
            raise Exception(f"Unexpected token: {token}")
        i += 1

    # sanity check
    assert i == len(postfix)
    if len(stack) != 1:
        raise Exception(
            f"Stack should only contain 1 element: {stack} "
            f"Check that the negative operator - is only used as a unary operator. "
            f"Infix input: {init_infix}"
        )

    return stack[0]


if __name__ == "__main__":

    # python -m evariste.envs.eq.utils

    def test_simplify():

        print("===== RUNNING SIMPLIFY TESTS ...")

        x = VNode("x")
        y = VNode("y")
        to_simplify = [
            [(x + ZERO) + (INode(4) + y) * 0, x],
            [(x ** 2) ** 0.5, abs(x)],
            [x - (x + 0), ZERO],
            [((x.exp().ln() + (y.inv()).inv()) ** 2) ** 0.5, abs(x + y)],
            [((x.ln().exp() + ((y.inv()).inv()) * (1 + 0)) ** 0.5) ** 2, x + y],
            [(-(-x) - x + (y.inv())) / (y.inv()), ONE],
            [(-x - (-x) + 1 * abs((y.inv()))) / abs(-(y.inv())), ONE],
            [x - 0 + (ONE.inv() * 1 - ONE), x],
            [ZERO.exp(), ONE],
            [ONE.ln(), ZERO],
        ]
        for node1, node2 in to_simplify:
            assert simplify_equation(node1).eq(node2)

        print("OK")

    def test_infix_to_node():

        from evariste.envs.eq.graph import ZERO, ONE, TWO, PI
        from evariste.envs.eq.graph import exp, ln, sqrt  # , Min, Max
        from evariste.envs.eq.graph import sin, cos
        from evariste.envs.eq.graph import sinh, cosh, tanh, atanh
        from evariste.envs.eq.graph import PI

        x = VNode("x")
        y = VNode("y")
        z = VNode("z")

        print("===== RUNNING INFIX_TO_NODE TESTS ...")

        def is_invalid(x):
            return x is None or x == math.inf or x == -math.inf or x != x

        infixes = [
            ("-3", INode(-3)),
            ("-(-3)", -(INode(-3))),
            ("g", VNode("g")),
            ("-(-F)", -(-VNode("F"))),
            ("-3+2", INode(-3) + 2),
            ("x < -3", x < INode(-3)),
            ("y < -(3)", y < -INode(3)),
            ("z < -(-3)", z < -INode(-3)),
            ("-x+y-z < -3", -x + y - z < INode(-3)),
            (
                "(((((-x) * ((1/(-(-5))) * (x + 0))) * x) + (4 + ((3 + x) + 2))) == (((x * (((1/(5)) * (-x)) * x)) + (x * (-(-(((1/(-(-5))) * (-x)) * x))))) + ((x + (x + 2)) + 4)))",
                (
                    (
                        ((-x) * (ONE / (-INode(-5)) * (x + ZERO)) * x)
                        + (INode(4) + ((3 + x) + TWO))
                    )
                    == (
                        (
                            (x * (((ONE / INode(5)) * (-x)) * x))
                            + (x * (-(-(((ONE / (-INode(-5))) * (-x)) * x))))
                        )
                        + ((x + (x + TWO)) + INode(4))
                    )
                ),
            ),
            ("exp(3/(-x))", exp(INode(3) / (-x))),
            ("(34+2*x)*exp(3/x)", (INode(34) + 2 * x) * exp(INode(3) / x)),
            ("12**2+x**0.5+(7+1/x)", INode(12) ** 2 + x ** 0.5 + (7 + ONE / x)),
            ("12**2+x**2+(7+1/x)", INode(12) ** 2 + x ** 2 + (7 + ONE / x),),
            ("abs(12+x+(-7))+14<= x", abs(12 + x + (INode(-7))) + 14 <= x),
            ("ln(cosh(2)) == atanh(1/x)", ln(cosh(TWO)) == atanh(ONE / x)),
            ("-3+x*ln(-1+4)", INode(-3) + x * (INode(-1) + INode(4)).ln()),
            ("2+3==5", TWO + INode(3) == INode(5)),
            ("2+PI==PI+2", TWO + PI == PI + TWO),
            ("2**0.5+3", TWO ** 0.5 + 3),
            ("cos(PI/2+PI/2)", cos(PI / 2 + PI / 2)),
            ("(-(-(-3)))", -(-(INode(-3)))),
            ("tanh(2 +(-3))", tanh(TWO + (INode(-3)))),
            ("-(1+2)+(x+2-3)", -(ONE + TWO) + (x + TWO - INode(3))),
            ("-(1+2)+(x+2+(-3))", -(ONE + TWO) + (x + TWO + INode(-3))),
            ("-(1-2)+(x+2-(-3))", -(ONE - TWO) + (x + TWO - INode(-3))),
            ("1+(-4)>z", ONE + INode(-4) > z),
            ("cos(1+sinh(3+2*4+x))", cos(ONE + sinh(INode(3) + TWO * INode(4) + x))),
            (
                "(cos(((PI * (1/2)) + (PI * (1/2)))) == cos(PI))",
                cos((PI * (ONE / 2)) + (PI * (ONE / 2))) == cos(PI),
            ),
            ("cos(-(PI/2))", cos(-(PI / TWO))),
            ("0 ≤ x⁻¹", ZERO <= x ** -1),
            ("0 ≤ (x)⁻¹", ZERO <= x ** -1),
            ("2⁻¹ + 2⁻¹ = 1", TWO ** -1 + TWO ** -1 == ONE),
            (
                "x * y * x⁻¹ * (x * z * x⁻¹) = x * (y * z) * x⁻¹",
                x * y * (x ** -1) * (x * z * (x ** -1)) == x * (y * z) * (x ** -1),
            ),
            (
                "x * y * x⁻¹ * (x - z * x⁻¹) = x * (y * z) * x⁻¹",
                x * y * (x ** -1) * (x - z * (x ** -1)) == x * (y * z) * (x ** -1),
            ),
            (
                "abs (cos (x) + (1 + x ^ 2 / 2)) ≤ 1",
                abs(cos(x) + (ONE + (x ** 2) / 2)) <= 1,
            ),
            ("x⁻¹ + y⁻¹ = (x + y) / (x * y)", x ** -1 + y ** -1 == (x + y) / (x * y)),
            ("(x * y) ^ -1 = y ^ -1 * x ^ -1", (x * y) ** -1 == (y ** -1) * (x ** -1)),
            ("log x⁻¹ = -log x", ln(x ** -1) == -ln(x)),
            ("2 ≠ 0", TWO != ZERO),
            ("-x < y", -x < y),
            ("x - y > 0", x - y > 0),
            ("x - y ** z > x ** 2", x - y ** z > x ** 2),
            ("x ** y ** z", x ** (y ** z)),
            ("x ** y + 2 ** z", x ** y + INode(2) ** z),
            ("x ** -1 + z ** 2", x.inv() + z.pow2()),
            ("2 ** (x + 1) > 1", INode(2) ** (x + 1) > 1),
            (
                "abs (cos (x) - (1 - x ^ 2 / 2)) ≤ 1",
                abs(cos(x) - (ONE - (x ** 2) / 2)) <= 1,
            ),
            ("cos(-PI/2)", cos(-PI / 2)),
            ("cos(-(PI/2))", cos(-(PI / 2))),
            ("-x**2", -(x ** 2)),
            ("sqrt (x * y) = sqrt (x) * sqrt (y)", sqrt(x * y) == sqrt(x) * sqrt(y)),
            ("-x+y-z", ((-x) + y) - z),
            ("-x-y-z", ((-x) - y) - z),
            ("y - x < 0", y - x < 0),
            ("y / -x = -(y / x)", y / (-x) == -(y / x)),
            ("-x*y", (-x) * y),
            ("-x **-y", -(x ** (-y))),
            ("-x**2", -(x ** 2)),
            ("-x**-1", -(x ** -1)),
            ("-sqrt(x) ≤ y", -sqrt(x) <= y),
            ("x⁻¹ - y⁻¹ = (y - x) / (x * y)", x ** -1 - y ** -1 == (y - x) / (x * y)),
            ("x⁻¹ = x", (x ** -1) == x),
            ("-x⁻¹ = -x", -(x ** -1) == -x),
            ("x⁻¹⁻¹ = x", (x ** -1) ** -1 == x),
            ("-x⁻¹⁻¹ = -x", -((x ** -1) ** -1) == -x),
            ("-(-x) = x", -(-x) == x),
            ("x = y⁻¹ * z", x == y ** -1 * z),
            ("real.tanh x = real.sinh x / real.cosh x", tanh(x) == sinh(x) / cosh(x)),
            ("real.pi < 3141593 / 1000000", PI < INode(3141593) / INode(1000000)),
            ("π < 3141593 / 1000000", PI < INode(3141593) / INode(1000000)),
            # ("(- - x) = x", -(-x) == x),  # TODO: fix
            # ("min (x) (y)", Min(x, y)),  # TODO: implement
            # ("max (x) (y)", Max(x, y)),  # TODO: implement
            ("--3", -INode(-3)),
            ("--(3)", --INode(3)),
            ("- x", -x),
            ("- - x", --x),
            ("- - -x", ---x),
            ("-x - -y = y - x", -x - -y == y - x),
            ("-x - - -y = -y - x", -x - --y == -y - x),
            ("-x - - ---y = -y - x", -x - ----y == -y - x),
            ("-x - - ---y = -y - x", -x - ----y == -y - x),
            ("-x --3", -x - -3),
            ("-x**2 - - ---3*y  **2", -(x ** 2) - ---INode(-3) * y ** 2),
            # ("min(x, y)", Min(x, y)), # TODO: fix
            ("min x y", Min(x, y)),
            ("min x (y)", Min(x, y)),
            ("min (x) y", Min(x, y)),
            ("min (x) (y)", Min(x, y)),
            ("max (x) (y)", Max(x, y)),
            ("max cos(x) (y + 2)", Max(cos(x), y + 2)),
            ("max (cos(x)) (y + 2)", Max(cos(x), y + 2)),
            ("min x cos(y)", Min(x, cos(y))),
            ("min (x) cos(y)", Min(x, cos(y))),
            ("max 3 exp(min (x) cos(y))", Max(INode(3), exp(Min(x, cos(y))))),
            ("max (3) exp(min (x) cos(y))", Max(INode(3), exp(Min(x, cos(y))))),
            ("x.gcd 1", GCD(x, ONE)),
            (
                "(2 + x).lcm (exp (x + y * z)) = (cos x).gcd (sin y)",
                LCM(2 + x, exp(x + y * z)) == GCD(cos(x), sin(y)),
            ),
            (
                "min x (y * z) = min x (min x y * min x z)",
                Min(x, y * z) == Min(x, (Min(x, y) * Min(x, z))),
            ),
        ]
        for infix, ref_node in infixes:
            # print("======")
            # print(infix)
            ref_node = ref_node.switch_pow()
            node = infix_to_node(infix)
            if ref_node is not None:
                if not node.eq(ref_node):
                    raise Exception(
                        f"\nFound:\n{node}\t\t{node.prefix()}"
                        f"\nexpected:\n{ref_node}\t\t{ref_node.prefix()}"
                    )
            node_ = infix_to_node(node.infix())
            if not node.eq(node_):
                raise Exception(
                    f"Found:\n{node_}    Prefix: ({node_.prefix()})\nexpected:\n"
                    f"{node}    Prefix: ({node.prefix()})"
                )
            for x in TEST_VALUES:
                if any(
                    s in infix
                    for s in [
                        "⁻¹",
                        "≠",
                        "≤",
                        "≥",
                        "y",
                        "z",
                        "-(-",
                        "real",
                        "π",
                        "g",
                        "- -",
                    ]
                ):
                    continue
                subst = {"x": x}
                subst_ = {"pi": math.pi, **subst} if "pi" in infix.lower() else subst
                y = ne.evaluate(infix_for_numexpr(infix), local_dict=subst_)
                y_ = node.evaluate(subst=subst)
                assert is_invalid(y) == is_invalid(y_)
                if is_invalid(y):
                    continue
                if y != y_ and abs(y - y_) / max(abs(y), abs(y_)) > 1e-6:
                    print(infix, node, y, y_)
                    raise RuntimeError(
                        f"Unexpected output in {x}. {infix} evaluates to {y}, "
                        f"but {node.infix()} evaluates to {y_}"
                    )

        print("OK")

    test_simplify()
    test_infix_to_node()
