# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, List, Set, Dict
from fractions import Fraction
from datetime import datetime
from enum import Enum
import sys
import math
import warnings
import numpy as np
import numexpr as ne
import re

from evariste.utils import timeout, MyTimeoutError


INTEGERS = [f"{i}" for i in range(10)]
VARIABLES = [f"x{i}" for i in range(10)]
CONSTANTS = [f"a{i}" for i in range(100)]
CONSTANTS.append("PI")

U_OPS = [
    "neg",
    "inv",
    "exp",
    "ln",
    "pow2",
    "sqrt",
    "abs",
    "cos",
    "sin",
    "tan",
    "atan",
    "acos",
    "asin",
    "cosh",
    "sinh",
    "tanh",
    "atanh",
    "acosh",
    "asinh",
]
B_OPS = ["add", "mul", "sub", "div", "max", "min", "%", "**", "gcd", "lcm"]
C_OPS = ["==", "!=", "<=", "<", "∣"]


class NodeType(Enum):
    UNDEFINED = 0
    INTEGER = 1
    VARIABLE = 2
    CONSTANT = 3
    UNARY = 4
    BINARY = 5
    COMPARISON = 6


UNDEFINED = NodeType.UNDEFINED
INTEGER = NodeType.INTEGER
VARIABLE = NodeType.VARIABLE
CONSTANT = NodeType.CONSTANT
UNARY = NodeType.UNARY
BINARY = NodeType.BINARY
COMPARISON = NodeType.COMPARISON


TOK_TYPE = {
    x: y
    for x, y in [(variable, VARIABLE) for variable in VARIABLES]
    + [(integer, INTEGER) for integer in INTEGERS]
    + [(constant, CONSTANT) for constant in CONSTANTS]
    + [(u_ops, UNARY) for u_ops in U_OPS]
    + [(b_ops, BINARY) for b_ops in B_OPS]
    + [(c_ops, COMPARISON) for c_ops in C_OPS]
}


def infix_for_numexpr(infix: str) -> str:
    new_infix = (
        infix.replace("PI", "pi")
        .replace("ln(", "log(")
        .replace("atan(", "arctan(")
        .replace("asin(", "arcsin(")
        .replace("acos(", "arccos(")
        .replace("atanh(", "arctanh(")
        .replace("asinh(", "arcsinh(")
        .replace("acosh(", "arccosh(")
    )
    # new_infix = re.sub(r"((?<![a-zA-Z])\d+)", r"\1.0", new_infix)  # fails if (say) 2.0 is in the input
    # new_infix = re.sub(r"abs\(-(\d+)\)", r"abs(\1)", new_infix)
    return new_infix


class NodeParseError(Exception):
    pass


class EqNotFoundVariable(Exception):
    pass


class EqMatchException(Exception):
    pass


class EqInvalidVarType(Exception):
    pass


class Node:

    VARIABLE_TYPES = [None, "int"]

    def __init__(
        self,
        ntype: NodeType = UNDEFINED,
        value=None,
        children: Optional[List] = None,
        variable_type: Optional[str] = None,
    ):
        self.type: NodeType = ntype
        self.value = value  # : Union[str, int] -- use int_value / op_value / etc.?

        self.children: List[Node] = [] if children is None else children
        self.variable_type: Optional[str] = variable_type
        self._prefix_len: Optional[int] = None

        # cached properties for has_vars / is_valid / is_rational / fraction
        self._has_vars: Optional[bool] = None
        self._is_valid: Optional[bool] = None
        self._is_rational: Optional[bool] = None
        self._fraction: Optional[Fraction] = None
        self._vtype: Optional[str] = None

        # store the node evaluation (only for constant nodes)
        self._const_eval: Tuple[bool, Optional[float]] = (False, None)

        # sanity checks
        assert type(ntype) is NodeType
        assert (ntype is UNDEFINED) == (value is None)
        assert (
            variable_type is None
            or self.is_var()
            and variable_type in Node.VARIABLE_TYPES
        )

    @property
    def int_value(self) -> int:
        assert isinstance(self.value, int)
        return self.value

    @property
    def str_value(self) -> str:
        assert isinstance(self.value, str)
        return self.value

    # @property
    # def vtype(self) -> Optional[str]:
    #     assert self.is_var()
    #     return self._vtype

    # @property
    # def children(self) -> List["Node"]:
    #     assert self._children is not None
    #     return self._children

    def is_int(self) -> bool:
        return self.type is INTEGER

    def is_var(self) -> bool:
        return self.type is VARIABLE

    def is_const(self) -> bool:
        return self.type is CONSTANT

    def is_unary(self) -> bool:
        return self.type is UNARY

    def is_binary(self) -> bool:
        return self.type is BINARY

    def is_comp(self) -> bool:
        return self.type is COMPARISON

    def push_child(self, child: "Node") -> None:
        assert self.children is not None
        self.children.append(child)

    def depth(self) -> int:
        if len(self.children) == 0:
            return 0
        return 1 + max(c.depth() for c in self.children)

    # def at_prefix_pos(self, prefix_pos: int) -> "Node":
    #     def _find(node: Node, cur_pos: int) -> Node:
    #         if cur_pos == prefix_pos:
    #             return node
    #         child_prefix_start = cur_pos + 1
    #         for child in node.children:
    #             child_len = child.prefix_len()
    #             if child_prefix_start <= prefix_pos < child_prefix_start + child_len:
    #                 return _find(child, child_prefix_start)
    #             child_prefix_start += child_len
    #         raise RuntimeError(f"Unexpected prefix position: {prefix_pos} in {self}")

    #     return _find(self, 0)

    def prefix_tokens(self) -> List[str]:
        if self.is_int():
            assert type(self.value) is int
            sign = "+" if self.value >= 0 else "-"
            return [sign, *str(abs(self.value))]
        else:
            tokens = [self.value]
            for c in self.children:
                tokens.extend(c.prefix_tokens())
        return tokens

    def _get_ops(self, ntype: NodeType, ops: Set[str]) -> None:
        if self.type is ntype:
            ops.add(self.value)
        for c in self.children:
            c._get_ops(ntype, ops)

    def get_unary_ops(self) -> Set[str]:
        res: Set[str] = set()
        self._get_ops(UNARY, res)
        return res

    def get_binary_ops(self) -> Set[str]:
        res: Set[str] = set()
        self._get_ops(BINARY, res)
        return res

    @classmethod
    def from_prefix_tokens(cls, tokens: List[str]) -> "Node":
        assert type(tokens) is list

        def aux(offset: int) -> Tuple["Node", int]:
            if offset >= len(tokens):
                raise NodeParseError(f"Missing token, parsing {' '.join(tokens)}")
            tok = tokens[offset]
            tok_type = TOK_TYPE.get(tok, None)  # + 1 2 -> None,
            if tok_type is BINARY:
                lhs, rhs_offset = aux(offset + 1)
                rhs, next_offset = aux(rhs_offset)
                return BNode(tok, lhs, rhs), next_offset
            if tok_type is COMPARISON:
                lhs, rhs_offset = aux(offset + 1)
                rhs, next_offset = aux(rhs_offset)
                return CNode(tok, lhs, rhs), next_offset
            elif tok_type is UNARY:
                term, next_offset = aux(offset + 1)
                return UNode(tok, term), next_offset
            elif tok_type is CONSTANT:
                return PNode(tok), offset + 1
            elif tok_type is VARIABLE:
                return VNode(tok), offset + 1
            elif tok_type is INTEGER:
                return INode(int(tok)), offset + 1
            else:
                if tok != "+" and tok != "-":
                    raise NodeParseError(
                        f"Unexpected token of {tok_type} type: {tok} // {tokens}"
                    )
                i, v, found = offset + 1, 0, False
                while i < len(tokens):
                    if tokens[i] not in INTEGERS:
                        break
                    v = v * 10 + int(tokens[i])
                    i += 1
                    found = True
                if not found:
                    raise NodeParseError(f"Missing digit, parsing {' '.join(tokens)}")
                if tok == "-":
                    v *= -1
                return INode(v), i

        node, last_offset = aux(0)
        if last_offset != len(tokens):
            raise NodeParseError(
                f"Didn't parse everything: {' '.join(tokens)}. "
                f"Stopped at length {last_offset}"
            )
        return node

    def prefix_len(self, refresh=False) -> int:
        if self._prefix_len is not None and not refresh:
            return self._prefix_len
        if self.is_int():
            assert type(self.value) is int
            self._prefix_len = 1 + len(str(abs(self.value)))
        else:
            self._prefix_len = 1 + sum(c.prefix_len(refresh) for c in self.children)
        return self._prefix_len

    def prefix(self) -> str:
        return " ".join(str(x) for x in self.prefix_tokens())

    def qtree_prefix(self) -> str:
        """Export to latex qtree format: prefix with Tree, use package qtree."""
        s = "[.$" + str(self.value) + "$ "
        for c in self.children:
            s += c.qtree_prefix()
        s += "]"
        return s

    def infix(self, old_format: bool = False) -> str:
        """
        Since rule names are generated with this function, old_format is used to not
        lose compatibility with previous models.
        New format is compatible with infix_to_node.
        """
        if self.is_int():
            assert len(self.children) == 0
            return str(self.value) if self.value >= 0 else f"({self.value})"
        elif len(self.children) == 0:
            return str(self.value)
        elif len(self.children) == 1:
            assert self.value in U_OPS
            c = self.children[0].infix(old_format)
            assert c[0] != "(" or c[-1] == ")"
            if self.value == "neg":
                if old_format:
                    return f"(-{c})"
                else:
                    return f"(-{c})" if c[0] == "(" else f"(-({c}))"
            elif self.value == "inv":
                return f"({c} ** -1)"
            elif self.value == "pow2":
                return f"({c} ** 2)"
            else:
                if old_format:
                    return f"{self.value}({c})"
                else:
                    # c = c if c[0] == "(" else f"({c})"
                    # return f"({self.value}{c})"
                    return f"{self.value}{c}" if c[0] == "(" else f"{self.value}({c})"
        elif len(self.children) == 2:
            c0 = self.children[0].infix(old_format)
            c1 = self.children[1].infix(old_format)
            assert self.value in B_OPS or self.value in C_OPS
            if self.value == "add":
                return f"({c0} + {c1})"
            if self.value == "sub":
                return f"({c0} - {c1})"
            elif self.value == "mul":
                return f"({c0} * {c1})"
            elif self.value == "%":
                return f"({c0} % {c1})"
            elif self.value == "div":
                return f"({c0} / {c1})"

            elif self.value == "**":
                return f"({c0} ** {c1})"
            elif self.value in ["min", "max", "gcd", "lcm"]:
                return f"({self.value} {c0} {c1})"
            elif self.value in C_OPS:
                return f"({c0} {self.value} {c1})"
            else:
                raise Exception(f"Unknown operator: {self.value}")
        else:
            raise Exception(f"Too many children: {len(self.children)}")

    LEAN_U_OP_MAPPING = {
        "abs": "abs",
        "ln": "real.log",
        "exp": "real.exp",
        "sqrt": "real.sqrt",
        "cos": "real.cos",
        "sin": "real.sin",
        "tan": "real.tan",
        "cosh": "real.cosh",
        "sinh": "real.sinh",
        "tanh": "real.tanh",
        "asin": "real.arcsin",
        "acos": "real.arccos",
        "atan": "real.arctan",
        "asinh": "real.arsinh",
        # "acosh": "real.arccos",
        # "atanh": "real.arctan",
    }

    LEAN_B_OP_MAPPING = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "==": "=",
        "!=": "≠",
        "<=": "≤",
        "<": "<",
        "∣": "∣",
        "%": "%",
        "**": "^",
    }

    def infix_lean(self, vtype: str = "real") -> str:
        var_type = {"nat": "ℕ", "int": "ℤ", "real": "ℝ"}[vtype]
        if self.is_int():
            assert len(self.children) == 0
            return f"({self.value}:{var_type})"
        elif len(self.children) == 0:
            if self.is_var():
                return str(self.value)
            elif self.value == "PI":
                return "real.pi"
            else:
                raise Exception(f"Not implemented: {self.value}")
        elif len(self.children) == 1:
            assert self.value in U_OPS
            c = self.children[0].infix_lean(vtype)
            if self.value == "neg":
                return f"(-{c})"
            elif self.value == "inv":
                return f"({c}⁻¹)"
                # raise Exception(f"Unexpected operator: {self.value} -- {str(self)}")
            elif self.value == "pow2":
                return f"({c}^2)"
                # raise Exception(f"Unexpected operator: {self.value} -- {str(self)}")
            elif self.value in self.LEAN_U_OP_MAPPING:
                op_infix = self.LEAN_U_OP_MAPPING[self.value].replace("real", vtype)
                return f"({op_infix} {c})"
            else:
                raise Exception(f"Not implemented: {self.value}")
        elif len(self.children) == 2:
            c0 = self.children[0].infix_lean(vtype)
            c1 = self.children[1].infix_lean(vtype)
            assert self.value in B_OPS or self.value in C_OPS
            if self.value == "**" and self.children[1].eq(INode(-1)):
                return f"({c0}⁻¹)"
            elif self.value in self.LEAN_B_OP_MAPPING:
                return f"({c0} {self.LEAN_B_OP_MAPPING[self.value]} {c1})"
            elif self.value in ["min", "max"]:
                return f"({self.value} {c0} {c1})"
            elif self.value in ["gcd", "lcm"]:
                assert vtype in ["nat", "int"], vtype
                return f"({vtype}.{self.value} {c0} {c1})"
            else:
                raise Exception(f"Unknown operator: {self.value}")
        else:
            raise Exception(f"Too many children: {len(self.children)}")

    def infix_latex(self) -> str:
        """
        TODO: add tests / add C_OPS handling
        """
        n_children = len(self.children)
        if n_children == 0:
            return str(self.value)
        elif n_children == 1:
            c = self.children[0].infix_latex()
            if self.value == "sqrt":
                s = "\\sqrt{" + c + "}"
            elif self.value == "pow2":
                s = f"({c})^2"
            elif self.value == "neg":
                s = f"-({c})"
            elif self.value == "inv":
                s = "\\frac{1}{" + c + "}"
            elif self.value == "abs":
                s = "\\lvert " + c + "\\rvert "
            elif self.value in [
                "ln",
                "exp",
                "cos",
                "sin",
                "tan",
                "cosh",
                "sinh",
                "tanh",
            ]:
                s = "\\"
                s += str(self.value)
                s += "(" + c + ")"
            else:
                s = str(self.value)
                s += "(" + c + ")"
            return s
        assert n_children == 2
        c0, c1 = self.children
        infix0 = c0.infix_latex()
        infix1 = c1.infix_latex()
        if self.value == "max":
            return f"\\max({infix0}, {infix1})"
        if self.value == "min":
            return f"\\min({infix0}, {infix1})"
        if self.value == "<=":
            return f"{infix0} \\leq {infix1}"
        if self.value == "!=":
            return f"{infix0} \\neq {infix1}"
        s = f"({infix0} {self.value} {infix1})"
        s = s.replace(" add ", "+")
        s = s.replace(" == ", " = ")
        s = s.replace(" sub ", "-")
        s = s.replace(" mul ", r" \cdot ")  # \times
        return s

    def __str__(self):
        return self.infix()

    def __repr__(self):
        return self.infix()

    def __len__(self):
        return 1 + sum(len(c) for c in self.children)

    def __bool__(self):
        raise Exception(f"Node '{self}' cannot be converted to a boolean expression")

    def switch_pow(self: "Node") -> "Node":
        if self.value == "pow2":
            return BNode("**", self.children[0].switch_pow(), INode(2))
        elif self.value == "inv":
            return BNode("**", self.children[0].switch_pow(), INode(-1))
        else:
            return Node(self.type, self.value, [c.switch_pow() for c in self.children])

    def eq(self, node) -> bool:
        """
        Check if two trees are exactly equal, i.e. two expressions are exactly equal (strong equality)
        """
        return (
            self.type == node.type
            and self.value == node.value
            and len(self.children) == len(node.children)
            and all(c1.eq(c2) for c1, c2 in zip(self.children, node.children))
        )

    def ne(self, node) -> bool:
        return not self.eq(node)

    def __eq__(self, node) -> "Node":  # type: ignore
        return CNode("==", self, node)

    def __ne__(self, node) -> "Node":  # type: ignore
        return CNode("!=", self, node)

    def __le__(self, node) -> "Node":
        node = autocast(node)
        return CNode("<=", self, node)

    def __lt__(self, node) -> "Node":
        node = autocast(node)
        return CNode("<", self, node)

    def __ge__(self, node) -> "Node":
        node = autocast(node)
        return CNode("<=", node, self)

    def __gt__(self, node) -> "Node":
        node = autocast(node)
        return CNode("<", node, self)

    def __add__(self, node) -> "Node":
        node = autocast(node)
        return BNode("add", self, node)

    def __radd__(self, node) -> "Node":
        node = autocast(node)
        return BNode("add", node, self)

    def __sub__(self, node) -> "Node":
        node = autocast(node)
        return BNode("sub", self, node)

    def __rsub__(self, node) -> "Node":
        node = autocast(node)
        return BNode("sub", node, self)

    def __mul__(self, node) -> "Node":
        node = autocast(node)
        return BNode("mul", self, node)

    def __rmul__(self, node) -> "Node":
        node = autocast(node)
        return BNode("mul", node, self)

    def __truediv__(self, node) -> "Node":
        node = autocast(node)
        return BNode("div", self, node)

    def __abs__(self) -> "Node":
        return UNode("abs", self)

    def __neg__(self) -> "Node":
        return UNode("neg", self)

    def __pow__(self, exponent) -> "Node":
        if isinstance(exponent, Node):
            return BNode("**", self, exponent)
        elif exponent == -1:
            return UNode("inv", self)
        elif exponent == 2:
            return UNode("pow2", self)
        elif exponent == 3:
            return self ** 2 * self
        elif exponent == 4:
            return (self ** 2) ** 2
        elif exponent == 0.5:
            return UNode("sqrt", self)
        else:
            raise Exception(f"Exponent not supported: {exponent}")

    def __mod__(self, node) -> "Node":
        node = autocast(node)
        return BNode("%", self, node)

    def __contains__(self, node) -> bool:
        """Return whether the tree contains a given subtree."""
        return self.eq(node) or any(node in c for c in self.children)

    def neg(self) -> "Node":
        return UNode("neg", self)

    def add(self, node: "Node") -> "Node":
        return BNode("add", self, node)

    def sub(self, node: "Node") -> "Node":
        return BNode("sub", self, node)

    def mul(self, node: "Node") -> "Node":
        return BNode("mul", self, node)

    def div(self, node: "Node") -> "Node":
        return BNode("div", self, node)

    def exp(self) -> "Node":
        return UNode("exp", self)

    def ln(self) -> "Node":
        return UNode("ln", self)

    def pow2(self) -> "Node":
        return UNode("pow2", self)

    def sqrt(self) -> "Node":
        return UNode("sqrt", self)

    def inv(self) -> "Node":
        return UNode("inv", self)

    def sin(self) -> "Node":
        return UNode("sin", self)

    def cos(self) -> "Node":
        return UNode("cos", self)

    def tan(self) -> "Node":
        return UNode("tan", self)

    def asin(self) -> "Node":
        return UNode("asin", self)

    def acos(self) -> "Node":
        return UNode("acos", self)

    def atan(self) -> "Node":
        return UNode("atan", self)

    def sinh(self) -> "Node":
        return UNode("sinh", self)

    def cosh(self) -> "Node":
        return UNode("cosh", self)

    def tanh(self) -> "Node":
        return UNode("tanh", self)

    def asinh(self) -> "Node":
        return UNode("asinh", self)

    def acosh(self) -> "Node":
        return UNode("acosh", self)

    def atanh(self) -> "Node":
        return UNode("atanh", self)

    def negation(self) -> "Node":
        assert self.is_comp()
        lhs, rhs = self.children
        if self.value == "==":
            return lhs != rhs
        elif self.value == "!=":
            return lhs == rhs
        elif self.value == "<":
            return lhs >= rhs
        elif self.value == "<=":
            return lhs > rhs
        elif self.value == "∣":
            raise Exception(f"Incompatible operator: '{self.value}'")
        else:
            raise Exception(f"Unknown operator: '{self.value}'")

    def clone(self) -> "Node":
        """Clone a tree."""
        return Node(self.type, self.value, [c.clone() for c in self.children])

    def replace(self, x: "Node", y: "Node") -> "Node":
        """
        Return a copy of the current tree, where we
        replaced the source tree by a target tree.
        """
        assert isinstance(x, Node)
        assert isinstance(y, Node)
        if self.eq(x):
            return y
        return Node(self.type, self.value, [c.replace(x, y) for c in self.children])

    def replace_at(self, x: "Node", prefix: int, counter: int = 0):
        if prefix == counter:
            return x
        new_children = []
        offset = 1
        for c in self.children:
            new_children.append(c.replace_at(x, prefix, counter + offset))
            offset += c.prefix_len()
        return Node(self.type, self.value, new_children)

    def find_prefix(self, x: "Node", counter: int = 0) -> int:
        """find one match for x in self"""
        if self.eq(x):
            return counter
        else:
            offset = 1
            for c in self.children:
                found = c.find_prefix(x, counter + offset)
                if found > 0:
                    return found
                offset += c.prefix_len()
        return -1

    def has_vars(self) -> bool:
        """
        Return whether the current node has variables.
        """
        if self._has_vars is None:
            self._has_vars = self.is_var() or any(c.has_vars() for c in self.children)
        return self._has_vars

    def get_vars(self) -> Set[str]:
        """
        Return a set with the variables in the equation.
        """

        def _get_vars(eq: Node, res: Set[str]) -> None:
            if eq.is_var():
                assert type(eq.value) is str
                assert len(eq.children) == 0
                res.add(eq.value)
            for c in eq.children:
                _get_vars(c, res)

        res: Set[str] = set()
        _get_vars(self, res)
        return res

    def set_var(self, name: str, value: "Node", mandatory: bool = True) -> "Node":
        """
        Set a variable in an equation.
        # TODO: also possible to use self.replace
        """
        assert type(name) is str
        assert isinstance(value, Node)
        found = []

        def _set_var(eq: Node):
            if eq.is_var() and eq.value == name:
                if eq.variable_type == "int" and not value.is_int():
                    raise EqInvalidVarType
                found.append(True)
                return value
            children = [_set_var(c) for c in eq.children]
            return Node(eq.type, eq.value, children)

        eq = _set_var(self)
        if len(found) == 0 and mandatory:
            raise EqNotFoundVariable(f"Variable {name} not found in {self}")

        return eq

    def set_vars(self, subst: Dict[str, "Node"], mandatory: bool = True) -> "Node":
        """
        Replace a set of variables.
        """
        eq = self
        for name, value in subst.items():
            eq = eq.set_var(name, value, mandatory=mandatory)
        return eq

    def _match(
        self, node: "Node", res: Dict[str, "Node"], variables: Optional["NodeSet"]
    ) -> bool:
        """
        Check if two trees have the same form (weak equality). Used to check whether
        a tree has a specific form, and whether one can apply a specific rule or not.
        E.g. A + B and C + D have the same form, the value of A, B, C and D do not matter.
        Input:
            self: (A + B) ** 2
            node: (x + (y + t)) ** 2
        Output:
            A: x
            B: y + t
        """
        if self.is_comp() != node.is_comp():
            return False

        if self.is_var() and (variables is None or self in variables):
            if self.value in res:
                return res[self.value].eq(node)
            if (
                self.variable_type is None
                or self.variable_type == "int"
                and node.is_int()
            ):
                res[self.value] = node
                return True
            return False
        elif self.is_var():
            return self.eq(node)

        if self.is_int():
            return self.eq(node)

        if self.is_const():
            return self.eq(node)

        assert self.is_unary() or self.is_binary() or self.is_comp()
        if self.value != node.value or len(self.children) != len(node.children):
            return False
        for c1, c2 in zip(self.children, node.children):
            if not c1._match(c2, res, variables):
                return False
        return True

    def match(
        self, node: "Node", variables: Optional["NodeSet"] = None
    ) -> Optional[Dict[str, "Node"]]:
        res: Dict[str, Node] = {}
        matched = self._match(node, res, variables)
        return res if matched else None

    def count_nested_exp(self) -> int:
        """
        Return the maximum number of nested exponential functions.
        Used to determine whether the evaluating with numexpr will blow.
        """
        EXP_OPERATORS = {"exp", "sinh", "cosh", "tanh", "**"}

        def traverse(node: Node) -> int:
            if node.is_var() or node.is_int() or node.is_const():
                return 0
            elif node.is_unary():
                v = 1 if node.value in EXP_OPERATORS else 0
                return v + traverse(node.children[0])
            else:
                assert node.is_binary() or node.is_comp()
                c0, c1 = node.children
                return max(traverse(c0), traverse(c1))

        return traverse(self)

    def evaluate(self, subst: Optional[Dict] = None) -> Optional[float]:
        if not self.can_evaluate():
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            print(
                f"WARNING {now} -- cannot evaluate expression {self.infix()}",
                file=sys.stderr,
                flush=True,
            )
            return None
        if self._const_eval[0]:
            assert not self.has_vars()
            return self._const_eval[1]
        try:
            res = self._evaluate(subst)
        except (MyTimeoutError, TypeError, NotImplementedError, KeyError) as e:
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            print(
                f"WARNING {now} -- evaluate exception of type "
                f"{type(e).__name__}: {str(e)} when evaluating {self.infix()}",
                file=sys.stderr,
                flush=True,
            )
            res = None
        if not self.has_vars():
            self._const_eval = (True, res)
        return res

    @timeout(2)
    def _evaluate(
        self, subst: Optional[Dict], none_if_nan_inf: bool = True
    ) -> Optional[float]:
        """
        Evaluate the node in a particular point.
        """
        if subst is None:
            subst = {"pi": math.pi}
        else:
            assert "pi" not in subst
            subst = dict(subst)
            subst["pi"] = math.pi

        bin_ops = self.get_binary_ops()
        for op in ["min", "max", "gcd", "lcm"]:
            if op in bin_ops:
                raise TypeError(f"{op} operator is not compatible with numexpr!")

        # ignore expressions likely to blow memory (i.e. too large numbers)
        if self.count_nested_exp() >= 4:
            return None if none_if_nan_inf else math.nan

        # convert to infix for numexpr evaluation
        infix = infix_for_numexpr(self.infix())
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            try:
                value = ne.evaluate(infix, local_dict=subst).item()
            except (
                ZeroDivisionError,
                OverflowError,
                AttributeError,
                TypeError,
                ValueError,
                KeyError,
                MemoryError,
            ):
                return None if none_if_nan_inf else math.nan

        if none_if_nan_inf and (math.isnan(value) or math.isinf(value)):
            return None

        return value

    def compute_is_valid(self, is_lean: bool = False) -> bool:
        """
        Test whether an expression is valid (i.e. does not contain any division by 0, or
        the log of a negative value).
        Can fail when mixed with variables, e.g. (1 / (0 * x)).

        if is_lean : log | sqrt (neg) = 0, x / 0 = 0
        """
        c_value = None
        # int / variable / constant
        if self.is_int() or self.is_var() or self.is_const():
            assert len(self.children) == 0
            return True

        # comparison or binary operator
        if self.is_comp() or self.is_binary():
            assert len(self.children) == 2
            if any(s.is_comp() for s in self.children):
                return False
            valid = all(c.is_valid(is_lean=is_lean) for c in self.children)
            if valid and self.value == "div" and not self.children[1].has_vars():
                if is_lean:
                    return True  # in lean 1/0 = 0
                c_value = self.children[1].evaluate()
                if c_value is None or abs(c_value) < 1e-8:
                    return False
            return valid

        # unary operator
        assert self.is_unary()
        assert len(self.children) == 1
        if self.children[0].is_comp():
            return False
        valid_child = self.children[0].is_valid(is_lean=is_lean)
        if not valid_child or self.has_vars():
            return valid_child

        # if we cannot evaluate this node or its child, the node is invalid
        if not is_lean:
            if self.evaluate() is None:
                return False
            c_value = self.children[0].evaluate()
            if c_value is None:
                return False

        # numerical checks
        if is_lean:
            return True
        assert c_value is not None
        op = self.value
        if op == "inv" and abs(c_value) < 1e-8:
            return False
        elif op == "ln" and c_value < 1e-8:
            return False
        elif op == "sqrt" and c_value < -1e-8:
            return False
        elif op == "tan":
            mod_val = np.fmod(c_value, math.pi)
            if (
                mod_val > 0
                and abs(mod_val - math.pi / 2) < 1e-8
                or mod_val < 0
                and abs(mod_val + math.pi / 2) < 1e-8
            ):
                return False
        elif op == "acos" and not (-1 <= c_value <= 1):
            return False
        elif op == "asin" and not (-1 <= c_value <= 1):
            return False
        elif op == "atanh" and not (-1 < c_value < 1):
            return False
        elif op == "acosh" and not (c_value >= 1):
            return False
        else:
            assert op in U_OPS

        return True

    def can_evaluate(self) -> bool:
        bin_ops = self.get_binary_ops()
        return not any(op in ["min", "max", "gcd", "lcm"] for op in bin_ops)

    def is_valid(self, vtype: str = "real", is_lean: bool = False) -> bool:
        if (vtype == "nat" or vtype == "int") and not self.can_evaluate():
            return True
        if self._is_valid is None:
            self._is_valid = self.compute_is_valid(is_lean)
        return self._is_valid

    def is_rational(self, vtype: str) -> bool:
        """
        Determine whether a number is rational. This function has multiple usages.
        - check whether the number is a rational number
        - check that we can call `self.fraction(vtype)` on `self`
        - check whether norm_num will succeed in mathlib
        The fonction matches the "standard" sense of being rational for `vtype=real`,
        and matches the Lean behavior for `nat` and `int`.
        """
        assert vtype in ["nat", "int", "real"]
        assert (self._is_rational is None) == (self._vtype is None)
        if self._is_rational is None:
            if self.has_vars() or self.is_comp() or self.is_const():
                res = False
            elif self.is_int():
                res = True
            elif self.is_unary():
                c = self.children[0]
                if vtype == "real":
                    res = c.is_rational(vtype) and (
                        self.value in ["neg", "pow2"]
                        or self.value == "inv"
                        and c.fraction(vtype) != 0
                    )
                elif vtype == "nat":
                    assert self.value == "sqrt"
                    res = False
                else:
                    assert vtype == "int"
                    # must be different from abs here otherwise norm_num fails
                    res = self.value != "abs" and c.is_rational(vtype)
            elif self.is_binary():
                assert self.value in B_OPS
                if vtype == "nat" and self.value == "lcm":
                    res = False
                else:
                    c1, c2 = self.children
                    res = c1.is_rational(vtype) and c2.is_rational(vtype)
                    if self.value == "div":
                        res = res and c2.fraction(vtype) != 0
                    elif self.value == "**":
                        res = res and c2.is_int()
            else:
                raise Exception(f"Unknown node type: {self}")
            self._is_rational = res
            self._vtype = vtype
        assert self._vtype == vtype
        return self._is_rational

    def fraction(self, vtype: str) -> Fraction:
        assert vtype in ["nat", "int", "real"]
        assert self.is_rational(vtype), self.infix()
        if self._fraction is None:
            if self.is_int():
                res = Fraction(self.value)
            elif self.is_unary():
                c = self.children[0].fraction(vtype)
                if self.value == "neg":
                    res = -c
                elif self.value == "inv":
                    res = 1 / c
                elif self.value == "abs":
                    res = abs(c)
                elif self.value == "pow2":
                    res = c ** 2
                else:
                    raise Exception(f"Unknown operator: {self.value}")
            else:
                assert self.is_binary()
                c1 = self.children[0].fraction(vtype)
                c2 = self.children[1].fraction(vtype)
                if self.value in ["gcd", "lcm", "%"]:
                    assert vtype in ["nat", "int"]
                    assert int(c1) == c1 and int(c2) == c2
                if self.value == "add":
                    res = c1 + c2
                elif self.value == "mul":
                    res = c1 * c2
                elif self.value == "div":
                    if vtype == "real":
                        res = c1 / c2
                    else:
                        if c2 == 0:
                            res = Fraction(0)
                        elif c2 > 0:
                            res = Fraction(math.floor(c1 / c2))
                        else:
                            res = -Fraction(math.floor(c1 / -c2))
                elif self.value == "sub":
                    res = c1 - c2
                    if vtype == "nat":
                        res = max(res, Fraction(0))
                elif self.value == "gcd":
                    res = Fraction(int(np.gcd(int(c1), int(c2))))
                elif self.value == "lcm":
                    res = Fraction(int(np.lcm(int(c1), int(c2))))
                elif self.value == "%":
                    res = Fraction((c1 % abs(c2)) if c2 != 0 else c1)
                elif self.value == "**":
                    res = Fraction(c1 ** c2)
                elif self.value == "min":
                    res = min(c1, c2)
                elif self.value == "max":
                    res = max(c1, c2)
                else:
                    raise Exception(f"Unknown operator: {self.value}")
            self._fraction = res
        return self._fraction

    def _norm_numify(self, vtype: str) -> "Node":
        if self.is_rational(vtype):
            try:
                frac = self.fraction(vtype)
            except ZeroDivisionError:
                pass
            else:
                if vtype == "int" or int(frac) == float(frac):
                    # TODO: Here some typing will be needed
                    return INode(int(frac))
        return Node(
            self.type,
            self.value,
            [c._norm_numify(vtype) for c in self.children],
            self.variable_type,
        )

    # def move_to(self, moves):
    #     """Move to a specific place in the tree."""
    #     n = self
    #     for m in moves:
    #         n = n.children[m]
    #     return n

    # def enum_leaves(self, position):
    #     res = []
    #     if self.is_var() or self.is_const():
    #         res.append([self.type, self.value, position])
    #     i = 0
    #     for n in self.children:
    #         p = position[:]
    #         p.append(i)
    #         res.extend(n.enum_leaves(p))
    #         i += 1
    #     return res


def eq_nodes_are_equal(a: List[Node], b: List[Node]) -> bool:
    if len(a) != len(b):
        return False
    return all(aa.eq(bb) for aa, bb in zip(a, b))


def INode(value: int) -> Node:
    assert type(value) is int
    return Node(INTEGER, value)


def VNode(value: str, variable_type: Optional[str] = None) -> Node:
    assert variable_type in Node.VARIABLE_TYPES
    return Node(VARIABLE, value, variable_type=variable_type)


def PNode(value: str) -> Node:
    assert value in CONSTANTS, value
    return Node(CONSTANT, value)


def UNode(value: str, x: Node) -> Node:
    assert value in U_OPS
    assert isinstance(x, Node)
    return Node(UNARY, value, [x])


def BNode(value: str, x: Node, y: Node) -> Node:
    assert value in B_OPS
    assert isinstance(x, Node)
    assert isinstance(y, Node)
    return Node(BINARY, value, [x, y])


def CNode(value: str, x: Node, y: Node) -> Node:
    assert value in C_OPS
    assert isinstance(x, Node)
    assert isinstance(y, Node)
    return Node(COMPARISON, value, [x, y])


def autocast(x):
    if isinstance(x, Node):
        return x
    elif type(x) is int:
        return INode(x)
    else:
        raise NotImplementedError(f"Unexpected type: {x}")


class NodeSet:
    def __init__(self, nodes: Optional[List[Node]] = None):
        self.nodes: List[Node] = []
        if nodes is not None:
            for node in nodes:
                self.add(node)

    def __len__(self):
        return len(self.nodes)

    def __contains__(self, node):
        assert isinstance(node, Node)
        return any(x.eq(node) for x in self.nodes)

    def __iter__(self):
        for node in self.nodes:
            yield node

    def __eq__(self, ns):
        assert isinstance(ns, NodeSet)
        return len(self) == len(ns) and all(x in ns for x in self)

    def __or__(self, ns):
        assert isinstance(ns, NodeSet)
        result = NodeSet()
        for node in self:
            result.add(node)
        for node in ns:
            result.add(node)
        return result

    def add(self, node):
        assert isinstance(node, Node)
        if not any(x.eq(node) for x in self.nodes):
            self.nodes.append(node)
        return self

    def extend(self, nodes: List[Node]):
        for node in nodes:
            self.add(node)
        return self


def Max(a: Node, b: Node):
    return BNode("max", a, b)


def Min(a: Node, b: Node):
    return BNode("min", a, b)


def GCD(a: Node, b: Node):
    return BNode("gcd", a, b)


def LCM(a: Node, b: Node):
    return BNode("lcm", a, b)


def exp(x: Node):
    return x.exp()


def ln(x: Node):
    return x.ln()


def sqrt(x: Node):
    return x ** 0.5


def sin(x: Node):
    return x.sin()


def cos(x: Node):
    return x.cos()


def tan(x: Node):
    return x.tan()


def asin(x: Node):
    return x.asin()


def acos(x: Node):
    return x.acos()


def atan(x: Node):
    return x.atan()


def sinh(x: Node):
    return x.sinh()


def cosh(x: Node):
    return x.cosh()


def tanh(x: Node):
    return x.tanh()


def asinh(x: Node):
    return x.asinh()


def acosh(x: Node):
    return x.acosh()


def atanh(x: Node):
    return x.atanh()


ZERO = INode(0)
ONE = INode(1)
TWO = INode(2)
NEG_ONE = INode(-1)

PI = PNode("PI")

A = VNode("A")
B = VNode("B")
C = VNode("C")
D = VNode("D")
E = VNode("E")
F = VNode("F")
G = VNode("G")
H = VNode("H")
I = VNode("I")
J = VNode("J")


a = VNode("a", variable_type="int")
b = VNode("b", variable_type="int")
c = VNode("c", variable_type="int")
d = VNode("d", variable_type="int")
e = VNode("e", variable_type="int")
f = VNode("f", variable_type="int")
minus_a = VNode("-a", variable_type="int")


RULE_VARS = [A, B, C, D, E, F, G, H, I, J, a, b, c, d, e, f, minus_a]


if __name__ == "__main__":

    # cd formal
    # python -m evariste.envs.eq.graph

    def test_operators():

        print("===== RUNNING OPERATORS TESTS ...")

        A = VNode("A")
        B = VNode("B")

        # comparison operators
        assert CNode("==", A, B).eq(A == B)
        assert CNode("!=", A, B).eq(A != B)
        assert CNode("<=", A, B).eq(A <= B)
        assert CNode("<", A, B).eq(A < B)
        assert (A >= B).eq(B <= A)
        assert (A > B).eq(B < A)

        # binary operators
        assert BNode("add", A, B).eq(A + B)
        assert BNode("mul", A, B).eq(A * B)

        # unary operators
        assert UNode("abs", A).eq(abs(A))
        assert UNode("neg", A).eq(-A)
        assert UNode("pow2", A).eq(A ** 2)
        assert UNode("sqrt", A).eq(A ** 0.5)
        assert UNode("inv", A).eq(A.inv())

        print("OK")

    def test_constructors():

        print("===== RUNNING CONSTRUCTORS TESTS ...")

        A = VNode("A")
        B = VNode("B")
        C = VNode("C")
        D = VNode("D")

        c0 = A + B
        c1 = BNode("add", A, B)
        c2 = Node(BINARY, "add", [A, B])

        e0 = abs((A + B) * C) + A ** 2
        e1 = Node(
            BINARY,
            "add",
            [
                Node(
                    UNARY,
                    "abs",
                    [Node(BINARY, "mul", [Node(BINARY, "add", [A, B]), C])],
                ),
                Node(UNARY, "pow2", [A]),
            ],
        )

        tests = [
            [c0, c0, True],
            [c1, c1, True],
            [c2, c2, True],
            [c0, c1, True],
            [c0, c2, True],
            [c1, c2, True],
            [c0, B + A, False],
            [c1, B + A, False],
            [c2, B + A, False],
            [e0, e1, True],
            [e0, abs((A + B) * C) + A ** 2, True],
            [e0, abs((A + B) * D) + A ** 2, False],
            [e0, abs((A * B) * C) + A ** 2, False],
            [e0, abs((A + B) * C) + A.inv(), False],
            [UNode("pow2", A + B), (A + B) ** 2, True],
            [UNode("sqrt", A + B), (A + B) ** 0.5, True],
            [UNode("inv", A + B), (A + B) ** -1, True],
            [UNode("inv", A + B), (A + B).inv(), True],
            [(A + B).exp(), UNode("exp", A + B), True],
            [(A + B).ln(), UNode("ln", A + B), True],
            [(A + B).pow2(), (A + B) ** 2, True],
            [(A + B).sqrt(), (A + B) ** 0.5, True],
            [(A + B).inv(), (A + B).inv(), True],
            [A + B + C, (A + B) + C, True],
            [A + B + C, A + (B + C), False],
            [A * (B.inv()), A * B ** -1, True],
            [A * (B.inv()), A / B, False],
            [A - B, A + (-B), False],
            [-A - B, -A + (-B), False],
            [A.ln() * INode(2).inv(), A.ln() / 2, False],
            [A.ln() * INode(2).inv(), A.ln() * INode(2) ** -1, True],
            [A.ln() / INode(2), A.ln() / 2, True],
            [A ** 3, A ** 2 * A, True],
            [A ** 4, (A ** 2) ** 2, True],
        ]
        tests += [
            (-A, A.neg(), True),
            (-A + B, A.neg().add(B), True),
            (-A - B, A.neg().sub(B), True),
            (-A * B, A.neg().mul(B), True),
            (-A / B, A.neg().div(B), True),
            (-A, UNode("neg", A), True),
            (-A + B, BNode("add", UNode("neg", A), B), True),
            (-A - B, BNode("sub", UNode("neg", A), B), True),
            (-A * B, BNode("mul", UNode("neg", A), B), True),
            (-A / B, BNode("div", UNode("neg", A), B), True),
        ]
        for eq0, eq1, y in tests:
            if eq0.eq(eq1) != y or eq1.eq(eq0) != y:
                raise Exception(f"Expected {y}, found {not y} ({eq0} == {eq1})")

        print("OK")

    def test_autocast():

        print("===== RUNNING AUTOCAST TESTS ...")

        A = VNode("A")
        ZERO = INode(0)

        tests = [
            [A + ZERO, A + 0],
            [A * ZERO, A * 0],
            [ZERO + A, 0 + A],
            [ZERO * A, 0 * A],
            [A < ZERO, A < 0],
            [A > ZERO, A > 0],
            [ZERO < A, 0 < A],
            [ZERO > A, 0 > A],
            [A <= ZERO, A <= 0],
            [A >= ZERO, A >= 0],
            [ZERO <= A, 0 <= A],
            [ZERO >= A, 0 >= A],
            [A / 1, A / INode(1)],
            [A / 2, A / INode(2)],
            [A - 2, A - INode(2)],
            [2 - A, INode(2) - A],
        ]
        for eq0, eq1 in tests:
            if eq0.ne(eq1) or eq1.ne(eq0):
                raise Exception(f"Expected {eq0} and {eq1} are not equal!")

        print("OK")

    def test_get_ops():

        print("===== RUNNING GET_OPS TESTS ...")

        A = VNode("A")
        B = VNode("B")
        C = VNode("C")

        tests = [
            [A + B, UNARY, set()],
            [A + B, BINARY, {"add"}],
            [A ** 2, UNARY, {"pow2"}],
            [A ** 2, BINARY, set()],
            [A ** 2 + INode(3) / B + C * 5, UNARY, {"pow2"}],
            [A ** 2 + INode(3) * B.inv() + C * 5, UNARY, {"inv", "pow2"}],
            [A ** 2 + INode(3) / B + C * 5, BINARY, {"add", "mul", "div"}],
            [A ** 2 + INode(3) / B - C * 5, BINARY, {"add", "mul", "div", "sub"}],
            [A ** 2 + INode(3) / B + C * 5, INTEGER, {3, 5}],
        ]
        for eq, ntype, y in tests:
            y_: Set[str] = set()
            eq._get_ops(ntype, y_)
            if y_ != y:
                raise Exception(f"Expected {y} but found {y_} ({eq})!")

        print("OK")

    def test_prefix_infix():

        print("===== RUNNING PREFIX INFIX TESTS ...")

        x1 = VNode("x1")
        x2 = VNode("x2")
        x3 = VNode("x3")
        x4 = VNode("x4")

        tests = [
            [-x1 * x2, ["mul", "neg", "x1", "x2"], "((-(x1)) * x2)"],
            [-(x1 * x2), ["neg", "mul", "x1", "x2"], "(-(x1 * x2))"],
            [sin(x1), ["sin", "x1"], "sin(x1)"],
            [sin(-x1), ["sin", "neg", "x1"], "sin(-(x1))"],
            [
                x1 + x2 * x3 + x4,
                ["add", "add", "x1", "mul", "x2", "x3", "x4"],
                "((x1 + (x2 * x3)) + x4)",
            ],
            [
                x1 ** 2 > (x2.inv()) ** 0.5,
                ["<", "sqrt", "inv", "x2", "pow2", "x1"],
                "(sqrt(x2 ** -1) < (x1 ** 2))",
            ],
            [
                BNode("max", x1 + x2, INode(0)) <= UNode("exp", (x3 + x4).inv()),
                [
                    "<=",
                    "max",
                    "add",
                    "x1",
                    "x2",
                    "+",
                    "0",
                    "exp",
                    "inv",
                    "add",
                    "x3",
                    "x4",
                ],
                "((max (x1 + x2) 0) <= exp((x3 + x4) ** -1))",
            ],
            [
                BNode("max", (x1 + x2).inv(), INode(2)),
                ["max", "inv", "add", "x1", "x2", "+", "2"],
                "(max ((x1 + x2) ** -1) 2)",
            ],
            [
                BNode("min", (x1 + x2).inv(), INode(2)),
                ["min", "inv", "add", "x1", "x2", "+", "2"],
                "(min ((x1 + x2) ** -1) 2)",
            ],
            [
                ((x1 ** 0.5) + x2).inv(),
                ["inv", "add", "sqrt", "x1", "x2"],
                "((sqrt(x1) + x2) ** -1)",
            ],
            [
                x1 + -x1 + INode(-3),
                ["add", "add", "x1", "neg", "x1", "-", "3"],
                "((x1 + (-(x1))) + (-3))",
            ],
            [
                abs((x1 + x2) * x3) + x1 ** 2,
                ["add", "abs", "mul", "add", "x1", "x2", "x3", "pow2", "x1"],
                "(abs((x1 + x2) * x3) + (x1 ** 2))",
            ],
        ]
        for eq, prefix, infix in tests:
            prefix_ = eq.prefix_tokens()
            try:
                deserialized = Node.from_prefix_tokens(eq.prefix_tokens())
                if not eq.eq(deserialized):
                    raise RuntimeError(f"{eq.prefix_tokens()}")
            except AssertionError as e:
                raise e
            infix_ = eq.infix()
            if prefix_ != prefix:
                raise Exception(f"Expected {prefix}, found {prefix_}")
            if infix_ != infix:
                raise Exception(f"Expected {infix}, found {infix_}")
            assert len(prefix) == eq.prefix_len()

        print("OK")

    def test_in():

        print("===== RUNNING IN TESTS ...")

        A = VNode("A")
        B = VNode("B")
        C = VNode("C")
        D = VNode("D")

        x = VNode("x")
        y = VNode("y")
        z = VNode("z")
        t = VNode("t")

        tests = [
            [A + C == B + C, A, True],
            [A + C == B + C, D, False],
            [A + C == B + C, B + C, True],
            [A + C == B + C, A + C, True],
            [A + C == B + C, A + B, False],
            [A + C == B + C, A + C == B + C, True],
            [A + C == B + C, A + C == B + D, False],
            [A * B > C + D ** 2, D ** 2, True],
            [A * B > C + D ** 2, C + D ** 2, True],
            [A * B > C + D ** 2, C + D, False],
            [A * B > C + D ** 2, A + B, False],
            [x + (y + INode(2)) ** 2 < (z + t) * (t + INode(1)), INode(2), True],
            [x + (y + INode(2)) ** 2 < (z + t) * (t + INode(1)), INode(3), False],
            [x + (y + INode(2)) ** 2 < (z + t) * (t + INode(1)), y + INode(2), True],
            [x + (y + INode(2)) ** 2 < (z + t) * (t + INode(1)), y + INode(3), False],
            [x + (y + INode(2)) ** 2 < (z + t) * (t + INode(1)), z + t, True],
            [x + (y + INode(2)) ** 2 < (z + t) * (t + INode(1)), z + y, False],
            [(z + t) * t > x + y ** 2, z + t, True],
            [(z + t) * t > x + y ** 2, t > x + y, False],
            [(z + t) * t > x + y ** 2, (z + t) * t, True],
        ]

        for query, key, y in tests:
            y_ = key in query
            if y_ != y:
                raise Exception(f"Expected {y}, found {y_}")

        print("OK")

    def test_replace():

        print("===== RUNNING REPLACE TESTS ...")

        A = VNode("A")
        B = VNode("B")
        C = VNode("C")
        D = VNode("D")

        x = VNode("x")
        y = VNode("y")
        z = VNode("z")
        t = VNode("t")

        tests = [
            [A + C == B + C, A, x, x + C == B + C],
            [A + C == B + C, x, A, A + C == B + C],
            [A + C == B + C, A + C, INode(2), INode(2) == B + C],
            [A * B > C + D ** 2, D ** 2, t, A * B > C + t],
            [(z + t) * t > x + y ** 2, (z + t) * t > x + y ** 2, B, B],
            [(z + t) * t > x + y ** 2, z + t, A, A * t > x + y ** 2],
            [(z + t) * t > x + (z + t) ** 2, z + t, A, A * t > x + A ** 2],
        ]

        for eq, src, tgt, y in tests:
            y_ = eq.replace(src, tgt)
            if y_.ne(y):
                raise Exception(f"Expected {y}, found {y_}")

        print("OK")

    def test_match():

        print("===== RUNNING MATCH TESTS ...")

        a = VNode("a", variable_type="int")
        b = VNode("b", variable_type="int")
        c = VNode("c", variable_type="int")

        A = VNode("A")
        B = VNode("B")
        C = VNode("C")
        D = VNode("D")

        x = VNode("x")
        y = VNode("y")
        z = VNode("z")
        t = VNode("t")

        tests = [
            [ONE, ONE, {}],
            [ONE + ONE, ONE, None],
            [ONE, ONE + ONE, None],
            [ONE + ONE, ONE + ONE, {}],
            [a + B, INode(7) + x, {"a": INode(7), "B": x}],
            [A + B, INode(7) + x, {"A": INode(7), "B": x}],
            [a + b, INode(7) + x, None],
            [
                A + C == B + C,
                INode(7) + INode(2) == INode(9) + INode(2),
                {"A": INode(7), "B": INode(9), "C": INode(2)},
            ],
            [
                A * B > C + D ** 2,
                x + (y + INode(2)) ** 2 < (z + t) * (t + INode(1)),
                {"A": (z + t), "B": (t + INode(1)), "C": x, "D": (y + INode(2))},
            ],
            [A * B > A + C ** 2, (z + t) * t > x + y ** 2, None],
            [
                A * B > A + C ** 2,
                (z + t) * t > (z + t) + y ** 2,
                {"A": (z + t), "B": t, "C": y},
            ],
            [A + C == B + C, INode(7) + INode(12) == INode(9) + INode(2), None],
            [INode(0), INode(0), {}],
            [INode(0), INode(1), None],
            [
                A + A + B,
                INode(2) + INode(2) + INode(12),
                {"A": INode(2), "B": INode(12)},
            ],
            [A + A + B, INode(2) + INode(10) + INode(12), None],
            [a + b, INode(7) + INode(12), {"a": INode(7), "b": INode(12)}],
            [(a + b) * a, (INode(7) + INode(12)) * INode(8), None],
            [
                (a + b) * a,
                (INode(7) + INode(12)) * INode(7),
                {"a": INode(7), "b": INode(12)},
            ],
            [
                (a + b) * c,
                (INode(7) + INode(12)) * INode(8),
                {"a": INode(7), "b": INode(12), "c": INode(8)},
            ],
            [abs(exp(A)), abs(exp(x + y)), {"A": x + y}],
        ]

        for rule, node, res in tests:
            res_ = rule.match(node)
            if (
                (res is None) != (res_ is None)
                or (res is not None and set(res.keys()) != set(res_.keys()))
                or (res is not None and any(v.ne(res_[k]) for k, v in res.items()))
            ):
                raise Exception(f"Expected {res}, found {res_}")
        print("OK")

    def test_get_vars():

        print("===== RUNNING GET VARS TESTS ...")

        # TODO: int vars tests

        A = VNode("A")
        B = VNode("B")
        C = VNode("C")
        D = VNode("D")

        tests = [
            [A + C == B + C, {"A", "B", "C"}],
            [A + A + A + A == B + A, {"A", "B"}],
            [A + (C + D.exp() * D) == C + A, {"A", "C", "D"}],
        ]

        for eq, found_vars in tests:
            _vars = eq.get_vars()
            if _vars != found_vars:
                raise Exception(f"Expected {found_vars}, found {_vars}")

        print("OK")

    def test_set_var():

        print("===== RUNNING SET VAR TESTS ...")

        a = VNode("a", variable_type="int")
        A = VNode("A")
        B = VNode("B")
        C = VNode("C")

        x = VNode("x")
        y = VNode("y")
        z = VNode("z")

        tests = [
            [A + C == B + C, {"B": x, "C": y + z}, A + (y + z) == x + (y + z)],
            [
                a + C == B + a,
                {"a": INode(3), "C": INode(2)},
                INode(3) + INode(2) == B + INode(3),
            ],
            [A.inv() + A ** 2, {"A": x + y + z}, (x + y + z).inv() + (x + y + z) ** 2],
            [A + B == B + A, {"C": x}, "EqNotFoundVariable"],
            [(A + B).exp() == A + A + A, {"C": x}, "EqNotFoundVariable"],
            [a, {"a": x}, "EqInvalidVarType"],
            [a, {"a": INode(3)}, INode(3)],
            [a + B == B + A, {"a": x}, "EqInvalidVarType"],
        ]

        for src, replacement, tgt in tests:
            try:
                for k, v in replacement.items():
                    src = src.set_var(k, v)
            except EqNotFoundVariable:
                assert tgt == "EqNotFoundVariable"
            except EqInvalidVarType:
                assert tgt == "EqInvalidVarType"
            else:
                if src.ne(tgt):
                    raise Exception(f"Expected {src}, found {tgt}")

        print("OK")

    def test_nodeset():

        print("===== RUNNING NodeSet TESTS ...")

        ns = NodeSet()
        x1 = VNode("x")
        x2 = VNode("x")
        y = VNode("y")
        assert x1 not in ns
        assert x2 not in ns
        ns.add(x1)
        assert len(ns) == 1
        assert x1 in ns
        assert x2 in ns
        ns = NodeSet([x1, x2, y])
        assert len(ns) == 2
        ns = NodeSet([x1 == y, x1 >= 0, y ** 2])
        assert (INode(0) <= VNode("x")) in ns
        assert (INode(0) <= VNode("y")) not in ns
        assert (y ** 2) in ns
        assert (x1 == y) in ns
        assert (x1 == y ** 2) not in ns
        lisnode = [0 <= x1, x1 == x2, x2 ** 2 >= y]
        ns.extend(lisnode)
        assert (0 <= x1) in ns
        assert (x2 ** 2 >= y) in ns
        assert (x1 == x2) in ns
        ns2 = NodeSet()
        list2 = [x1 == y, x1 >= 0, y ** 2, x1 == x2, x2 ** 2 >= y]
        ns2.extend(list2)
        assert ns2 == ns
        ns3 = NodeSet([node for node in ns2])
        assert ns3 == ns2

        print("OK")

    def test_eval():

        print("===== RUNNING EVAL TESTS ...")

        x = VNode("x0")
        y = VNode("x1")
        dico = {"x0": 0.28402043, "x1": 0.49593}
        for ops in U_OPS:
            node = UNode(ops, x)
            if ops != "acosh":
                assert node.evaluate(dico) is not None
            else:
                assert node.evaluate({"x0": 1.2344}) is not None
        for ops in B_OPS:
            node = BNode(ops, x, y)
            if ops not in [
                "min",
                "max",
                "**",
                "gcd",
                "lcm",
            ]:  # TODO: add "**" in numexpr even if the handling is weird.
                assert node.evaluate(dico) is not None

        tests = [[x + y, {"x0": 3, "x1": 2}, 5]]
        for eq, subst, y in tests:
            y_ = eq.evaluate(subst)
            if abs(y_ - y) > 1e-6:
                raise ValueError(f"Found {y_}, expected {y} for {eq} in {subst}!")

        print("OK")

    def test_count_nested_exp():

        print("===== RUNNING COUNT NESTED EXP TESTS ...")

        ONE = INode(1)
        x = VNode("x0")
        y = VNode("x1")

        tests = [
            (ONE, 0),
            (ONE + ONE, 0),
            (ONE + ONE + ONE, 0),
            (ONE + ONE + ONE.exp(), 1),
            ((ONE + ONE.exp()).exp(), 2),
            (x.exp().exp().exp(), 3),
            ((x.exp().exp() + y.exp()).exp(), 3),
            (((x.exp().exp() + y.exp()).exp() + 1).exp(), 4),
        ]
        for eq, count in tests:
            count_ = eq.count_nested_exp()
            if count_ != count:
                raise ValueError(f"Found {count_}, expected {count} for {eq}!")

        print("OK")

    def test_is_valid():

        print("===== RUNNING IS VALID TESTS ...")

        ZERO = INode(0)
        ONE = INode(1)
        PI = PNode("PI")
        x = VNode("x0")
        y = VNode("x1")

        valids = [
            ZERO,
            ONE,
            ONE.ln(),
            (PI / 2).cos(),
            (ONE * 10).exp(),
            ONE / (PI / 2).sin(),
            (PI / 2).sin() ** -1,
            (ONE / (PI / 2).sin()).exp(),
            (2 * (PI / 2)).tan(),
            ONE / (x - x),
            ONE / (x - y),
            (x - x) ** -1,
            (x - y) ** -1,
        ]
        invalids = [
            ONE.ln().ln(),
            ZERO * ZERO.ln(),
            (ONE * 10).exp().exp(),
            (PI / 2).tan(),
            (3 * PI / 2).tan(),
            (-3 * PI / 2).tan(),
            ONE / (PI / 2).cos(),
            ONE / PI.sin(),
            (2 * (PI / 4)).tan(),
            ONE / ZERO,
            ONE / (18 - 18 * ONE),
            ONE / ZERO.sin(),
            ZERO.sin() ** -1,
            ONE / ONE.ln(),
            (ONE / ONE.ln()).exp(),
            (ONE / ZERO).inv(),
            (ONE / ZERO).sin(),
            (ONE / ZERO).sin().inv(),
            (ONE / ZERO).inv().sin(),
            (PI / 2).cos() * (PI / 2).tan(),
            (x > 1).exp(),
            (ONE < 2).cos(),
            (x.sin() < 2) + (y.cos() < 3),
            (x > 1) > y,
            (x > 1) > y.cos(),
            (x > 1).sin() > y.cos(),
        ]

        for eq in valids:
            if not eq.is_valid():
                raise ValueError(f"Expression {eq} should be valid!")

        for eq in invalids:
            if eq.is_valid():
                raise ValueError(f"Expression {eq} should be invalid!")

        print("OK")

    def test_fractions():

        print("===== RUNNING FRACTIONS TESTS ...")

        ZERO = INode(0)
        ONE = INode(1)
        TWO = INode(2)
        PI = PNode("PI")
        x = VNode("x0")
        y = VNode("x1")

        rational: List[Tuple[Node, Fraction]] = [
            (ZERO, Fraction(0)),
            (ONE, Fraction(1)),
            (TWO, Fraction(2)),
            (INode(-2), Fraction(-2)),
            (INode(-2), -Fraction(2)),
            (-INode(2), Fraction(-2)),
            (-INode(2), -Fraction(2)),
            (INode(2) / INode(3), Fraction(2, 3)),
            (INode(2) - INode(3), -Fraction(1)),
            (INode(1) / INode(2) - INode(1) / INode(3), Fraction(1, 6)),
            ((ONE * 5 - ZERO + 8).inv(), Fraction(1, 13)),
            (((ONE * 5 - ZERO + 8).inv() + 2).inv(), Fraction(13, 27)),
        ]
        irrational = [x, x + y, ZERO <= ONE, PI, ZERO * x, INode(1) / INode(0)]

        int_rational = [(-INode(6), Fraction(-6, 1))]

        for eq, frac in rational:
            assert eq.is_rational("real")
            assert eq.fraction("real") == frac, (eq.fraction("real"), frac)

        for eq in irrational:
            assert not eq.is_rational("real")

        for eq, frac in int_rational:
            assert eq.fraction("int") is not None, (eq, frac)
            eq.fraction("int") == frac

        print("OK")

    def test_infix_lean():

        print("===== RUNNING INFIX LEAN TESTS ...")

        x = VNode("x0")
        y = VNode("x1")

        eqs = [
            (
                INode(2) * BNode("max", x, y) * x ** INode(2),
                "(((2:ℝ) * (max x0 x1)) * (x0 ^ (2:ℝ)))",
            ),
            (
                INode(2) * Min(x, y) * x ** INode(2),
                "(((2:ℝ) * (min x0 x1)) * (x0 ^ (2:ℝ)))",
            ),
            (INode(2).cos(), "(real.cos (2:ℝ))"),
        ]
        for eq, infix in eqs:
            assert eq.infix_lean() == infix, (eq.infix_lean(), infix)

        print("OK")

    def test_switch_pow():

        print("==== RUNNING SWITCH POW ...")

        A = VNode("A")
        B = VNode("B")
        x = VNode("x")
        y = VNode("y")
        z = VNode("z")
        exprs = [
            [
                ((A ** -1 + x).cos() + (y ** 2).sin()).sin(),
                ((A ** INode(-1) + x).cos() + (y ** INode(2)).sin()).sin(),
            ],
            [
                (((x ** 2) * B).sin() ** -1 + (z ** 2).sin()).sin(),
                (
                    ((x ** INode(2)) * B).sin() ** INode(-1) + (z ** INode(2)).sin()
                ).sin(),
            ],
        ]
        for expr, expr_sub in exprs:
            assert isinstance(expr, Node)
            assert isinstance(expr_sub, Node)
            assert expr_sub.eq(expr.switch_pow()), (expr, expr_sub)

        print("OK")

    test_operators()
    test_constructors()
    test_autocast()
    test_get_ops()
    test_prefix_infix()
    test_in()
    test_replace()
    test_match()
    test_get_vars()
    test_set_var()
    test_nodeset()
    test_eval()
    test_count_nested_exp()
    test_is_valid()
    test_fractions()
    test_infix_lean()
    test_switch_pow()

    print("All tests clear")
