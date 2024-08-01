# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Union, Tuple, List, Set, Dict
from functools import cached_property
from collections import defaultdict
from dataclasses import dataclass
import re
import math
import itertools

from evariste.envs.eq.graph import (
    EqMatchException,
    Node,
    INode,
    VNode,
    RULE_VARS,
    NodeSet,
)
from evariste.envs.eq.graph import ZERO, ONE, PI, exp, ln, sin, cos

NODE_VARS = NodeSet(RULE_VARS)


"""
Right now unable to apply a disjunction of cases
and hence to deal with stuff like
prove: A ** 2 + B ** 2 > 0 with assumption A != B.
"""


# def get_lean_rule_name(
#     lhs: Optional[Node] = None,
#     rhs: Optional[Node] = None,
#     node: Optional[Node] = None,
#     hyps: Optional[List[Node]] = None,
# ) -> str:
#     assert (lhs is None) == (rhs is None) != (node is None)
#     s = f"T||{lhs}|{rhs}" if node is None else f"A||{node}"
#     if hyps is not None:
#         s = s + "||" + "|".join(str(hyp) for hyp in hyps)
#     s = s.replace(" ", "_")
#     s = s.replace("|", "_")
#     s = s.replace("+", "_add_")
#     s = s.replace("-", "_sub_")
#     s = s.replace("*", "_mul_")
#     s = s.replace("/", "_div_")
#     s = s.replace("<=", "_le_")
#     s = s.replace("<", "_lt_")
#     s = s.replace("==", "_eq_")
#     s = s.replace("!=", "_neq_")
#     s = re.sub(r"[^a-zA-Z0-9_]", "", s)
#     s = re.sub(r"_+", "_", s)
#     h = hashlib.sha256(s.encode("utf-8")).hexdigest()
#     return f"{s}__{h[:8]}"


@dataclass
class LeanRule:
    label: str
    statement: str
    args: Optional[List] = None
    left: Optional[Node] = None
    right: Optional[Node] = None
    node: Optional[Node] = None
    hyps: Optional[List[Node]] = None
    mandatory_vars: Optional[List[str]] = None
    is_simp: bool = False

    @property
    def is_t_rule(self) -> bool:
        return self.left is not None

    @property
    def is_a_rule(self) -> bool:
        return self.node is not None

    def get_vars(self, implicit_vars: Optional[bool]) -> List[str]:
        assert self.args is not None
        res: List[str] = []
        for arg in self.args:
            if arg[0] == "var":
                _, vname, implicit, _vtype = arg
                if implicit_vars is None or implicit_vars == implicit:
                    res.append(vname)
        return res

    @cached_property
    def implicit_vars(self) -> List[str]:
        return self.get_vars(implicit_vars=True)

    @cached_property
    def explicit_vars(self) -> List[str]:
        return self.get_vars(implicit_vars=False)

    @cached_property
    def all_vars(self) -> List[str]:
        return self.get_vars(implicit_vars=None)

    def __post_init__(self):

        assert " " not in self.label, self.label

        # no hypotheses
        if self.hyps is None:
            self.hyps = []

        # all unary operators
        unary_ops: Set[str] = set()
        for hyp in self.hyps:
            unary_ops |= hyp.get_unary_ops()

        # either tranformation or assertion rule
        assert (self.left is None) == (self.right is None) != (self.node is None)
        if self.node is None:
            assert isinstance(self.left, Node)
            assert isinstance(self.right, Node)
            assert self.left.ne(self.right)
            unary_ops |= self.left.get_unary_ops()
            unary_ops |= self.right.get_unary_ops()
        else:
            assert isinstance(self.node, Node)
            unary_ops |= self.node.get_unary_ops()

        # check no pow2 / inv
        assert unary_ops.isdisjoint({"pow2", "inv"})

        # no more checks if no arguments
        if self.args is None:
            return

        # args check
        for arg in self.args:
            assert isinstance(arg, tuple)
            assert arg[0] in ["instance", "type", "var", "hyp"]
        assert sum(int(arg[0] == "hyp") for arg in self.args) == len(self.hyps)

        # variables check
        _all_vars: Set[str] = set()
        if self.is_t_rule:
            assert self.left is not None and self.right is not None
            _all_vars |= self.left.get_vars()
            _all_vars |= self.right.get_vars()
        else:
            assert self.node is not None
            _all_vars |= self.node.get_vars()
        assert self.hyps is not None
        for hyp in self.hyps:
            _all_vars |= hyp.get_vars()
        assert set(self.all_vars) == _all_vars, self.label

        # implicit variables check
        found_implicit: Set[str] = set()
        for var_names in re.findall(
            r"{(?P<vars>[a-zA-Z0-9₀-₉_ ]+) : [a-zA-Z0-9₀-₉ℕℤℚℝℂα_]+}", self.statement
        ):
            found_implicit |= set(var_names.split())
        assert len(found_implicit) == len(self.implicit_vars)
        _impl = set(self.implicit_vars)
        _expl = set(self.explicit_vars)
        assert len(_impl & _expl) == 0
        assert _impl | _expl == set(self.all_vars)

        # mandatory variables check (used for rare (buggy?) rules in Lean)
        if self.mandatory_vars is not None:
            assert len(self.mandatory_vars) == len(set(self.mandatory_vars))
            assert all(v in self.explicit_vars for v in self.mandatory_vars)

        # hypotheses check
        assert all(isinstance(hyp, Node) for hyp in self.hyps)
        assert len(self.hyps) == len({h.prefix() for h in self.hyps})

        # check proper rule type
        th = self.statement.split(" → ")[-1]
        assert ("=" in th or "↔" in th) == self.is_t_rule, self.label


class TRule:
    def __init__(
        self,
        left: Node,
        right: Node,
        hyps: Optional[List[Node]] = None,
        rule_type: Optional[str] = None,
        arith_name: Optional[str] = None,
        lean_rule: Optional[LeanRule] = None,
    ):
        hyps = [] if hyps is None else hyps
        assert isinstance(left, Node)
        assert isinstance(right, Node)
        assert all(isinstance(hyp, Node) and hyp.is_comp() for hyp in hyps), hyps
        assert (rule_type in {"lean", "imo"}) == (lean_rule is not None)
        if arith_name is not None:
            assert len(hyps) == 0
            assert rule_type == "arith"
            assert arith_name in ["addition", "multiplication", "fraction", "negation"]
            assert len(hyps) == 0
        self.left: Node = left
        self.right: Node = right
        self.hyps: List[Node] = hyps
        self.rule_type = rule_type
        self.arith_name = arith_name
        self.arithmetic = arith_name is not None
        self.l_vars: Set[str] = left.get_vars()
        self.r_vars: Set[str] = right.get_vars()
        self.h_vars: Set[str] = set().union(*[hyp.get_vars() for hyp in hyps])  # type: ignore
        self.all_vars: Set[str] = self.l_vars | self.r_vars | self.h_vars
        self.lean_rule = lean_rule
        assert left.prefix() != right.prefix(), self.name

    @classmethod
    def create_arith(cls, left: Node, right: Node, name: str):
        assert isinstance(left, Node) and isinstance(right, Node)
        return cls(left, right, rule_type="arith", arith_name=name)

    @cached_property
    def name(self) -> str:
        """
        Return a name for the current rule, based on its content.
        """
        if self.lean_rule is not None:
            return self.lean_rule.label
        s = f"T||{self.left.infix(old_format=True)}|{self.right.infix(old_format=True)}"
        if len(self.hyps) > 0:
            hyps = "|".join(hyp.infix(old_format=True) for hyp in self.hyps)
            s = f"{s}||{hyps}"
        name = s.replace(" ", "_")
        return name

    def __str__(self):
        s = f'Rule "{self.name}":'
        s += f"\n\t{self.left}"
        s += f"\n\t{self.right}"
        if self.arithmetic:
            s += f"\n\tarith: {self.arith_name}"
        return s

    def get_unary_ops(self) -> Set[str]:
        """Get unary operators."""
        return self.left.get_unary_ops() | self.right.get_unary_ops()

    def get_binary_ops(self) -> Set[str]:
        """Get binary operators."""
        return self.left.get_binary_ops() | self.right.get_binary_ops()

    def apply(self, eq: Node, fwd: bool, prefix_pos: int):
        """
        Apply a rule to an equation.
        Return a set of variables to replace.
        """
        assert eq.get_vars().isdisjoint(self.all_vars), (eq.get_vars(), self.all_vars)
        assert isinstance(eq, Node), f"eq is {eq} and not instance of Node"
        assert type(fwd) is bool, f"{type(fwd)} should be bool"
        assert type(prefix_pos) is int, f"{type(prefix_pos)} should be int"

        src, tgt = (self.left, self.right) if fwd else (self.right, self.left)
        found: List[Dict[str, Node]] = []

        # retrieve variables
        src_vars = self.l_vars if fwd else self.r_vars
        tgt_vars = self.r_vars if fwd else self.l_vars
        hyp_vars = self.h_vars

        def _apply(eq, cur_pos: int) -> Node:
            # we reached the node to replace -- apply the rule
            if cur_pos == prefix_pos:
                match = src.match(eq)
                if match is None:
                    raise EqMatchException(
                        f"No match for {src} at {eq} // "
                        f"{' '.join(eq.prefix_tokens())} // {prefix_pos}"
                    )
                found.append(match)
                new_eq = tgt.clone()
                for name, value in match.items():
                    if name in tgt_vars:
                        new_eq = new_eq.set_var(name, value)
                return new_eq

            child_prefix_start = cur_pos + 1
            children = []
            matched = False
            for i, c in enumerate(eq.children):
                child_len = c.prefix_len()
                if child_prefix_start <= prefix_pos < child_prefix_start + child_len:
                    assert not matched, "several match shouldn't be possible"
                    matched = True
                    children.append(_apply(c, child_prefix_start))
                else:
                    children.append(c)
                child_prefix_start += child_len
            if not matched:
                raise EqMatchException(f"Couldn't find prefix_pos {prefix_pos}")
            return Node(eq.type, eq.value, children=children)

        new_eq = _apply(eq, cur_pos=0)
        if len(found) != 1:
            raise EqMatchException(f"Found {len(found)} != 1 match")
        match = found[0]

        # hypotheses
        hyps = []
        for h in self.hyps:
            for k, v in match.items():
                assert type(k) is str and isinstance(v, Node)
                h = h.replace(VNode(k), v)
            hyps.append(h)

        return {
            "eq": new_eq,
            "match": match,
            "to_fill": (tgt_vars | hyp_vars) - src_vars,
            "hyps": hyps,
        }

    def _eligible(
        self,
        pattern: Node,
        eq: Node,
        prefix_pos: int,
        res: List[Tuple[int, Node]],
        fast: bool = False,
    ) -> None:
        """
        """
        assert isinstance(eq, Node)

        # check whether the rule apply
        if pattern.match(eq, variables=NODE_VARS) is not None:
            res.append((prefix_pos, eq))
            if fast:
                return

        # search in children
        prefix_pos = prefix_pos + 1
        for c in eq.children:
            self._eligible(pattern, c, prefix_pos, res)
            prefix_pos += c.prefix_len()
            if fast and len(res) > 0:
                return

    def eligible(
        self,
        eq: Node,
        fwd: bool,
        override_pattern: Optional[Node] = None,
        fast: bool = False,
    ) -> List[Tuple[int, Node]]:
        default_pattern = self.left if fwd else self.right
        if override_pattern is None:
            pattern = default_pattern
        else:
            # the overriding pattern must be more restrictive
            # NOTE: actually not with negative INT matching for Lean
            # assert default_pattern.match(override_pattern) is not None
            pattern = override_pattern
        res: List[Tuple[int, Node]] = []
        self._eligible(pattern=pattern, eq=eq, prefix_pos=0, res=res, fast=fast)
        return res


class ARule:
    def __init__(
        self,
        node: Node,
        hyps: Optional[List[Node]] = None,
        rule_type: Optional[str] = None,
        lean_rule: Optional[LeanRule] = None,
    ):
        hyps = [] if hyps is None else hyps
        assert isinstance(node, Node) and node.is_comp()
        assert all(isinstance(hyp, Node) and hyp.is_comp() for hyp in hyps)
        assert (rule_type in {"lean", "imo"}) == (lean_rule is not None)
        self.node: Node = node
        self.hyps: List[Node] = hyps
        self.rule_type = rule_type
        self.n_vars: Set[str] = node.get_vars()
        self.h_vars: Set[str] = set().union(*[hyp.get_vars() for hyp in hyps])  # type: ignore
        self.all_vars: Set[str] = self.n_vars | self.h_vars
        self.lean_rule = lean_rule

    @cached_property
    def name(self) -> str:
        """
        Return a name for the current rule, based on its content.
        """
        if self.lean_rule is not None:
            return self.lean_rule.label
        s = f"A||{self.node.infix(old_format=True)}"
        if len(self.hyps) > 0:
            hyps = "|".join(hyp.infix(old_format=True) for hyp in self.hyps)
            s = f"{s}||{hyps}"
        name = s.replace(" ", "_")
        return name

    def __str__(self):
        s = f'Rule "{self.name}":'
        s += f"\n\t{self.node}"
        if self.hyps:
            s += f"\nHyps:"
            for h in self.hyps:
                s += f"\n\t{h}"
        return s

    def get_unary_ops(self):
        """Get unary operators."""
        return self.node.get_unary_ops()

    def get_binary_ops(self):
        """Get binary operators."""
        return self.node.get_binary_ops()

    def apply(self, eq: Node):

        assert eq.get_vars().isdisjoint(self.all_vars)

        match = self.node.match(eq)
        if match is None:
            raise EqMatchException(f"{self.node} does not match {eq}")

        # hypotheses
        hyps = []
        for h in self.hyps:
            for k, v in match.items():
                assert type(k) is str and isinstance(v, Node)
                h = h.replace(VNode(k), v)
            hyps.append(h)

        return {
            "match": match,
            "hyps": hyps,
            "to_fill": self.h_vars - self.n_vars,
        }


Rule = Union[TRule, ARule]


def eval_numeric(node: Node, subst: Dict[str, float]) -> Optional[bool]:
    """
    Evaluate numerically whether a statement is true.
    Returns:
        - True if we are confident that the statement is true
        - False if we are confident that the statement is False
        - None otherwise
    NOTE: this is not perfectly accurate and should only be used for checks / tests.
    """
    assert node.is_comp()
    assert len(node.children) == 2
    lhs, rhs = node.children
    v0 = lhs.evaluate(subst)
    v1 = rhs.evaluate(subst)
    if v0 is None or v1 is None:
        return None
    is_equal = abs(v0 - v1) < 1e-12
    is_diff = abs(v0 - v1) > 1e-6
    if not is_equal and not is_diff:
        return None
    if node.value == "∣":  # Not handled by eval_numeric
        return None
    if node.value == "==":
        return is_equal
    elif node.value == "!=":
        return is_diff
    elif node.value == "<":
        return v0 < v1 - 1e-6
    elif node.value == "<=":
        return v0 < v1 - 1e-6 or is_equal
    else:
        raise RuntimeError(f"Unknown operator: {node.value}")


def eval_assert_with_rule(
    node: Node, rules: List[ARule], vtype: str = "real"
) -> Union[None, bool, str, ARule]:
    """
    Evaluate an assertion given assertion rules.
    Returns
        __NUMERIC__ is the expression is a true fraction inequality
        or both sides are equal with a purely evaluation of the fraction
        ARule if the expression can be solved by a ARule without hypotheses
        False if the equation is false
        None if we cannot determine the correctness of the assertion.
    """
    assert node.is_comp()
    assert len(node.children) == 2

    # if the two children are equal up to a numerical evaluation
    c0 = node.children[0]
    c1 = node.children[1]
    if (
        c0._norm_numify(vtype).eq(c1)
        or c1._norm_numify(vtype).eq(c0)
        or c0._norm_numify(vtype).eq(c1._norm_numify(vtype))
    ):
        return False if node.value == "<" else "__NUMERIC__"

    # if the comparison operator is division, we don't know
    if node.value == "∣":
        return None

    # check if we match an assertion rule
    node_neg = node.negation()
    for rule in rules:
        assert isinstance(rule, ARule)
        if len(rule.hyps) == 0 and rule.node.match(node) is not None:
            return rule
        if len(rule.hyps) == 0 and rule.node.match(node_neg) is not None:
            return False

    # if both sides are fractions, evaluate the correctness
    c0 = node.children[0]
    c1 = node.children[1]
    if c0.is_rational(vtype) and c1.is_rational(vtype):
        assert c0.fraction(vtype) is not None
        assert c1.fraction(vtype) is not None
        if node.value == "==":
            return "__NUMERIC__" if c0.fraction(vtype) == c1.fraction(vtype) else False
        elif node.value == "!=":
            return "__NUMERIC__" if c0.fraction(vtype) != c1.fraction(vtype) else False
        elif node.value == "<=":
            return "__NUMERIC__" if c0.fraction(vtype) <= c1.fraction(vtype) else False
        else:
            assert node.value == "<"
            return "__NUMERIC__" if c0.fraction(vtype) < c1.fraction(vtype) else False

    # otherwise, we do not know
    return None


def eval_assert(node: Node, rules: List[ARule], vtype: str = "real") -> Optional[bool]:
    """
    Evaluate an assertion given assertion rules.
    Returns
        True if the equation is true
        False if the equation is false
        None if we cannot determine the correctness of the assertion.

    """
    res = eval_assert_with_rule(node, rules, vtype)
    if res == "__NUMERIC__" or isinstance(res, ARule):
        return True
    assert res is False or res is None
    return res


TEST_VALUES: List[float] = [
    -math.pi,
    -2.1,
    -math.pi / 2,
    -math.pi / 4,
    -1.3,
    -0.4,
    0.0,
    0.3924,
    0.6364,
    1.0,
    1.1313,
    math.pi / 4,
    math.pi / 2,
    1.81,
    2.12,
    math.pi,
]


def get_test_substs(var_names: Set[str]) -> List[Dict[str, float]]:
    names: List[str] = sorted(list(var_names))
    n_vars = len(names)
    substs = list(itertools.product(TEST_VALUES, repeat=n_vars))
    assert all(len(x) == n_vars for x in substs)
    return [{names[i]: v[i] for i in range(n_vars)} for v in substs]


def test_eligible():

    print("===== RUNNING ELIGIBLE EQUIV TESTS ...")

    A = VNode("A")
    B = VNode("B")
    C = VNode("C")
    x = VNode("x")
    y = VNode("y")

    rule = TRule(A + A + B, 2 * A + B)
    tests = [
        [
            (x + y) ** 2 + (x + y) ** 2 + (x + x + y),
            True,
            [
                (0, (((x + y) ** 2) + ((x + y) ** 2)) + ((x + x) + y)),
                (10, (x + x) + y),
            ],
        ],
        [(x + y) ** 2 + (x + y) ** 2 + (x + x + y), False, []],
        [(x + y) ** 2 + (2 * x + y) ** 2 + (x + y), True, []],
        [(x + y) ** 2 + (2 * x + y) ** 2 + (x + y), False, [(7, 2 * x + y)]],
    ]
    for eq, fwd, ref in tests:
        out = rule.eligible(eq, fwd)
        assert len(out) == len(ref)
        for (out_prefix, out_subtree), (ref_prefix, ref_subtree) in zip(out, ref):
            assert out_prefix == ref_prefix
            assert out_subtree.eq(ref_subtree)

    test = [
        [-(x ** 2 + y + -(-y)) + x + -((-y) ** 2), [A, -(-A)], True, 1],
        [(2 * x + y) - (x + y ** 2), [A - B, (A + C) - (B + C)], True, 0],
        [(2 * x + y) - (x + y), [A - B, (A + C) - (B + C)], True, 1],
    ]

    for eq, _rule, fwd, n_eligible in test:
        rule = TRule(*_rule)
        nodes = rule.eligible(eq, not fwd)
        assert len(nodes) == n_eligible

    print("OK")


def test_apply_rule():

    print("===== RUNNING APPLY RULE TESTS ...")

    ZERO = INode(0)
    a = VNode("a", variable_type="int")
    b = VNode("b", variable_type="int")
    c = VNode("c", variable_type="int")
    d = VNode("d", variable_type="int")
    A = VNode("A")
    B = VNode("B")
    C = VNode("C")
    x = VNode("x")
    x0 = VNode("x0")
    y = VNode("y")
    z = VNode("z")
    t = VNode("t")

    all_tests = []

    rule = TRule(A + B, B + A)
    tests = [
        [x + y == z, 1, y + x == z, {"A": x, "B": y}, []],
        [x + y == z + t, 1, y + x == z + t, {"A": x, "B": y}, []],
        [x + y == z + t, 4, x + y == t + z, {"A": z, "B": t}, []],
    ]
    all_tests.append((rule, tests))

    rule = TRule(exp(A + B), exp(A) * exp(B))
    tests = [
        [exp(x + y), 0, exp(x) * exp(y), {"A": x, "B": y}, []],
        [exp(x + t + y), 0, (x + t).exp() * exp(y), {"A": x + t, "B": y}, []],
        [exp(exp(x + y)), 1, exp(exp(x) * exp(y)), {"A": x, "B": y}, []],
        [
            abs((x.ln() + y).exp().ln()),
            2,
            abs((x.ln().exp() * exp(y)).ln()),
            {"A": x.ln(), "B": y},
            [],
        ],
    ]
    all_tests.append((rule, tests))

    rule = TRule(A, abs(A), [A >= 0])
    tests = [
        [x, 0, abs(x), {"A": x}, [x >= 0]],
        [x + y, 0, abs(x + y), {"A": x + y}, [x + y >= 0]],
    ]
    all_tests.append((rule, tests))

    rule = TRule(ln(A * B), ln(A) + ln(B), [A > 0, B > 0])
    tests = [
        [
            ln(x ** 2 * y),
            0,
            ln(x ** 2) + ln(y),
            {"A": x ** 2, "B": y},
            [x ** 2 > 0, y > 0],
        ],
    ]
    all_tests.append((rule, tests))

    for rule, tests in all_tests:
        for src, prefix_pos, tgt, match, hyps in tests:
            for fwd in [True, False]:
                if not fwd:
                    src, tgt = tgt, src
                res = rule.apply(src, fwd, prefix_pos)
                assert res["eq"].eq(tgt)
                assert res["to_fill"] == set()
                assert res["match"].keys() == match.keys()
                for k, v in match.items():
                    assert res["match"][k].eq(v)
                assert len(hyps) == len(res["hyps"])
                for h_hyp, h_ref in zip(hyps, res["hyps"]):
                    assert h_hyp.eq(h_ref)
                # check that we can revert rule
                assert rule.apply(tgt, not fwd, prefix_pos)["eq"].eq(src)

    tests = [
        [TRule(ZERO, A - A), ZERO, 0, {}, {"A"}],
        [TRule(A - B, (A + C) - (B + C)), x - y, 0, {"A": x, "B": y}, {"C"}],
        [
            TRule.create_arith(a + b, c, name="addition"),
            INode(3) + INode(4),
            0,
            {"a": INode(3), "b": INode(4)},
            {"c"},
        ],
        [
            TRule.create_arith(c, a + b, name="addition"),
            INode(10),
            0,
            {"c": INode(10)},
            {"a", "b"},
        ],
        [
            TRule((A * B) ** 2, A ** 2 * B ** 2),
            x + x + x + ((y + z) * INode(2)) ** 2,
            6,
            {"A": y + z, "B": INode(2)},
            set(),
        ],
        [
            TRule((A * B) ** 2, A ** 2 * B ** 2),
            x + x + x + ((y + z) * INode(2)) ** 2,
            5,
            None,
            None,
        ],
        [TRule(A * ZERO, ZERO), x * ZERO, 0, {"A": x}, set()],
        [TRule(A + B, B + A), -(x0 + x0), 1, {"A": x0, "B": x0}, set()],
        [
            TRule(a * (b.inv()), c * (d.inv())),
            ln(INode(29) * (INode(1).inv())),
            1,
            {"a": INode(29), "b": INode(1)},
            {"c", "d"},
        ],
    ]
    for rule, eq, prefix_pos, match, to_fill in tests:
        assert (match is None) == (to_fill is None)
        try:
            res = rule.apply(eq, True, prefix_pos)
        except EqMatchException:
            assert match is None, "should not have failed"
            continue
        assert match is not None, "should have failed"
        assert res["match"].keys() == match.keys()
        assert res["to_fill"] == to_fill
        for k, v in match.items():
            assert res["match"][k].eq(v)

    print("OK")


def test_duplicated_rules(rules: List[Rule], allow_sym: bool = False):

    print(f"===== RUNNING DUPLICATED RULES TESTS (allow_sym={allow_sym}) ...")

    counts: Dict[Tuple[str, ...], List[str]] = defaultdict(list)

    for rule in rules:
        # TRule
        if isinstance(rule, TRule):
            s_x = [rule.left.prefix(), rule.right.prefix()]
            if not allow_sym:
                s_x = sorted(s_x)
        # ARule
        else:
            assert isinstance(rule, ARule)
            s_x = [rule.node.prefix()]
        s_hyps = sorted([hyp.prefix() for hyp in rule.hyps])
        k = tuple(s_x + s_hyps)
        counts[k].append(rule.name)

    n_dupl = 0
    for k, v in counts.items():
        if len(v) > 1:
            print(f"Found {len(v)} rules for {k}:")
            for name in v:
                print(f"\t{name}")
            n_dupl += len(v)

    if all(len(v) == 1 for v in counts.values()):
        assert n_dupl == 0
        print(f"OK for {len(rules)} rules")
    else:
        assert n_dupl > 1
        print(f"{n_dupl} rules have at least one duplicate!")


def test_valid_transform_rules_e_numeric(
    rules_t_e: List[TRule], allow_none: bool = False
):

    print("===== RUNNING VALID TRANSFORM RULE E NUMERIC TESTS ...")

    n_tests = 0

    def _run_test(rule: TRule):

        nonlocal n_tests

        lhs = rule.left
        rhs = rule.right
        assert not lhs.is_comp()
        assert not rhs.is_comp()

        # skip arithmetic rules
        if rule.arithmetic:
            return

        # no variables
        if not lhs.has_vars() and not rhs.has_vars():
            lhs_value = lhs.evaluate()
            rhs_value = rhs.evaluate()
            if (
                lhs_value is None
                or rhs_value is None
                or abs(lhs_value - rhs_value) >= 1e-12
            ):
                raise Exception(
                    f"Numerical verification failed: lhs={lhs_value} rhs={rhs_value}"
                )
            return

        # numerical check
        c_one_none, c_two_none, c_true, c_false, c_n = 0, 0, 0, 0, 0
        for subst in get_test_substs(rule.all_vars):
            # check hypotheses
            if not all(eval_numeric(hyp, subst) is True for hyp in rule.hyps):
                continue
            c_n += 1
            n_tests += 1
            v0 = lhs.evaluate(subst)
            v1 = rhs.evaluate(subst)
            if v0 is None and v1 is None:
                c_two_none += 1
            elif v0 is None or v1 is None:
                c_one_none += 1
            elif abs(v0 - v1) <= 1e-12:
                c_true += 1
            else:
                c_false += 1
        if c_n == 0:
            raise ValueError(f"No valid test found for {rule}")
        if (
            c_true == 0
            or c_false > 0
            or (c_one_none + c_two_none > 0 and not allow_none)
        ):
            raise ValueError(
                f"Rule {rule} seems wrong: "
                f"one_none={c_one_none} two_none={c_two_none} "
                f"true={c_true} false={c_false}"
            )

    n_succ = 0
    failed: List[str] = []

    # for each rule
    for rule in rules_t_e:

        b_ops = rule.left.get_binary_ops() | rule.right.get_binary_ops()
        if "min" in b_ops or "max" in b_ops or "%" in b_ops:
            continue

        # run test
        try:
            _run_test(rule)
            n_succ += 1
        except (ValueError, Exception) as e:
            failed.append(rule.name)
            print(f"TEST FAILURE: {rule.name} -- {e}")

    print(f"OK for {n_succ}/{len(rules_t_e)} transformation rules E ({n_tests} tests)")
    if len(failed) > 0:
        print(f"{len(failed)} rules did not pass the tests:")
        for k in failed:
            print(f"\t{k}")


def test_valid_transform_rules_c_numeric(
    rules_t_c: List[TRule], allow_none: bool = False
):

    print("===== RUNNING VALID TRANSFORM RULE C NUMERIC TESTS ...")

    n_tests = 0

    def _run_test(rule: TRule):

        nonlocal n_tests

        lhs = rule.left
        rhs = rule.right
        assert lhs.is_comp()
        assert rhs.is_comp()
        assert not rule.arithmetic

        # skip if min/max operators
        assert "min" not in lhs.get_binary_ops()
        assert "min" not in rhs.get_binary_ops()
        assert "max" not in lhs.get_binary_ops()
        assert "max" not in rhs.get_binary_ops()

        # numerical check
        c_one_none, c_two_none, c_true, c_false, c_n = 0, 0, 0, 0, 0
        for subst in get_test_substs(rule.all_vars):
            # check hypotheses
            if not all(eval_numeric(hyp, subst) is True for hyp in rule.hyps):
                continue
            c_n += 1
            n_tests += 1
            # eval
            v0 = eval_numeric(lhs, subst=subst)
            v1 = eval_numeric(rhs, subst=subst)
            if v0 is None and v1 is None:
                c_two_none += 1
            elif v0 is None or v1 is None:
                c_one_none += 1
            else:
                assert type(v0) is bool and type(v1) is bool
                if v0 == v1:
                    c_true += 1
                else:
                    c_false += 1
        if c_n == 0:
            raise ValueError(f"No valid test found for {rule}")
        if (
            c_true == 0
            or c_false > 0
            or (c_one_none + c_two_none > 0 and not allow_none)
        ):
            raise ValueError(
                f"Rule {rule} seems wrong: "
                f"one_none={c_one_none} two_none={c_two_none} "
                f"true={c_true} false={c_false}"
            )

    n_succ = 0
    failed: List[str] = []

    # for each rule
    for rule in rules_t_c:

        b_ops = rule.left.get_binary_ops() | rule.right.get_binary_ops()
        if "min" in b_ops or "max" in b_ops or "%" in b_ops:
            continue

        # run test
        try:
            _run_test(rule)
            n_succ += 1
        except (ValueError, Exception) as e:
            failed.append(rule.name)
            print(f"TEST FAILURE: {rule.name} -- {e}")

    print(f"OK for {n_succ}/{len(rules_t_c)} transformation rules C ({n_tests} tests)")
    if len(failed) > 0:
        print(f"{len(failed)} rules did not pass the tests:")
        for k in failed:
            print(f"\t{k}")


def test_valid_assert_rules_numeric(rules_a: List[ARule], allow_none: bool = False):

    print("===== RUNNING VALID ASSERT RULE NUMERIC TESTS ...")

    n_tests = 0

    def _run_test(rule: ARule):

        nonlocal n_tests

        eq = rule.node
        assert eq.is_comp()
        assert len(eq.children) == 2
        lhs, rhs = eq.children

        # skip if max operator
        bin_ops = eq.get_binary_ops()
        if "min" in bin_ops or "max" in bin_ops:
            return

        # numerical check
        c_one_none, c_two_none, c_true, c_false, c_n = 0, 0, 0, 0, 0
        for subst in get_test_substs(rule.all_vars):
            # check hypotheses
            if not all(eval_numeric(hyp, subst) is True for hyp in rule.hyps):
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
                    valid = v0 < v1 - 1e-6 or abs(v0 - v1) <= 1e-12
                elif eq.value == "==":
                    valid = abs(v0 - v1) <= 1e-12
                else:
                    assert eq.value == "!="
                    valid = abs(v0 - v1) > 1e-6
                if valid:
                    c_true += 1
                else:
                    c_false += 1
        if c_n == 0:
            raise ValueError(f"No valid test found for {rule}")
        if (
            c_true == 0
            or c_false > 0
            or (c_one_none + c_two_none > 0 and not allow_none)
        ):
            raise ValueError(
                f"Rule {rule} seems wrong: "
                f"one_none={c_one_none} two_none={c_two_none} "
                f"true={c_true} false={c_false}"
            )

    n_succ = 0
    failed: List[str] = []

    # for each rule
    for rule in rules_a:

        b_ops = rule.node.get_binary_ops()
        if "min" in b_ops or "max" in b_ops or "%" in b_ops:
            continue

        # run test
        try:
            _run_test(rule)
            n_succ += 1
        except (ValueError, Exception) as e:
            failed.append(rule.name)
            print(f"TEST FAILURE: {rule.name} -- {e}")

    print(f"OK for {len(rules_a)} assertion rules ({n_tests} tests)")
    if len(failed) > 0:
        print(f"{len(failed)} rules did not pass the tests:")
        for k in failed:
            print(f"\t{k}")


def test_eval_assert(rules_a: List[ARule]):

    print("===== RUNNING EVAL ASSERT TESTS ...")

    ONE_T = INode(1_000)
    ONE_B = INode(1_000_000_000)
    BIG = INode(10 ** 1000)
    A = VNode("A")
    B = VNode("B")

    tests = [
        (A == A, True),
        (A + B == A + B, True),
        (exp(A) == exp(A), True),
        (A == ln(exp(A)), None),
    ]

    tests += [
        (exp(A) > 0, True),
        (exp(A) < 0, False),
        (exp(A) >= 0, True),
        (exp(A) <= 0, False),
        (exp(A) == ZERO, False),
        (exp(A) != ZERO, True),
        (exp(A) > 1, None),
        (exp(A) < 1, None),
        (exp(A) >= 1, None),
        (exp(A) <= 1, None),
        (exp(A) == ONE, None),
        (exp(A) != ONE, None),
    ]

    tests += [
        (ZERO == ZERO, True),
        (ZERO != ZERO, False),
        (ZERO <= ZERO, True),
        (ZERO < ZERO, False),
        (ZERO == ONE, False),
        (ZERO != ONE, True),
        (ZERO <= ONE, True),
        (ZERO >= ONE, False),
        (ZERO < ONE, True),
        (ZERO > ONE, False),
        (cos(PI / 2) == ZERO, None),  # corresponding assertion rule not provided
        (cos(PI / 2) != ZERO, None),  # corresponding assertion rule not provided
        (sin(PI / 2) == ZERO, None),  # corresponding assertion rule not provided
        (sin(PI / 2) != ZERO, None),  # corresponding assertion rule not provided
        (INode(3) / 2 == INode(6) / 4, True),
        (INode(3) / 2 != INode(6) / 4, False),
        (INode(3) / 2 <= INode(6) / 4, True),
        (INode(3) / 2 >= INode(6) / 4, True),
        (INode(3) / 2 < INode(6) / 4, False),
        (INode(3) / 2 > INode(6) / 4, False),
        (ONE / 2 > ONE / 3, True),
        (ONE / 2 < ONE / 3, False),
        (INode(2) / INode(3) == -INode(-4) / 6, True),
        (INode(2) - INode(3) == -ONE, True),
        (INode(2) - INode(3) == ONE, False),
        (INode(2) - INode(3) <= ONE, True),
        (INode(3) - INode(2) >= -ONE, True),
        (ONE / INode(2) - ONE / INode(3) == ONE / 6, True),
        ((ONE * 5 - ZERO + 8).inv() == ONE / 13, True),
        (((ONE * 5 - ZERO + 8).inv() + 2) == INode(27) / 13, True),
        (((ONE * 5 - ZERO + 8).inv() + 2).inv() == INode(13) / 27, True),
        (ONE * 2 < ONE.exp(), None),
        (ONE * 13 / 17 < ONE * 77 / 100, True),
        (ONE * 17 / 13 > ONE * 100 / 77, True),
        (ONE / 4 - ONE / 3 == -ONE / 12, True),
    ]

    tests += [
        # 10^3
        (ONE / ONE_T == ONE / (ONE_T + 1), False),
        (ONE / ONE_T != ONE / (ONE_T + 1), True),
        (ONE / ONE_T <= ONE / (ONE_T + 1), False),
        (ONE / ONE_T >= ONE / (ONE_T + 1), True),
        (ONE / ONE_T < ONE / (ONE_T + 1), False),
        (ONE / ONE_T > ONE / (ONE_T + 1), True),
        # 10^9
        (ONE / ONE_B == ONE / (ONE_B + 1), False),
        (ONE / ONE_B != ONE / (ONE_B + 1), True),
        (ONE / ONE_B <= ONE / (ONE_B + 1), False),
        (ONE / ONE_B >= ONE / (ONE_B + 1), True),
        (ONE / ONE_B < ONE / (ONE_B + 1), False),
        (ONE / ONE_B > ONE / (ONE_B + 1), True),
        # 10^1000
        (ONE / BIG == ONE / (BIG + 1), False),
        (ONE / BIG != ONE / (BIG + 1), True),
        (ONE / BIG <= ONE / (BIG + 1), False),
        (ONE / BIG >= ONE / (BIG + 1), True),
        (ONE / BIG < ONE / (BIG + 1), False),
        (ONE / BIG > ONE / (BIG + 1), True),
        (BIG == BIG, True),
        (BIG != BIG, False),
        (BIG <= BIG, True),
        (BIG >= BIG, True),
        (BIG < BIG, False),
        (BIG > BIG, False),
        (BIG == BIG + 1, False),
        (BIG != BIG + 1, True),
        (BIG <= BIG + 1, True),
        (BIG >= BIG + 1, False),
        (BIG < BIG + 1, True),
        (BIG > BIG + 1, False),
        (ZERO == ONE / BIG, False),
        (ZERO != ONE / BIG, True),
        (ZERO <= ONE / BIG, True),
        (ZERO >= ONE / BIG, False),
        (ZERO < ONE / BIG, True),
        (ZERO > ONE / BIG, False),
        (BIG.exp() == ZERO, False),
        (BIG.exp() != ZERO, True),
        (BIG.exp() <= ZERO, False),
        (BIG.exp() >= ZERO, True),
        (BIG.exp() < ZERO, False),
        (BIG.exp() > ZERO, True),
    ]

    for expr, y in tests:
        y_ = eval_assert(expr, rules_a)
        if y != y_:
            print(
                f"TEST FAILURE: eval_assert error when evaluating {expr} -- "
                f"Expecting {y}, found {y_}"
            )

    print("OK")


def test_eval_assert_nat_int():

    print("===== RUNNING EVAL ASSERT TESTS NAT / INT ...")

    all_tests = {
        "int": [
            (INode(-7) / INode(3) == -INode(3), True),
            (INode(7) / INode(3) == INode(2), True),
            (INode(-9) / INode(-10) == INode(1), True),
            (INode(9) / INode(-10) == INode(0), True),
            (-INode(9) / (-INode(10)) == INode(1), True),
            (-INode(8) % (-INode(5)) == INode(2), True),
            (-INode(8) % (INode(5)) == INode(2), True),
            (INode(9) / INode(10) == INode(0), True),
            # INode(-2) ** INode(-3),
            (INode(4) / INode(0) == INode(0), None),
            (INode(4) / (INode(1) - INode(1)) == INode(0), None),
            # (INode(4) / (INode(1) / INode(2)) == INode(0), True),
            (INode(4) % INode(0) == INode(4), True),
            (INode(12) % INode(3) == INode(0), True),
        ],
        "nat": [
            (INode(5) - INode(10) + INode(5) == INode(5), True),
            (INode(5) / INode(10) == INode(0), True),
            (INode(5) ** INode(3) == INode(125), True),
        ],
    }

    rule = ARule(VNode("x0") == VNode("x0"))
    for vtype, tests in all_tests.items():
        for expr, y in tests:
            y_ = eval_assert(expr, [rule], vtype=vtype)
            # y_ is None for a division by 0 but it is not an issue in Lean.
            if y != y_:
                print(
                    f"TEST FAILURE: eval_assert error when evaluating ({vtype}) {expr} -- "
                    f"Expecting {y}, found {y_}"
                )

    print("OK")


if __name__ == "__main__":

    test_eligible()
    test_apply_rule()
    test_eval_assert_nat_int()
    print(f"All tests clear")

    # cd formal
    # python -m evariste.envs.eq.env    # python -m evariste.envs.eq.generation
    # python -m evariste.envs.eq.utils
    # python -m evariste.envs.eq.graph
    # python -m evariste.envs.eq.identities
    # python -m evariste.envs.eq.rules    # python -m evariste.envs.eq.rules_default
    # python -m evariste.envs.eq.rules_lean
    # python -m evariste.envs.eq.rules_lean
