# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, List, Set, Dict
from dataclasses import dataclass, field
from functools import cached_property
import traceback
import numpy as np

from params import Params
from evariste.envs.eq.graph import Node, INode, VNode, NodeType
from evariste.envs.eq.graph import U_OPS, B_OPS, C_OPS, VARIABLES, CONSTANTS
from evariste.envs.eq.graph import INTEGER, VARIABLE, CONSTANT, UNARY, BINARY
from evariste.envs.eq.rules import TRule, ARule
from evariste.envs.eq.utils import get_divs, gcd


@dataclass
class EquationsEnvArgs(Params):
    vtype: str = "real"
    max_int: int = field(
        default=10, metadata={"help": "Maximum absolute value of sampled int leaves"},
    )
    positive: bool = field(
        default=False,
        metadata={"help": "Sampled int leaves in expressions are always positive"},
    )
    pos_hyps: bool = field(
        default=False,
        metadata={"help": "Sampled int leaves in hypotheses are always positive"},
    )
    non_null: bool = field(
        default=True, metadata={"help": "Sampled int leaves are non zero"}
    )
    fill_max_ops: int = field(
        default=4,
        metadata={"help": "Maximum numbers of operators to fill expressions"},
    )
    n_vars: int = field(default=5, metadata={"help": "Number of variables"})
    n_consts: int = field(default=0, metadata={"help": "Number of constants"})
    unary_ops_str: str = field(
        default="neg,inv", metadata={"help": "Unary operators. Empty string for all."}
    )
    binary_ops_str: str = field(
        default="add,sub,mul,div",
        metadata={"help": "Binary operators. Empty string for all."},
    )
    comp_ops_str: str = field(
        default="==,!=,<=,<",
        metadata={"help": "Comparison operators. Empty string for all."},
    )
    leaf_probs_str: str = field(
        default="0.5,0.5,0",
        metadata={
            "help": "Leaf probabilities (integer, variable, parameter). Must sum to 1"
        },
    )

    @cached_property
    def unary_ops(self) -> List[str]:
        if self.unary_ops_str == "":
            ops = U_OPS
        else:
            ops = [x for x in self.unary_ops_str.split(",") if len(x) > 0]
            assert len(ops) == len(set(ops))
            assert all(op in U_OPS for op in ops)
        return ops

    @cached_property
    def binary_ops(self) -> List[str]:
        if self.binary_ops_str == "":
            ops = B_OPS
        else:
            ops = [x for x in self.binary_ops_str.split(",") if len(x) > 0]
            assert len(ops) == len(set(ops))
            assert all(op in B_OPS for op in ops)
        return ops

    @cached_property
    def comp_ops(self) -> List[str]:
        if self.comp_ops_str == "":
            ops = C_OPS
        else:
            ops = [x for x in self.comp_ops_str.split(",") if len(x) > 0]
            assert len(ops) == len(set(ops))
            assert all(op in C_OPS for op in ops)
        return ops

    @cached_property
    def leaf_probs(self) -> Tuple[float, float, float]:
        p = [float(x) for x in self.leaf_probs_str.split(",")]
        assert len(p) == 3
        assert all(x >= 0 for x in p) and sum(p) == 1
        assert (self.n_vars > 0) == (p[1] > 0)
        return p[0], p[1], p[2]

    def __post_init__(self):
        assert self.vtype in ["nat", "int", "real"]
        _ = self.unary_ops
        _ = self.binary_ops
        _ = self.leaf_probs


@dataclass
class EqGraphSamplerParams(Params):
    max_prefix_len: int = field(
        default=-1, metadata={"help": "Maximum prefix length (-1 to disable)"}
    )
    skip_non_equal_nodes: bool = field(
        default=True, metadata={"help": "Do not samples nodes of the form A != B"}
    )
    depth_weight: float = field(
        default=0,
        metadata={
            "help": "Depth weight (0 to disable, positive to encourage deeper proof trees)"
        },
    )
    size_weight: float = field(
        default=0,
        metadata={
            "help": "Size weight (0 to disable, positive to encourage larger proof trees)"
        },
    )
    sd_ratio_weight: float = field(
        default=0, metadata={"help": "Size/depth weight (0 to disable)"},
    )
    prefix_len_weight: float = field(
        default=0,
        metadata={
            "help": "Prefix length weight (0 to disable, negative for shorter equations)"
        },
    )
    rule_weight: float = field(
        default=0,
        metadata={
            "help": "Rule weight (0 to disable, positive to encourage rare rules)"
        },
    )

    def _check_and_mutate_args(self):
        assert self.max_prefix_len == -1 or self.max_prefix_len > 0


class EquationEnv:

    MAX_OPS = 100

    def __init__(
        self,
        vtype: str,
        max_int: int,
        positive: bool,
        pos_hyps: bool,
        non_null: bool,
        fill_max_ops: int,
        unary_ops: List[str],
        binary_ops: List[str],
        comp_ops: List[str],
        n_vars: int = 1,
        n_consts: int = 0,
        leaf_probs: Tuple[float, float, float] = (0.0, 1.0, 0.0),  # (int, var, param)
        seed=None,
    ):
        self.vtype = vtype
        self.max_int = max_int
        self.positive = positive
        self.pos_hyps = pos_hyps
        self.non_null = non_null
        self.fill_max_ops = fill_max_ops
        assert self.vtype in ["nat", "int", "real"]

        # variables
        assert n_vars <= len(VARIABLES)
        self.n_vars = n_vars
        self.vars = [VARIABLES[i] for i in range(n_vars)]

        # parameters
        self.consts = []
        self.n_consts = n_consts
        assert n_consts <= len(CONSTANTS)
        for i in range(n_consts):
            self.consts.append(CONSTANTS[i])

        # leaf probabilities
        assert (
            len(leaf_probs) == 3
            and all(p >= 0 for p in leaf_probs)
            and sum(leaf_probs) == 1
        ), f"{leaf_probs} {type(leaf_probs)}"
        self.leaf_probs = np.array(leaf_probs, dtype=np.float32)
        assert (self.leaf_probs[1] > 0) == (n_vars > 0)
        assert (self.leaf_probs[2] > 0) == (n_consts > 0)

        # operators / tree distributions
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.comp_ops = comp_ops
        self.unary = len(self.unary_ops) > 0
        self.distrib = self.generate_dist(EquationEnv.MAX_OPS)
        assert all(o in U_OPS for o in unary_ops)
        assert all(o in B_OPS for o in binary_ops)
        assert all(o in C_OPS for o in comp_ops)

        # operators compatibility
        allowed_ops = {
            "real": {
                "unary": ",".join(U_OPS),
                "binary": "add,sub,mul,div,**",
                "comp": "==,!=,<=,<",
            },
            "nat": {
                "unary": "sqrt",
                "binary": "add,sub,mul,div,%,min,max,**,gcd,lcm",
                "comp": "==,!=,<=,<,∣",
            },
            "int": {
                "unary": "neg,abs",
                "binary": "add,sub,mul,div,%,min,max",
                "comp": "==,!=,<=,<,∣",
            },
        }
        allowed_unary = set(allowed_ops[vtype]["unary"].split(","))
        allowed_binary = set(allowed_ops[vtype]["binary"].split(","))
        allowed_comp = set(allowed_ops[vtype]["comp"].split(","))
        assert set(unary_ops).issubset(allowed_unary), (unary_ops, allowed_unary)
        assert set(binary_ops).issubset(allowed_binary), (binary_ops, allowed_binary)
        assert set(comp_ops).issubset(allowed_comp), (comp_ops, allowed_comp)

        # environment random generator
        assert seed is None or type(seed) is int and seed >= 0
        self.rng = np.random.RandomState(seed)

    @staticmethod
    def build(args: EquationsEnvArgs, seed: Optional[int] = None):
        assert type(args) is EquationsEnvArgs
        return EquationEnv(
            vtype=args.vtype,
            max_int=args.max_int,
            positive=args.positive,
            pos_hyps=args.pos_hyps,
            non_null=args.non_null,
            fill_max_ops=args.fill_max_ops,
            unary_ops=args.unary_ops,
            binary_ops=args.binary_ops,
            comp_ops=args.comp_ops,
            n_vars=args.n_vars,
            n_consts=args.n_consts,
            leaf_probs=args.leaf_probs,
            seed=seed,
        )

    def set_rng(self, rng):
        old_rng, self.rng = self.rng, rng
        return old_rng

    def get_integer(self, positive: bool, non_null: bool):
        """
        Generate a random integer.
        """
        if positive and non_null:
            v = self.rng.randint(1, self.max_int + 1)
        elif positive:
            v = self.rng.randint(0, self.max_int + 1)
        elif non_null:
            s = self.rng.randint(1, 2 * self.max_int + 1)
            v = s if s <= self.max_int else (self.max_int - s)
        else:
            v = self.rng.randint(-self.max_int, self.max_int + 1)
        return int(v)

    def generate_leaf(
        self,
        positive: bool,
        non_null: bool,
        imposed_type: Optional[NodeType] = None,
        avoid: Optional[Set[str]] = None,
    ):
        """
        Generate a random leaf.
        """
        avoid = avoid or set()
        leaf_types: List[NodeType] = [INTEGER, VARIABLE, CONSTANT]

        if imposed_type is None:
            n_type = self.rng.choice(leaf_types, p=self.leaf_probs)  # type: ignore
        else:
            assert imposed_type in leaf_types
            n_type = imposed_type

        if n_type is INTEGER:
            value = self.get_integer(positive, non_null)
        elif n_type is VARIABLE:
            v = [vv for vv in self.vars if vv not in avoid]
            rand_idx = self.rng.randint(len(v))
            value = v[rand_idx]
        else:
            assert n_type is CONSTANT
            value = self.consts[self.rng.randint(len(self.consts))]
        return n_type, value

    def generate_operator(self, arity: int) -> Tuple[NodeType, str]:
        """
        Generate a random operator.
        """
        assert arity in [1, 2]
        value = self.rng.choice(self.unary_ops if arity == 1 else self.binary_ops)
        ntype = UNARY if arity == 1 else BINARY
        return ntype, value

    def generate_dist(self, max_ops: int) -> List[List[int]]:
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(n, 0) = 0
            D(0, e) = 1
            D(n, e) = D(n, e - 1) + p_1 * D(n - 1, e) + D(n - 1, e + 1)
        p1 =  if binary trees, 1 if unary binary
        """
        p1 = 1 if self.unary else 0
        # enumerate possible trees
        D: List[List[int]] = [[0] + [1 for _ in range(1, 2 * max_ops + 1)]]
        for n in range(1, 2 * max_ops + 1):  # number of operators
            s = [0]
            for e in range(1, 2 * max_ops - n + 1):  # number of empty nodes
                s.append(s[e - 1] + p1 * D[n - 1][e] + D[n - 1][e + 1])
            D.append(s)
        assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
        return D

    def sample_next_pos(self, n_empty: int, n_ops: int) -> Tuple[int, int]:
        """
        Sample the position of the next node (binary case).
        Sample a position in {0, ..., `n_empty` - 1}.
        """
        assert n_empty > 0
        assert n_ops > 0
        scores = []
        if self.unary:
            for i in range(n_empty):
                scores.append(self.distrib[n_ops - 1][n_empty - i])
        for i in range(n_empty):
            scores.append(self.distrib[n_ops - 1][n_empty - i + 1])
        probs = [p / self.distrib[n_ops][n_empty] for p in scores]
        p = np.array(probs, dtype=np.float64)
        e: int = self.rng.choice(len(p), p=p)
        arity = 1 if self.unary and e < n_empty else 2
        e %= n_empty
        return e, arity

    def _generate_expr(
        self,
        n_ops: int,
        positive: Optional[bool],
        non_null: Optional[bool],
        avoid: Set[str],
    ) -> Node:
        """
        Generate a random expression.
        """
        assert n_ops <= EquationEnv.MAX_OPS
        positive = self.positive if positive is None else positive
        non_null = self.non_null if non_null is None else non_null
        tree = Node()
        empty_nodes = [tree]
        next_en = 0
        n_empty = 1
        while n_ops > 0:
            next_pos, arity = self.sample_next_pos(n_empty, n_ops)
            for n in empty_nodes[next_en : next_en + next_pos]:
                n.type, n.value = self.generate_leaf(
                    positive=positive, non_null=non_null, avoid=avoid
                )
            next_en += next_pos
            (
                empty_nodes[next_en].type,
                empty_nodes[next_en].value,
            ) = self.generate_operator(arity)
            for _ in range(arity):
                e = Node()
                empty_nodes[next_en].push_child(e)
                empty_nodes.append(e)
            n_empty += arity - 1 - next_pos
            n_ops -= 1
            next_en += 1
        for n in empty_nodes[next_en:]:
            n.type, n.value = self.generate_leaf(
                positive=positive, non_null=non_null, imposed_type=None, avoid=avoid
            )
            assert not (isinstance(n.value, int) and n.value < 0) or not positive
        return tree

    def generate_expr(
        self,
        n_ops: int,
        positive: Optional[bool] = None,
        non_null: Optional[bool] = None,
        avoid: Optional[Set[str]] = None,
    ) -> Node:
        avoid = avoid or set()
        while True:
            expr = self._generate_expr(n_ops, positive, non_null, avoid)
            if expr.is_valid(self.vtype):
                assert not expr.is_comp(), expr
                return expr

    def fill_expr(self, var_names: Set[str]) -> Dict[str, Node]:
        assert type(var_names) is set
        assert all(type(name) is str for name in var_names)
        to_fill: Dict[str, Node] = {}
        for name in sorted(var_names):
            n_ops = self.rng.randint(self.fill_max_ops + 1)
            while True:
                eq = self.generate_expr(n_ops)
                # skip invalid constant expressions
                if eq.has_vars():
                    break
                if self.vtype == "real" and eq.evaluate() is not None:
                    break
                elif self.vtype != "real":
                    if not eq.can_evaluate() or eq.evaluate() is not None:
                        break
            to_fill[name] = eq
        return to_fill

    def fill_arith(
        self,
        var_names: Set[str],
        filled: Dict[str, Node],
        arith_name: str,
        fwd: bool,
        match: Dict[str, Node],
    ) -> Dict[str, Node]:
        assert arith_name in ["addition", "multiplication", "fraction", "negation"]

        # a + b -> c
        if arith_name == "addition" and fwd:
            assert match.keys() == {"a", "b"}
            if "c" in filled:
                assert (
                    filled["c"].int_value == match["a"].int_value + match["b"].int_value
                )
                return {}
            else:
                assert "c" in var_names
                return {"c": INode(match["a"].int_value + match["b"].int_value)}

        # c -> a + b
        if arith_name == "addition" and not fwd:
            assert match.keys() == {"c"}
            c = match["c"].int_value
            if "a" in filled and "b" in filled:
                assert c == filled["a"].int_value + filled["b"].int_value
                return {}
            elif "a" in filled and not "b" in filled:
                return {"b": INode(c - filled["a"].int_value)}
            elif "b" in filled and not "a" in filled:
                return {"a": INode(c - filled["b"].int_value)}
            else:
                b = int(self.rng.randint(-10, 11))
                return {"a": INode(c - b), "b": INode(b)}

        if arith_name == "multiplication" and fwd:
            assert match.keys() == {"a", "b"}
            if "c" in filled:
                assert (
                    filled["c"].int_value == match["a"].int_value * match["b"].int_value
                )
                return {}
            else:
                assert "c" in var_names
                return {"c": INode(match["a"].int_value * match["b"].int_value)}

        if arith_name == "multiplication" and not fwd:
            assert match.keys() == {"c"}
            c = match["c"].int_value

            if "a" in filled and "b" in filled:
                assert c == filled["a"].int_value * filled["b"].int_value
                return {}
            elif "a" in filled and not "b" in filled:
                assert c % filled["a"].int_value == 0
                return {"b": INode(c // filled["a"].int_value)}
            elif "b" in filled and not "c" in filled:
                assert c % filled["b"].int_value == 0
                return {"a": INode(c // filled["b"].int_value)}
            else:
                a = int(self.rng.choice(get_divs(c)))
                assert c % a == 0
                return {"a": INode(a), "b": INode(c // a)}

        # a / b -> c / d
        if arith_name == "fraction" and fwd:
            assert match.keys() == {"a", "b"}
            a = match["a"].int_value
            b = match["b"].int_value
            if "c" in filled and "d" in filled:
                assert a * filled["d"].int_value == b * filled["c"].int_value
                return {}
            elif "c" in filled and "d" not in filled:
                assert (filled["c"].int_value * b) % a == 0
                return {"c": INode(filled["c"].int_value * b // a)}
            elif "d" in filled and "c" not in filled:
                assert (filled["d"].int_value * a) % b == 0
                return {"c": INode(filled["d"].int_value * a // b)}
            else:
                x = int(self.rng.randint(2, 11))
                return {"c": INode(x * a), "d": INode(x * b)}

        if arith_name == "fraction" and not fwd:
            assert match.keys() == {"c", "d"}
            c = match["c"].int_value
            d = match["d"].int_value
            if "a" in filled and "b" in filled:
                assert d * filled["a"].int_value == c * filled["b"].int_value
                return {}
            elif "a" in filled and "b" not in filled:
                assert (filled["a"].int_value * d) % c == 0
                return {"b": INode(filled["a"].int_value * d // c)}
            elif "b" in filled and "a" not in filled:
                assert (filled["b"].int_value * c) % d == 0
                return {"a": INode(filled["b"].int_value * c // d)}
            else:
                x = gcd(c, d)
                return {"a": INode(c // x), "b": INode(d // x)}

        # -INode(a) -> INode(-a)
        if arith_name == "negation" and fwd:
            assert match.keys() == {"a"}
            if "-a" in filled:
                assert filled["-a"].int_value == -match["a"].int_value
                return {}
            return {"-a": INode(-match["a"].int_value)}

        # INode(-a) -> -INode(a)
        if arith_name == "negation" and not fwd:
            assert match.keys() == {"-a"}
            if "a" in filled:
                assert filled["a"].int_value == -match["-a"].int_value
                return {}
            return {"a": INode(-match["-a"].int_value)}

        raise RuntimeError("Did not find any arith_name")

    def apply_t_rule(
        self,
        eq: Node,
        rule: TRule,
        fwd: bool,
        prefix_pos: int,
        to_fill: Dict[str, Node],
    ):
        """
        Apply a transformation rule to an equation.
        """
        assert isinstance(eq, Node)
        assert isinstance(rule, TRule)
        res = rule.apply(eq, fwd, prefix_pos)
        # assert to_fill.keys() <= res["to_fill"], (list(to_fill.keys()), res["to_fill"])

        # generate missing expressions
        if rule.arithmetic is False:
            to_fill = {**to_fill, **self.fill_expr(res["to_fill"] - to_fill.keys())}

        # generate missing values
        if rule.arithmetic is True:
            assert rule.arith_name is not None
            to_fill = {
                **to_fill,
                **self.fill_arith(
                    var_names=res["to_fill"] - to_fill.keys(),
                    filled=to_fill,
                    arith_name=rule.arith_name,
                    fwd=fwd,
                    match=res["match"],
                ),
            }

        # apply filled variables
        assert to_fill is not None
        # assert to_fill.keys() == res["to_fill"], (to_fill, res["to_fill"], rule.name)
        eq = res["eq"].set_vars(to_fill, mandatory=False)
        hyps = [hyp.set_vars(to_fill, mandatory=False) for hyp in res["hyps"]]

        assert res["match"].keys() | to_fill.keys() == rule.all_vars
        return {
            "eq": eq,
            "match": res["match"],
            "to_fill": to_fill,
            "hyps": hyps,
        }

    def apply_a_rule(self, eq: Node, rule: ARule, to_fill: Optional[Dict[str, Node]]):
        """
        Apply an assertion rule to an equation.
        """
        assert isinstance(eq, Node)
        assert isinstance(rule, ARule)
        res = rule.apply(eq)

        # generate random expressions
        if to_fill is None:
            to_fill = self.fill_expr(res["to_fill"])

        # apply filled variables
        assert to_fill.keys() == res["to_fill"]
        hyps = [hyp.set_vars(to_fill, mandatory=False) for hyp in res["hyps"]]

        assert res["match"].keys() | to_fill.keys() == rule.all_vars
        return {
            "match": res["match"],
            "to_fill": to_fill,
            "hyps": hyps,
        }
