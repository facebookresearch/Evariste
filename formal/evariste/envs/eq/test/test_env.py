# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
from typing import Tuple, List, Dict
import traceback

from evariste.envs.eq.graph import Node, INode, VNode, U_OPS
from evariste.envs.eq.env import EquationEnv
from evariste.envs.eq.rules import TRule


def test_apply_rule():

    # cd formal
    # python -m evariste.envs.eq.env
    env = EquationEnv(
        vtype="real",
        max_int=100,
        positive=False,
        pos_hyps=False,
        non_null=False,
        fill_max_ops=3,
        n_vars=2,
        n_consts=0,
        unary_ops=U_OPS,
        binary_ops=["add", "mul", "sub", "div"],
        comp_ops=["==", "!=", "<=", "<"],
        seed=0,
    )

    print("===== RUNNING APPLY_RULE TESTS ...")

    ZERO = INode(0)
    A = VNode("A")
    B = VNode("B")
    C = VNode("C")
    x = VNode("x")
    y = VNode("y")
    z = VNode("z")
    a = VNode("a", variable_type="int")
    b = VNode("b", variable_type="int")
    c = VNode("c", variable_type="int")
    d = VNode("d", variable_type="int")
    minus_a = VNode("-a", variable_type="int")

    tests = [
        [
            TRule(A + B, B + A),
            x + y,
            True,
            0,
            {},
            {"eq": (y + x), "match": {"A": x, "B": y}, "to_fill": {}},
        ],
        [
            TRule(A + B, B + A),
            x + (y + z),
            True,
            2,
            {},
            {"eq": (x + (z + y)), "match": {"A": y, "B": z}, "to_fill": {}},
        ],
        [
            TRule(ZERO, A + (-A)),
            x + (ZERO + z),
            True,
            3,
            {"A": x + x},
            {
                "eq": (x + (((x + x) + (-(x + x))) + z)),
                "match": {},
                "to_fill": {"A": (x + x)},
            },
        ],
        [
            TRule((A * 2).ln(), 2 * A.ln(), hyps=[A > 0]),
            ((x + y) * 2).ln(),
            True,
            0,
            {},
            {
                "eq": 2 * (x + y).ln(),
                "match": {"A": x + y},
                "to_fill": {},
                "hyps": [x + y > 0],
            },
        ],
        [
            TRule.create_arith(a + b, c, "addition"),
            INode(3) + INode(2),
            True,
            0,
            {},
            {
                "eq": INode(5),
                "match": {"a": INode(3), "b": INode(2)},
                "to_fill": {"c": INode(5)},
            },
        ],
        [
            TRule.create_arith(a + b, c, "addition"),
            INode(10),
            False,
            0,
            {"a": INode(3), "b": INode(7)},
            {
                "eq": INode(3) + INode(7),
                "match": {"c": INode(10)},
                "to_fill": {"a": INode(3), "b": INode(7)},
            },
        ],
        [
            TRule.create_arith(a * b, c, "multiplication"),
            INode(3) * INode(2),
            True,
            0,
            {},
            {
                "eq": INode(6),
                "match": {"a": INode(3), "b": INode(2)},
                "to_fill": {"c": INode(6)},
            },
        ],
        [
            TRule.create_arith(a * b, c, "multiplication"),
            INode(128),
            False,
            0,
            {"a": INode(32), "b": INode(4)},
            {
                "eq": INode(32) * INode(4),
                "match": {"c": INode(128)},
                "to_fill": {"a": INode(32), "b": INode(4)},
            },
        ],
        [
            TRule.create_arith(a * (b.inv()), c * (d.inv()), "fraction"),
            INode(3) * INode(2).inv(),
            True,
            0,
            {"c": INode(30), "d": INode(20)},
            {
                "eq": INode(30) * INode(20).inv(),
                "match": {"a": INode(3), "b": INode(2)},
                "to_fill": {"c": INode(30), "d": INode(20)},
            },
        ],
        [
            TRule.create_arith(a * (b.inv()), c * (d.inv()), "fraction"),
            INode(30) * INode(20).inv(),
            False,
            0,
            {"a": INode(3), "b": INode(2)},
            {
                "eq": INode(3) * INode(2).inv(),
                "match": {"c": INode(30), "d": INode(20)},
                "to_fill": {"a": INode(3), "b": INode(2)},
            },
        ],
        [
            TRule.create_arith(-a, minus_a, "negation"),
            -INode(3),
            True,
            0,
            {},
            {"eq": INode(-3), "match": {"a": INode(3)}, "to_fill": {"-a": INode(-3)},},
        ],
        [
            TRule.create_arith(-a, minus_a, "negation"),
            INode(-10),
            False,
            0,
            {},
            {
                "eq": -INode(10),
                "match": {"-a": INode(-10)},
                "to_fill": {"a": INode(10)},
            },
        ],
        [
            TRule(ZERO, A - A, hyps=[B == ZERO]),
            x + (ZERO + z),
            True,
            3,
            {"A": y, "B": 2 * y},
            {"eq": x + (y - y + z), "match": {}, "to_fill": {"A": y, "B": 2 * y}},
        ],
        [
            TRule(B * A, ZERO, hyps=[A == ZERO]),
            x + (ZERO + z),
            False,
            3,
            {"A": y, "B": 2 * y},
            {
                "eq": x + ((2 * y) * y + z),
                "match": {},
                "to_fill": {"A": y, "B": 2 * y},
            },
        ],
    ]
    for rule, eq, fwd, prefix_pos, to_fill, ref in tests:
        out = env.apply_t_rule(eq, rule, fwd, prefix_pos, to_fill)
        assert out["eq"].eq(ref["eq"]), (out["eq"], ref["eq"])
        assert out["match"].keys() == ref["match"].keys()
        assert out["to_fill"].keys() == ref["to_fill"].keys()
        for k, v in out["match"].items():
            assert v.eq(ref["match"][k])
        for k, v in out["to_fill"].items():
            assert v.eq(ref["to_fill"][k])
        if "hyps" in ref:
            assert len(ref["hyps"]) == len(out["hyps"])
            for h_hyp, h_ref in zip(out["hyps"], ref["hyps"]):
                assert h_hyp.eq(h_ref)
        if rule.arithmetic:
            env.fill_arith(set(), out["to_fill"], rule.arith_name, fwd, out["match"])
            out2 = env.apply_t_rule(out["eq"], rule, not fwd, prefix_pos, out["match"])
            for k, v in out2["match"].items():
                assert v.eq(out["to_fill"][k])
            for k, v in out2["to_fill"].items():
                assert v.eq(out["match"][k])
            assert out2["eq"].eq(eq)

    for i in [4, 0, -10]:
        eq = INode(i)
        rule = TRule.create_arith(-a, minus_a, "negation")
        out = env.apply_t_rule(eq, rule, False, 0, to_fill={})["eq"]
        assert out.eq(-(INode(-i)))

    # mul_eq_zero_of_right
    t_rule = TRule(B * A, ZERO, hyps=[A == ZERO])
    TESTS: List[Dict[str, Node]] = [
        {},
        {"B": x},
        {"A": y},
        {"A": y, "B": x},
        {"A": INode(3)},
        {"A": INode(3), "B": INode(4)},
    ]
    for to_fill in TESTS:
        _ = env.apply_t_rule(
            ZERO, rule=t_rule, fwd=False, prefix_pos=0, to_fill=to_fill,
        )

    a, b, c = VNode("a"), VNode("b"), VNode("c")
    t_rule = TRule(a * b, c, rule_type="arith", arith_name="multiplication")
    TESTS_2: List[Tuple[Dict[str, Node], Dict[str, Node], bool]] = [
        ({"a": INode(2)}, {"b": INode(5)}, False),
        ({"b": INode(3)}, {"b": INode(2)}, True),
    ]
    for to_fill, filled, should_fail in TESTS_2:
        try:
            applied = env.apply_t_rule(
                INode(10), rule=t_rule, fwd=False, prefix_pos=0, to_fill=to_fill,
            )
            assert applied["to_fill"]["b"].eq(filled["b"])
            failed = False
        except AssertionError as e:
            failed = True
            traceback.print_exc()
        if failed != should_fail:
            raise RuntimeError(f"failed={failed} but should_fail={should_fail}")
    print("OK")
