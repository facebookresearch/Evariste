# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Union, List, Dict

from evariste.envs.eq.graph import Node
from evariste.envs.eq.rules import TRule, ARule, Rule
from evariste.envs.eq.rules_default import RULES as RULES_DEFAULT
from evariste.envs.eq.rules_lean_manual import LEAN_CUSTOM_RULES as CUSTOM_RULES

from evariste.envs.eq.rules_lean import (
    RULES_REAL as RULES_LEAN_REAL,
    RULES_IMO as RULES_LEAN_IMO,
    RULES_NAT as RULES_LEAN_NAT,
    RULES_INT as RULES_LEAN_INT,
)

# from evariste.envs.eq.rules_lean_manual import RULES as RULES_LEAN_MANUAL

lean_id_rule = CUSTOM_RULES[0]
assert isinstance(lean_id_rule.left, Node)
assert isinstance(lean_id_rule.right, Node)
LEAN__IDENTITY = TRule(
    left=lean_id_rule.left,
    right=lean_id_rule.right,
    hyps=lean_id_rule.hyps,
    rule_type="lean",
    lean_rule=lean_id_rule,
)

RULES_LEAN_REAL = RULES_LEAN_REAL + [LEAN__IDENTITY]
RULES_LEAN_INT = RULES_LEAN_INT + [LEAN__IDENTITY]
RULES_LEAN_NAT = RULES_LEAN_NAT + [LEAN__IDENTITY]


ALL_RULES: Dict[str, List[Rule]] = {
    "default": RULES_DEFAULT,
    "lean_real": RULES_LEAN_REAL,
    "lean_nat": RULES_LEAN_NAT,
    "lean_int": RULES_LEAN_INT,
    "imo": RULES_LEAN_IMO,
    # "lean_manual": RULES_LEAN_MANUAL,
}
ALL_T_RULES: Dict[str, List[TRule]] = {
    "default": [rule for rule in RULES_DEFAULT if isinstance(rule, TRule)],
    "lean_real": [rule for rule in RULES_LEAN_REAL if isinstance(rule, TRule)],
    "lean_nat": [rule for rule in RULES_LEAN_NAT if isinstance(rule, TRule)],
    "lean_int": [rule for rule in RULES_LEAN_INT if isinstance(rule, TRule)],
    "imo": [
        rule for rule in RULES_LEAN_IMO if isinstance(rule, TRule)
    ]  # is empty for now but added for sanity checks
    # "lean_manual": [rule for rule in RULES_LEAN_MANUAL if isinstance(rule, TRule)],
}
ALL_A_RULES: Dict[str, List[ARule]] = {
    "default": [rule for rule in RULES_DEFAULT if isinstance(rule, ARule)],
    "lean_real": [rule for rule in RULES_LEAN_REAL if isinstance(rule, ARule)],
    "lean_nat": [rule for rule in RULES_LEAN_NAT if isinstance(rule, ARule)],
    "lean_int": [rule for rule in RULES_LEAN_INT if isinstance(rule, ARule)],
    "imo": [rule for rule in RULES_LEAN_IMO if isinstance(rule, ARule)],
    # "lean_manual": [rule for rule in RULES_LEAN_MANUAL if isinstance(rule, ARule)],
}


# check no duplicated labels
for v in ALL_RULES.values():
    names = {rule.name for rule in v}
    assert len(names) == len(set(names))


NAME_TO_RULE: Dict[str, Union[ARule, TRule]] = {}
NAME_TO_TYPE: Dict[str, str] = {}
for rules in ALL_RULES.values():
    for rule in rules:
        if rule.name in NAME_TO_RULE:
            prev_rule = NAME_TO_RULE[rule.name]
            assert prev_rule.lean_rule is not None
            assert rule.lean_rule is not None
            assert prev_rule.lean_rule.statement == rule.lean_rule.statement, (
                rule.name,
                prev_rule.lean_rule.statement,
                rule.lean_rule.statement,
            )
            # TODO: also check left/right/node
        rule_type = "t" if isinstance(rule, TRule) else "a"
        assert rule.name not in NAME_TO_TYPE or NAME_TO_TYPE[rule.name] == rule_type
        NAME_TO_RULE[rule.name] = rule
        NAME_TO_TYPE[rule.name] = rule_type

# sanity checks
assert len(NAME_TO_RULE) == len(NAME_TO_TYPE)
for k, v in ALL_RULES.items():
    assert len(v) == len(ALL_T_RULES[k]) + len(ALL_A_RULES[k])


if __name__ == "__main__":
    print(f"Found {len(NAME_TO_RULE)} rules in total:")
    for k, v in ALL_RULES.items():
        print(
            f"{k:<8} {len(v):>4} rules: {len(ALL_T_RULES[k])} transform "
            f"+ {len(ALL_A_RULES[k])} asserts"
        )
    all_lean = {
        rule.name
        for rule in ALL_RULES["lean_real"]
        + ALL_RULES["lean_nat"]
        + ALL_RULES["lean_int"]
    }
    print(f"All Lean: {len(all_lean)}")
    i = 0
    for name in all_lean:
        if "_proof" in name:
            print(name)
            i += 1
    print(f"Weird rule counter: {i}")
    print("SIMP RULES")
    for z in ["nat", "int", "real"]:
        print(z)
        print("#############")
        simp_rules = [
            r
            for r in ALL_RULES[f"lean_{z}"]
            if (r.lean_rule is not None and r.lean_rule.is_simp)
        ]

        for rule in simp_rules:
            if isinstance(rule, ARule):
                print(rule)
