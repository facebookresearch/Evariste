# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Union, Tuple, List, Set, Dict
from collections import defaultdict
from pathlib import Path
import re
import traceback

from evariste import json as json
from evariste.clusters.utils import clusterify_path
from evariste.envs.eq.graph import Node, RULE_VARS
from evariste.envs.eq.rules import LeanRule, TRule, ARule, Rule
from evariste.envs.eq.utils import infix_to_node


LEAN_SUBSCRIPTS = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]
LEAN_HYP_NAMES = []
for i in range(100):
    s = f"h{i}"
    for k, v in enumerate(LEAN_SUBSCRIPTS):
        s = s.replace(str(k), v)
    LEAN_HYP_NAMES.append(s)


LEAN_THEOREM_TEMPLATE = """theorem {name}{var_names}{hyps} :
  {statement} :=
begin
  sorry
end
"""

IMPORT_HEADER = """import common
open_locale big_operators
open_locale euclidean_geometry
open_locale nat
open_locale topological_space
"""


_LEAN_THEOREMS_PATHS: List[str] = [
    "YOUR_PATH",
]

LEAN_THEOREMS_PATHS: List[Path] = [
    path
    for path in ([Path(clusterify_path(x)) for x in _LEAN_THEOREMS_PATHS])
    if path.is_file()
]


def rule_to_lean(rule: Rule, implicit_vars: bool) -> str:
    """
    Used to convert an Equation rule to a Lean rule. Only compatible with real numbers so far.
    """
    # variables
    if len(rule.all_vars) == 0:
        var_names = ""
    else:
        var_names = " ".join(sorted(rule.all_vars))
        if implicit_vars:
            var_names = f"\n  {{{var_names} : ℝ}}"
        else:
            var_names = f"\n  ({var_names} : ℝ)"

    # hypotheses
    hyps: List[str] = []
    assert len(hyps) + 1 < len(LEAN_HYP_NAMES)
    for i, hyp in enumerate(rule.hyps):
        hyps.append(f"\n  ({LEAN_HYP_NAMES[i]} : {hyp.infix_lean()})")

    # statement
    if isinstance(rule, TRule):
        statement = f"{rule.left.infix_lean()} = {rule.right.infix_lean()}"
    else:
        assert isinstance(rule, ARule)
        statement = rule.node.infix_lean()

    return LEAN_THEOREM_TEMPLATE.format(
        name=rule.name, var_names=var_names, hyps="".join(hyps), statement=statement,
    )


def export_to_lean(rules: List[Rule], implicit_vars: bool, path: str):

    # if os.path.isfile(path):
    #     with open(path, "r") as f:
    #         lines = [line.rstrip() for line in f]
    #         print(f"Reloaded {len(lines)} lines from {path}")
    #     th_in_file = {x.split()[1] for x in lines if x.startswith("theorem ")}
    # new_theorems: Set[str] = set()

    s: List[str] = []

    for rule in rules:
        s.append(rule_to_lean(rule, implicit_vars=implicit_vars))

    print("\n".join(s))

    with open(path, "w") as f:
        f.write(IMPORT_HEADER + "\n\n")
        f.write("\n".join(s))


def get_types(s: str) -> Optional[Tuple[str, str]]:
    match = re.match(r"^∀ {[a-zA-Z0-9₀-₉_α]+ : Type [a-zA-Z0-9₀-₉_]+}", s)
    if match is None:
        return None
    end = match.span()[1]
    return s[end:], s[:end]


def get_instances(s: str) -> Optional[Tuple[str, str]]:
    match = re.match(r"^\[_inst_[0-9]+ : [a-zA-Z0-9₀-₉α_ .]+]", s)
    if match is None:
        return None
    end = match.span()[1]
    return s[end:], s[:end]


def get_vars(s: str, implicit: bool) -> Optional[Tuple[str, List[str], str]]:
    if implicit:
        match = re.match(
            r"^(∀ )?{(?P<vars>[a-zA-Z0-9₀-₉_ ]+) : (?P<type>[a-zA-Z0-9₀-₉ℕℤℚℝℂα_]+)}", s
        )
    else:
        match = re.match(
            r"^(∀ )?\((?P<vars>[a-zA-Z0-9₀-₉_ ]+) : (?P<type>[a-zA-Z0-9₀-₉ℕℤℚℝℂα_]+)\)",
            s,
        )
    if match is None:
        return None
    var_names = match.groupdict()["vars"].split()
    vars_types = match.groupdict()["type"]
    assert len(set(var_names)) == len(var_names)
    return s[match.span()[1] :], var_names, vars_types


def get_expressions(s: str) -> Optional[Tuple[str, str]]:
    s = s.strip()
    match = re.match(r"^[≤≥<>=≠↔^a-zA-Z0-9₀-₉()⁻¹ ∣%.+*/-]+", s)
    if match is None:
        return None
    end = match.span()[1]
    return s[end:].strip(), s[:end].strip()


def parse_lean_statement(
    statement: str,
) -> Tuple[
    Set[str],
    List[Union[Tuple[str], Tuple[str, str], Tuple[str, str, bool, str]]],
    List[str],
    str,
]:

    all_vars: Set[str] = set()

    args: List[Union[Tuple[str], Tuple[str, str], Tuple[str, str, bool, str]]] = []
    expressions: List[str] = []

    s = statement

    while len(s) > 0:

        init_s = s
        # strip white spaces and commas
        if s[0] in [" ", ","]:
            s = s[1:]
            continue

        # get types stuff
        match_t = get_types(s)
        if match_t is not None:
            s, t = match_t
            args.append(("type", t))
            continue

        # get instances stuff
        match_i = get_instances(s)
        if match_i is not None:
            s, inst = match_i
            args.append(("instance", inst))
            continue

        # get implicit or explicit variables
        for implicit in [True, False]:
            match_v = get_vars(s, implicit=implicit)
            if match_v is not None:
                s, var_names, vars_type = match_v
                assert all_vars.isdisjoint(set(var_names))
                all_vars.update(var_names)
                for v in var_names:
                    args.append(("var", v, implicit, vars_type))
        if s != init_s:
            continue

        # get expressions
        match_e = get_expressions(s)
        if match_e is not None:
            s, expr = match_e
            expressions.append(expr)
            args.append(("hyp",))

        # get implications
        if len(s) > 0 and s[0] == "→":
            expressions.append(s[0])
            s = s[1:].strip()

        if init_s == s:
            break

    assert s == "", (s, statement)
    assert len(expressions) % 2 == 1, expressions
    assert args[-1][0] == "hyp"
    args.pop()

    hyps: List[str] = []
    for i, expr in enumerate(expressions):
        assert (expr == "→") == ((i % 2) == 1)
        if expr != "→":
            hyps.append(expr)
    concl = hyps.pop()

    return all_vars, args, hyps, concl


def extract_instances(instances: List[str]) -> List[str]:
    res: List[str] = []
    for x in instances:
        assert re.fullmatch(r"\[_inst_[0-9]+ : [a-zA-Z0-9₀-₉α_ .]+]", x)
        assert x.count(":") == 1
        inst0 = x[1:-1].split(" : ")[1]
        inst = inst0.split()
        assert len(inst) >= 2
        if len(inst) == 2:
            inst_ = inst[0]
        else:  # taking care of covariance contravariance etc.
            inst_ = inst0
        res.append(inst_)
    return res


RELOADED_LEAN_THEOREMS: Optional[Dict[str, str]] = None


SIMP_LEAN_THEOREMS: Optional[Set[str]] = None


def reload_lean_theorems() -> Tuple[Dict[str, str], Set[str]]:

    global RELOADED_LEAN_THEOREMS, SIMP_LEAN_THEOREMS

    if RELOADED_LEAN_THEOREMS is not None:
        assert SIMP_LEAN_THEOREMS is not None
        return RELOADED_LEAN_THEOREMS, SIMP_LEAN_THEOREMS

    # retrieve theorems path
    if len(LEAN_THEOREMS_PATHS) == 0:
        raise FileNotFoundError("Could not find original Lean statements.")
    reloaded: Dict[str, str] = {}
    simps = set()
    for path in LEAN_THEOREMS_PATHS:
        with open(path, "r") as f:
            for line in f:
                x = json.loads(line.rstrip())
                s_name = x["decl_name"]
                s_type = x["decl_type"]
                if x.get("is_simp", None) == "tt":
                    simps.add(x["decl_name"])
                assert s_name not in reloaded, s_name
                reloaded[s_name] = s_type

    RELOADED_LEAN_THEOREMS = reloaded
    SIMP_LEAN_THEOREMS = simps
    return reloaded, simps


RELOADED_RULES: Dict[str, Optional[Dict[str, Rule]]] = {
    "nat": None,
    "int": None,
    "real": None,
}


def parse_mathlib_statements(vtype: str, verbose: bool = False) -> Dict[str, Rule]:
    global RELOADED_RULES
    if RELOADED_RULES[vtype] is None:
        RELOADED_RULES[vtype] = _parse_mathlib_statements(vtype, verbose=verbose)
    return RELOADED_RULES[vtype]  # type: ignore


# fmt: off
TO_SKIP_WORDS_ALL: Dict[str, Set[str]] = {
    "nat": {"exp", "log", "real", "cos", "sin", "tan", "cosh", "sinh", "tanh", "arcsin", "arccos", "arctan", "arsinh", "arccos", "arctan", "lucas_lehmer.mersenne_int_ne_zero", "int.neg_one_pow_ne_zero","int.one_nonneg","int.one_pos"},
    "int": {"exp", "log", "real", "cos", "sin", "tan", "cosh", "sinh", "tanh", "arcsin", "arccos", "arctan", "arsinh", "arccos", "arctan", "sqrt", "nat."},
    "real": {"%", "∣","int.","char.","nat."},
}
TO_SKIP_INSTANCES_ALL : Dict[str, Set[str]] = {
    "nat": {"linear_ordered_comm_ring", "add_group", "div_inv_monoid", "comm_ring", "comm_group", "integral_domain", "linear_ordered_ring", "euclidean_domain", "ring", "division_ring", "ordered_ring", "group", "linear_ordered_add_comm_group", "is_R_or_C", "has_inv", "field", "linear_ordered_field", "linear_ordered_comm_group", "group_with_zero", "comm_group_with_zero", "ordered_add_comm_group", "sub_neg_monoid", "add_comm_group"},
    "int": {"div_inv_monoid", "comm_group", "division_ring", "group", "is_R_or_C", "has_inv", "field", "linear_ordered_field", "linear_ordered_comm_group", "group_with_zero", "comm_group_with_zero", "canonically_ordered_comm_semiring", "canonically_ordered_add_monoid", "ordered_comm_group","covariant_class α α has_mul.mul has_le.le", "contravariant_class α α has_mul.mul has_lt.lt", "linear_ordered_comm_monoid_with_zero"},
    "real": {"covariant_class α α has_mul.mul has_le.le", "canonically_ordered_add_monoid", "contravariant_class α α has_mul.mul has_lt.lt", "gcd_monoid", "canonically_ordered_comm_semiring", "linear_ordered_comm_monoid_with_zero"},
}
TO_SKIP_TYPES_ALL : Dict[str, Set[str]] = {
    "nat": {"ℝ", "ℤ"},
    "int": {"ℕ", "ℝ"},
    "real": {"ℕ", "ℤ"},
}
ALLOWED_WORDS_ALL : Dict[str, Set[str]] = {
    "nat": {"nat", "%", "∣", "gcd", "lcm"},
    "int": {"int", "%", "∣"},
    "real": {"real", "cos", "sin", "tan", "cosh", "sinh", "tanh", "arcsin", "arccos", "arctan", "arsinh", "arccos", "arctan"},
}
# fmt: on


def _parse_mathlib_statements(vtype: str, verbose: bool) -> Dict[str, Rule]:

    assert vtype in ["nat", "int", "real"]

    if verbose:
        print(
            f"Automatically building rules from mathlib statements (vtype={vtype}) ..."
        )
    reloaded_statements, simp_lemmas = reload_lean_theorems()
    if verbose:
        print(f"Reloaded {len(reloaded_statements)} Lean statements.")

    TO_SKIP_LABELS: Set[str] = {
        "norm_num.adc_zero",
        "pow_pos_iff",
        "int.sq_ne_two_mod_four",
    }

    # fmt: off
    TO_SKIP_WORDS = {"decidable", "complex", "deriv", "convex", "Ioo", "Icc", "Ico", "Ici", "function", "list", "bit", "card", "ordinal", "set", "measure", "polynomial", "length", "ennreal", "pgame", "ereal", "pgames", "eint", "punit", "pgame", "enat", "znum", "zmod", "order_dual", "onote", "fin.", "pell","bool.", "multiplicative.linear_ordered_comm_group_with_zero._proof_16"}
    TO_SKIP_INSTANCES = {"canonically_ordered_monoid", "canonically_linear_ordered_add_monoid", "canonically_linear_ordered_monoid", "cancel_comm_monoid", "boolean_ring", "left_cancel_monoid", "right_cancel_monoid", "contravariant_class α α has_mul.mul has_le.le", "invertible", "right_cancel_semigroup", "left_cancel_semigroup", "covariant_class α α has_mul.mul has_lt.lt", "category_theory.limits.has_zero_object", "category_theory.limits.has_zero_morphisms", "category_theory.category", "complete_lattice", "semilattice_inf_bot", "onote.NF", "encodable", "subsingleton", "G.impartial", "primcodable", "rack", "char_p", "linear_ordered_comm_group_with_zero"}
    TO_SKIP_TYPES = {"ℚ", "ℂ", "nnreal", "pnat", "ennreal", "num", "ereal", "pgames", "eint", "punit", "pgame", "game", "unit", "many_one_degree", "enat", "znum", "onote", "circle_deg1_lift", "gaussian_int"}
    ALLOWED_WORDS = {"pi", "abs", "log", "exp", "sqrt", "min", "max"}
    # fmt: on

    TO_SKIP_WORDS |= TO_SKIP_WORDS_ALL[vtype]
    TO_SKIP_INSTANCES |= TO_SKIP_INSTANCES_ALL[vtype]
    TO_SKIP_TYPES |= TO_SKIP_TYPES_ALL[vtype]
    ALLOWED_WORDS |= ALLOWED_WORDS_ALL[vtype]

    MANDATORY_VARS: Dict[str, List[str]] = {"mul_mul_div": ["a"]}

    def to_ignore(label: str, statement: str) -> bool:
        if label == "dist_bdd_within_interval":
            return True  # TODO: handle (variables with names size > 1)
        return any(s in label or s in statement for s in TO_SKIP_WORDS)

    auto_parsed_rules: Dict[str, Rule] = {}

    # stats
    total = 0
    skipped: Dict[str, List[str]] = defaultdict(list)

    # for each statement

    for label, statement in reloaded_statements.items():

        total += 1

        if label in TO_SKIP_LABELS:
            skipped["bad_labels"].append(label)
            continue

        # ignore because content clearly not compatible
        if to_ignore(label, statement):
            skipped["not_compatible"].append(label)
            continue

        try:
            # try to parse the statement
            all_vars, args, hyps, concl = parse_lean_statement(statement)

            skip = False

            # skip if invalid words
            for expr in [*hyps, concl]:
                skip = skip or any(
                    len(w) > 1 and w not in ALLOWED_WORDS
                    for w in re.findall(r"[a-zA-Z]+", expr)
                )
            if skip:
                skipped["not_allowed_words"].append(label)
                continue

            # check instances
            instances = [arg[1] for arg in args if arg[0] == "instance"]  # type: ignore
            for inst in extract_instances(instances):
                if "group" in inst and not (
                    "add" in inst or "semigroup" in inst or "group_with_zero" in inst
                ):
                    skip = True
                skip = skip or (inst in TO_SKIP_INSTANCES)
            if skip:
                skipped["not_allowed_instances"].append(label)
                continue

            # check var types
            # TODO: add N / Z / Q
            skip = any(arg[0] == "var" and arg[3] in TO_SKIP_TYPES for arg in args)  # type: ignore
            if skip:
                skipped["not_allowed_var_types"].append(label)
                continue

        except AssertionError:
            skipped["parse_assertion_error"].append(label)
            continue

        def bad_pow(s):
            """
            Anything is authorized for nat and real.
            Only ** 2 is allowed for int.
            """
            s = s.replace("^", "**")
            s = s.replace(" ", "")
            return re.search(r"\*\*([013-9a-zA-Z(-]|2[0-9]+)", s) is not None

        def is_eq(s: str) -> bool:
            return re.search(f"[=≠<>≤≥∣]", s) is not None

        def too_many_eq(s: str):
            assert "→" not in s
            return any(x.count("=") >= 2 for x in s.split("↔"))

        # skip bad conditions
        if len(all_vars) > 10:
            skipped["too_many_vars"].append(label)
            continue
        if any(hyp[0] == "(" or hyp[-1] == ")" for hyp in hyps):
            skipped["parentheses_in_hyps"].append(label)
            continue
        if not all(is_eq(hyp) for hyp in hyps) or not is_eq(concl):
            skipped["not_all_equations"].append(label)
            continue
        if any(too_many_eq(hyp) for hyp in hyps) or too_many_eq(concl):
            skipped["too_many_eq"].append(label)
            continue
        if concl.count("↔") > 1:
            skipped["multiple_iff_in_concl"].append(label)
            continue
        if vtype == "int" and (bad_pow(concl) or any(bad_pow(hyp) for hyp in hyps)):
            skipped["bad_pow"].append(label)
            continue

        def rename_vars(s: str, var_map: Dict[str, str]) -> str:
            for src, tgt in var_map.items():
                s = re.sub(rf"(?<![a-zA-Z0-9₀-₉_]){src}(?![a-zA-Z0-9₀-₉_])", tgt, s)
            return s

        # rename variables
        s_vars: List[str] = [x[1] for x in args if x[0] == "var"]  # type: ignore
        vmap: Dict[str, str] = {v: str(RULE_VARS[i]) for i, v in enumerate(s_vars)}
        args = [(x[0], vmap[x[1]], x[2], x[3]) if x[0] == "var" else x for x in args]  # type: ignore
        hyps = [rename_vars(hyp, vmap) for hyp in hyps]
        concl = rename_vars(concl, vmap)

        # hypotheses nodes
        hyp_nodes: List[Node] = [infix_to_node(hyp) for hyp in hyps]

        # mandatory variables (rare)
        mandatory_vars: Optional[List[str]] = None
        if label in MANDATORY_VARS:
            mandatory_vars = [vmap[x] for x in MANDATORY_VARS[label]]

        rule_type = "lean"

        try:
            # transformation rule
            if "↔" in concl or "=" in concl:
                left, right = concl.split(" ↔ " if "↔" in concl else " = ")
                if left.count("(") != left.count(")"):
                    assert left.count("(") - 1 == left.count(")") and left[0] == "("
                    left = left[1:]
                if right.count("(") != right.count(")"):
                    assert right.count("(") == right.count(")") - 1 and right[-1] == ")"
                    right = right[:-1]
                l_node = infix_to_node(left)
                r_node = infix_to_node(right)
                if l_node.eq(r_node):
                    skipped["left_equals_right"].append(label)
                    continue
                lean_rule = LeanRule(
                    label=label,
                    statement=statement,
                    left=l_node,
                    right=r_node,
                    hyps=hyp_nodes,
                    args=args,
                    mandatory_vars=mandatory_vars,
                    is_simp=label in simp_lemmas,
                )
                auto_parsed_rules[label] = TRule(
                    left=l_node,
                    right=r_node,
                    hyps=hyp_nodes,
                    rule_type=rule_type,
                    lean_rule=lean_rule,
                )

            # assertion rule
            else:
                node = infix_to_node(concl)
                lean_rule = LeanRule(
                    label=label,
                    statement=statement,
                    node=node,
                    hyps=hyp_nodes,
                    args=args,
                    mandatory_vars=mandatory_vars,
                    is_simp=label in simp_lemmas,
                )
                auto_parsed_rules[label] = ARule(
                    node, hyps=hyp_nodes, rule_type=rule_type, lean_rule=lean_rule
                )

        except AssertionError as e:
            skipped["new_assertion_error"].append(label)
            continue

    # stats
    n_skipped = sum(len(v) for v in skipped.values())
    assert n_skipped + len(auto_parsed_rules) == total
    if verbose:
        print("")
        print(
            f"Total   : {total}\n"
            f"Parsed  : {len(auto_parsed_rules)}\n"
            f"Skipped : {n_skipped}\n"
        )
        for k, v in skipped.items():
            print(f"    {k:<25} : {len(v)}")
        print("")
        for k, v in skipped.items():
            if len(v) < 100:
                print(f"    {k}: " + ", ".join(v))
        print("\n=====\n")

    return auto_parsed_rules


def auto_parse_comparison():
    """
    Compare manually parsed rules with automatically parsed ones.
    """
    from evariste.envs.eq.rules_lean_manual import RULES as MANUALLY_PARSED_RULES

    # automatically reload rules from mathlib
    auto_parsed_rules = parse_mathlib_statements("real")
    print(f"Automatically parsed: {len(auto_parsed_rules)} rules")

    # reload manually parsed statements
    manually_parsed_rules: Dict[str, Rule] = {
        rule.lean_rule.label: rule for rule in MANUALLY_PARSED_RULES
    }
    print(f"Manually parsed: {len(manually_parsed_rules)} rules")

    # check that everything we parsed before was properly parsed
    for label in manually_parsed_rules.keys():
        if label not in auto_parsed_rules:
            print(f"Failed to automatically parse: {label}")
            continue
        rule1 = auto_parsed_rules[label]
        rule2 = manually_parsed_rules[label]
        try:
            assert rule1.lean_rule.statement == rule2.lean_rule.statement
            assert (rule1.lean_rule.node is None) == (rule2.lean_rule.node is None)
            if rule1.lean_rule.node is None:
                l1 = rule1.lean_rule.left
                l2 = rule2.lean_rule.left
                r1 = rule1.lean_rule.right
                r2 = rule2.lean_rule.right
                if (
                    l1.ne(l2)
                    or r1.ne(r2)
                    or len(rule1.hyps) != len(rule2.hyps)
                    or any(h1.ne(h2) for h1, h2 in zip(rule1.hyps, rule2.hyps))
                ):
                    print("")
                    print(f"Mismatch on {label}")
                    print("Auto parse:")
                    for hyp in rule1.hyps:
                        print(f"\t{hyp}")
                    print(f"Left : {l1}")
                    print(f"Right: {r1}")
                    print("")
                    print("Manual parse:")
                    for hyp in rule2.hyps:
                        print(f"\t{hyp}")
                    print(f"Left : {l2}")
                    print(f"Right: {r2}")
                    print("")
                    raise RuntimeError(f"Mismatch on {label}")
            else:
                n1 = rule1.lean_rule.node
                n2 = rule2.lean_rule.node
                if (
                    n1.ne(n2)
                    or len(rule1.hyps) != len(rule2.hyps)
                    or any(h1.ne(h2) for h1, h2 in zip(rule1.hyps, rule2.hyps))
                ):
                    print("")
                    print(f"Mismatch on {label}")
                    print("Auto parse:")
                    for hyp in rule1.hyps:
                        print(f"\t{hyp}")
                    print(f"Statement: {n1}")
                    print("")
                    print("Manual parse:")
                    for hyp in rule2.hyps:
                        print(f"\t{hyp}")
                    print(f"Statement: {n2}")
                    print("")
                    raise RuntimeError(f"Mismatch on {label}")
        except AssertionError:
            print(f"AUTO PARSE != MANUAL PARSE FOR {label}")
            print(traceback.format_exc())
            continue

    print("OK")


if __name__ == "__main__":

    def run_parse_tests():
        print("===== RUNNING LEAN PARSING TESTS ...")

        def check_same(y_, y):
            all_vars, args, hyps, concl = y
            all_vars_, args_, hyps_, concl_ = y_
            if all_vars != all_vars_:
                raise RuntimeError(
                    f"Different variables. Found:\n{all_vars_}\nexpected:\n{all_vars}"
                )
            if args != args_:
                raise RuntimeError(
                    f"Different arguments. Found:\n{args_}\nexpected:\n{args}"
                )
            if hyps != hyps_:
                raise RuntimeError(
                    f"Different hypotheses. Found:\n{hyps_}\nexpected:\n{hyps}"
                )
            if concl != concl_:
                raise RuntimeError(
                    f"Different conclusions. Found:\n{concl_}\nexpected:\n{concl}"
                )

        TESTS = [
            (
                "∀ (x y : ℝ), real.tanh x = real.sinh x / real.cosh x",
                (
                    {"x", "y"},
                    [("var", "x", False, "ℝ"), ("var", "y", False, "ℝ")],
                    [],
                    "real.tanh x = real.sinh x / real.cosh x",
                ),
            ),
            (
                "∀ (x : ℝ), ∀ {y : ℝ}, real.tanh x = real.sinh x / real.cosh x",
                (
                    {"x", "y"},
                    [("var", "x", False, "ℝ"), ("var", "y", True, "ℝ")],
                    [],
                    "real.tanh x = real.sinh x / real.cosh x",
                ),
            ),
            (
                "∀ {R : Type x} [_inst_1 : distrib R] (a b c : R), a * (b + c) = a * b + a * c",
                (
                    {"a", "b", "c"},
                    [
                        ("type", "∀ {R : Type x}"),
                        ("instance", "[_inst_1 : distrib R]"),
                        ("var", "a", False, "R"),
                        ("var", "b", False, "R"),
                        ("var", "c", False, "R"),
                    ],
                    [],
                    "a * (b + c) = a * b + a * c",
                ),
            ),
            (
                "(a b c : R), a * (b + c) = a * b + a * c",
                (
                    {"a", "b", "c"},
                    [
                        ("var", "a", False, "R"),
                        ("var", "b", False, "R"),
                        ("var", "c", False, "R"),
                    ],
                    [],
                    "a * (b + c) = a * b + a * c",
                ),
            ),
            (
                "∀ {α : Type u} [_inst_1 : ordered_ring α] {a b c : α}, b ≤ a → c ≤ 0 → a * c ≤ b * c",
                (
                    {"a", "b", "c"},
                    [
                        ("type", "∀ {α : Type u}"),
                        ("instance", "[_inst_1 : ordered_ring α]"),
                        ("var", "a", True, "α"),
                        ("var", "b", True, "α"),
                        ("var", "c", True, "α"),
                        ("hyp",),
                        ("hyp",),
                    ],
                    ["b ≤ a", "c ≤ 0"],
                    "a * c ≤ b * c",
                ),
            ),
            (
                "∀ {α : Type u} [_inst_1 : linear_ordered_add_comm_group α] {a b c : α}, a ≤ b → b ≤ c → abs b ≤ max (abs a) (abs c)",
                (
                    {"a", "b", "c"},
                    [
                        ("type", "∀ {α : Type u}"),
                        ("instance", "[_inst_1 : linear_ordered_add_comm_group α]"),
                        ("var", "a", True, "α"),
                        ("var", "b", True, "α"),
                        ("var", "c", True, "α"),
                        ("hyp",),
                        ("hyp",),
                    ],
                    ["a ≤ b", "b ≤ c"],
                    "abs b ≤ max (abs a) (abs c)",
                ),
            ),
            (
                "∀ {M₀ : Type u_1} [_inst_1 : mul_zero_class M₀] {a : M₀}, a = 0 → ∀ (b : M₀), a * b = 0",
                (
                    {"a", "b"},
                    [
                        ("type", "∀ {M₀ : Type u_1}"),
                        ("instance", "[_inst_1 : mul_zero_class M₀]"),
                        ("var", "a", True, "M₀"),
                        ("hyp",),
                        ("var", "b", False, "M₀"),
                    ],
                    ["a = 0"],
                    "a * b = 0",
                ),
            ),
            (
                "real.cos (real.pi / 8) = real.sqrt (2 + real.sqrt 2) / 2",
                (
                    set(),
                    [],
                    [],
                    "real.cos (real.pi / 8) = real.sqrt (2 + real.sqrt 2) / 2",
                ),
            ),
            (
                "∀ {R : Type u_2} [_inst_1 : ordered_semiring R] {a : R} {n m : ℕ}, 1 < a → n < m → a ^ n < a ^ m",
                (
                    {"a", "n", "m"},
                    [
                        ("type", "∀ {R : Type u_2}"),
                        ("instance", "[_inst_1 : ordered_semiring R]"),
                        ("var", "a", True, "R"),
                        ("var", "n", True, "ℕ"),
                        ("var", "m", True, "ℕ"),
                        ("hyp",),
                        ("hyp",),
                    ],
                    ["1 < a", "n < m"],
                    "a ^ n < a ^ m",
                ),
            ),
            (
                "∀ {G₀ : Type u_2} [_inst_1 : group_with_zero G₀] {x : G₀}, x ≠ 0 → ∀ {y z : G₀}, y = z * x → y / x = z",
                (
                    {"x", "y", "z"},
                    [
                        ("type", "∀ {G₀ : Type u_2}"),
                        ("instance", "[_inst_1 : group_with_zero G₀]"),
                        ("var", "x", True, "G₀"),
                        ("hyp",),
                        ("var", "y", True, "G₀"),
                        ("var", "z", True, "G₀"),
                        ("hyp",),
                    ],
                    ["x ≠ 0", "y = z * x"],
                    "y / x = z",
                ),
            ),
        ]
        n_valid = 0
        for x, y in TESTS:
            # print("====")
            # print(x)
            y_ = parse_lean_statement(x)
            try:
                check_same(y_, y)
                n_valid += 1
            except RuntimeError as e:
                print(f"Different outputs: {str(e)}")
        print(f"OK for {n_valid}/{len(TESTS)} tests")

    run_parse_tests()
    parse_mathlib_statements("real", verbose=True)
    auto_parse_comparison()
    parse_mathlib_statements("nat", verbose=True)
    parse_mathlib_statements("int", verbose=True)
