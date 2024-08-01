# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple, List, Dict
from collections import defaultdict

from evariste.envs.eq.rules import Rule, TRule, ARule
from evariste.envs.eq.rules_lean_manual import RULES as MANUAL_RULES
from evariste.envs.eq.lean_utils import reload_lean_theorems, parse_mathlib_statements


REPLACE_BY: Dict[str, str] = {
    "norm_num.one_succ": "one_add_one_eq_two",
    "pell.n_lt_a_pow": "nat.lt_pow_self",
    "add_add_neg_cancel'_right": "add_neg_cancel_comm_assoc",
    "add_eq_zero_iff_eq_neg": "eq_neg_iff_add_eq_zero",
    "add_eq_zero_iff_neg_eq": "neg_eq_iff_add_eq_zero",
    "add_halves": "add_halves'",
    "le_of_add_le_add_left": "add_le_add_iff_left",  # TODO Maybe to remove
    "add_left_cancel_iff": "add_right_inj",
    "add_right_cancel_iff": "add_left_inj",
    "lt_of_add_lt_add_left": "add_lt_add_iff_left",  # TODO Maybe to remove
    "canonically_ordered_semiring.mul_le_mul_left'": "mul_le_mul_left'",
    "canonically_ordered_semiring.one_le_pow_of_one_le": "one_le_pow_of_one_le",
    "one_le_pow_of_one_le'": "one_le_pow_of_one_le",  # TODO warning, not exactly equivalent, but ok for Z N Q R
    # "canonically_ordered_semiring.pow_le_one": "pow_le_one_of_le_one",  # TODO same
    "nat.pow_le_pow_of_le_left": "canonically_ordered_semiring.pow_le_pow_of_le_left",
    "canonically_ordered_semiring.pow_pos": "pow_pos",
    "canonically_ordered_semiring.zero_lt_one": "zero_lt_one",
    "zero_lt_one'": "zero_lt_one",  # Requires to keep nontrivial in the vocab.
    "eq.ge": "ge_of_eq",
    "self_le_add_left": "le_add_self",
    "self_le_add_right": "le_self_add",
    "le_zero_iff": "nonpos_iff_eq_zero",  # TODO not equivalent for ok for N Z R Q
    "nat.le_zero_iff": "nonpos_iff_eq_zero",
    "lt_add_of_lt_of_nonneg'": "lt_add_of_lt_of_nonneg",
    "lt_mul_of_lt_of_one_le'": "lt_mul_of_lt_of_one_le",
    "tactic.ring.mul_assoc_rev": "mul_assoc",
    "mul_mono_nonneg": "mul_le_mul_of_nonneg_right",
    "nat.mul_ne_zero": "mul_ne_zero",
    "nat.lt_iff_add_one_le": "nat.add_one_le_iff",
    "nat.mod_mul_right_div_self": "nat.div_mod_eq_mod_mul_div",
    "nat.pow_lt_pow_of_lt_right": "pow_lt_pow",
    "one_le_mul_right": "one_le_mul_of_one_le_of_one_le",
    "one_le_mul": "one_le_mul_of_one_le_of_one_le",
    "one_lt_mul_of_lt_of_le'": "one_lt_mul_of_lt_of_le",
    "zero_lt_iff": "pos_iff_ne_zero",  # TODO Not equivalent but ok for N and co
    "tactic.ring.pow_add_rev": "pow_add",
    "rat.zero_ne_one": "zero_ne_one",
    "two_ne_zero": "two_ne_zero'",
    "zero_le'": "zero_le",  # TODO Not equivalent but ok for N and co
    "zero_le_one'": "zero_le_one",  # TODO Not equivalent but ok for R,N,Z,Q
    "sub_nonneg_of_le": "sub_nonneg",
    "sub_pos_of_lt": "sub_pos",
    "tactic.ring.add_neg_eq_sub": "sub_eq_add_neg",
    "sub_add_eq_sub_sub": "sub_sub",
    "neg_pos_of_neg": "neg_pos",
    "nonneg_of_neg_nonpos": "neg_nonpos",
    "neg_nonneg": "neg_nonneg_of_nonpos",
    "neg_mul_eq_neg_mul_symm": "neg_mul_eq_neg_mul",
    "pos_of_neg_neg": "neg_lt_zero",
    "neg_eq_zero_sub": "zero_sub",
    "neg_one_mul": "neg_eq_neg_one_mul",
    "ordered_ring.mul_nonneg": "mul_nonneg",
    "mul_neg_eq_neg_mul_symm": "neg_mul_eq_mul_neg",
    "ordered_ring.mul_lt_mul_of_pos_right": "mul_lt_mul_of_pos_right",
    "ordered_ring.mul_lt_mul_of_pos_left": "mul_lt_mul_of_pos_left",
    "mul_mono_nonpos": "mul_le_mul_of_nonpos_right",
    "ordered_ring.mul_le_mul_of_nonneg_right": "mul_le_mul_of_nonneg_right",
    "ordered_ring.mul_le_mul_of_nonneg_left": "mul_le_mul_of_nonneg_left",
    "lt_sub_left_of_add_lt": "lt_sub_iff_add_lt'",
    "lt_sub_right_of_add_lt": "lt_sub_iff_add_lt",
    "lt_of_sub_neg": "sub_lt_zero",
    "lt_neg_add_of_add_lt": "lt_neg_add_iff_add_lt",
    "lt_add_of_sub_right_lt": "sub_lt_iff_lt_add",
    "lt_add_of_sub_left_lt": "sub_lt_iff_lt_add'",
    "lt_add_of_neg_add_lt_right": "neg_add_lt_iff_lt_add_right",
    "lt_add_of_neg_add_lt": "neg_add_lt_iff_lt_add",
    "int.zero_mod": "euclidean_domain.zero_mod",
    "int.zero_div": "euclidean_domain.zero_div",
    "int.mul_div_cancel_left": "euclidean_domain.mul_div_cancel_left",
    "int.mul_div_cancel": "euclidean_domain.mul_div_cancel",
    "int.mul_div_assoc": "euclidean_domain.mul_div_assoc",
    "int.mod_zero": "euclidean_domain.mod_zero",
    "int.mod_self": "euclidean_domain.mod_self",
    "int.mod_one": "euclidean_domain.mod_one",
    "int.mod_def": "euclidean_domain.mod_eq_sub_mul_div",
    "int.mod_add_div'": "euclidean_domain.mod_add_div'",
    "int.mod_add_div": "euclidean_domain.mod_add_div",
    "int.eq_div_of_mul_eq_right": "euclidean_domain.eq_div_of_mul_eq_right",
    "int.div_zero": "euclidean_domain.div_zero",
    "int.div_self": "euclidean_domain.div_self",
    "int.div_add_mod'": "euclidean_domain.div_add_mod'",
    "int.div_add_mod": "euclidean_domain.div_add_mod",
    "dvd_add_iff_right": "dvd_add_right",
    "dvd_add_iff_left": "dvd_add_left",
    "cancel_factors.neg_subst": "norm_num.mul_pos_neg",
    "add_sub_assoc": "add_sub",
    "linear_ordered_add_comm_group.add_lt_add_left": "add_lt_add_left",
    "real.zero_lt_one": "zero_lt_one",  # We need nontrivial for this one
    "real.mul_pos": "mul_pos",
    "is_R_or_C.two_ne_zero": "two_ne_zero'",
    "fpow_two_pos_of_ne_zero": "sq_pos_of_ne_zero",
    "fpow_two_nonneg": "sq_nonneg",
    "div_le_div_of_le_of_nonneg": "div_le_div_of_le",
    "sub_self_div_two": "sub_half",
    "same_sub_div": "one_sub_div",
    "tactic.ring_exp.mul_coeff_pf_one_mul": "one_mul",
    "same_add_div": "one_add_div",
    "neg_div'": "neg_div",
    "mul_self_inj_of_nonneg": "mul_self_inj",
    "tactic.ring_exp.mul_coeff_pf_mul_one": "mul_one",
    "mul_div_assoc'": "mul_div_assoc",
    "is_R_or_C.mul_inv_cancel": "mul_inv_cancel",
    "is_R_or_C.inv_zero": "inv_zero",
    "with_zero.inv_zero": "inv_zero",  # "with_zero.inv_zero" should be general but weirdly Lean does not accept it for (0:real)
    "norm_num.inv_one": "inv_one",
    "inv_neg": "neg_inv",
    "inv_eq_one_div": "one_div",
    "norm_num.inv_div_one": "one_div",
    "norm_num.inv_div": "inv_div",
    "zero_div": "euclidean_domain.zero_div",  # More general but one can does not completely includes the other.
    "mul_div_cancel_left": "euclidean_domain.mul_div_cancel_left",
    "mul_div_cancel": "euclidean_domain.mul_div_cancel",
    "div_zero": "euclidean_domain.div_zero",
    "div_two_sub_self": "half_sub",
    "div_sub_same": "div_sub_one",
    "div_sub_div_same": "sub_div",
    "div_self": "euclidean_domain.div_self",
    "div_neg_eq_neg_div": "div_neg",
    "div_eq_mul_one_div": "mul_one_div",
    "div_add_same": "div_add_one",
    "add_self_div_two": "half_add_self",
    "le_add_of_neg_add_le": "neg_add_le_iff_le_add",
    "le_add_of_neg_add_le_right": "neg_add_le_iff_le_add",
    "le_neg_add_of_add_le": "le_neg_add_iff_add_le",
    "le_sub_right_of_add_le": "le_sub_iff_add_le",
    "le_sub_left_of_add_le": "le_sub_iff_add_le'",
    "le_of_sub_nonpos": "sub_nonpos",
    "dvd_of_dvd_neg": "dvd_neg",
    "dvd_of_neg_dvd": "neg_dvd",
    "div_add_div_same": "add_div",
    "gpow_one": "pow_one",
    "real.rpow_one": "pow_one",
    "tactic.ring_exp.simple_pf_var_one": "pow_one",
    "gpow_zero": "pow_zero",
    "real.rpow_zero": "pow_zero",
    "canonically_ordered_semiring.pow_le_one": "pow_le_one_of_le_one",
}


def filter_duplicated_rules(
    rules: List[Rule], allow_sym: bool = False, verbose: bool = False
) -> List[Rule]:
    def pp(s: str):
        if verbose:
            print(s)

    pp(f"===== FILTERING DUPLICATED RULES (allow_sym={allow_sym}) ...")
    pp(f"Currently {len(rules)} rules")

    theorems, _ = reload_lean_theorems()

    assert all(rule.lean_rule is not None for rule in rules)

    #
    # first filtering. removing rules with exactly identical statements
    #
    counts: Dict[str, List[str]] = defaultdict(list)
    for rule in rules:
        assert rule.lean_rule is not None
        counts[rule.lean_rule.statement].append(rule.name)
        assert theorems[rule.name] == rule.lean_rule.statement
    new_rules: List[Rule] = []
    n_skipped = 0
    for rule in rules:
        assert rule.lean_rule is not None
        k = rule.lean_rule.statement
        assert k in counts
        if len(counts[k]) == 1:
            new_rules.append(rule)
        else:
            order = [(len(name), name.count("."), name) for name in counts[k]]
            to_select = sorted(order)[0][-1]
            if to_select == rule.name:
                new_rules.append(rule)
                n_skipped += len(order) - 1
            else:
                # pp(f"Skipped {rule.name} (same statement as {to_select})")
                continue
    assert len(new_rules) + n_skipped == len(rules)
    rules = new_rules
    pp(f"Removed identical rules -- now {len(rules)} rules")

    #
    # second filtering -- manual using REPLACE_BY
    #
    new_rules = []
    curr_rule_names: List[str] = []
    for rule in rules:
        assert rule.lean_rule is not None
        curr_rule_names.append(rule.lean_rule.label)
    assert len(curr_rule_names) == len(set(curr_rule_names))
    for rule in rules:
        assert rule.lean_rule is not None
        name = rule.lean_rule.label
        if name in REPLACE_BY:
            assert REPLACE_BY[name] in curr_rule_names, name
            # pp(f"Skipping {name} (will use {REPLACE_BY[name]})")
        else:
            new_rules.append(rule)
    rules = new_rules
    pp(f"Removed replacable rules -- now {len(rules)} rules")

    #
    # final check
    #
    counts2: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
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
        counts2[tuple(s_x + s_hyps)].append(rule.name)
    n_dupl = 0
    for k2, v2 in counts2.items():
        if len(v2) > 1:
            pp(f"Found {len(v2)} rules for {k2}:")
            mlen = max(len(name) for name in v2)
            for name in v2:
                suffix = " " * (mlen - len(name))
                pp(f"\t{name}{suffix}   \t {theorems[name]}")
            n_dupl += len(v2)
            pp("\n")

    if all(len(v) == 1 for v in counts2.values()):
        assert n_dupl == 0
        pp(f"OK for {len(rules)} rules -- No duplicates.")
    else:
        assert n_dupl > 1
        pp(f"{n_dupl} rules have at least one duplicate!")

    return rules


try:
    verbose = __name__ == "__main__"

    # real
    auto_parsed = parse_mathlib_statements("real")
    RULES_REAL: List[Rule] = [r for r in auto_parsed.values() if r.rule_type == "lean"]
    RULES_REAL = sorted(RULES_REAL, key=lambda rule: rule.name)
    RULES_REAL = filter_duplicated_rules(RULES_REAL, verbose=verbose)

    RULES_IMO: List[Rule] = [r for r in auto_parsed.values() if r.rule_type == "imo"]
    RULES_IMO = sorted(RULES_IMO, key=lambda rule: rule.name)
    RULES_IMO = filter_duplicated_rules(RULES_IMO, verbose=verbose)

    # nat
    auto_parsed_nat_rules = parse_mathlib_statements("nat")
    RULES_NAT: List[Rule] = list(auto_parsed_nat_rules.values())
    RULES_NAT = sorted(RULES_NAT, key=lambda rule: rule.name)
    RULES_NAT = filter_duplicated_rules(RULES_NAT, verbose=verbose)

    # int
    auto_parsed_nat_rules = parse_mathlib_statements("int")
    RULES_INT: List[Rule] = list(auto_parsed_nat_rules.values())
    RULES_INT = sorted(RULES_INT, key=lambda rule: rule.name)
    RULES_INT = filter_duplicated_rules(RULES_INT, verbose=verbose)

except FileNotFoundError:
    print(f"Lean theorems not found! Falling back to manually parsed rules!")
    RULES_REAL = MANUAL_RULES
    RULES_NAT = []
    RULES_INT = []
    RULES_IMO = []
