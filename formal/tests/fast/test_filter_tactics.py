# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import numpy as np

from evariste.backward.env.lean.filter_tactics import clean_tactic
from evariste.datasets.lean import TacticFilter
from evariste.backward.env.core import ModelExpansion, PreEnvExpansion
from evariste.backward.env.lean.graph import LeanTheorem, LeanContext
from evariste.backward.env.lean.env import LeanExpanderEnv
from evariste.backward.env.lean.tokenizer import LeanTokenizer
from evariste.model.data.dictionary import B_CMD_WORD, E_CMD_WORD


def test_clean_tactics():

    # input, no split, split
    # fmt: off
    TESTS = [

        # forbidden tactics
        (
            "classical",
            None,
            None,
        ),
        (
            "classical,classical, classical",
            None,
            None,
        ),
        (
            "rotate",
            None,
            None,
        ),
        (
            "tactic.swap",
            None,
            None,
        ),
        (
            "swap ,classical",
            None,
            None,
        ),
        (
            "clear h",
            None,
            None,
        ),
        (
            "bla, clear h",
            None,
            "bla",
        ),
        (
            "ring, classical",
            "ring",
            "ring",
        ),
        (
            "nontriviality,classical,norm_num1",
            "norm_num1",
            "norm_num1",
        ),
        (
            "inhabit ℂ",
            None,
            None,
        ),
        (
            "classical , erw h₂",
            "erw h₂",
            "erw h₂",
        ),
        (
            "classical,bla, classical",
            "bla",
            "bla",
        ),
        (
            "OKclassical,bla, classical",
            "OKclassical, bla",
            "OKclassical",
        ),
        (
            "OKclassical,bla, {  classical}",
            "OKclassical, bla, {}",
            "OKclassical",
        ),
        (
            "aclear h",
            "aclear h",
            "aclear h",
        ),
        (
            "bla, aclear h",
            "bla, aclear h",
            "bla",
        ),
        (
            "rotate 3, rw h",
            "rw h",
            "rw h",
        ),
        (
            "rotate 3, rotate, rw h",
            "rw h",
            "rw h",
        ),
        (
            "rotate 3, rw h, rotate 2",
            "rw h",
            "rw h",
        ),
        (
            "rotate 3, rotate, rotate 2",
            None,
            None,
        ),
        (
            "nontriviality, nontriviality, nontriviality",
            None,
            None,
        ),
        (
            "nontriviality ennreal, nontriviality R, nontriviality ℂ, rw h",
            "rw h",
            "rw h",
        ),
        (
            "nontriviality, rw h, nontriviality",
            "rw h",
            "rw h",
        ),
        (
            "nontriviality (fin 0 ↪ β), rw h, nontriviality",
            "rw h",
            "rw h",
        ),

        # unbalanced tactics
        (
            "atry { sdf",
            None,
            None,
        ),
        (
            "try { sdf",
            None,
            None,
        ),
        (
            "rw h ]",
            "rw h",
            "rw h",
        ),
        (
            "rw h, ]",
            "rw h",
            "rw h",
        ),

        # tactic split
        (
            "{   aa ,  bb  }",
            "{aa, bb}",
            "aa",
        ),
        (
            "classical, push_cast [finset.sum_range_succ ] at h₁, induction a using nat.strong_induction_on , symmetry",
            "push_cast [finset.sum_range_succ] at h₁, induction a using nat.strong_induction_on, symmetry",
            "push_cast [finset.sum_range_succ] at h₁",
        ),
        (
            "symmetry,classical , simp only[pow_succ', pow_succ', *, add_mul] at * {contextual := tt}",
            "symmetry, simp only [pow_succ', pow_succ', *, add_mul] at * {contextual := tt}",
            "symmetry",
        ),
        (
            "nontriviality, field_simp [h₁, h₂, h₃, h₄] at* {contextual := tt}",
            "field_simp [h₁, h₂, h₃, h₄] at * {contextual := tt}",
            "field_simp [h₁, h₂, h₃, h₄] at * {contextual := tt}",
        ),
        (
            "classical, split_ifs at h₀ ,split,nlinarith,nlinarith,nlinarith! ,nlinarith! []",
            "split_ifs at h₀, split, nlinarith, nlinarith, nlinarith!, nlinarith!",
            "split_ifs at h₀",
        ),
        (
            "solve_by_elim, nontriviality, field_simp [h₀,mul_comm],subst v, norm_num [ complex.ext_iff]",
            "solve_by_elim, field_simp [h₀, mul_comm], subst v, norm_num [complex.ext_iff]",
            "solve_by_elim",
        ),
        (
            "classical, split_ifs at h₀ ,split,nlinarith,nlinarith,nlinarith! ,nlinarith! []",
            "split_ifs at h₀, split, nlinarith, nlinarith, nlinarith!, nlinarith!",
            "split_ifs at h₀",
        ),
        (
            "iterate{ iterate {rcases h₀ }, norm_num [] at h₀ }",
            "iterate {iterate {rcases h₀}, norm_num at h₀}",
            "rcases h₀",
        ),
        (
            "iterate 6 { classical, rw nat.gcd_rec }",
            "iterate 6 {rw nat.gcd_rec}",
            "rw nat.gcd_rec",
        ),
        (
            "iterate 6 { rw nat.gcd_rec }",
            "iterate 6 {rw nat.gcd_rec}",
            "rw nat.gcd_rec"
        ),
        (
            "iterate 6{ rcases}",
            "iterate 6 {rcases}",
            "rcases"
        ),
        (
            "iterate{ iterate {rcases h_1 with (⟨_|rfl⟩ |rfl) }, norm_num [] at h₀ }",
            "iterate {iterate {rcases h_1 with (⟨_|rfl⟩ |rfl)}, norm_num at h₀}",
            "rcases h_1 with (⟨_|rfl⟩ |rfl)",
        ),
        (
            "all_goals { have s_tail_mth_eq, from this.trans s_succ_mth_eq }",
            "all_goals {have s_tail_mth_eq, from this.trans s_succ_mth_eq}",
            "have s_tail_mth_eq",
        ),
        ("all_goals { repeat { some, stuff }, lol }", None, "some"),
        (
            "apply mul_pos <|> apply add_pos_of_nonneg_of_pos",
            "apply mul_pos",
            "apply mul_pos",
        ),
        (
            "repeat { apply mul_pos <|> apply add_pos_of_nonneg_of_pos }",
            None,
            "apply mul_pos",
        ),
        (
            "any_goals { repeat { apply mul_pos <|> apply add_pos_of_nonneg_of_pos } }; assumption",
            None,
            "apply mul_pos",
        ),
        (
            "bla { apply mul_pos <|> apply add_pos_of_nonneg_of_pos }",
            None,
            None,
        ),
        (
            "try {dunfold bind}",
            None,
            "dunfold bind",
        ),

        # normalization
        (
            "rewrite h",
            "rw h",
            "rw h",
        ),
        (
            "nth_rewrite 1 h, rewrite this at h",
            "nth_rewrite 1 h, rw this at h",
            "nth_rewrite 1 h",
        ),
        (
            "erewrite h",
            "erw h",
            "erw h",
        ),
        (
            "rewrite h1,erewrite h2,rewrite h3",
            "rw h1, erw h2, rw h3",
            "rw h1",
        ),
        (
            "nlinarith[ sq (b * b)]",
            "nlinarith [sq (b * b)]",
            "nlinarith [sq (b * b)]",
        ),
        (
            "have :=h₀ 3 (by norm_num)",
            "have := h₀ 3 (by norm_num)",
            "have := h₀ 3 (by norm_num)",
        ),
        (
            "have:=h₀ 3 (by norm_num)",
            "have := h₀ 3 (by norm_num)",
            "have := h₀ 3 (by norm_num)",
        ),
        (
            "have:= h₀ 3 (by norm_num)",
            "have := h₀ 3 (by norm_num)",
            "have := h₀ 3 (by norm_num)",
        ),
        (
            "simp [is_open_Ioo] {  contextual:=tt  }",
            "simp [is_open_Ioo] {contextual := tt}",
            "simp [is_open_Ioo] {contextual := tt}",
        ),
        (
            "push_cast[nat.lt_succ_iff_lt_or_eq,two_mul ] at*",
            "push_cast [nat.lt_succ_iff_lt_or_eq, two_mul] at *",
            "push_cast [nat.lt_succ_iff_lt_or_eq, two_mul] at *",
        ),
        (
            "classical, norm_num [ abs]",
            "norm_num [abs]",
            "norm_num [abs]",
        ),
        (
            "nlinarith! []",
            "nlinarith!",
            "nlinarith!"
        ),
        (
            "linarith ! []",
            "linarith!",
            "linarith!"
        ),
        (
            "nlinarith []",
            "nlinarith",
            "nlinarith"
        ),
        (
            "nlinarith only []",
            "nlinarith only",
            "nlinarith only",
        ),
        (
            "norm_num []",
            "norm_num",
            "norm_num"
        ),
        (
            "norm_num [ abs ]",
            "norm_num [abs]",
            "norm_num [abs]"
        ),
        (
            "simp[lol]at*",
            "simp [lol] at *",
            "simp [lol] at *",
        ),
        (
            "norm_num[lol] at*",
            "norm_num [lol] at *",
            "norm_num [lol] at *",
        ),
        (
            "norm_num[lol]at*",
            "norm_num [lol] at *",
            "norm_num [lol] at *",
        ),
        (
            "xx[lol]]at*",
            "xx [lol]",
            "xx [lol]",
        ),
        (
            "simp[dsf   ,     sdfsd  ]",
            "simp [dsf, sdfsd]",
            "simp [dsf, sdfsd]",
        ),
        (
            "simp[ (a) , (b + c)  ]",
            "simp [(a), (b + c)]",
            "simp [(a), (b + c)]",
        ),
        (
            "simp[ SMOD,  sdfsd]",
            "simp [ SMOD, sdfsd]",
            "simp [ SMOD, sdfsd]",
        ),
        (
            "rcases h_1 with (⟨  _|rfl   ⟩ |rfl)}",
            "rcases h_1 with (⟨_|rfl⟩ |rfl)",
            "rcases h_1 with (⟨_|rfl⟩ |rfl)",
        ),
        (
            "cases ( nat.sqrt _ )",
            "cases (nat.sqrt _)",
            "cases (nat.sqrt _)",
        ),
        (
            "classical, nontriviality ℤ,rw [pow_zero, mul_comm, mul_assoc] , intro h ,classical",
            "rw [pow_zero, mul_comm, mul_assoc], intro h",
            "rw [pow_zero, mul_comm, mul_assoc]",
        ),
        (
            "h -2 + (-1 + x)-2",
            "h - 2 + (-1 + x) - 2",
            "h - 2 + (-1 + x) - 2",
        ),
        (
            "-2",
            "-2",
            "-2",
        ),
        (
            "-2:ℤ",
            "-2 : ℤ",
            "-2 : ℤ",
        ),
        (
            "(-2:ℤ)",
            "(-2 : ℤ)",
            "(-2 : ℤ)",
        ),
        (
            "a=-1",
            "a = -1",
            "a = -1",
        ),
        (
            "a=10%2",
            "a = 10 % 2",
            "a = 10 % 2",
        ),
        (
            "suffices : (⌊  ((( 3:ℝ)/ (8:  ℝ)) / (-(( 2:ℝ)/ (5 :ℝ)))) ⌋ = -1), norm_num*",
            "suffices : (⌊(((3 : ℝ) / (8 : ℝ)) / (-((2 : ℝ) / (5 : ℝ))))⌋ = -1), norm_num *",
            "suffices : (⌊(((3 : ℝ) / (8 : ℝ)) / (-((2 : ℝ) / (5 : ℝ))))⌋ = -1)",
        ),
        (
            "use ((x -y) *(x- z))+ ((y- z)* (y - z)) ,",
            "use ((x - y) * (x - z)) + ((y - z) * (y - z))",
            "use ((x - y) * (x - z)) + ((y - z) * (y - z))",
        ),
        (
            "a* x ^ 4 + b * y ^ 4=42",
            "a * x ^ 4 + b * y ^ 4 = 42",
            "a * x ^ 4 + b * y ^ 4 = 42",
        ),
        (
            "simp only [←mul_assoc, *,←one_mul, mul_one]",
            "simp only [← mul_assoc, *, ← one_mul, mul_one]",
            "simp only [← mul_assoc, *, ← one_mul, mul_one]",
        ),
        (
            "x≥0",
            "x ≥ 0",
            "x ≥ 0",
        ),
        (
            "ℝ≥0",
            "ℝ≥0",
            "ℝ≥0",
        ),
        (
            "0≤x↔0>y",
            "0 ≤ x ↔ 0 > y",
            "0 ≤ x ↔ 0 > y",
        ),
        (
            "inits_core_eq (a::l)",
            "inits_core_eq (a::l)",
            "inits_core_eq (a::l)",
        ),
        (
            "use ∑ a in s.to_finset, (s.count a / k) • (a ::ₘ 0)",
            "use ∑ a in s.to_finset, (s.count a / k) • (a ::ₘ 0)",
            "use ∑ a in s.to_finset",
        ),
        (
            "∀ x ∈ {x | 0 < x ∧ x ^ 2 = sqrt 2 ^ x}.to_finset, x ∈ {x | 0 < x ∧ x ^ 2 = sqrt 2 ^ x}",
            "∀ x ∈ {x | 0 < x ∧ x ^ 2 = sqrt 2 ^ x}.to_finset, x ∈ {x | 0 < x ∧ x ^ 2 = sqrt 2 ^ x}",
            "∀ x ∈ {x | 0 < x ∧ x ^ 2 = sqrt 2 ^ x}.to_finset",
        ),
    ]
    # fmt: on

    filter_no_split = TacticFilter(split_tactic=False, no_inhabit=True)
    filter_split = TacticFilter(split_tactic=True, no_inhabit=True)

    for x, y_no_split, y_split in TESTS:
        z_no_split = clean_tactic(x, filter_no_split)
        z_split = clean_tactic(x, filter_split)
        # print(f"====== no split -- {x}")
        # print(f"y: {y_no_split}")
        # print(f"z: {z_no_split}")
        # print("=== split")
        # print(f"y: {y_split}")
        # print(f"z: {z_split}")
        assert z_no_split == y_no_split
        assert z_split == y_split


def test_filter_tactics():

    LeanTokenizer.build("char")

    # Tuple [ tactics_with_log_probs, filters, expected_tactics ]
    tests = [
        (
            [
                ("norm_num; linarith", 0.1),
                ("norm_num", 0.2),
                ("norm_num <|> bla", 0.3),
                ("rw  h", 0.35),
            ],
            TacticFilter(sum_tactic_scores=False, split_tactic=True),
            {"norm_num": 0.3, "rw h": 0.35},
        ),
        (
            [
                ("norm_num; linarith", 0.1),
                ("norm_num", 0.2),
                ("norm_num <|> bla", 0.3),
                ("rw  h", 0.35),
            ],
            TacticFilter(sum_tactic_scores=True, split_tactic=True),
            {"norm_num": 0.6, "rw h": 0.35},
        ),
        (
            [
                ("norm_num;   linarith", 0.1),
                ("norm_num", 0.2),
                ("norm_num <|> bla", 0.3),
                ("rw  h", 0.35),
            ],
            TacticFilter(sum_tactic_scores=True, split_tactic=False),
            {"norm_num; linarith": 0.1, "norm_num": 0.5, "rw h": 0.35},
        ),
    ]

    for tactics, filters, y_tactics in tests:
        y_tactics = dict(y_tactics.items())
        model_exp = ModelExpansion(
            exp_duration=0.0,
            gpu_duration=0.0,
            error=None,
            log_critic=0.0,
            tactics=[[B_CMD_WORD, *t, E_CMD_WORD] for t, _ in tactics],
            log_priors=[float(np.log(p)) for _, p in tactics],
        )
        LeanExpanderEnv.do_filter_model_expansion(filters, model_exp)
        PreEnvExpansion.from_model_expansion(
            LeanTheorem("test_thm", LeanContext(namespaces=set()), state=None),
            model_exp,
        )
        assert len(model_exp.tactics) == len(y_tactics)
        z_tactics = {
            "".join(tactic_chars[1:-1]): np.exp(lp)
            for tactic_chars, lp in zip(model_exp.tactics, model_exp.log_priors)
        }
        assert y_tactics.keys() == z_tactics.keys(), (y_tactics, z_tactics)
        for k, v in y_tactics.items():
            assert abs(v - z_tactics[k]) < 1e-9, (k, (y_tactics, z_tactics))
