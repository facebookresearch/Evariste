# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from typing import Optional
import re

from params import Params, ConfStore


ORELSE_CHAR = "🦞"


@dataclass
class TacticFilter(Params):

    # sum tactic log-probs if they are identical after cleaning. max otherwise
    sum_tactic_scores: bool = False

    # split tactics (with "," or ";") if bracket balancing allows it
    split_tactic: bool = False

    no_swap: bool = True
    no_rotate: bool = True
    no_classical: bool = True
    no_nontriviality: bool = True
    no_inhabit: bool = False

    # forbidden tokens for expander
    no_try: bool = True
    no_clear: bool = True
    no_repeat: bool = True


ConfStore["tac_filter_no_split"] = TacticFilter(split_tactic=False)
ConfStore["tac_filter_split"] = TacticFilter(split_tactic=True)


# delimiters are taken from:
# https://github.com/leanprover/vscode-lean/blob/master/src/abbreviation/abbreviations.json
open_delimiters = {
    "{": "}",
    "⦃": "⦄",
    "[": "]",
    "⟦": "⟧",
    "⟨": "⟩",
    "(": ")",
    "⟮": "⟯",
    "‹": "›",
    "«": "»",
    "⁅": "⁆",
    "⌊": "⌋",
    "⌈": "⌉",
}
close_delimiters = {v: k for k, v in open_delimiters.items()}


def unwrap_tactic(s: str) -> str:
    s = s.strip()
    regexes = [
        re.compile(r"^ *{(.*)} *$"),
        re.compile(r"^try *{(.*)} *$"),
        re.compile(r"^repeat *{(.*)} *$"),
        re.compile(r"^abstract *{(.*)} *$"),
        re.compile(r"^all_goals *{(.*)} *$"),
        re.compile(r"^any_goals *{(.*)} *$"),
        re.compile(r"^iterate *{(.*)} *$"),
        re.compile(r"^iterate +\d+ *{(.*)} *$"),
    ]
    for regex in regexes:
        new = regex.sub(r"\1", s).strip()
        if new != s:
            return re.sub(r"[, ]+$", "", new)  # rstrip spaces and ","
    return s


def truncate_tactic(tactic: str, split: bool) -> Optional[str]:
    """
    Truncate the tactic.
    If `split` is `True`, truncate on one of these symbols: ";", ",", "<|>"
    Always ensure that delimiters are balanced. Returns `None` otherwise.
    """

    # special treatment for the "orelse" operator: "<|>"
    # always remove the right-hand side
    if "<|>" in tactic:
        assert ORELSE_CHAR not in tactic, tactic
        tactic = tactic.replace("<|>", ORELSE_CHAR)

    counts = {k: 0 for k in open_delimiters.keys()}

    for i, c in enumerate(tactic):

        # opening delimiter
        if c in open_delimiters:
            counts[c] += 1

        # closing delimiter
        elif c in close_delimiters:
            cd = close_delimiters[c]
            counts[cd] -= 1
            if counts[cd] < 0:
                # check whether the tactic is valid without this unexpected closing
                # delimiter. if yes, return it. otherwise, the tactic is broken
                counts[cd] = 0
                if sum(counts.values()) == 0:
                    return tactic[:i].strip()
                else:
                    return None

        # tactic split
        elif c in {";", ","} and split or c == ORELSE_CHAR:
            # if delimiters are all closed, truncate the tactic
            if sum(counts.values()) == 0:
                return tactic[:i].strip()

    # tactic should be balanced
    is_balanced = sum(counts.values()) == 0
    return tactic.strip() if is_balanced else None


def unwrap_and_split(tactic: str) -> Optional[str]:
    """
    Iteratively split and unwrap the tactic, until we cannot shorten it.
    """
    s = tactic.strip()
    while True:
        last = s
        s = unwrap_tactic(s)
        if s != last:
            continue
        new_s = truncate_tactic(s, split=True)
        if new_s is None:
            return None
        s = new_s
        if s == last:
            break
    # if we did not manage to get rid of all <|> operators, do not return anything
    if ORELSE_CHAR in s:
        return None
    return s


def remove_empty_brackets(s: str) -> str:
    s = re.sub(r"(n?linarith) *(!?)(( only)?) *\[]", r"\1\2\3", s)
    s = re.sub(r"(norm_num1?) *\[]", r"\1", s)
    return s.strip()


def replace_abbreviations(s: str) -> str:
    s = re.sub(r"(?<![a-zA-Z0-9_.])rewrite *", "rw ", s)
    s = re.sub(r"(?<![a-zA-Z0-9_.])erewrite *", "erw ", s)
    return s


def remove_tactic(s: str, name: str) -> str:
    regexes = {
        "swap": re.compile(r"(?<![a-zA-Z0-9_.])swap[ ,]*"),
        "tactic.swap": re.compile(r"(?<![a-zA-Z0-9_.])tactic.swap[ ,]*"),
        "rotate": re.compile(r"(?<![a-zA-Z0-9_.])rotate *(\d+)?[ ,]*"),
        "classical": re.compile(r"(?<![a-zA-Z0-9_.])classical[ ,]*"),
        "nontriviality": re.compile(
            r"(?<![a-zA-Z0-9_.])nontriviality( [a-zA-Z0-9ℕℤℚℝℂα]+| \([^)]+\))?[ ,]*"
        ),
        "inhabit": re.compile(
            r"(?<![a-zA-Z0-9_.])inhabit( [a-zA-Z0-9ℕℤℚℝℂα]+| \([^)]+\))?[ ,]*"
        ),
    }
    s = regexes[name].sub("", s)
    return re.sub(r"[ ,]+$", "", s)


def normalize_tactic(s: str) -> str:
    """
    Normalize a tactic. Be careful about notations.
    Notations can be found with:
    ```
    grep -IrEih '(^|")notation .*' \
        ~/lean_proof_check/_target/deps/mathlib/src/* \
        > lean_notations_exceptions`
    ```
    """

    # remove successive spaces
    s = re.sub(r"  +", " ", s.strip())

    # "rw h at*" -> "rw h at *"
    # "simp [h]at*" -> "simp [h] at *"
    s = re.sub(r"([^a-zA-Z0-9_.]) *at\*", r"\1 at *", s)

    # no space after "[" ("norm_num[ abs]" -> "norm_num [abs]")
    # no spaces before "]" ("norm_num [abs ]" -> "norm_num [abs]")
    s = re.sub(r"([a-z0-9]) *\[", r"\1 [", s)
    s = re.sub(r"\[ +(?=[^xA-Z])", r"[", s)
    s = re.sub(r" +]", "]", s)

    # no space after "{" and before "}"
    s = re.sub(r"([a-zA-Z0-9_]) *{", r"\1 {", s)
    s = re.sub(r"{ +", r"{", s)
    s = re.sub(r" +}", "}", s)

    # no space after opening / before closing delimiters
    s = re.sub(r"([(⟨⌊⌈⁅‹⟮⟦⦃]) +", r"\1", s)
    s = re.sub(r" +([⦄⟧⟯›⌉⌋⟩)])", r"\1", s)

    # add spaces around ":" -- careful about :: and :=
    s = re.sub(r"([a-zA-Z0-9ℕℤℚℝℂα]) *:(?!:)", r"\1 :", s)
    s = re.sub(r"(?<!:): *([a-zA-Z0-9ℕℤℚℝℂα])", r": \1", s)

    # add spaces around common operators -- [≠←↔%] are safe (not in notations)
    s = re.sub("=-", "= -", s)  # the string "= -" is not in the training set
    s = re.sub(r" *:= *", r" := ", s)
    s = re.sub(r"(?<=[a-zA-Z0-9_)])([=<>≤≥→+*/-])([a-zA-Z0-9_( ]|$)", r" \1\2", s)
    s = re.sub(r"(?<=[a-zA-Z0-9_) ])([=<>≤≥→+*/-])(?=[a-zA-Z0-9_(])", r"\1 ", s)
    s = re.sub(r"([a-zA-Z0-9_)]) *([≠←↔%])", r"\1 \2", s)
    s = re.sub(r"([≠←↔%]) *([a-zA-Z0-9_(])", r"\1 \2", s)

    # handle cases where "-" is the negation operator
    s = re.sub(r"(?<![a-zA-Z0-9_) ])( *)- +", r"\1-", s)

    # remove spaces before ",", add one space after ","
    s = re.sub(r" +,", ",", s)
    s = re.sub(r", *", ", ", s)

    return re.sub(r"[ ,]+$", "", s.strip())


def clean_tactic(tactic: str, filters: TacticFilter) -> Optional[str]:

    assert ORELSE_CHAR not in tactic, tactic

    # remove empty square brackets
    tactic = remove_empty_brackets(tactic)

    # replace rewrite / erewrite abbreviations
    tactic = replace_abbreviations(tactic)

    # remove some unwanted tactics
    if filters.no_swap:
        tactic = remove_tactic(tactic, "swap")
        tactic = remove_tactic(tactic, "tactic.swap")
    if filters.no_rotate:
        tactic = remove_tactic(tactic, "rotate")
    if filters.no_classical:
        tactic = remove_tactic(tactic, "classical")
    if filters.no_nontriviality:
        tactic = remove_tactic(tactic, "nontriviality")
    if filters.no_inhabit:
        tactic = remove_tactic(tactic, "inhabit")
    # optionally unwrap and take the first sub-tactic.
    # ensure that the delimiters are balanced
    if filters.split_tactic:
        s = unwrap_and_split(tactic)
    else:
        s = truncate_tactic(tactic, split=False)
    if s is None or s == "" or ORELSE_CHAR in s:
        return None
    tactic = s

    # this should have been discarded by expander
    if filters.no_try and re.search(r"(^|\W)try *{", tactic) is not None:
        return None
    if filters.no_clear and re.search(r"(^|\W)clear ", tactic) is not None:
        return None
    if filters.no_repeat and re.search(r"(^|\W)repeat *{", tactic) is not None:
        return None

    # normalize tactic
    tactic = normalize_tactic(tactic)

    return None if tactic == "" else tactic


if __name__ == "__main__":

    # python -m evariste.backward.env.lean.filter_tactics

    to_clean = [
        "classical , erw h₂",
        "symmetry,classical , simp only[pow_succ', pow_succ', *, add_mul] at * {contextual := tt}",
        "nontriviality, field_simp [h₁, h₂, h₃, h₄] at* {contextual := tt}",
        "clasical, split_ifs at h₀ ,split,nlinarith,nlinarith,nlinarith! ,nlinarith! []",
        "ring, classical",
        "tactic.swap",
        "rotate",
        "swap ,classical",
        "solve_by_elim, nontriviality, field_simp [h₀,mul_comm],subst v, norm_num [ complex.ext_iff]",
        "nlinarith[ sq (b * b)]",
        "nontriviality,classical,norm_num1",
        "have :=h₀ 3 (by norm_num)",
        "have:=h₀ 3 (by norm_num)",
        "have:= h₀ 3 (by norm_num)",
        "simp [is_open_Ioo] {contextual:=tt}",
        "classical, split_ifs at h₀ ,split,nlinarith,nlinarith,nlinarith! ,nlinarith! []",
        "classical, push_cast [finset.sum_range_succ ] at h₁, induction a using nat.strong_induction_on , symmetry",
        "push_cast[nat.lt_succ_iff_lt_or_eq,two_mul ] at*",
        "classical, norm_num [ abs]",
        "iterate 6 { classical, rw nat.gcd_rec }",
        "iterate{ iterate {rcases h_1 with (⟨_|rfl⟩ |rfl) }, norm_num [] at h₀ }",
        "nlinarith! []",
        "linarith ! []",
        "nlinarith []",
        "norm_num []",
        "norm_num [ abs ]",
        "iterate 6 { rw nat.gcd_rec }",
        "iterate 6{ rcases}",
        "atry { sdf",
        "try { sdf",
        "clear h",
        "aclear h",
        "bla, clear h",
        "bla, aclear h",
        "all_goals { have s_tail_mth_eq, from this.trans s_succ_mth_eq }",
        "all_goals { repeat { some, stuff }, lol }",
        "simp[lol]at*",
        "norm_num[lol] at*",
        "norm_num[lol]at*",
        "xx[lol]]at*",
        "simp[dsf   ,     sdfsd  ]",
        "simp[ (a) , (b + c)  ]",
        "simp[ =dsf,  sdfsd]",
        "simp[ SMOD,  sdfsd]",
        "classical",
        "classical,classical, classical,",
        "classical,bla, classical,",
        "OKclassical,bla, classical,",
        "OKclassical,bla, {  classical},",
        ",classical,classical, classical,",
        "repeat { apply mul_pos <|> apply add_pos_of_nonneg_of_pos }",
        "any_goals { repeat { apply mul_pos <|> apply add_pos_of_nonneg_of_pos } }; assumption",
        "apply mul_pos <|> apply add_pos_of_nonneg_of_pos",
        "bla { apply mul_pos <|> apply add_pos_of_nonneg_of_pos }",
        "try {dunfold bind}",
        "rcases h_1 with (⟨  _|rfl   ⟩ |rfl)}",
        "cases ( nat.sqrt _ )",
        "classical, nontriviality ℤ,rw [pow_zero, mul_comm, mul_assoc] , intro h ,classical",
        "suffices : (⌊  ((( 3:ℝ)/ (8:  ℝ)) / (-(( 2:ℝ)/ (5 :ℝ)))) ⌋ =- 1), norm_num*,",
        "use ((x -y) *(x- z)) + ((y - z) * (y - z)) , simp[pow_succ, mul_assoc, int.mul_mod_right] at * {contextual := tt}",
        "a* x ^ 4 + b * y ^ 4=42",
        "simp only [←mul_assoc, *, one_mul, mul_one]",
        "0≤x↔0>y",
        "inits_core_eq (a::l)",
        "rotate 3, rw h",
        "rotate 3, rotate, rw h",
        "rotate 3, rw h, rotate 2",
        "rotate 3, rotate, rotate 2",
        "use ∑ a in s.to_finset, (s.count a / k) • (a ::ₘ 0)",
        "iterate 6 { rw nat.gcd_rec }",
        "bla, clear h",
        "nontriviality ennreal, nontriviality R, nontriviality ℂ, rw h",
        "nlinarith only []",
        "nlinarith ! only []",
        "nlinarith! only []",
        "{aa, bb}",
        "{ aa, bb }",
        "rewrite h",
        "erewrite h",
        "ext; simp [h₀]; ring,",
        "a=10%2",
        "∀ x ∈ {x | 0 < x ∧ x ^ 2 = sqrt 2 ^ x}.to_finset, x ∈ {x | 0 < x ∧ x ^ 2 = sqrt 2 ^ x}",
        "bla <|> lol",
        "have : ∀ x, 0 < f x",
    ]

    for s_ in to_clean:
        filters_ = TacticFilter(split_tactic=True)
        print("======")
        print(s_)
        print(normalize_tactic(s_))
        print(clean_tactic(s_, filters_))
