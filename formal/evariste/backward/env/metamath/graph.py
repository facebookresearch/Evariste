# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Tuple, Dict, Optional, Set, Sequence
import re

from evariste.model.data.dictionary import (
    B_CMD_WORD,
    E_CMD_WORD,
    B_SUBST_WORD,
    M_SUBST_WORD,
    E_SUBST_WORD,
    B_HYP_WORD,
    E_HYP_WORD,
    B_GOAL_WORD,
    E_GOAL_WORD,
    SPECIAL_WORDS,
)
from evariste.backward.graph import Tactic, Theorem, Token, Hypothesis
from evariste.backward.graph import MalformedTactic, MalformedTheorem


MMSubstitutions = Dict[Token, str]  # (token, substitute)


_SPECIAL_WORDS = set(SPECIAL_WORDS)

MM_WRONG_TAC = "Malformed tactic"
MM_MANY_SUBS = "Multiple subs for"
MM_UNK_LABEL = "Unknown label:"
MM_HYPS_MISMATCH = "Substitutions and f_hyps don't match:"
MM_DISJ_V = "Disjoint violation:"
MM_NEW_DISJ = "Tactic would lead to new disjoint in mand_vars:"
MM_WRONG = "Tactic produces wrong output :"
MM_SUB_SYNTAX = "Subgoal has invalid syntax:"

mm_error_str = [
    ("WRONG_OUTPUT", rf"^{MM_WRONG}"),
    ("NEW_DISJOINT", rf"^{MM_NEW_DISJ}"),
    ("DISJOINT_VIO", rf"^{MM_DISJ_V}"),
    ("UNKNOWN_LABEL", rf"^{MM_UNK_LABEL}"),
    ("SUB_F_HYPS_MISMATCH", rf"^{MM_HYPS_MISMATCH}"),
    ("BAD_TACTIC", rf"^{MM_WRONG_TAC}"),
    ("MULTIPLE_SUBS", rf"^{MM_MANY_SUBS}"),
    ("BAD_SYNTAX", rf"^{MM_SUB_SYNTAX}"),
]


def get_error(error: str):
    for name, regex in mm_error_str:
        if re.match(regex, error):
            return name
    print("UNKNOWN ERROR", error)
    return "UNK_ERROR"


class MMTactic(Tactic):
    @staticmethod
    def from_error(error_msg: str, tokens: Optional[List[str]] = None) -> "MMTactic":
        return MMTactic(
            "", {}, valid=False, malformed=True, error_msg=error_msg, tokens=tokens
        )

    def get_error_code(self) -> str:
        assert self.error_msg is not None
        return get_error(self.error_msg)

    @staticmethod
    def wrap_subs(name: Token, term: str) -> List[Token]:
        return [B_SUBST_WORD, name, M_SUBST_WORD, *term.split(), E_SUBST_WORD]

    def tokenize(self) -> List[Token]:
        assert self.is_valid
        # fmt: off
        return [
            B_CMD_WORD, self.label,
            *sum([
                MMTactic.wrap_subs(old_token, self.subs[old_token])
                for old_token in sorted(self.subs.keys())
            ], []),
            E_CMD_WORD
        ]
        # fmt: on

    def update_subs(self, to_update: Dict[str, str]):
        assert self.is_valid and not self.malformed
        self.subs.update(to_update)
        Tactic.__init__(
            self,
            is_valid=True,
            unique_str=" ".join(self.tokenize()),
            error_msg=None,
            malformed=False,
        )  # re-hash

    @staticmethod
    def from_tokens(tokens: List[Token]) -> "MMTactic":
        assert type(tokens) is list
        try:
            subs: Dict[Token, List[Token]] = {}
            if len(tokens) == 0:
                raise MalformedTactic(f"Empty tactic")
            if (tokens[0], tokens[-1]) != (B_CMD_WORD, E_CMD_WORD):
                raise MalformedTactic(f"{MM_WRONG_TAC} {' '.join(tokens)}")
            label = tokens[1]
            i = 2
            try:
                while tokens[i] == B_SUBST_WORD:
                    old_token = tokens[i + 1]
                    if tokens[i + 2] != M_SUBST_WORD:
                        raise MalformedTactic(f"{MM_WRONG_TAC} {' '.join(tokens)}")
                    i += 3
                    new_tokens = []
                    while tokens[i] != E_SUBST_WORD:
                        new_tokens.append(tokens[i])
                        i += 1
                    if old_token in subs:
                        raise MalformedTactic(f"{MM_MANY_SUBS} '{old_token}'")
                    subs[old_token] = new_tokens
                    i += 1
            except IndexError:
                raise MalformedTactic(f"{MM_WRONG_TAC} {' '.join(tokens)}")
            if i != len(tokens) - 1:
                raise MalformedTactic(f"{MM_WRONG_TAC} {' '.join(tokens)}")
            for old_token, substituted in subs.items():
                if old_token in _SPECIAL_WORDS or _SPECIAL_WORDS.intersection(
                    substituted
                ):
                    raise MalformedTactic(f"{MM_WRONG_TAC} {' '.join(tokens)}")
            return MMTactic(
                label, {x: " ".join(y) for x, y in subs.items()}, valid=True
            )
        except MalformedTactic as e:
            return MMTactic.from_error(error_msg=str(e), tokens=tokens)

    def to_dict(self, light=False):
        if self.is_valid:
            return {"label": self.label, "subs": self.subs}
        return {"error_msg": self.error_msg}

    @classmethod
    def from_dict(cls, data):
        return cls(data["label"], data["subs"])

    def __init__(
        self,
        label: str,
        subs: MMSubstitutions,
        valid=True,
        malformed=False,
        error_msg=None,
        tokens: Optional[List[str]] = None,
    ):
        assert type(label) is str and type(subs) is dict
        assert all(type(k) is str and type(v) is str for k, v in subs.items())
        assert valid == (error_msg is None)
        assert valid == (tokens is None)
        self.is_valid = valid
        self._label = label
        self._subs = subs
        if self.is_valid:
            unique_str = " ".join(self.tokenize())
        else:
            assert tokens is not None
            unique_str = " ".join(tokens)
        Tactic.__init__(
            self, valid, unique_str=unique_str, error_msg=error_msg, malformed=malformed
        )

    @property
    def label(self) -> Token:
        assert self.is_valid, self.error_msg
        return self._label

    @property
    def subs(self) -> MMSubstitutions:
        assert self.is_valid
        return self._subs

    def __repr__(self):
        if not self.is_valid:
            return f"<INVALID_TACTIC> {self.error_msg}"
        return (
            f"Label: {self.label}\n"
            + f"Substitutions:\n\t"
            + "\t".join(f"{k} -> {v}\n" for k, v in self.subs.items())
        )

    def get_diff(self, other):
        """
        Outputs first diff
        @param other:
        @return:
        """
        if self == other:
            return None
        if self.label != other.label:
            return "label", None

        for x, y in self.subs.items():
            this_y = other.subs[x]
            if y != this_y:
                if len(y) != len(this_y):
                    return x, min(len(y), len(this_y))
                for k, (a, b) in enumerate(zip(y, this_y)):
                    if a != b:
                        return x, k


class MMTheorem(Theorem):
    def to_dict(self, light=False):
        return {
            "conclusion": self.conclusion,
            "hyps": self.hyps,
            # set json serialization is annoying
            "mand_vars": list(self.mand_vars) if self.mand_vars is not None else None,
            "mand_disj": list(self.mand_disj) if self.mand_disj is not None else None,
        }

    @classmethod
    def from_dict(cls, data):
        data["mand_vars"] = (
            set(data["mand_vars"]) if data["mand_vars"] is not None else None
        )
        if data["mand_disj"] is not None:
            # this avoids unhashable type list ...
            data["mand_disj"] = set([(x, y) for x, y in data["mand_disj"]])
        return cls(**data)

    def conc_in_hyp(self) -> bool:
        return self.conclusion in set([h for _, h in self.hyps])

    def get_proving_hyp(self) -> str:
        for i, (_, h) in enumerate(self.hyps):
            if self.conclusion == h:
                return f"E_HYP_{i}"
        raise RuntimeError("didn't find matching hyp")

    def tokenize(self) -> List[Token]:
        # fmt: off
        return [
            B_GOAL_WORD,
            # include hypothesis if any
            *sum([[B_HYP_WORD, *hyp.split(), E_HYP_WORD] for _, hyp in self.hyps], []),
            *self.conclusion.split(),
            E_GOAL_WORD
        ]
        # fmt: on

    @classmethod
    def from_tokens(cls, tokens: List[Token]):
        assert type(tokens) is list

        def exception(s: str):
            raise MalformedTheorem(f"{s}: {' '.join(tokens)}")

        if len(tokens) == 0 or (tokens[0], tokens[-1]) != (B_GOAL_WORD, E_GOAL_WORD):
            raise exception("invalid theorem delimitors")

        if tokens.count(B_HYP_WORD) != tokens.count(E_HYP_WORD):
            raise exception("invalid hypotheses delimitors")

        hyps: List[str] = []
        i = 1
        try:
            while tokens[i] == B_HYP_WORD:
                hyp_content = []
                if tokens[i + 1] == E_HYP_WORD:
                    raise exception("empty hypothesis")
                i += 1
                try:
                    while tokens[i] != E_HYP_WORD:
                        hyp_content.append(tokens[i])
                        i += 1
                except IndexError:
                    raise exception("unfinished hypothesis")
                hyps.append(" ".join(hyp_content))
                i += 1
            if i >= len(tokens) - 1:
                raise exception("unfinished hypothesis")
        except IndexError:
            raise exception("unfinished hypothesis")

        hyp_toks: List[str] = sum([hyp.split() for hyp in hyps], [])
        all_toks = set(hyp_toks + tokens[i:-1])
        special_words = _SPECIAL_WORDS.intersection(all_toks)
        if len(special_words) > 0:
            raise exception(f"unexpected special words ({special_words})")

        conclusion = " ".join(tokens[i:-1])
        return cls(conclusion, [(None, h) for h in hyps])

    def __init__(
        self,
        conclusion: str,
        hyps: Sequence[Hypothesis],
        train_label: Optional[str] = None,
        mand_vars: Optional[Set[Token]] = None,
        mand_disj: Optional[Set[Tuple[Token, Token]]] = None,
    ):
        assert type(conclusion) is str, conclusion
        assert type(hyps) is list
        assert all((n is None or type(n) is str) and type(h) is str for n, h in hyps)

        super().__init__(conclusion, hyps, train_label)
        self.mand_vars = mand_vars
        self.mand_disj = mand_disj

    def __repr__(self):
        s = []
        if len(self.hyps) > 0:
            s = ["Hypotheses:"]
            for name, hyp in self.hyps:
                s.append(f"\t{hyp}" if name is None else f"\t{name} {hyp}")
        s.append(f"Conclusion: {self.conclusion}")
        return "\n".join(s)


def get_subst(
    goal: MMTheorem, tactics: List[MMTactic]
) -> Tuple[List[List[Token]], List[List[Token]], List[List[Token]]]:
    """
    @param goal:
    @param tactics:
    @return: a tuple containing the (mandatory, pred, all) subst for each tactics
    """
    mand_subst = []
    pred_subst = []
    all_subst = []

    predictable = set(goal.conclusion.split(" "))
    for tactic in tactics:
        # add substitutions
        this_mand_subst = []
        this_pred_subst = []
        for name, term in tactic.subs.items():
            seq = MMTactic.wrap_subs(name, term)
            if name in predictable:
                this_pred_subst.extend(seq)
            else:
                this_mand_subst.extend(seq)

        mand_subst.append(this_mand_subst)
        pred_subst.append(this_pred_subst)
        all_subst.append(this_mand_subst + this_pred_subst)

    assert len(mand_subst) == len(pred_subst) == len(all_subst) == len(tactics)
    return mand_subst, pred_subst, all_subst
