# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, asdict
from typing import List, Optional, Set
import re

from evariste.logger import create_logger
from evariste.model.data.dictionary import (
    B_CMD_WORD,
    E_CMD_WORD,
    B_GOAL_WORD,
    E_GOAL_WORD,
    B_NS_WORD,
    E_NS_WORD,
    M_STACK_WORD,
    SPECIAL_WORDS,
    NEWLINE_WORD,
)
from evariste.backward.graph import Tactic, Theorem, Token
from evariste.backward.graph import MalformedTactic, MalformedTheorem
from evariste.backward.env.tokenized_tactic import TokenizedTactic
from lean_cluster.instance import LeanInstanceDied
import evariste.backward.env.lean.tokenizer as lean_tok


logger = create_logger(None)


_SPECIAL_WORDS = set(SPECIAL_WORDS)


@dataclass
class LeanState:
    session: str
    node_id: int
    n_subgoals: Optional[int] = None
    n_metavars: Optional[int] = None


@dataclass
class LeanContext:
    """Describes the context in which a theorem is declared (open namespaces, imports, open_locale).
    These are passed down identically to children LeanTheorems, and used as additional inputs to the transformer
    """

    namespaces: Set[str]
    # will be available later
    # locales : Set[str]
    # imports : Set[str]


lean_error_str = [
    (LeanInstanceDied, fr"^{LeanInstanceDied}"),
    ("MALFORMED_TACTIC", "^MALFORMED "),
    ("INVALID_CASE", "^Invalid `case`"),
    ("AMBIGUOUS_OVERLOAD", "^ambiguous overload"),
    (
        "APPLY_INSTANCE_FAILED",
        "^apply_instance tactic fail, target is not a type class",
    ),
    ("CASES_FAILED", "^cases tactic failed"),
    ("EXACT_FAILED", "^exact tactic failed, type mismatch"),
    ("INSTANTIATE_GOAL_FAILED", "^failed to instantiate goal with"),
    ("SYNTHETIZE_TYPE_CLASS_FAILED", "^failed to synthesize type class instance"),
    ("FUNCTION_EXPECTED", "^function expected at"),
    ("GET_LOCAL_TAC_FAILED", "^get_local tactic failed"),
    ("INVALID_TAC_UNIFY_FAILED", "^invalid apply tactic, failed to unify"),
    ("INVALID_FIELD_NOTATION", "^invalid field notation"),
    ("INVALID_SIMP_LEMMA", "^invalid simplification lemma"),
    ("INVALID_TYPE_ASCRIPTION", "^invalid type ascription"),
    ("NO_EXTENSIONALITY_RULE", "^no applicable extensionality rule found"),
    ("NO_OVERLOADS", "^none of the overloads are applicable"),
    ("ONLY_KANDPI_SUPPORTED", "^only constants and Pi types are supported"),
    ("UNSOLVED_GOALS", "^tactic failed, there are unsolved goals"),
    ("CHANGE_FAILED", "^tactic.change failed"),
    ("TYPE_MISMATCH_APP", "^type mismatch at application"),
    ("INVALID_APP", "^invalid (.*) application,"),
    ("UNEXPECTED_TOKEN", "(.*)error: unexpected token"),
    ("FAILED_TO_SIMPLIFY", "(.*)failed to simplify"),
    (
        "MK_INSTANCE_FAILED_GENERATE",
        "^tactic.mk_instance failed to generate instance for",
    ),
    ("LINARITH_NO_CONTRADICTION", "^linarith failed to find a contradiction"),
    ("INFER_FAILED_UNK_VAR", "^infer type failed, unknown variable"),
    ("TAC_TIMEOUT", "^tactic timeout"),
    # these would also catch things that are before
    ("INVALID_WORD", r"^invalid (?P<word>\w+)"),
    ("X_EXPECTED", "(.*) expected"),
    ("X_TAC_FAILED", r"^(?P<word>\w+) tactic failed"),
    ("UNKNOWN_WORD", r"^unknown (?P<word>\w+)"),
    ("NEVER_GOT_RESULT", "^NEVER_GOT_RESULT"),
    ("FORBIDDEN_TOKEN", "^Forbidden token "),
    ("BIG_SUBGOAL", "^Big subgoal"),
    ("TOO_MANY_SUBGOALS", "^Too many subgoals"),
    ("TOO_MANY_METAVARS", "^Too many meta variables"),
    ("TOO_MANY_REPEATED_HYPS", "^Too many repeated hypothesis names."),
    ("TOO_MANY_INST", "^Too many instances"),
    ("ASYNC_CONST", "^Async constant"),
    ("OTHER_TIMEOUT", "timeout"),
]
lean_error_regexes = [re.compile(x) for n, x in lean_error_str]


def get_error(error: str):
    for name, regex in zip(lean_error_str, lean_error_regexes):
        if regex.match(error):
            return name[0]
    return "UNK_ERROR"


class LeanTactic(TokenizedTactic):
    def __init__(
        self,
        tactic: str,
        valid=True,
        malformed=False,
        error_msg=None,
        tokens: Optional[List[str]] = None,
        duration: Optional[float] = None,
    ):
        super(LeanTactic, self).__init__(
            tactic, valid, malformed, error_msg, tokens, duration
        )

    def tokenize(self) -> List[Token]:
        assert (not self.malformed) and lean_tok.tokenizer is not None
        return lean_tok.tokenizer.encode(self._tactic)

    def tokenize_error(self) -> List[Token]:
        assert self.error_msg is not None and lean_tok.tokenizer is not None, (
            self.error_msg,
            repr(lean_tok.tokenizer),
        )
        return lean_tok.tokenizer.encode(self.error_msg)

    @staticmethod
    def from_tokens(tokens: List[Token]) -> "LeanTactic":
        assert lean_tok.tokenizer is not None
        try:
            if len(tokens) == 0:
                raise MalformedTactic(f"MALFORMED Empty tactic")
            if (tokens[0], tokens[-1]) != (B_CMD_WORD, E_CMD_WORD):
                raise MalformedTactic(f"MALFORMED Missing b_e cmd {' '.join(tokens)}")
            if _SPECIAL_WORDS.intersection(set(tokens[1:-1])):
                raise MalformedTactic(f"MALFORMED tactic {' '.join(tokens)}")
            return LeanTactic(lean_tok.tokenizer.decode(tokens[1:-1]))
        except MalformedTactic as e:
            return LeanTactic.from_error(error_msg=str(e), tokens=tokens)

    @staticmethod
    def from_error(error_msg: str, tokens: Optional[List[str]] = None) -> "LeanTactic":
        return LeanTactic(
            "", valid=False, error_msg=error_msg, tokens=tokens, malformed=True
        )

    def get_error_code(self) -> str:
        assert self.is_error() and self.error_msg is not None
        return get_error(self.error_msg)


class LeanTheorem(Theorem):
    """
    For now, store everything in conclusion. Goals are not being split in the env.
    """

    def to_dict(self, light=False):
        return {
            "conclusion": self.conclusion,
            "state": asdict(self.state) if not (light or self.state is None) else None,
            "context.namespaces": list(self.context.namespaces),
            "fingerprint": self.fingerprint,
            "past_tactics": [pt.to_dict() for pt in self.past_tactics],
        }

    def tokenize(self) -> List[Token]:
        assert lean_tok.tokenizer is not None
        context = [
            B_NS_WORD,
            *lean_tok.tokenizer.encode(" ".join(sorted(self.context.namespaces))),
            E_NS_WORD,
        ]
        return [
            B_GOAL_WORD,
            *context,
            *lean_tok.tokenizer.encode(self.conclusion),
            E_GOAL_WORD,
        ] + sum([[M_STACK_WORD, *tac.tokenize()] for tac in self.past_tactics], [])

    @classmethod
    def from_tokens(cls, tokens: List[Token]):
        assert type(tokens) is list and lean_tok.tokenizer is not None

        def exception(s: str):
            return MalformedTheorem(f"{s}: {' '.join(tokens)}")

        if len(tokens) == 0 or (tokens[0], tokens[-1]) != (B_GOAL_WORD, E_GOAL_WORD):
            raise exception("invalid theorem delimitors")

        try:
            end_ns = tokens.index(E_NS_WORD)
        except ValueError:
            raise exception("Missing E_NS_WORD")
        if tokens[1] != B_NS_WORD:
            raise exception("Missing B_NS_WORD")
        ns_tok = tokens[2:end_ns]
        ns = lean_tok.tokenizer.decode(ns_tok)
        open_namespaces = set(ns.split())
        conclusion_tok = tokens[end_ns + 1 : -1]
        uniq_tok = set(conclusion_tok)
        special_words_without_newline = set(_SPECIAL_WORDS)
        special_words_without_newline.remove(NEWLINE_WORD)
        if special_words_without_newline.intersection(uniq_tok):
            raise exception(
                f"unexpected special words "
                f"({special_words_without_newline.intersection(uniq_tok)})"
            )
        conclusion = lean_tok.tokenizer.decode(conclusion_tok)
        return cls(
            conclusion, state=None, context=LeanContext(namespaces=open_namespaces)
        )

    @classmethod
    def from_dict(cls, data):
        state = LeanState(**data["state"]) if data["state"] is not None else None
        context = LeanContext(namespaces=set(data["context.namespaces"]))
        tactics = (
            [LeanTactic.from_dict(pt) for pt in data["past_tactics"]]
            if data["past_tactics"] is not None
            else None
        )
        return cls(
            data["conclusion"],
            context=context,
            state=state,
            past_tactics=tactics,
            fingerprint=data["fingerprint"],  # crash if 'fingerprint' is not here
        )

    def __init__(
        self,
        conclusion: str,
        context: LeanContext,
        state: Optional[LeanState],
        past_tactics: Optional[List[LeanTactic]] = None,
        hyps=None,
        fingerprint: Optional[str] = None,
    ):
        assert type(conclusion) is str, conclusion
        assert hyps is None
        # Nodes sometimes are pretty printed similarly but are different
        # We have to give them a different hash in the tree to avoid wrong proofs
        # For example linarith can fail or succeed on two nodes that have equal pp but != pp_all

        super().__init__(conclusion, [], given_unique_str=fingerprint)
        self.state = state
        self.context = context
        self.fingerprint = fingerprint
        self.past_tactics = past_tactics or []

    def __repr__(self):
        return f"{self.__class__.__name__}(conclusion={self.conclusion!r}, context={self.context})"
