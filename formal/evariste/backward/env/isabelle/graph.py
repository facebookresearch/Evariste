# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, List, Dict, Set
from dataclasses import dataclass, asdict

from evariste.logger import create_logger
from evariste.model.data.dictionary import (
    B_CMD_WORD,
    E_CMD_WORD,
    B_GOAL_WORD,
    E_GOAL_WORD,
    B_NS_WORD,
    E_NS_WORD,
    SPECIAL_WORDS,
    NEWLINE_WORD,
)
from evariste.backward.graph import Tactic, Theorem, Token
from evariste.backward.graph import MalformedTactic, MalformedTheorem
from evariste.backward.env.tokenized_tactic import TokenizedTactic
import evariste.backward.env.isabelle.tokenizer as isabelle_tok

logger = create_logger(None)


_SPECIAL_WORDS = set(SPECIAL_WORDS)


@dataclass
class IsabelleState:
    session: str
    node_id: int
    n_subgoals: Optional[int] = None
    n_metavars: Optional[int] = None


@dataclass
class IsabelleContext:
    """
    Describes the context in which a theorem is declared.
    These will be passed down to children IsabelleTheorems, and used as additional input
    For now, only include the relative path, whether the source comes from afp or std, and the theorem declaration
    """

    # source = "afp" or "std"
    # source: str
    # relative path to the afp or std origin.
    # e.g., "thys/FunWithFunctions/FunWithFunctions.thy"
    relative_path: str
    # theorem declaration, without breakline characters or consecutive spaces
    # theorem_declaration: str


def get_error(error: str):
    # TODO: Make this error recognition better and more robust
    return error


class IsabelleTactic(TokenizedTactic):
    def __init__(
        self,
        tactic: str,
        valid=True,
        malformed=False,
        error_msg=None,
        tokens: Optional[List[str]] = None,
        duration: Optional[float] = None,
    ):
        super(IsabelleTactic, self).__init__(
            tactic, valid, malformed, error_msg, tokens, duration
        )

    def tokenize(self) -> List[Token]:
        assert (not self.malformed) and isabelle_tok.tokenizer is not None
        return isabelle_tok.tokenizer.encode(self._tactic)

    def tokenize_error(self) -> List[Token]:
        assert self.error_msg is not None and isabelle_tok.tokenizer is not None, (
            self.error_msg,
            repr(isabelle_tok.tokenizer),
        )
        return isabelle_tok.tokenizer.encode(self.error_msg)

    @staticmethod
    def from_tokens(tokens: List[Token]) -> "IsabelleTactic":
        assert isabelle_tok.tokenizer is not None
        try:
            if len(tokens) == 0:
                raise MalformedTactic("MALFORMED Empty tactic")
            if (tokens[0], tokens[-1]) != (B_CMD_WORD, E_CMD_WORD):
                raise MalformedTactic(f"MALFORMED Missing b_e cmd {' '.join(tokens)}")
            if _SPECIAL_WORDS.intersection(set(tokens[1:-1])):
                raise MalformedTactic(f"MALFORMED tactic {' '.join(tokens)}")
            return IsabelleTactic(isabelle_tok.tokenizer.decode(tokens[1:-1]))
        except MalformedTactic as e:
            return IsabelleTactic.from_error(error_msg=str(e), tokens=tokens)

    @staticmethod
    def from_error(
        error_msg: str, tokens: Optional[List[str]] = None
    ) -> "IsabelleTactic":
        return IsabelleTactic(
            "", valid=False, error_msg=error_msg, tokens=tokens, malformed=True
        )

    def get_error_code(self) -> str:
        assert self.is_error() and self.error_msg is not None
        return get_error(self.error_msg)


class IsabelleTheorem(Theorem):
    """
    Store everything in conclusion. Goals are not being split in the environment.
    """

    def __init__(
        self,
        conclusion: str,
        context: IsabelleContext,
        state: Optional[IsabelleState],
        hyps=None,
        fingerprint: Optional[str] = None,
    ):
        assert isinstance(conclusion, str)
        assert hyps is None

        super().__init__(conclusion, [], given_unique_str=fingerprint)
        self.state = state
        self.context = context
        self.fingerprint = fingerprint

    def __repr__(self):
        return f"{self.__class__.__name__}(conclusion={self.conclusion!r}, context={self.context})"

    def to_dict(self, light=False):
        return {
            "conclusion": self.conclusion,
            "state": asdict(self.state) if not (light or self.state is None) else None,
            "context.relative_path": self.context.relative_path,
            "fingerprint": self.fingerprint,
        }

    def tokenize(self) -> List[Token]:
        assert isabelle_tok.tokenizer is not None

        context = [
            B_NS_WORD,
            *isabelle_tok.tokenizer.encode(self.context.relative_path),
            E_NS_WORD,
        ]
        return [
            B_GOAL_WORD,
            *context,
            *isabelle_tok.tokenizer.encode(self.conclusion),
            E_GOAL_WORD,
        ]

    @classmethod
    def from_tokens(cls, tokens: List[Token]):
        assert isinstance(tokens, list) and isabelle_tok.tokenizer is not None

        def exception(s: str):
            return MalformedTheorem(f"{s}: {' '.join(tokens)}")

        if not tokens or (tokens[0], tokens[-1]) != (B_GOAL_WORD, E_GOAL_WORD):
            raise exception("invalid theorem delimiters")

        try:
            end_relative_path = tokens.index(E_NS_WORD)
        except ValueError:
            raise exception("missing E_NS_WORD")
        if tokens[1] != B_NS_WORD:
            raise exception("missing B_NS_WORD")

        relative_path_tok = tokens[2:end_relative_path]
        relative_path = isabelle_tok.tokenizer.decode(relative_path_tok)
        conclusions_tok = tokens[end_relative_path + 1 : -1]
        uniq_tok = set(conclusions_tok)
        special_words_without_newline = set(_SPECIAL_WORDS)
        special_words_without_newline.remove(NEWLINE_WORD)
        if special_words_without_newline.intersection(uniq_tok):
            raise exception(
                f"unexpected special words "
                f"({special_words_without_newline.intersection(uniq_tok)})"
            )
        conclusion = isabelle_tok.tokenizer.decode(conclusions_tok)
        return cls(
            conclusion, state=None, context=IsabelleContext(relative_path=relative_path)
        )

    @classmethod
    def from_dict(cls, data: Dict):
        state = IsabelleState(**data["state"]) if data["state"] is not None else None
        context = IsabelleContext(relative_path=data["context.relative_path"])
        return cls(
            data["conclusion"],
            context=context,
            state=state,
            fingerprint=data["fingerprint"],
        )
