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
    SPECIAL_WORDS,
    NEWLINE_WORD,
)
from evariste.backward.graph import Tactic, Theorem, Token
from evariste.backward.graph import MalformedTactic, MalformedTheorem


class TokenizedTactic(Tactic):
    def __init__(
        self,
        tactic: str,
        valid=True,
        malformed=False,
        error_msg=None,
        tokens: Optional[List[str]] = None,
        duration: Optional[float] = None,
    ):
        assert type(tactic) is str
        assert valid == (error_msg is None)
        assert valid == (tokens is None)
        self.is_valid = valid
        self._tactic = tactic
        if not malformed:
            unique_str = self._tactic
        else:
            assert tokens is not None
            unique_str = " ".join(tokens)
        Tactic.__init__(
            self, valid, unique_str=unique_str, error_msg=error_msg, malformed=malformed
        )

        # set by env for usage stats
        self.uses_theorems: Set[int] = set()
        self.n_theorems: int = -1
        self.duration: Optional[float] = duration

    def __repr__(self):
        return self._tactic if self.is_valid else "INVALID_TACTIC"

    def to_dict(self, light=True):
        if self.is_valid:
            assert self.error_msg is None
            return {"str": self._tactic}
        else:
            assert self.error_msg is not None
            return {"error_msg": self.error_msg, "str": self._tactic}

    @classmethod
    def from_dict(cls, data):
        valid = "error_msg" not in data
        error_msg = data.get("error_msg", None)
        tokens = None if valid else []  # tokens are not dumped -> use empty list
        return cls(tactic=data["str"], valid=valid, error_msg=error_msg, tokens=tokens)
