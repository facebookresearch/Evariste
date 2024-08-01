# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Dict, Optional

from evariste.backward.graph import Theorem, Tactic, Token


class MyTheorem(Theorem):
    def __init__(self, conclusion: str):
        super().__init__(conclusion, [])

    def tokenize(self):
        pass

    @classmethod
    def from_tokens(cls, tokens):
        raise NotImplementedError

    def to_dict(self, light=False):
        pass

    @classmethod
    def from_dict(cls, data):
        pass

    def __repr__(self):
        return self.conclusion


class MyTactic(Tactic):
    def __init__(self, unique_str: str):
        super().__init__(True, unique_str, error_msg=None, malformed=False)

    def tokenize(self):
        pass

    @staticmethod
    def from_error(error_msg: str, tokens: Optional[List[str]] = None) -> "MyTactic":
        raise NotImplementedError

    @staticmethod
    def from_tokens(tokens):
        raise NotImplementedError

    def to_dict(self, light=False) -> Dict:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: Dict):
        raise NotImplementedError

    def tokenize_error(self) -> List[Token]:
        raise NotImplementedError
