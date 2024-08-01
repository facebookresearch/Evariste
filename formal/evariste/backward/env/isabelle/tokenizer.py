# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from evariste.clusters.utils import clusterify_path
from abc import ABC, abstractmethod
from typing import Optional, List, Set
import os
import re
import youtokentome as yttm


tokenizer: Optional["IsabelleTokenizer"] = None


class IsabelleTokenizer(ABC):
    @abstractmethod
    def encode(self, sentence: str) -> List[str]:
        pass

    @abstractmethod
    def decode(self, tokens: List[str]) -> str:
        pass

    @staticmethod
    def build(version: Optional[str]) -> "IsabelleTokenizer":
        global tokenizer
        print(f"Building Isabelle tokenizer ({version}) ...")
        if version == "bpe_arxiv_lean_utf8_20k":
            tokenizer = IsabelleTokenizerBPEArxivUTF8(bpe_path="", split_digits=False,)
        else:
            raise RuntimeError(f"Unknown tokenizer: {version}")
        assert tokenizer is not None
        return tokenizer


class IsabelleTokenizerBPEArxivUTF8(IsabelleTokenizer):
    def __init__(self, bpe_path: str, split_digits: bool, alone_digits: bool = False):
        bpe_path = clusterify_path(bpe_path)
        assert os.path.isfile(bpe_path)
        self.bpe: Optional[yttm.BPE] = None  # Lazy for pickling purposes
        self.bpe_path = bpe_path
        self.split_digits = split_digits
        self.alone_digits = alone_digits
        self.vocab: Set[str] = set()
        assert not alone_digits or split_digits

    def init_bpe(self):
        if self.bpe is None:
            self.bpe = yttm.BPE(model=self.bpe_path)
            self.vocab = set(self.bpe.vocab())
            assert "🐙" not in self.vocab
            print(
                f"Initialized BPE from {self.bpe_path} -- "
                f"Found {len(self.vocab)} words."
            )

    def split_unk(self, tokens: List[str]) -> List[str]:
        res: List[str] = []
        for tok in tokens:
            if tok in self.vocab:
                res.append(tok)
            else:
                res.extend(list(tok))
        return res

    def encode(self, s: str) -> List[str]:
        self.init_bpe()
        assert self.bpe is not None
        s = s.replace("\n", "🐙 ")
        tokens = self.bpe.encode(s, yttm.OutputType.SUBWORD)
        if self.split_digits:
            str_tokens = " ".join(tokens)
            if self.alone_digits:
                str_tokens = re.sub(r"(▁?[0-9]+)", r" \1 ", str_tokens).strip()
            str_tokens = re.sub(r"(?<=[0-9])(?=[0-9])", " ", str_tokens)
            tokens = str_tokens.split()

        # some UNKs may occur because of the custom digit split.
        # split them at character level.
        tokens = self.split_unk(tokens)

        # # debug sanity check (\s and not " " because we lose the \t)
        # detok = self.decode(tokens)
        # assert detok == re.sub(r"\s+", " ", s), (detok, re.sub(r"\s+", " ", s))

        tokens = ["<NEWLINE>" if tok == "🐙" else tok for tok in tokens]
        return tokens

    def decode(self, tokens: List[str]) -> str:
        s = "".join(tokens).replace("▁", " ").strip()
        s = s.replace("<NEWLINE> ", "\n")
        return s
