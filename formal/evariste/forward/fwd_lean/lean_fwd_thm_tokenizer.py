# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
from dataclasses import dataclass
from typing import List, Dict

from evariste.backward.env.lean.graph import LeanTheorem, LeanContext, _SPECIAL_WORDS
from evariste.backward.env.lean.tokenizer import LeanTokenizer
from evariste.backward.graph import MalformedTheorem
from evariste.datasets import LeanDatasetConf
from evariste.model.data.dictionary import (
    B_NS_WORD,
    E_NS_WORD,
    NEWLINE_WORD,
    REPEAT_GOAL_WORD,
    GOAL_STATEMENT_WORD,
)
from params import Params


def hyp_tok(i: int) -> str:
    assert i <= 100
    return f"<HYP_{i}>"


@dataclass
class CompressCfg(Params):
    # if true, compress hyps by using special tokens to repeat goals hyps.
    compress_hyps: bool = False
    compress_goal_statement: bool = False
    hyp_tok_in_goal: bool = False
    repeat_goal: bool = False

    def _check_and_mutate_args(self):
        self.check()

    def check(self):
        if self.hyp_tok_in_goal or self.repeat_goal or self.compress_goal_statement:
            assert self.compress_hyps

        if not self.compress_hyps:
            assert not self.hyp_tok_in_goal
            assert not self.repeat_goal
            assert not self.compress_goal_statement


class LeanFwdThmTokenizer:
    def __init__(
        self,
        lean_tok: LeanTokenizer,
        dataset: LeanDatasetConf,
        compress_cfg: CompressCfg,
    ):
        assert lean_tok is not None
        self.lean_tok = lean_tok
        self.dataset = dataset
        self.compress_cfg = compress_cfg

    def _get_hyp2tok(self, goal: LeanTheorem) -> Dict[str, str]:
        assert "⊢" in goal.conclusion, goal.conclusion
        goal_hyps, goal_statement = goal.conclusion.split("⊢")
        goal_hyps = goal_hyps.splitlines(True)  # keeping \n in hyps
        hyp2tok = {}
        for i, hyp in enumerate(goal_hyps):
            if hyp in hyp2tok:
                # repeated hyp
                continue
            hyp2tok[hyp] = hyp_tok(i)
        if self.compress_cfg.compress_goal_statement:
            hyp2tok[f"⊢{goal_statement}"] = GOAL_STATEMENT_WORD
        return hyp2tok

    def encode_thm(self, thm: LeanTheorem, goal: LeanTheorem) -> List[str]:
        # for the moment this object should exists only if we compress hyps
        assert self.compress_cfg.compress_hyps

        return self._encode(thm, goal, is_goal=False)

    def _encode(self, thm: LeanTheorem, goal: LeanTheorem, is_goal: bool) -> List[str]:
        if (
            thm.conclusion == goal.conclusion
            and not is_goal
            and self.compress_cfg.repeat_goal
        ):
            return [REPEAT_GOAL_WORD]
        hyp2tok = self._get_hyp2tok(goal)
        splitted = thm.conclusion.split("⊢")
        assert len(splitted) == 2, (splitted, thm)
        hyps, statement = splitted
        statement = f"⊢{statement}"
        hyps = hyps.splitlines(True)  # keeping \n in hyps
        if not self.dataset.pp_full_names:
            context = [
                B_NS_WORD,
                *self.lean_tok.encode(" ".join(sorted(thm.context.namespaces))),
                E_NS_WORD,
            ]
        else:
            context = []
        tokens = []
        buffer = []

        def _empty_buffer_and_fill_toks():
            if not buffer:
                return
            joined = "".join(buffer)
            toks = self.lean_tok.encode(joined)
            tokens.extend(toks)
            buffer[:] = []

        for hyp in hyps:
            if hyp in hyp2tok:
                _empty_buffer_and_fill_toks()
                tokens.append(hyp2tok[hyp])
                if is_goal:
                    # we still add hyp tokens in goal
                    buffer.append(hyp)
            else:
                buffer.append(hyp)

        if (
            self.compress_cfg.compress_goal_statement
            and statement in hyp2tok
            and not is_goal
        ):
            _empty_buffer_and_fill_toks()
            tokens.append(hyp2tok[statement])
        else:
            buffer.append(statement)
            _empty_buffer_and_fill_toks()
        return [
            *context,
            *tokens,
        ]

    def decode_thm(self, tokens: List[str], goal: LeanTheorem) -> LeanTheorem:
        if self.compress_cfg.repeat_goal and tokens == [REPEAT_GOAL_WORD]:
            generated = copy.deepcopy(goal)
            generated.state = None
            return generated
        # for the moment this object should exists only if we compress hyps
        assert self.compress_cfg.compress_hyps

        def error(s):
            return MalformedTheorem(f"{s}: {' '.join(tokens)}")

        special_words_without_newline = set(_SPECIAL_WORDS)
        special_words_without_newline.remove(NEWLINE_WORD)

        if len(tokens) == 0:
            raise error("No tokens")

        if not self.dataset.pp_full_names:
            try:
                end_ns = tokens.index(E_NS_WORD)
            except ValueError:
                raise error("Missing E_NS_WORD")
            if tokens[0] != B_NS_WORD:
                raise error("Missing B_NS_WORD")
            ns_tok = tokens[1:end_ns]
            ns = self.lean_tok.decode(ns_tok)
            open_namespaces = set(ns.split())
            context = LeanContext(namespaces=open_namespaces)
            tokens = tokens[end_ns + 1 :]
        else:
            context = LeanContext(namespaces=set())

        if len(tokens) == 0:
            raise error("No tokens")

        tok2hyp = {tok: hyp for hyp, tok in self._get_hyp2tok(goal).items()}

        detokenized = []
        buffer = []

        def _empty_buffer_and_fill_detokenized():
            if not buffer:
                return
            uniq_tok = set(buffer)
            if special_words_without_newline.intersection(uniq_tok):
                raise error(
                    f"unexpected special words "
                    f"({special_words_without_newline.intersection(uniq_tok)})"
                )

            trailing_newline = False
            if buffer[-1] == NEWLINE_WORD:
                trailing_newline = True  # not handled correctly by tokenizer since it
                # removes trailing whitespace
                buffer.pop(-1)

            decoded = self.lean_tok.decode(buffer)
            detokenized.append(decoded)
            if trailing_newline:
                detokenized.append("\n")
            buffer[:] = []

        for tok in tokens:
            if tok in tok2hyp:

                if tok == GOAL_STATEMENT_WORD:
                    assert self.compress_cfg.compress_goal_statement

                _empty_buffer_and_fill_detokenized()
                detokenized.append(tok2hyp[tok])
            elif tok.startswith("<HYP_"):
                raise error(f"{len(tok2hyp)} hyps in goal, and detected token {tok}")
            else:
                buffer.append(tok)
        _empty_buffer_and_fill_detokenized()

        concl = "".join(detokenized)

        return LeanTheorem(conclusion=concl, state=None, context=context)

    def encode_goal(self, goal: LeanTheorem) -> List[str]:
        # for the moment this object should exists only if we compress hyps
        assert self.compress_cfg.hyp_tok_in_goal
        return self._encode(goal, goal, is_goal=True)
