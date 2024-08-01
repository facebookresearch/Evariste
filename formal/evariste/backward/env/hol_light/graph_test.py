# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest

from evariste.model.data.dictionary import (
    B_CMD_WORD,
    E_CMD_WORD,
    B_HYP_WORD,
    M_HYP_WORD,
    E_HYP_WORD,
    B_GOAL_WORD,
    E_GOAL_WORD,
    SPECIAL_WORDS,
)
from evariste.envs.hl.tokenizer import tokenize_hl, detokenize_hl
from evariste.backward.graph import MalformedTheorem
from evariste.backward.env.hol_light.graph import HLTactic, HLTheorem


@pytest.mark.slow
class TestHLTactic:
    def test_init(self):

        # test 1
        tactic1 = HLTactic("REWRITE_TAC[MULT_AC;LE_ADD]")
        tactics1 = [
            HLTactic("REWRITE_TAC[MULT_AC    ;LE_ADD]"),
            HLTactic("    REWRITE_TAC[MULT_AC   ;LE_ADD]   "),
        ]
        detok1 = "REWRITE_TAC [ MULT_AC ; LE_ADD ]"

        # test 2
        tactic2 = HLTactic("EXISTS_TAC `{a:real^1,b}`")
        tactics2 = [
            HLTactic("EXISTS_TAC     `{a:real^   1,b}`"),
            HLTactic("   EXISTS_TAC `{a:real   ^1,   b}`"),
        ]
        detok2 = "EXISTS_TAC `{ a : real ^ 1 , b }`"

        for tactic, tactics, detok in [
            (tactic1, tactics1, detok1),
            (tactic2, tactics2, detok2),
        ]:
            assert tactic.is_valid
            assert tactic.detokenized == detok
            assert all(t == tactic and t.detokenized == detok for t in tactics)

    def test_tokenize_detokenize(self):
        # test 1
        tactic1 = HLTactic("REWRITE_TAC[MULT_AC;LE_ADD]")
        tokens1 = [
            *tokenize_hl("REWRITE_TAC [ MULT_AC ; LE_ADD ]"),
        ]

        # test 2
        tactic2 = HLTactic("EXISTS_TAC `{a:real^1,b}`")
        tokens2 = [
            *tokenize_hl("EXISTS_TAC ` { a : real ^ 1 , b } `"),
        ]

        for tactic, tokens in [(tactic1, tokens1), (tactic2, tokens2)]:
            assert tactic.is_valid
            assert tactic.tokenize() == tokens
            assert HLTactic(detokenize_hl(tokens)).tokenize() == tokens
            assert HLTactic.from_tokens([B_CMD_WORD, *tokens, E_CMD_WORD]) == tactic

    def test_detokenize_fails(self):
        tokens = [
            B_CMD_WORD,
            *tokenize_hl("REWRITE_TAC [ MULT_AC ; LE_ADD ]"),
            E_CMD_WORD,
        ]
        for i, token in enumerate(tokens):
            if token in set(SPECIAL_WORDS):
                _tokens = tokens[:i] + tokens[i + 1 :]
                assert not HLTactic.from_tokens(_tokens).is_valid
        for i in range(len(tokens)):
            for token in SPECIAL_WORDS:
                _tokens = tokens[:i] + [token] + tokens[i:]
                assert not HLTactic.from_tokens(_tokens).is_valid


@pytest.mark.slow
class TestHLTheorem:
    def test_init(self):
        conclusion = "m * n DIV p <= (m * n) DIV p"
        hyps = [("h0", "~(p = 0)"), (None, "~(p = 1)"), (None, "a"), ("h1", "b")]
        theorem = HLTheorem(conclusion, hyps)
        assert HLTheorem(conclusion, hyps[::-1]) == theorem
        assert HLTheorem(conclusion, hyps[:1]) != theorem

        for conclusion in [
            "m * n DIV p <= (m * n) DIV p",
            "m * n DIV p<=(m * n) DIV p",
            "m * n DIV p <= (m*n) DIV p",
            "m*n DIV p<=(m*n) DIV p",
        ]:
            for c0 in ["~(p = 0)", "~(p=0)", "~   (p=  0)"]:
                for c1 in ["~(p = 1)", "~(p=1)", "~   (  p    =  1  )"]:
                    for shuffle in [False, True]:
                        hyps = [
                            ("h0", c0),
                            (None, c1),
                            (None, "a"),
                            ("h1", "b"),
                        ]
                        hyps = hyps[::-1] if shuffle else hyps
                        _theorem = HLTheorem(conclusion, hyps)
                        assert _theorem == theorem
                        assert _theorem.conclusion == "m * n DIV p <= ( m * n ) DIV p"
                        assert _theorem.hyps[0] == (None, "a")
                        assert _theorem.hyps[1] == (None, "~ ( p = 1 )")
                        assert _theorem.hyps[2] == ("h0", "~ ( p = 0 )")
                        assert _theorem.hyps[3] == ("h1", "b")

    def test_tokenize_detokenize(self):
        for hyp_name in [None, "bla"]:
            conclusion = "m * n DIV p <= (m * n) DIV p"
            hyps = [(hyp_name, "~(p = 0)")]
            theorem = HLTheorem(conclusion, hyps)
            tokens = [
                B_GOAL_WORD,
                *[
                    B_HYP_WORD,
                    hyp_name,
                    M_HYP_WORD,
                    *tokenize_hl(hyps[0][1]),
                    E_HYP_WORD,
                ],
                *tokenize_hl(conclusion),
                E_GOAL_WORD,
            ]
            tokens = [x for x in tokens if x is not None]
            assert theorem.tokenize() == tokens
            assert HLTheorem.from_tokens(tokens) == theorem

    def test_detokenize_fails(self):
        conclusion = "m * n DIV p <= (m * n) DIV p"
        hyps = [("h0", "~(p = 0)")]
        tokens = [
            B_GOAL_WORD,
            *[B_HYP_WORD, hyps[0][0], M_HYP_WORD, *tokenize_hl(hyps[0][1]), E_HYP_WORD],
            *tokenize_hl(conclusion),
            E_GOAL_WORD,
        ]
        for i, token in enumerate(tokens):
            if token in set(SPECIAL_WORDS):
                _tokens = tokens[:i] + tokens[i + 1 :]
                with pytest.raises(MalformedTheorem):
                    assert not HLTheorem.from_tokens(_tokens)
        for i in range(len(tokens)):
            for token in SPECIAL_WORDS:
                _tokens = tokens[:i] + [token] + tokens[i:]
                with pytest.raises(MalformedTheorem):
                    assert not HLTheorem.from_tokens(_tokens)
