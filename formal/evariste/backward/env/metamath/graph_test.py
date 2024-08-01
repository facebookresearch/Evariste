# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest

from evariste.model.data.dictionary import (
    B_HYP_WORD,
    E_HYP_WORD,
    B_GOAL_WORD,
    E_GOAL_WORD,
    B_CMD_WORD,
    E_CMD_WORD,
    B_SUBST_WORD,
    M_SUBST_WORD,
    E_SUBST_WORD,
    SPECIAL_WORDS,
)
from evariste.backward.graph import MalformedTheorem
from evariste.backward.env.metamath.graph import MMTactic, MMTheorem

# fmt: off
TACTIC_TOKENS = [
    B_CMD_WORD, 'syl',
    *sum([
        [B_SUBST_WORD, x, M_SUBST_WORD, x, E_SUBST_WORD]
        for x in ['ch', 'ph', 'ps']
    ], []),
    E_CMD_WORD
]
TH_CONCLUSION = '|- ( ph -> ch )'
TH_HYPS = [
    '|- ( ph -> ( ps -> ch ) )',
    '|- ps',
]
TH_TOKENS = [
    B_GOAL_WORD,
    B_HYP_WORD, *TH_HYPS[0].split(), E_HYP_WORD,
    B_HYP_WORD, *TH_HYPS[1].split(), E_HYP_WORD,
    *TH_CONCLUSION.split(),
    E_GOAL_WORD
]
# fmt: on


class TestMMTactic:
    def test_init(self):
        tactic = MMTactic("syl", {"ch": "ch", "ph": "ph", "ps": "ps"})
        assert tactic.is_valid
        assert tactic.label == "syl"
        assert tactic.subs == {x: x for x in ["ph", "ps", "ch"]}
        tactics = [
            MMTactic("syl", {"ch": "ch", "ph": "ph", "ps": "ps"}),
            MMTactic("syl", {"ch": "ch", "ps": "ps", "ph": "ph"}),
            MMTactic("syl", {"ps": "ps", "ch": "ch", "ph": "ph"}),
        ]
        assert all(t == tactic for t in tactics)
        assert MMTactic("syl", {"ch": "ps", "ph": "ph", "ps": "ch"}) != tactic
        assert MMTactic("syl", {"ch": "ch", "ph": "ph", "ps": "BLA"}) != tactic

    def test_tokenize_detokenize(self):
        tactic = MMTactic("syl", {"ch": "ch", "ph": "ph", "ps": "ps"})
        assert tactic.is_valid
        assert tactic.tokenize() == TACTIC_TOKENS
        assert MMTactic.from_tokens(TACTIC_TOKENS) == tactic

    def test_detokenize_fails(self):
        for i, token in enumerate(TACTIC_TOKENS):
            if token in set(SPECIAL_WORDS):
                _tokens = TACTIC_TOKENS[:i] + TACTIC_TOKENS[i + 1 :]
                assert not MMTactic.from_tokens(_tokens).is_valid
        for i in range(len(TACTIC_TOKENS)):
            for token in SPECIAL_WORDS:
                _tokens = TACTIC_TOKENS[:i] + [token] + TACTIC_TOKENS[i:]
                assert not MMTactic.from_tokens(_tokens).is_valid


class TestMMTheorem:
    def test_init(self):
        theorem = MMTheorem(hyps=[(None, h) for h in TH_HYPS], conclusion=TH_CONCLUSION)
        assert theorem.conclusion == TH_CONCLUSION
        assert [h for _, h in theorem.hyps] == sorted(TH_HYPS)
        # fmt: off
        swapped_hyps = [
            B_GOAL_WORD,
            B_HYP_WORD, *TH_HYPS[1].split(), E_HYP_WORD,
            B_HYP_WORD, *TH_HYPS[0].split(), E_HYP_WORD,
            *TH_CONCLUSION.split(),
            E_GOAL_WORD
        ]
        # fmt: on
        assert MMTheorem.from_tokens(swapped_hyps) == theorem

    def test_tokenize_detokenize(self):
        theorem1 = MMTheorem(
            hyps=[(None, h) for h in TH_HYPS], conclusion=TH_CONCLUSION
        )
        theorem2 = MMTheorem.from_tokens(TH_TOKENS)
        assert theorem1 == theorem2
        assert theorem1.conclusion == TH_CONCLUSION
        assert theorem2.conclusion == TH_CONCLUSION
        assert theorem1.hyps == theorem2.hyps
        assert theorem1.tokenize() == TH_TOKENS
        assert theorem2.tokenize() == TH_TOKENS

    def test_detokenize_fails(self):
        # Any of the reserved words can be missing
        for i, token in enumerate(TH_TOKENS):
            if token in set(SPECIAL_WORDS):
                _tokens = TH_TOKENS[:i] + TH_TOKENS[i + 1 :]
                with pytest.raises(MalformedTheorem):
                    MMTheorem.from_tokens(_tokens)
        for i in range(len(TH_TOKENS)):
            for token in SPECIAL_WORDS:
                _tokens = TH_TOKENS[:i] + [token] + TH_TOKENS[i:]
                with pytest.raises(MalformedTheorem):
                    MMTheorem.from_tokens(_tokens)
