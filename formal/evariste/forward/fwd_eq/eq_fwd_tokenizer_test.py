# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import Counter

import pytest

from evariste.forward.fwd_eq.conftest import ForwardGraphsAndTactics
from evariste.forward.fwd_eq.eq_fwd_tokenizer import EqFwdTokenizer
from evariste.model.data.dictionary import B_HYP_WORD, E_HYP_WORD, B_NODE_WORD


def test_tokenize_graph(eq_fwd_graphs: ForwardGraphsAndTactics):
    fwd_tokenizer = EqFwdTokenizer(is_generation=False)
    for graph, _ in eq_fwd_graphs:
        assert graph.fwd_goal.global_hyps is not None
        assert graph.generated_thms is not None
        result = fwd_tokenizer.tokenize_graph(graph)
        counts = Counter(result)
        print(" ".join(result))
        assert counts[B_HYP_WORD] == 0
        assert counts[E_HYP_WORD] == 0
        assert counts[B_NODE_WORD] == len(graph.fwd_goal.global_hyps) + len(
            graph.generated_thms
        )


def test_tokenize_graph_gen(eq_gen_graphs: ForwardGraphsAndTactics):
    fwd_tokenizer = EqFwdTokenizer(is_generation=True)
    for graph, _ in eq_gen_graphs:
        assert graph.fwd_goal.global_hyps is not None
        assert graph.generated_thms is not None
        result = fwd_tokenizer.tokenize_graph(graph)
        print(" ".join(result))
        counts = Counter(result)
        assert counts[B_HYP_WORD] == 0
        assert counts[E_HYP_WORD] == 0
        assert counts[B_NODE_WORD] == len(graph.fwd_goal.global_hyps) + len(
            graph.generated_thms
        )


def test_fail_when_wrong_gen_bool(
    eq_fwd_graphs: ForwardGraphsAndTactics, eq_gen_graphs: ForwardGraphsAndTactics,
):
    fwd_tokenizer = EqFwdTokenizer(is_generation=False)
    for graph, _ in eq_gen_graphs:
        with pytest.raises(AssertionError):
            _ = fwd_tokenizer.tokenize_graph(graph)

    fwd_tokenizer = EqFwdTokenizer(is_generation=True)
    for graph, _ in eq_fwd_graphs:
        with pytest.raises(AssertionError):
            _ = fwd_tokenizer.tokenize_graph(graph)


def test_tok_parse_command(eq_fwd_graphs: ForwardGraphsAndTactics):
    fwd_tokenizer = EqFwdTokenizer(is_generation=False)
    for graph, fwd_tactic in eq_fwd_graphs:
        command = fwd_tokenizer.tokenize_command(fwd_tactic)
        obtained_fwd_tactic = fwd_tokenizer.detokenize_command(command, graph)
        assert obtained_fwd_tactic == fwd_tactic


def test_tok_parse_command_gen(eq_gen_graphs: ForwardGraphsAndTactics):
    fwd_tokenizer = EqFwdTokenizer(is_generation=True)
    for graph, fwd_tactic in eq_gen_graphs:
        command = fwd_tokenizer.tokenize_command(fwd_tactic)
        obtained_fwd_tactic = fwd_tokenizer.detokenize_command(command, graph)
        assert obtained_fwd_tactic == fwd_tactic
