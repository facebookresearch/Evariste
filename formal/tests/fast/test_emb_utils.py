# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List

import torch
import numpy as np

from evariste.model.data.dictionary import Dictionary
from evariste.model.embedder_utils import construct_sequences, create_graph_tensors

VOCAB_SIZE = 100
MIN_TOK = 50


def test_construct_embedding_sequences():
    embedded_statements_ = torch.randn(4, 5).float()
    statements_positions_ = [(0, 0), (0, 1), (0, 2), (1, 0)]
    sequences_len = torch.tensor([3, 1])

    x = construct_sequences(
        embedded_statements_, statements_positions_, sequences_len=sequences_len
    )
    assert x.shape == (2, 3, 5)
    for i, j in [(1, 1), (1, 2)]:
        assert x[i, j].abs().max().item() == 0
    for i, j in statements_positions_:
        assert x[i, j].abs().max().item() != 0


def _random_sample(
    rng: np.random.RandomState, n_nodes: int, children: List[int], include_goal: bool
):
    assert all(c < n_nodes for c in children)
    if include_goal:
        types = [2] + [1] * (n_nodes - 1)
    else:
        types = [1] * n_nodes
    return {
        "graph": [
            rng.randint(MIN_TOK, VOCAB_SIZE, size=10).tolist() for _ in range(n_nodes)
        ],
        "token": f"tok_{rng.randint(0, VOCAB_SIZE)}",
        "children": children,
        "types": types,
    }


def _dummy_dico():
    return Dictionary.create_from_vocab_counts(
        {f"tok_{i}": i for i in range(VOCAB_SIZE)}
    )


def test_create_graph_tensors():
    dico = _dummy_dico()
    rng = np.random.RandomState(42)
    sample_1 = _random_sample(rng, n_nodes=3, children=[0, 2, 0], include_goal=False)
    sample_2 = _random_sample(rng, n_nodes=6, children=[0, 3, 5], include_goal=False)
    tensors = create_graph_tensors([sample_1, sample_2], dico)
    assert np.allclose(tensors["graph_ids"], [0, 0, 0, 1, 1, 1])
    assert np.allclose(
        tensors["dec_ptr_pos"], [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]
    )
    assert np.allclose(tensors["dec_ptr_tgt"], [0, 2, 0, 0, 3, 5])
    assert tensors["enc_ptr_pos"] == [(0, i) for i in range(3)] + [
        (1, i) for i in range(6)
    ]
    assert np.allclose(tensors["enc_typ_idx"], [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]])


def test_create_graph_tensors_with_goal():
    dico = _dummy_dico()
    rng = np.random.RandomState(42)
    sample_1 = _random_sample(rng, n_nodes=3, children=[0, 2, 0], include_goal=True)
    sample_2 = _random_sample(rng, n_nodes=6, children=[0, 3, 5], include_goal=True)
    tensors = create_graph_tensors([sample_1, sample_2], dico)
    assert np.allclose(tensors["graph_ids"], [0, 0, 0, 1, 1, 1])
    assert np.allclose(
        tensors["dec_ptr_pos"], [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]
    )
    assert np.allclose(tensors["dec_ptr_tgt"], [0, 2, 0, 0, 3, 5])

    assert tensors["enc_ptr_pos"] == [(0, i) for i in range(3)] + [
        (1, i) for i in range(6)
    ]
    assert np.allclose(tensors["enc_typ_idx"], [[2, 1, 1, 0, 0, 0], [2, 1, 1, 1, 1, 1]])
