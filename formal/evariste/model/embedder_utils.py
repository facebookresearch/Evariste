# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Tuple
import torch

from ..model.data.dictionary import EOS_WORD, EMB_WORD

TYPE_PAD_IDX = 0
PTR_PAD_IDX = -1


def construct_sequences(
    embedded_statements: torch.Tensor,
    positions: List[Tuple[int, int]],
    sequences_len: torch.Tensor,
    full: bool = True,
) -> torch.Tensor:
    """
    Construct sequences of embeddings that will be fed to the encoder.

    params:
    embedded_statements: torch.Tensor
        (n_statements, emb_dim)
        Batch of Embedded statements (can be assertion or hypothesis).
    positions: List[Tuple[int, int]]
        give the (x, y) position in the sequences batch
        for each statement embedding in embedded_statements batch
    sequences_len: torch.Tensor
        (n_sequences,)
        Length of each sequence
    full: bool
        Whether the tensor must be full (without "blanks" between embeddings)
    return torch.Tensor
        Batch of reconstructed sequences of embeddings
    """
    n_statements, emb_dim = embedded_statements.size()
    bs = len(sequences_len)
    max_len = max(sequences_len)

    # input check
    assert len(positions) == n_statements
    assert not full or n_statements == sequences_len.sum().item()
    assert not full or {
        (i, j) for i, l in enumerate(sequences_len) for j in range(l)
    } == set(positions)

    idx = torch.LongTensor(positions)
    sequences = embedded_statements.new_zeros((bs, max_len, emb_dim))
    sequences[idx[:, 0], idx[:, 1]] = embedded_statements

    return sequences


def create_padded_batch(x, xlen, pad_index):
    """
    Create a padded batch.
    """
    # input check
    bs = len(xlen)
    max_len = xlen.max()
    assert len(x) == bs
    assert max(len(y) for y in x) == max_len, (x, xlen)

    # create padded tensor
    batch = torch.full((bs, max_len), pad_index, dtype=torch.long)
    for i in range(bs):
        assert len(x[i]) == xlen[i]
        batch[i, : xlen[i]] = torch.LongTensor(x[i])

    return batch


def create_graph_tensors(items, dico):
    """
    Create mask tensors for a step (create a new node given a graph).
    Each item is a dictionary that contains:
        - graph
            - the list of terms in the graph
        - output: the command to generate the next node. can be:
            - </s> f_variable </s>
            - </s> theorem_label hyp_0_node ... hyp_{n-1}_node </s>
        - parents
            - a list (potentially empty) of parent node hypotheses
    """

    emb_len = []  # statement lengths
    enc_len = []  # graph sizes
    dec_len = []  # output command lengths

    emb_tok_idx = []  # embedder token IDs
    dec_tok_idx = []  # decoder token IDs
    enc_ptr_pos = []  # encoder pointer positions
    dec_ptr_pos = []  # decoder pointer positions
    dec_ptr_tgt = []  # IDs of parent nodes (local graph ID)
    graph_ids = []  # seq_id for each pointer

    enc_typ_idx = []  # types IDs fed to the encoder

    for seq_id, item in enumerate(items):

        # size of the graph
        n_nodes = len(item["graph"])
        n_children = len(item["children"])
        assert n_nodes >= 0

        # update embedder / encoder / decoder sequence lengths
        emb_len += [len(s) for s in item["graph"]]
        enc_len.append(n_nodes)
        dec_len.append(2 + n_children)  # + 2 because of </s> + token prefixes

        # embedder and decoder input tokens
        emb_tok_idx += item["graph"]
        dec_seq = [EOS_WORD, item["token"]] + [EMB_WORD] * n_children
        dec_tok_idx.append([dico.index(token) for token in dec_seq])

        # encoder input pointers and types
        # TODO: rename? it's more embedding positions in encoder input sequence
        enc_ptr_pos += [(seq_id, i) for i in range(n_nodes)]
        enc_typ_idx.append(item["types"])

        # decoder input tokens / pointers
        # i + 2 in dec_ptr_pos because of "</s> LABEL" prefixes
        assert all(0 <= pid < n_nodes for pid in item["children"])
        dec_ptr_pos += [(seq_id, i + 2) for i in range(len(item["children"]))]
        dec_ptr_tgt += item["children"]
        graph_ids.extend([seq_id] * len(item["children"]))

    # input lengths
    emb_len = torch.LongTensor(emb_len)
    enc_len = torch.LongTensor(enc_len)
    dec_len = torch.LongTensor(dec_len)

    # embedder and decoder tokens input batch
    emb_tok_idx = create_padded_batch(emb_tok_idx, emb_len, dico.pad_index)
    dec_tok_idx = create_padded_batch(dec_tok_idx, dec_len, dico.pad_index)
    enc_typ_idx = create_padded_batch(enc_typ_idx, enc_len, TYPE_PAD_IDX)
    assert (dec_tok_idx == dico.emb_index).sum() == len(dec_ptr_pos) == len(dec_ptr_tgt)

    return {
        "emb_len": emb_len,
        "enc_len": enc_len,
        "dec_len": dec_len,
        "emb_tok_idx": emb_tok_idx,
        "dec_tok_idx": dec_tok_idx,
        "enc_ptr_pos": enc_ptr_pos,
        "dec_ptr_pos": dec_ptr_pos,
        "dec_ptr_tgt": torch.LongTensor(dec_ptr_tgt),
        "graph_ids": torch.LongTensor(graph_ids),
        "enc_typ_idx": enc_typ_idx,
    }


def mask_from_seq_len(seq_len):
    alen = torch.arange(seq_len.max().item(), dtype=torch.long, device=seq_len.device)
    mask = alen[None] < seq_len[:, None]
    return mask


def gather_with_mask(source, index, index_mask):
    out = source.new_zeros(size=list(index.shape) + [source.size(2)])
    cols = index[index_mask]
    rows = row_ids_from_mask(index_mask)
    out[index_mask] = source[rows, cols]
    return out


def row_ids_from_mask(mask):
    """
    given a 2d mask, return a 1d sequence where the i th element is the row_id of
    the i th non masked item in the mask.
    """
    seq_ids = torch.arange(mask.size(0)).unsqueeze(-1).expand_as(mask)[mask]
    return seq_ids
