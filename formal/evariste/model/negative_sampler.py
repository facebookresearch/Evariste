# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import defaultdict
from logging import getLogger
from typing import List, Dict
import numpy as np
import torch
import torch.distributed as dist

from evariste.model.data.dictionary import Dictionary
from evariste.model.utils import to_cuda, batch_sequences, get_knn_pytorch, get_embs


logger = getLogger()


class NegativeSampler:
    """
    Sample random theorem sequences for negative sampling training.
    """

    def __init__(
        self,
        labels: List[str],
        sequences_lst: List[List[str]],
        dico: Dictionary,
        worst_offenders: float,
        dim: int,
        fp16: bool,
        tf_build_emb: str,
        tokens_per_batch: int = 10000,
    ):

        assert type(worst_offenders) is float and 0 <= worst_offenders <= 1
        assert type(labels) is list and all(type(x) is str for x in labels)
        assert type(sequences_lst) is list and all(
            type(x) is list for x in sequences_lst
        )
        assert len(labels) == len(sequences_lst)
        assert tf_build_emb in ["first", "mean", "max"]

        logger.info(f"Indexing {len(sequences_lst)} sequences...")

        # sort sequences by length
        lab_seqs = sorted(list(zip(labels, sequences_lst)), key=lambda x: len(x[1]))
        labels = [label for label, _ in lab_seqs]
        sequences = [" ".join(seq) for _, seq in lab_seqs]

        # id2seq / seq2id
        n_dup = 0
        self.seq2id: Dict[str, int] = {}
        self.seq2labels = defaultdict(list)  # can be duplicates
        for label, seq in zip(labels, sequences):
            self.seq2labels[seq].append(label)
            if seq in self.seq2id:
                # logger.warning(f"Duplicated sequence: {seq}")
                n_dup += 1
                continue
            sid = len(self.seq2id)
            self.seq2id[seq] = sid
        self.id2seq = {i: s for s, i in self.seq2id.items()}
        self.id2tensor = {
            i: torch.LongTensor([dico.index(w) for w in s.split()])
            for i, s in self.id2seq.items()
        }
        assert len(self.id2seq) == len(self.seq2id) == len(self.id2tensor)
        logger.info(f"Indexed {len(self.id2seq)} sequences. Found {n_dup} duplicates.")

        self.size = len(self.id2seq)
        self.dtype = torch.half if fp16 else torch.float
        self.dico = dico
        self.dim = dim
        self.worst_offenders = worst_offenders
        self.tf_build_emb = tf_build_emb

        # estimated sequences embeddings. embeddings are not perfectly
        # accurate, but are updated any time a sequence is sampled.
        self.embeddings = torch.randn(self.size, dim).cuda().to(dtype=self.dtype)
        self.last_updated = torch.full((self.size,), -1, dtype=torch.long).cuda()

        # create batches with similar number of tokens and (almost) no padding
        self.batch_idx = []
        cur_idx = []
        cur_len = 0
        for i in range(self.size):
            cur_idx.append(i)
            cur_len += len(self.id2tensor[i])
            if cur_len > tokens_per_batch or i == self.size - 1:
                self.batch_idx.append(cur_idx)
                cur_idx = []
                cur_len = 0
        assert sum([len(x) for x in self.batch_idx]) == self.size
        assert set(range(self.size)) == set(sum(self.batch_idx, []))

        # distributed mode
        if dist.is_initialized():
            self.init_distributed()

    def init_distributed(self):
        """
        Preprocess data for distributed mode.
        """
        assert dist.is_initialized()

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.dist_batches = []
        self.dist_idx = []
        self.dist_n = []

        for rank in range(self.world_size):
            batches = self.batch_idx[rank :: self.world_size]
            idx = sum(batches, [])
            self.dist_batches.append(batches)
            self.dist_idx.append(torch.cuda.LongTensor(idx))
            self.dist_n.append(len(idx))

        assert sum(self.dist_n) == self.size
        logger.info(
            f"Process {self.rank} ({self.world_size} total). "
            f"Assigned to {len(self.dist_batches[self.rank])} batches "
            f"({self.dist_n[self.rank]} sequences)."
        )

    def __len__(self):
        return self.size

    def get_negative_ids(self, sequences: List, sample_size: int):
        """
        Get negative sample IDs for a batch of sequences.
        """
        assert type(sequences) is list
        assert sample_size >= len(sequences)
        assert all(type(seq) is list for seq in sequences)
        global_seq_ids = [self.seq2id[" ".join(seq)] for seq in sequences]
        bs = len(sequences)

        # map sequence ID to position in the global batch
        id2pos: Dict[int, int] = {}
        for sid in global_seq_ids:
            if sid not in id2pos:
                id2pos[sid] = len(id2pos)

        # positive sequence IDs / target positions
        pos2id = {v: k for k, v in id2pos.items()}
        pos_ids = [pos2id[i] for i in range(len(pos2id))]
        local_tgt_ids = [id2pos[sid] for sid in global_seq_ids]

        # negative sequence IDs. either random, or worst offenders. if there are N
        # different sequences, we only need to sample  sample_size - N negative samples,
        # as all elements  are merged across the batch
        n_neg = sample_size - len(id2pos)
        if np.random.rand() < self.worst_offenders:
            _, worst_idx = get_knn_pytorch(self.embeddings, pos_ids, n_neg)
            assert worst_idx.size() == (len(id2pos), n_neg)
            neg_ids = worst_idx.transpose(0, 1).view(-1).cpu().numpy()
            # get `n_neg` unique IDs
            indexes = np.unique(neg_ids, return_index=True)[1]
            neg_ids = [neg_ids[index] for index in sorted(indexes)][:n_neg]

        else:
            # negative sampling probabilities
            p = np.ones((self.size,), dtype=np.float32)
            p[pos_ids] = 0
            p = p / p.sum()
            neg_ids = np.random.choice(a=self.size, size=(n_neg,), p=p, replace=False)

        # sanity checks
        assert len(set(id2pos.keys()) & set(neg_ids)) == 0
        assert len(neg_ids) == n_neg
        assert max(local_tgt_ids) == len(pos_ids) - 1
        assert len(pos_ids) + len(neg_ids) == sample_size
        assert len(set(pos_ids) | set(neg_ids)) == sample_size

        return local_tgt_ids, global_seq_ids, pos_ids, neg_ids

    def get_negative_batch(self, sequences: List, sample_size: int):
        """
        Get negative sample batch.
        """
        # get negative sample IDs for a batch of sequences
        local_tgt_ids, global_tgt_ids, pos_ids, neg_ids = self.get_negative_ids(
            sequences, sample_size
        )
        seq_ids = np.concatenate([pos_ids, neg_ids])

        # retrieve sequences
        sequences = [self.id2tensor[i] for i in seq_ids]
        assert len(sequences) == sample_size

        # build batch
        batch, lengths = batch_sequences(sequences, pad_index=self.dico.pad_index)
        return (
            torch.LongTensor(local_tgt_ids),
            torch.LongTensor(global_tgt_ids),
            torch.LongTensor(seq_ids),
            batch,
            lengths,
        )

    def update_embeddings(self, indices, embeddings):
        """
        Update sequence embeddings.
        """
        bs, dim = embeddings.size()
        assert dim == self.dim
        assert indices.size() == (bs,)

        # TODO: can be improved if distributed mode
        with torch.no_grad():
            self.last_updated[self.last_updated >= 0] += 1
            self.last_updated[indices] = 0
            self.embeddings[indices] = embeddings.to(self.dtype)

    @torch.no_grad()
    def embed_sequences(self, embedder, batches):
        """
        Embed sequences.
        """
        embeddings = []
        for batch in batches:
            sequences = [self.id2tensor[i] for i in batch]
            x, xlen = batch_sequences(sequences, pad_index=self.dico.pad_index)
            x, xlen = to_cuda(x, xlen)
            embedded = embedder("fwd", causal=False, tokens=x, lengths=xlen)
            assert embedded.size() == (len(x), xlen.max().item(), self.dim)
            embedded = get_embs(embedded, xlen, self.tf_build_emb)
            embeddings.append(embedded.detach())
        embeddings = torch.cat(embeddings, dim=0)
        return embeddings.to(self.dtype)

    @torch.no_grad()
    def update_all_embeddings(self, embedder):
        """
        Update all sequence embeddings with an embedder.
        """
        logger.info(f"Generating embeddings for {self.size} statements ...")

        # compute embeddings
        embeddings = self.embed_sequences(embedder, self.batch_idx)
        assert embeddings.size() == (self.size, self.dim)
        assert embeddings.size() == self.embeddings.size()

        # update embeddings
        self.embeddings = embeddings
        self.last_updated.fill_(0)
        logger.info(f"Updated {self.size} statements embeddings.")

    @torch.no_grad()
    def update_all_embeddings_distributed(self, embedder):
        """
        Update all sequence embeddings with an embedder in a distributed way.
        """
        assert dist.is_initialized()

        rank = self.rank
        world_size = self.world_size
        batches = self.dist_batches[rank]

        logger.info(
            f"Generating embeddings for {self.size} statements (distributed mode) ...\n"
            f"Process {rank} ({world_size} total). "
            f"Processing {len(batches)} of {len(self.batch_idx)} batches "
            f"({self.dist_n[rank]} of {self.size} sequences)."
        )

        # compute embeddings for this process
        embs = self.embed_sequences(embedder, batches)
        assert embs.size() == (self.dist_n[rank], self.dim)

        # pad embeddings (distributed requires that tensors have the same shape)
        max_embs = max(self.dist_n)
        n_pad = max_embs - len(embs)
        if n_pad > 0:
            embs = torch.cat([embs, embs.new_zeros(n_pad, self.dim)], 0)

        # gather embeddings
        buffer = [embs.new_zeros((max_embs, self.dim)) for _ in range(world_size)]
        torch.distributed.all_gather(buffer, embs)
        for idx, buff in zip(self.dist_idx, buffer):
            # TODO: remove assert
            assert len(buff) <= len(idx) or buff[len(idx) :].abs().sum().item() == 0
            self.embeddings.index_copy_(0, idx, buff[: len(idx)])
        self.last_updated.fill_(0)
        logger.info(f"Updated {self.size} statements embeddings.")
