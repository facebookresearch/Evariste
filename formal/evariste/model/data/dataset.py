# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import math
import numpy as np
import torch


logger = getLogger()


class StreamDataset(object):
    def __init__(self, sent, pos, bs, bptt, eos_index):
        """
        Prepare batches for data iterator.
        """
        self.eos = eos_index

        # checks
        assert len(pos) == (sent == self.eos).sum()
        assert len(pos) == (sent[pos[:, 1]] == self.eos).sum()

        n_tokens = len(sent)
        n_batches = math.ceil(n_tokens / (bs * bptt))
        t_size = n_batches * bptt * bs

        buffer = np.zeros(t_size, dtype=sent.dtype) + self.eos
        buffer[t_size - n_tokens :] = sent
        buffer = buffer.reshape((bs, n_batches * bptt)).T
        self.data = np.zeros((n_batches * bptt + 1, bs), dtype=sent.dtype) + self.eos
        self.data[1:] = buffer

        self.bs = bs
        self.bptt = bptt
        self.n_tokens = n_tokens
        self.n_batches = n_batches
        self.n_sentences = len(pos)
        self.lengths = torch.LongTensor(bs).fill_(bptt)

    def __len__(self):
        """
        Number of sentences in the dataset.
        """
        return self.n_sentences

    def select_data(self, a, b):
        """
        Only select a subset of the dataset.
        """
        if not (0 <= a < b <= self.n_batches):
            logger.warning(f"Invalid split values: {a} {b} - {self.n_batches}")
            return
        assert 0 <= a < b <= self.n_batches
        logger.info(f"Selecting batches from {a} to {b} ...")

        # sub-select
        self.data = self.data[a * self.bptt : b * self.bptt]
        self.n_batches = b - a
        self.n_sentences = (self.data == self.eos).sum().item()

    def get_iterator(self, shuffle, infinite, subsample=1):
        """
        Return a sentences iterator.
        """
        while True:
            indexes = (np.random.permutation if shuffle else range)(
                self.n_batches // subsample
            )
            for i in indexes:
                a = self.bptt * i
                b = self.bptt * (i + 1)
                batch = self.data[a:b].astype(np.int64).T
                assert batch.shape == (self.bs, self.bptt)
                yield {"x": torch.from_numpy(batch).clone(), "xlen": self.lengths}
            if not infinite:
                break
