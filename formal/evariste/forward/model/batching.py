# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from typing import List, Callable, Iterator, Tuple, Optional, TypeVar, cast

import torch
from torch import Tensor

from evariste.model.embedder_utils import PTR_PAD_IDX


@dataclass
class EncoderInputs:
    enc_inp: torch.Tensor  # (bs, max_len), long
    enc_len: torch.Tensor  # (bs,), long
    forbidden: Optional[List[torch.Tensor]] = None

    @property
    def batch_size(self):
        return self.enc_inp.size(0)

    @property
    def max_len(self):
        return self.enc_inp.size(1)

    @property
    def size(self):
        return self.max_len * self.batch_size

    @property
    def sort_value(self):
        return self.max_len

    def sub_batch(self, start: int, end: int) -> "EncoderInputs":
        forbidden = self.forbidden[start:end] if self.forbidden else None
        enc_inp = self.enc_inp[start:end]
        enc_len = self.enc_len[start:end]
        max_len = int(enc_len.max().item())  # mypy
        enc_inp = enc_inp[:, :max_len]
        return EncoderInputs(enc_inp=enc_inp, enc_len=enc_len, forbidden=forbidden)

    def to(self, device, non_blocking=False) -> "EncoderInputs":
        enc_inp = self.enc_inp.to(device, non_blocking=non_blocking)
        enc_len = self.enc_len.to(device, non_blocking=non_blocking)
        forbidden: Optional[List[torch.Tensor]]
        if self.forbidden is not None:
            forbidden = [
                x.to(device=device, non_blocking=non_blocking) for x in self.forbidden
            ]
        else:
            forbidden = None
        return EncoderInputs(enc_inp=enc_inp, enc_len=enc_len, forbidden=forbidden)


def estimate_tokens_per_batch(samples: List[EncoderInputs]) -> int:
    bs = len(samples)
    max_len = max(s.max_len for s in samples)
    return bs * max_len


def estimate_mem_per_batch(samples: List[EncoderInputs]) -> int:
    bs = len(samples)
    max_len = max(s.max_len for s in samples)
    return bs * (max_len ** 2)


def batch(samples: List[EncoderInputs], pad_index: int) -> EncoderInputs:
    assert len(samples) > 0
    have_forbidden = samples[0].forbidden is not None

    enc_inp = batch_tensor([s.enc_inp for s in samples], pad=pad_index)
    enc_len = torch.cat([s.enc_len for s in samples])

    forbidden: Optional[List[Tensor]]
    if have_forbidden:
        forbidden = []
        for s in samples:
            assert s.forbidden is not None  # mypy
            forbidden.extend([f for f in s.forbidden])
    else:
        assert all(s.forbidden is None for s in samples)
        forbidden = None

    return EncoderInputs(enc_inp=enc_inp, enc_len=enc_len, forbidden=forbidden,)


def batch_tensor(tensors: List[Tensor], pad: int):
    """
    tensors are supposed to be all of batch 1.
    """
    assert len(tensors) > 0
    max_len = max([t.size(1) for t in tensors])
    bs = len(tensors)

    batched = tensors[0].new_full((bs, max_len), pad)
    for idx, tensor in enumerate(tensors):
        assert tensor.size(0) == 1  # batch size 1
        batched[idx, : tensor.size(1)] = tensor[0]
    return batched


def group_samples_by_size(
    sample_idxs: List[int],
    samples: List[EncoderInputs],
    max_size: int,
    size_fn: Callable[[List[EncoderInputs]], int],
    max_batch_size: int,
) -> Iterator[Tuple[List[int], List[EncoderInputs]]]:
    assert len(samples) == len(sample_idxs)
    start = 0
    batch: List[EncoderInputs] = []
    batch_idx: List[int] = []
    sorted_samples = sorted(zip(sample_idxs, samples), key=lambda x: x[1].sort_value)
    sample_idxs, samples = unzip_mypy(sorted_samples)

    for idx, sample in enumerate(samples):
        size = size_fn(samples[start : idx + 1])
        if batch_idx and (size > max_size or len(batch) == max_batch_size):
            yield batch_idx, batch
            start = idx
            batch = []
            batch_idx = []
        batch.append(sample)
        batch_idx.append(sample_idxs[idx])
    yield batch_idx, batch


X = TypeVar("X")
Y = TypeVar("Y")


def unzip_mypy(input: List[Tuple[X, Y]]) -> Tuple[List[X], List[Y]]:
    x_tuple, y_tuple = list(zip(*input))
    x = cast(List[X], list(x_tuple))
    y = cast(List[Y], list(y_tuple))
    return x, y
