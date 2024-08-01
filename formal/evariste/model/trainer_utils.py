# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Optional
import numpy as np

from evariste.model.data.dictionary import Dictionary


def word_shuffle(x: List[int], shuffle: float, dico: Dictionary, rng):
    """
    Randomly shuffle input words.
    """
    if shuffle == 0:
        return x
    assert shuffle > 1
    eos = dico.conf.eos_index
    assert x[0] == x[-1] == eos

    # shuffle words, except begin and end tokens
    x = x[1:-1]
    noise = rng.uniform(0, shuffle, size=(len(x),))
    scores = np.arange(len(x)) + noise
    permutation = scores.argsort()

    return [eos, *[x[i] for i in permutation], eos]


def word_dropout(x: List[int], dropout_p: float, dico: Dictionary, rng):
    """
    Randomly drop input words.
    """
    if dropout_p == 0:
        return x
    assert 0 < dropout_p < 1
    eos = dico.conf.eos_index
    assert x[0] == x[-1] == eos

    # define words to drop
    x = x[1:-1]
    keep = rng.rand(len(x)) >= dropout_p

    # we need to have at least one word in the sentence (in addition to eos tokens)
    if not keep.any():
        keep[rng.randint(len(keep))] = True

    return [eos, *[x[i] for i in range(len(x)) if keep[i]], eos]


def word_blank(x: List[int], blank_p: float, dico: Dictionary, rng):
    """
    Randomly blank input words.
    """
    if blank_p == 0:
        return x
    assert 0 < blank_p < 1
    eos = dico.conf.eos_index
    assert x[0] == x[-1] == eos

    # define words to blank
    x = x[1:-1]
    keep = rng.rand(len(x)) >= blank_p

    new_x = [x[i] if keep[i] else dico.conf.mask_index for i in range(len(x))]
    return [eos, *new_x, eos]


def add_noise(
    seq: List[int], params, dico: Dictionary, rng=Optional[Dictionary]
) -> List[int]:
    """
    Add noise to input sequences for Denoising Auto-Encoder training.
    """
    assert type(seq) is list
    rng = np.random if rng is None else rng

    # add noise
    seq = word_shuffle(seq, params.input_noise.shuffle, dico, rng=rng)
    seq = word_dropout(seq, params.input_noise.dropout, dico, rng=rng)
    seq = word_blank(seq, params.input_noise.blank, dico, rng=rng)

    return seq
