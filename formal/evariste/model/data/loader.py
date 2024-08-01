# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import numpy as np
import torch

from evariste.model.data.dictionary import UNK_WORD, Dictionary


logger = getLogger()


def process_binarized(data, params):
    """
    Process a binarized dataset and log main statistics.
    """
    dico = Dictionary(data["dico_id2word"], data["dico_word2id"], data["dico_counts"])

    # check that data is in uint16 if the vocabulary size if less than 2^16
    assert (
        (data["sentences"].dtype == np.uint16)
        and (len(dico) < 1 << 16)
        or (data["sentences"].dtype == np.int32)
        and (1 << 16 <= len(dico) < 1 << 31)
    )

    # log dataset stats
    n_words = len(data["sentences"]) - len(data["positions"])
    n_unk = sum(data["unk_words"].values())
    logger.info(
        f"{n_words} words ({len(dico)} unique) in {len(data['positions'])} sentences. "
        f"{n_unk} unknown words ({len(data['unk_words'])} unique) covering {100. * n_unk / (n_words):.2f}% of the data."
    )

    # clip vocabulary size
    if params.max_vocab != -1:
        assert params.max_vocab > 0
        logger.info(f"Selecting {params.max_vocab} most frequent words ...")
        dico.max_vocab(params.max_vocab)
        data["sentences"][data["sentences"] >= params.max_vocab] = dico.index(UNK_WORD)
        unk_count = (data["sentences"] == dico.index(UNK_WORD)).sum()
        logger.info(
            f"Now {unk_count} unknown words covering {100. * unk_count / (n_words):.2f}% of the data."
        )

    # select words with a minimum number of occurrences
    if params.min_count > 0:
        logger.info(f"Selecting words with >= {params.min_count} occurrences ...")
        dico.min_count(params.min_count)
        data["sentences"][data["sentences"] >= len(dico)] = dico.index(UNK_WORD)
        unk_count = (data["sentences"] == dico.index(UNK_WORD)).sum()
        logger.info(
            f"Now {unk_count} unknown words covering {100. * unk_count / (n_words):.2f}% of the data."
        )

    # move data to uint16 if the vocabulary size if less than 2^16
    if (data["sentences"].dtype == np.int32) and (len(dico) < 1 << 16):
        logger.info("Less than 65536 words. Moving data from int32 to uint16 ...")
        data["sentences"] = data["sentences"].astype(np.uint16)

    # update data dictionary
    data["dico"] = dico
    del data["dico_id2word"]
    del data["dico_word2id"]
    del data["dico_counts"]

    return data


def load_binarized(path, params):
    """
    Load a binarized dataset.
    """
    assert path.endswith(".pth")
    if params.debug.train:
        path = path.replace("train", "valid")

    # if available, load dataset splits
    if params.slurm_conf.multi_gpu:
        split_path = f"{path[:-4]}.{params.slurm_conf.local_rank}.pth"
        if os.path.isfile(split_path):
            assert params.split_data is False
            path = split_path

    # load data
    assert os.path.isfile(path), path
    logger.info(f"Loading data from {path} ...")
    data = torch.load(path)
    data = process_binarized(data, params)

    return data
