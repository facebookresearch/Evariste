# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Binarize informal data.

Example: python data/vocab.txt data/train.txt
vocab.txt: 1stline=word, 2ndline=count
"""

import os
import sys

from evariste.logger import create_logger
from evariste.model.data.dictionary import Dictionary


if __name__ == "__main__":

    logger = create_logger(None, 0)

    voc_path = sys.argv[1]
    txt_path = sys.argv[2]
    bin_path = sys.argv[2] + ".pth"
    assert os.path.isfile(voc_path)
    assert os.path.isfile(txt_path)

    dico = Dictionary.create_from_vocab_path(voc_path)
    logger.info("")

    data = Dictionary.index_data(txt_path, bin_path, dico)
    logger.info(
        "%i words (%i unique) in %i sentences."
        % (
            len(data["sentences"]) - len(data["positions"]),
            len(data["dico_id2word"]),
            len(data["positions"]),
        )
    )
    if len(data["unk_words"]) == 0:
        logger.info("No unknown words.")
    else:
        logger.info(
            "%i unknown words (%i unique), covering %.2f%% of the data."
            % (
                sum(data["unk_words"].values()),
                len(data["unk_words"]),
                sum(data["unk_words"].values())
                * 100.0
                / (len(data["sentences"]) - len(data["positions"])),
            )
        )
        for w, c in sorted(data["unk_words"].items(), key=lambda x: x[1])[::-1][:30]:
            logger.info("%s: %i" % (w, c))
