# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import youtokentome as yttm
import pandas
import tempfile


with tempfile.NamedTemporaryFile(mode="w+") as f:
    df = pandas.read_csv("")
    for row in df.iloc:
        if row.split == "train":
            if type(row.cleaned_goal) == str:
                f.write(row.cleaned_goal + "\n")
            f.write(row.human_tactic_code + "\n")
    f.flush()

    yttm.BPE.train(
        data=f.name, vocab_size=50000, model="",
    )
    # if we re-use this, use cat
    print("Appending all.txt to train file in the slowest way possible.")
    g = open("", "r")
    f.write(g.read() + "\n")
    f.flush()

    yttm.BPE.train(
        data=f.name, vocab_size=50000, model="",
    )
