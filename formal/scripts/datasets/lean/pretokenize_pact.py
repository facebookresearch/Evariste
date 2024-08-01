# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import re
import fire

from evariste import json as json
from evariste.backward.env.lean.tokenizer import LeanTokenizer
from evariste.model.data.dictionary import SQUID_TOK

TASKS = {
    "next_lemma_prediction",
    "premise_classification",
    "classify_premise",
    "proof_step_classification",
    "proof_term_elab",
    "proof_term_prediction",
    "result_elab",
    "skip_proof",
    "theorem_name_prediction",
    "ts_elab",
    "type_prediction",
    "subexpr_type_prediction",
}

FIELDS = {
    "classify_locals",
    "classify_premise",
    "elab_goal",
    "elab_proof_term",
    "goal_and_premise",
    "goal",
    "name",
    "next_lemma",
    "proof_term",
    "result_elab",
    "result",
    "skip_proof",
    "type",
    "typed_expression",
}


def main(dirpath: str, suffix: str = "", confirm: bool = True):

    # check path / split
    assert os.path.isdir(dirpath)
    split = os.path.basename(dirpath)
    assert split in {"train", "valid", "test"}

    # reload Lean tokenizer
    tokenizer = LeanTokenizer.build("bpe_arxiv_lean_utf8_20k_single_digits_no_unk")

    # look for files to process
    to_process = []
    for fname in os.listdir(dirpath):
        fpath = os.path.join(dirpath, fname)
        if not os.path.isfile(fpath):
            continue
        if split == "train":
            if not re.match(r".*\.json\.[0-7]$", fname):
                continue
        else:
            if not fname.endswith(".json"):
                continue
        if not fname.endswith(str(suffix)):
            continue
        to_process.append(fpath)
    already_processed = [x for x in to_process if os.path.isfile(x + ".tok")]
    print(
        f"Found {len(to_process)} files. "
        f"{len(already_processed)} already processed. "
    )

    if confirm:
        diff = set(to_process) - set(already_processed)
        print(f"About to process {len(diff)} files:")
        for x in sorted(diff):
            print(f"\t{x}")
        print("Continue? [yN]")
        if input().lower() != "y":
            exit()

    # process files
    for fpath in to_process:

        # retrieve task
        fname = os.path.basename(fpath)
        task = fname.split(".")[0]
        assert task in TASKS, task

        # skip if already done
        processed_path = f"{fpath}.tok"
        assert os.path.isfile(processed_path) == (fpath in already_processed)
        if fpath in already_processed:
            continue

        # reload data
        print(f"===== Loading data from {fpath} ...")
        data = []
        with open(fpath, "r") as f:
            for line in f:
                sample = json.loads(line)
                assert len(sample) == 2
                data.append(sample)
        print(f"Read {len(data)} lines.")

        # check available keys and retrieve input / output fields
        keys = set(data[0].keys())
        available = FIELDS & keys
        assert len(keys) == 2, keys
        assert all(set(x.keys()) == keys for x in data)
        assert len(available) == 2, (task, available)

        with open(processed_path, "w") as f:

            for sample in data:

                # retrieve fields
                x = {}
                for k in available:
                    s = sample[k]
                    assert isinstance(s, (str, list))
                    if isinstance(s, list):
                        assert (
                            task == "proof_step_classification"
                            and k == "classify_locals"
                        )
                        s = f" {SQUID_TOK} ".join(sample[k])
                    s = re.sub(r"PREDICT|<EXPR>|</EXPR>", f" {SQUID_TOK} ", s)
                    x[k] = re.sub(r" +", " ", s.strip())

                # tokenize & store
                x = {k: " ".join(tokenizer.encode(v)) for k, v in x.items()}
                f.write(f"{json.dumps(x)}\n")

        print(f"Processed and exported {len(data)} lines to {processed_path}")


if __name__ == "__main__":
    fire.Fire(main)
