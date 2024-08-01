# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Create a train / valid / test split of HOL-Light theorems and export the vocabulary.
"""
from typing import List, Dict
import os
from evariste import json as json
import random
import argparse
import numpy as np
from logging import getLogger
from collections import defaultdict

from params.params import bool_flag
from evariste.logger import create_logger
from evariste.envs.hl.utils import get_theorem_tokens
from evariste.backward.env.hol_light.graph import HLTheorem


logger = getLogger()


def get_parser():

    parser = argparse.ArgumentParser(description="Create HOL-Light splits")
    parser.add_argument(
        "--dataset_dir", type=str, default="", help="Dataset path",
    )
    parser.add_argument(
        "--valid_test_size",
        type=int,
        default=500,
        help="Number of theorems in the validation and test sets",
    )
    parser.add_argument(
        "--valid_never_used",
        type=bool_flag,
        default=True,
        help="Do not select valid/test theorems if they are used to prove another theorem.",
    )
    parser.add_argument(
        "--valid_no_duplicates",
        type=bool_flag,
        default=True,
        help="Ensure that valid/test theorems are unique and are not proved multiple times under another name",
    )
    return parser


def build_vocab(theorems: Dict[str, Dict], dataset_dir: str):
    """
    Build and export vocabulary.
    """
    assert os.path.isdir(dataset_dir)

    # build
    vocab = {}
    for _, theorem in theorems.items():
        th_vocab = get_theorem_tokens(theorem)
        for tok, count in th_vocab.items():
            vocab[tok] = vocab.get(tok, 0) + count
    logger.info(f"Found {len(vocab)} unique words, {sum(vocab.values())} total.")

    # export
    vocab_path = os.path.join(dataset_dir, "vocab")
    sorted_vocab = sorted(vocab.items(), key=lambda x: (-x[1], x[0]))
    if os.path.isfile(vocab_path):
        logger.info(f"Vocabulary already found in {vocab_path}")
        with open(vocab_path, "r") as f:
            lines = [line.rstrip().split() for line in f]
            assert len(lines) == len(sorted_vocab) and all(
                w1 == w2 and c1 == int(c2)
                for (w1, c1), (w2, c2) in zip(sorted_vocab, lines)
            )
    else:
        with open(vocab_path, "w") as f:
            for word, count in sorted_vocab:
                assert " " not in word
                f.write(f"{word} {count}\n")
        logger.info(f"Exported vocabulary to {vocab_path}")


def create_splits(
    theorems: Dict[str, Dict],
    valid_test_size: int,
    valid_never_used: bool,
    valid_no_duplicates: bool,
    dataset_dir: str,
) -> Dict[str, List[str]]:
    """
    Split theorems across train / valid / test sets.
    """
    assert os.path.isdir(dataset_dir)
    assert valid_test_size > 0

    # theorem usage
    counts = {name: 0 for name in theorems.keys()}
    for _, theorem in theorems.items():
        for step in theorem["steps"]:
            for tok in step["tactic"].split():
                if tok in counts:
                    counts[tok] += 1
    logger.info(f"{len(counts)} theorems.")
    for count in range(10):
        c = sum(int(v == count) for v in counts.values())
        logger.info(f"{c:4} are used {count} times.")
    c = sum(int(v >= 10) for v in counts.values())
    logger.info(f"{c} are used >= 10 times.")

    # theorems that appear multiple times (same statement / hypotheses).
    # if `valid_no_duplicates` is True, we prevent statements to be in the
    # valid/test sets if they have a duplicate (potentially in the train).
    # for instance: equncom/equncomVD, rbaib/rbaibOLD, neeq12d/neeq12dOLD, etc.
    statement_counts = defaultdict(list)
    for name, theorem in theorems.items():
        th = theorem["steps"][0]["goal"]
        th = HLTheorem(conclusion=th["concl"], hyps=th["hyps"])
        statement_counts[th].append(name)
    n_dupl = sum(1 for v in statement_counts.values() if len(v) > 1)
    with_duplicates = set(sum([v for v in statement_counts.values() if len(v) > 1], []))
    logger.info(
        f"Found {n_dupl} duplicated statements. "
        f"Involves {len(with_duplicates)} theorems."
    )
    for _, v in statement_counts.items():
        if len(v) > 1:
            print(f"\t{v}")

    # select candidate theorems for the validation and test sets
    test_candidates = {
        label
        for label, count in counts.items()
        if (
            (valid_never_used is False or count == 0)
            and (valid_no_duplicates is False or label not in with_duplicates)
        )
    }
    logger.info(f"Found {len(test_candidates)} candidates for the valid/test sets.")

    train_labels = []
    valid_labels = []
    test_labels = []

    # shuffle theorems
    all_labels = sorted(theorems.keys())
    rng = np.random.RandomState(42)
    rng.shuffle(all_labels)

    # for each theorem
    for label in all_labels:

        # not a candidate valid / test theorem
        if label not in test_candidates:
            train_labels.append(label)

        # if valid / test not created yet, fill them, otherwise train
        elif len(valid_labels) < valid_test_size:
            valid_labels.append(label)
        elif len(test_labels) < valid_test_size:
            test_labels.append(label)
        else:
            train_labels.append(label)

    logger.info(
        f"{len(train_labels)} / {len(valid_labels)} / {len(test_labels)} "
        f"train / valid / test theorems ({len(all_labels)} total)."
    )

    # sanity check
    assert len(valid_labels) == len(test_labels) == valid_test_size
    assert len(train_labels) + len(valid_labels) + len(test_labels) == len(all_labels)

    splits = {
        "train": train_labels,
        "valid": valid_labels,
        "test": test_labels,
    }

    # export theorem splits. if files exist, check that they are identical
    for split, theorems in splits.items():
        path = os.path.join(dataset_dir, f"split.{split}")
        if os.path.isfile(path):
            logger.info(f"{split} split already exists in {path}")
            with open(path, "r", encoding="utf-8") as f:
                reloaded = [x.rstrip() for x in f]
            assert reloaded == theorems
        else:
            logger.info(f"Saving {split} split to {path}")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(theorems) + "\n")

    return splits


if __name__ == "__main__":

    # parse arguments
    parser = get_parser()
    args = parser.parse_args()

    # check arguments
    args.dataset_path = os.path.join(args.dataset_dir, "dataset.json")
    assert os.path.isdir(args.dataset_dir)
    assert os.path.isfile(args.dataset_path)

    # create logger
    log_path = f"{os.getcwd()}/hol_light_create_splits.{random.randint(0, 100000)}.log"
    logger = create_logger(log_path)
    logger.info(f"Logging processing results in {log_path} ...")

    # load theorems
    logger.info(f"Loading theorems from {args.dataset_path} ...")
    with open(args.dataset_path, "r") as f:
        theorems = {}
        for line in f:
            line = json.loads(line.rstrip())
            name = line["name"]
            assert line.keys() == {"name", "filename", "line", "steps"}, line.keys()
            if name in theorems:
                logger.warning(
                    f"{name} ({line['filename']}) already found in "
                    f"{theorems[name]['filename']} -- Taking last one."
                )
            theorems[name] = line
    logger.info(f"Reloaded {len(theorems)} theorems from {args.dataset_path}")

    # build and export vocabulary
    build_vocab(theorems=theorems, dataset_dir=args.dataset_dir)

    # create splits
    splits = create_splits(
        theorems=theorems,
        valid_test_size=args.valid_test_size,
        valid_never_used=args.valid_never_used,
        valid_no_duplicates=args.valid_no_duplicates,
        dataset_dir=args.dataset_dir,
    )

    # split duplicate statistics
    stat_counts = defaultdict(lambda: defaultdict(list))
    for split, labels in splits.items():
        for label in labels:
            th = theorems[label]["steps"][0]["goal"]
            th = HLTheorem(conclusion=th["concl"], hyps=th["hyps"])
            stat_counts[split][th].append(label)
    c1 = len(set(stat_counts["train"].keys()) & set(stat_counts["valid"].keys()))
    c2 = len(set(stat_counts["train"].keys()) & set(stat_counts["test"].keys()))
    logger.info(f"{c1} valid statements also in the training set.")
    logger.info(f"{c2} test statements also in the training set.")
