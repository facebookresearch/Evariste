# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Given a metamath dataset folder, create a folder with a subset of the theorems
for fast data loading and debugging.
"""

import os
import pickle
import random
import argparse

from evariste.logger import create_logger


def get_parser():

    parser = argparse.ArgumentParser(description="Preprocess Metamath dataset")
    # data paths
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="resources/metamath/DATASET",
        help="Dataset path",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=100,
        help="Number of theorems in the train / valid / test",
    )

    return parser


if __name__ == "__main__":

    # parse arguments
    parser = get_parser()
    args = parser.parse_args()

    # check arguments
    assert os.path.isdir(args.dataset_dir)
    assert args.subset_size >= 1

    # create logger
    log_path = os.path.join(
        os.getcwd(), f"metamath_create_subset.{random.randint(0, 100000)}.log"
    )
    logger = create_logger(log_path, console_level="debug")

    # subset directory
    args.subset_dir = os.path.join(args.dataset_dir, str(args.subset_size))
    if not os.path.isdir(args.subset_dir):
        os.mkdir(args.subset_dir)

    # load all theorems
    all_labels = set()
    splits = {}
    for split in ["train", "valid", "test"]:
        path = os.path.join(args.dataset_dir, f"split.{split}")
        with open(path, "r", encoding="utf-8") as f:
            labels = [x.rstrip() for x in f]
            assert len(labels) == len(set(labels))
            assert not any(label in all_labels for label in labels)
            all_labels |= set(labels)
        splits[split] = labels
        logger.info(f"Reloaded {len(labels)} from {path}")
        assert len(labels) >= args.subset_size
    new_splits = {k: set(v[: args.subset_size]) for k, v in splits.items()}

    # create sub-splits
    for split in ["train", "valid", "test"]:
        path = os.path.join(args.subset_dir, f"split.{split}")
        if os.path.isfile(path):
            logger.info(f"{path} split already exists. Nothing to do.")
            continue
        with open(path, "w", encoding="utf-8") as f:
            for label in splits[split][: args.subset_size]:
                f.write(label + "\n")
        logger.info(f"Wrote {args.subset_size} labels to {path}")

    # load proof trees
    new_path = os.path.join(args.subset_dir, "proof_trees.pkl")
    if os.path.isfile(new_path):
        logger.info(f"{new_path} proof trees already exists. Nothing to do.")
    else:
        path = os.path.join(args.dataset_dir, "proof_trees.pkl")
        logger.info(f"Loading proof trees from {path} ...")
        with open(path, "rb") as f:
            proof_trees = pickle.load(f)
            assert all(label in all_labels for label in proof_trees.keys())
        logger.info(f"Loaded {len(proof_trees)} proof trees.")
        new_proof_trees = {
            label: proof_tree
            for label, proof_tree in proof_trees.items()
            if any(label in new_splits[split] for split in ["train", "valid", "test"])
        }
        with open(new_path, "wb") as f:
            pickle.dump(new_proof_trees, f)
        logger.info(f"Wrote {len(new_proof_trees)} proof trees to {new_path}")
