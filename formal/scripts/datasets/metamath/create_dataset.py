# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Create a train / valid / test split of Metamath theorems and export the datasets.
"""

import os
from evariste import json as json
import random
import argparse
import itertools
import pickle
import numpy as np
from logging import getLogger
from collections import defaultdict

from params.params import bool_flag
from evariste.utils import TermIndexer
from evariste.logger import create_logger
from evariste.envs.mm.env import MetamathEnv
from evariste.envs.mm.utils import (
    Node_a_p,
    decompress_all_proofs,
    simplify_proof_tree,
    has_cycle,
)
from evariste.envs.mm.state_history import StatementHistory
from evariste.backward.env.metamath import MMTheorem
from evariste.envs.mm.utils import remove_syntactic


logger = getLogger()


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
        "--database_path",
        type=str,
        default="resources/metamath/set.mm/set.mm",
        help="Metamath database",
    )

    # data
    parser.add_argument(
        "--data", choices=["", "proof_trees", "compressed_steps"], help="Data",
    )
    parser.add_argument(
        "--build_splits",
        type=bool_flag,
        default=True,
        help="Build dataset splits (otherwise, reload from files)",
    )
    parser.add_argument(
        "--remove_syntactic",
        type=bool_flag,
        default=False,
        help="Remove syntactic nodes",
    )

    parser.add_argument(
        "--syntactic_theorems",
        type=bool_flag,
        default=False,
        help="Keep syntactic theorems",
    )

    # Metamath parameters
    parser.add_argument(
        "--start_label",
        type=str,
        default="",
        help="Decompress and verify from a given label",
    )
    parser.add_argument("--stop_label", type=str, default="", help="Stop label")

    # logging
    parser.add_argument(
        "--api_log_level", type=str, default="debug", help="API log level"
    )
    parser.add_argument(
        "--console_log_level", type=str, default="debug", help="Console log level"
    )

    # valid / test proofs properties
    parser.add_argument(
        "--valid_test_size",
        type=int,
        default=1000,
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
        default=False,
        help="Ensure that valid/test theorems are unique and are not proved multiple times under another name",
    )

    return parser


def split_theorems(
    mm_env: MetamathEnv,
    valid_test_size: int,
    valid_never_used: bool,
    valid_no_duplicates: bool,
    dataset_dir: str,
):
    """
    Split theorems across train / valid / test sets.
    """
    assert os.path.isdir(dataset_dir)
    assert valid_test_size > 0

    # theorem usage
    counts = {label: 0 for label in mm_env.decompressed_proofs.keys()}
    for proof in mm_env.decompressed_proofs.values():
        for token in proof:
            if token in counts:
                counts[token] += 1
    logger.info(f"{len(counts)} theorems.")
    for count in range(10):
        c = sum(int(v == count) for v in counts.values())
        logger.info(f"{c:4} are used {count} times.")
    c = sum(int(v >= 10) for v in counts.values())
    logger.info(f"{c} are used >= 10 times.")

    # theorems that appear multiple times (same statement / $e hypotheses).
    # if `valid_no_duplicates` is True, we prevent statements to be in the
    # valid/test sets if they have a duplicate (potentially in the train).
    # for instance: equncom/equncomVD, rbaib/rbaibOLD, neeq12d/neeq12dOLD, etc.
    statement_counts = defaultdict(list)
    for label, (_, assertion) in mm_env.labels.items():
        theorem = MMTheorem(
            conclusion=" ".join(assertion["tokens"]),
            hyps=[(None, " ".join(h)) for h in assertion["e_hyps"]],
        )
        statement_counts[theorem].append(label)
    n_dupl_stat = sum(1 for v in statement_counts.values() if len(v) > 1)
    with_duplicates = set(sum([v for v in statement_counts.values() if len(v) > 1], []))
    logger.info(
        f"Found {n_dupl_stat} duplicate statements / $e hypotheses. "
        f"Involves {len(with_duplicates)} labels."
    )

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
    all_labels = list(mm_env.decompressed_proofs.keys())
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

    # export theorem splits. if files exist, check that they are identical
    for split, theorems in zip(
        ["train", "valid", "test"], [train_labels, valid_labels, test_labels]
    ):
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

    return train_labels, valid_labels, test_labels


def reload_splits(mm_env: MetamathEnv, dataset_dir: str):
    """
    Reload dataset splits from the disk.
    """
    labels = {}
    for name in ["train", "valid", "test"]:
        path = os.path.join(dataset_dir, f"split.{name}")
        assert os.path.isfile(path)
        logger.info(f"Loading {name} split from {path}")
        with open(path, "r", encoding="utf-8") as f:
            labels[name] = [x.rstrip() for x in f]
        assert all(x in mm_env.decompressed_proofs for x in labels[name])
    logger.info(
        f"Reloaded {sum(len(v) for v in labels.values())} / "
        f"{len(mm_env.decompressed_proofs)} labels from split files."
    )
    return labels["train"], labels["valid"], labels["test"]


def build_proof_trees(mm_env: MetamathEnv, args):
    """
    Compute proof trees.
    Compress them and compute the number of nodes / depth of each node.
    """
    proof_trees = {}
    skip_type = 0
    skip_cycle = 0
    skip_syntactic = 0

    for i, label in enumerate(mm_env.decompressed_proofs.keys()):

        # retrieve assertion
        label_type, assertion = mm_env.labels[label]
        assert label_type == "$p"
        decompressed_proof = mm_env.decompressed_proofs[label]

        # build proof tree
        root = mm_env.build_proof_tree(
            decompressed_proof, assertion=assertion, return_all_stacks=False
        )["root_node"]
        if not isinstance(root, Node_a_p):
            logger.warning(
                f'Proof tree for "{label}" is not of type Node_a_p. Ignoring theorem.'
            )
            skip_type += 1
            continue

        # skip syntactic proof trees
        if root.is_syntactic and not args.syntactic_theorems:
            logger.warning(
                f'Proof tree for "{label}" proves a syntactic statement. Ignoring theorem.'
            )
            skip_syntactic += 1
            continue

        # compute the number of nodes / depth before simplification, as it sometimes creates
        # a cyclic graph and infinite loops (e.g. onfrALTlem2, a9e2nd, uun0.1)
        root = simplify_proof_tree(root)
        if has_cycle(root):
            logger.error(f'Proof tree for "{label}" has a cycle. Ignoring theorem.')
            skip_cycle += 1
            continue
        root.set_nodes_and_depth()

        proof_trees[label] = root

        # stats
        if i % 100 == 0:
            logger.info(
                f"Processed proof tree {i + 1}/{len(mm_env.decompressed_proofs)}: {label:<15}"
            )

    # remove syntactic nodes
    if args.remove_syntactic:
        logger.info(f"Removing syntactic nodes ...")
        proof_trees = {k: remove_syntactic(v) for k, v in proof_trees.items()}
        assert not any(v is None for v in proof_trees.values())

    # log summary
    logger.info(
        f"Processed {len(proof_trees)} proof trees. "
        f"{skip_type + skip_cycle + skip_syntactic} were ignored "
        f"({skip_type} had of bad type, {skip_cycle} had a cycle, "
        f"{skip_syntactic} were syntactic)."
    )

    name = f"proof_trees.syntactic{int(not args.remove_syntactic)}.pkl"
    path = os.path.join(args.dataset_dir, name)
    with open(path, "wb") as f:
        pickle.dump(proof_trees, f)
    logger.info(f"Saved {len(proof_trees)} proof trees to {path}")


def build_compressed_proofs_dataset(mm_env: MetamathEnv, args):
    """
    Compressed bottom-up generation steps with statement history.

    We build the graph step-by-step, by adding elements into the stack and
    poping them when an assertion is applied.
    """
    labels = list(mm_env.decompressed_proofs.keys())

    # use a dictionary of terms to save space
    indexer = TermIndexer()

    # build proof for each label
    data = {}
    for i, label in enumerate(mm_env.decompressed_proofs.keys()):

        # retrieve assertion
        label_type, assertion = mm_env.labels[label]
        decompressed_proof = mm_env.decompressed_proofs[label]
        assert label_type == "$p"

        # compression with statement history
        stat_history = StatementHistory(
            min_stat_counts=1, min_proof_len=2, sample_subproof=False,
        )
        out = mm_env.build_proof_tree(
            decompressed_proof,
            assertion=assertion,
            return_all_stacks=True,
            stat_history=stat_history,
        )
        root_node = out["root_node"]
        compressed_proof = out["compressed_proof"]
        all_stacks = out["all_stacks"]

        # sanity check
        assert root_node.statement == assertion["tokens"]
        assert len(compressed_proof) == len(all_stacks)
        assert (
            list(
                itertools.chain.from_iterable(
                    [
                        x["tokens"] if x["is_subproof"] else [x["token"]]
                        for x in compressed_proof
                    ]
                )
            )
            == decompressed_proof
        )

        # use a dictionary of terms to save space
        # index stack terms, subproofs, and $e hypotheses
        for stack in all_stacks:
            for term in stack:
                indexer.add(term)
        for subproof in stat_history.get_available_subproofs():
            indexer.add(stat_history.proof2stat[subproof])
        for hyp in root_node.e_hyps.values():
            indexer.add(" ".join(hyp))

        # create data
        data[label] = []

        e_hyps = {k: indexer.get(" ".join(v)) for k, v in root_node.e_hyps.items()}

        for stack, x in zip(all_stacks, compressed_proof):

            # generate a token, or a previously generated statement (subproof)
            is_subproof = x["is_subproof"]
            output = indexer.get(x["statement"]) if is_subproof else x["token"]

            # add step
            data[label].append(
                {
                    "stack": [indexer.get(term) for term in stack],
                    "e_hyps": e_hyps,
                    "subproofs": [indexer.get(x) for x in x["available"]],
                    "output": output,
                    "is_subproof": is_subproof,
                }
            )

        # stats
        if i % 100 == 0:
            logger.info(
                f"Processed {len(data[label]):>3} compressed steps for proof "
                f"{i + 1}/{len(mm_env.decompressed_proofs)}: {label:<15} - "
                f"{sum(len(v) for v in data.values())} compressed steps"
            )

    # summary
    logger.info(
        f"Computed {sum(len(v) for v in data.values())} compressed steps "
        f"for {len(mm_env.decompressed_proofs)} proofs."
    )

    # export indexer and compressed steps
    name = "compressed_steps"
    path = os.path.join(args.dataset_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(indexer.id2term))
        f.write("\n")
        for label in labels:
            if label not in data:
                continue
            f.write(json.dumps({"label": label, name: data[label]}))
            f.write("\n")
        logger.info(f"Wrote {len(labels)} compressed steps to {path}")


if __name__ == "__main__":

    # parse arguments
    parser = get_parser()
    args = parser.parse_args()

    # check arguments
    assert os.path.isfile(args.database_path)
    assert os.path.isdir(args.dataset_dir)

    # create logger
    log_path = os.path.join(
        os.getcwd(), f"metamath_create_dataset.{random.randint(0, 100000)}.log"
    )
    logger = create_logger(log_path, console_level=args.console_log_level)
    logger.info(f"Logging processing results in {log_path} ...")

    # create Metamath instance
    mm_env = MetamathEnv(
        filepath=args.database_path,
        start_label=args.start_label,
        stop_label=args.stop_label,
        rename_e_hyps=True,
        decompress_proofs=False,
        verify_proofs=False,
        log_level=args.api_log_level,
    )
    mm_env.process()

    # decompress proofs
    decompress_all_proofs(mm_env)

    # check vocabulary
    tmp_vocab = defaultdict(int)
    for proof in mm_env.decompressed_proofs.values():
        for token in proof:
            tmp_vocab[token] += 1
    unk_labels = {
        t
        for t in tmp_vocab
        if (
            t not in mm_env.labels
            and t not in mm_env.fs.frames[0].f_labels
            and "E_HYP_" not in t
        )
    }
    logger.info(
        f"Found {len(tmp_vocab)} unique words, {sum(tmp_vocab.values())} total.\n"
        f"{len(unk_labels)}/{len(tmp_vocab)} tokens in the vocabulary are not an "
        f"assertion label, not a $f label in the outermost scope, and not a renamed "
        f"hypothesis: {', '.join(unk_labels)}"
    )

    # create train / valid / test theorem splits
    if args.build_splits:
        train_labels, valid_labels, test_labels = split_theorems(
            mm_env,
            args.valid_test_size,
            args.valid_never_used,
            args.valid_no_duplicates,
            args.dataset_dir,
        )
    else:
        train_labels, valid_labels, test_labels = reload_splits(
            mm_env, args.dataset_dir
        )

    # split duplicate statistics
    stat_counts = defaultdict(lambda: defaultdict(list))
    for split, theorems in zip(
        ["train", "valid", "test"], [train_labels, valid_labels, test_labels]
    ):
        for label in theorems:
            assertion = mm_env.labels[label][1]
            theorem = MMTheorem(
                conclusion=" ".join(assertion["tokens"]),
                hyps=[(None, " ".join(h)) for h in assertion["e_hyps"]],
            )
            stat_counts[split][theorem].append(label)
    c1 = len(set(stat_counts["train"].keys()) & set(stat_counts["valid"].keys()))
    c2 = len(set(stat_counts["train"].keys()) & set(stat_counts["test"].keys()))
    logger.info(f"{c1} valid statements also in the training set.")
    logger.info(f"{c2} test statements also in the training set.")

    # compressed graph for efficient bottom-up generation
    if args.data == "proof_trees":
        build_proof_trees(mm_env, args)

    # compressed steps for efficient bottom-up generation
    if args.data == "compressed_steps":
        build_compressed_proofs_dataset(mm_env, args)
