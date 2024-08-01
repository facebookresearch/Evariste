# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Use a statement history to shorten generated proofs.
During generation, the model can decide to output a token, or to directly
re-generate a previously generated node. The statement history can decide
to consider only node that appear a minimum number of times, or that have
a minimum proof length. Storing more subproofs provides faster generation,
but may confuse the model (hyper-parameters to tune).

Usage:
    # --min_stat_counts 1 --min_proof_len 2 -> stores everything
    # results in more stored proofs, and equally long compressions
    python -m evariste.envs.mm.state_history --n_labels 5000 --min_stat_counts 1 --min_proof_len 2

    # --min_stat_counts 3 --min_proof_len 2 -> only nodes that appeared 3 times
    # results in less stored proofs, but longer compressions
    python -m evariste.envs.mm.state_history --n_labels 5000 --min_stat_counts 3 --min_proof_len 2
"""

import os
import argparse
import itertools
import logging
import numpy as np
from collections import OrderedDict

from .metamath import MetamathEnv, decompress_ints
from .metamath_utils import decompress_all_proofs, Node_a_p
from ..logger import create_logger
from ..trie import Trie


logger = logging.getLogger()


class StatementHistory:
    def __init__(self, min_stat_counts=1, min_proof_len=2, sample_subproof=False):
        """
        Statement history.
        Store all statements with their associated proofs, for future fast regeneration.
        """
        self.stat2proof = OrderedDict()
        self.proof2stat = OrderedDict()
        self.stat2count = OrderedDict()
        self.proof_trie = Trie()

        # restrict what subproofs can be used
        self.min_proof_len = min_proof_len  # proofs with a minimum length
        self.min_stat_counts = (
            min_stat_counts  # proofs with minimum statement occurrences
        )
        assert min_stat_counts >= 1 and min_proof_len >= 1

        # sample subproofs, or take the longest one
        self.sample_subproof = sample_subproof

    def __len__(self):
        """
        Number of stored statements. Can be different than the number
        of proofs, if two different proofs led to the same statement.
        """
        return len(self.get_available_subproofs())

    def __str__(self):
        """
        Print stored proofs / statements.
        """
        s = ""
        for stat, count in self.stat2count.items():
            length = len(self.stat2proof[stat])
            s += f"{count:>3} - {length:>3} tokens: {stat}\n"
        return s

    def __repr__(self):
        return str(self)

    def add_node(self, node):
        """
        Add a new node. The node is the result of a $a or $p label.
        """
        assert type(node) is Node_a_p
        stat = " ".join(node.statement)
        proof = " ".join(node.proof)

        # ignore too short proofs
        if len(node.proof) < self.min_proof_len:
            return

        # sanity check
        assert (stat in self.stat2proof) == (stat in self.stat2count)
        assert (stat in self.stat2proof) or (proof not in self.proof2stat)

        # add proof to trie
        self.proof_trie.add(node.proof)

        # update statement IDs / statement count
        self.stat2count[stat] = self.stat2count.get(stat, 0) + 1

        # add proof. if the statement exists, select the shortest proof
        if stat not in self.stat2proof:
            self.stat2proof[stat] = node.proof
        elif len(node.proof) < len(self.stat2proof[stat]):
            logger.warning(
                f"Found a shorter proof for {stat}: "
                f"{len(self.stat2proof[stat])} -> {len(node.proof)}"
            )
            self.stat2proof[stat] = node.proof

        # add statement
        assert proof not in self.proof2stat or self.proof2stat[proof] == stat
        self.proof2stat[proof] = stat

    def get_available_subproofs(self):
        """
        Return indexed subproofs that can be used for fast decoding.
        """
        return OrderedDict(
            [
                (proof, None)
                for proof, stat in self.proof2stat.items()
                if self.stat2count[stat] >= self.min_stat_counts
            ]
        )

    def get_subproof(self, tokens):
        """
        Given the list of future tokens, look for previously indexed subproofs.
        """
        subproofs = self.get_available_subproofs()
        available = [self.proof2stat[sp] for sp in subproofs.keys()]

        # no subproofs available
        if len(subproofs) == 0:
            return None, available

        # retrieve candidate subproofs
        candidates = []
        for sub_len in self.proof_trie.find_prefixes(tokens):
            subproof = " ".join(tokens[:sub_len])
            if subproof in subproofs:
                statement = self.proof2stat[subproof]
                candidates.append(
                    {
                        "statement": statement,
                        "proof_tok": tokens[:sub_len],
                        "proof_len": sub_len,
                    }
                )

        # no candidates
        if len(candidates) == 0:
            return None, available

        # sample candidate subproofs, or take the longest one
        if self.sample_subproof:
            subproof = candidates[np.random.randint(0, len(candidates))]
        else:
            subproof = candidates[-1]

        return subproof, available


def get_ref_compressed_proof(mm_env, name):
    """
    Get the length of a compressed proof using Metamath reference compression.
    """
    proof = mm_env.compressed_proofs[name]
    assertion = mm_env.labels[name][1]
    labels = assertion["f_hyps_labels"] + assertion["e_hyps_labels"]
    m = len(labels)
    assert proof[0] == "("
    ep = proof.index(")")
    labels += proof[1:ep]
    n = len(proof[1:ep])
    compressed_ints = "".join(proof[ep + 1 :])
    decompressed_ints = decompress_ints(compressed_ints)
    _, subproofs = mm_env.decompress_proof(
        assertion, mm_env.compressed_proofs[name], return_subproofs=True
    )
    proof_idx = [
        (labels[i] if i < m + n else subproofs[i - (m + n)]) if i >= 0 else i
        for i in decompressed_ints
    ]
    assert (
        list(
            itertools.chain.from_iterable(
                [x if type(x) is list else [x] for x in proof_idx if x != -1]
            )
        )
        == mm_env.decompressed_proofs[name]
    )
    return proof_idx, subproofs


def get_parser():

    parser = argparse.ArgumentParser(description="Metamath")
    parser.add_argument(
        "--database_path",
        type=str,
        default="/resources/metamath/set.mm/set.mm",
        # default='resources/metamath/set.mm/set.mm',
        help="Metamath database path",
    )
    parser.add_argument(
        "--n_labels",
        type=int,
        default=100,
        help="Number of labels to consider to compute statistics (-1 for everything)",
    )
    parser.add_argument(
        "--min_stat_counts",
        type=int,
        default=1,
        help="Minimum number of subproof counts (should be >= 1)",
    )
    parser.add_argument(
        "--min_proof_len",
        type=int,
        default=2,
        help="Minimum proof length (should be >= 2)",
    )
    parser.add_argument(
        "--console_log_level", type=str, default="debug", help="Console log level"
    )
    return parser


if __name__ == "__main__":

    # parse arguments
    parser = get_parser()
    args = parser.parse_args()
    assert os.path.isfile(args.database_path)
    assert args.n_labels == -1 or args.n_labels > 0
    assert args.min_stat_counts >= 1
    assert args.min_proof_len >= 2

    # create logger
    logger = create_logger(None, console_level=args.console_log_level)

    # create Metamath instance
    mm_env = MetamathEnv(
        filepath=args.database_path,
        rename_e_hyps=True,
        decompress_proofs=False,
        verify_proofs=False,
        log_level="info",
    )
    mm_env.process()
    decompress_all_proofs(mm_env)

    results = []

    for name, (label_type, assertion) in mm_env.labels.items():

        # skip axioms / stop if enough results
        assert label_type in ["$a", "$p"]
        if label_type == "$a":
            continue
        if 0 < args.n_labels <= len(results):
            break

        decompressed_proof = mm_env.decompressed_proofs[name]
        logger.info(
            f"===== {len(results) + 1:>3} Compressing {name} ({len(decompressed_proof)} tokens) ..."
        )

        # reference compression
        ref_compressed_proof, ref_subproofs = get_ref_compressed_proof(mm_env, name)
        ref_compressed_len = len([x for x in ref_compressed_proof if x != -1])
        logger.info(
            f"Reference: {len(ref_subproofs)} subproofs. "
            f"Proof length: {ref_compressed_len} "
            f"({len(ref_compressed_proof)} including -1 indexing tokens)."
        )

        # compression with statement history
        stat_history = StatementHistory(
            min_stat_counts=args.min_stat_counts,
            min_proof_len=args.min_proof_len,
            sample_subproof=False,
        )
        out = mm_env.build_proof_tree(
            proof=decompressed_proof, stat_history=stat_history, assertion=assertion
        )
        compressed_proof = out["compressed_proof"]

        # sanity check / compression results / log results
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
        logger.info(
            f"S-History: {len(stat_history)} subproofs. "
            f"Proof length: {len(compressed_proof)} "
        )
        results.append(
            {
                "decompressed_len": len(decompressed_proof),
                "ref_compressed_len": ref_compressed_len,
                "ref_subproofs": len(ref_subproofs),
                "new_compressed_len": len(compressed_proof),
                "new_subproofs": len(stat_history),
            }
        )
        if ref_compressed_len > len(compressed_proof):
            logger.warning(f"Reference compressed proof unexpectedly longer.")

    # compression results / comparison with reference compression
    logger.info("===== Summary =====")
    pl_ratio = [
        x["new_compressed_len"] / x["ref_compressed_len"]
        for x in results
        if x["ref_compressed_len"] > 0
    ]
    sp_ratio = [
        x["new_subproofs"] / x["ref_subproofs"]
        for x in results
        if x["ref_subproofs"] > 0
    ]
    logger.info(
        f"Average proof length ratio: {np.mean(pl_ratio):.3f} (+/- {np.std(pl_ratio):.3f})"
    )
    logger.info(
        f"Average subproofs ratio: {np.mean(sp_ratio):.3f} (+/- {np.std(sp_ratio):.3f})"
    )
    logger.info(
        f"Ref compressed tokens: {sum(x['ref_compressed_len'] for x in results)}"
    )
    logger.info(
        f"New compressed tokens: {sum(x['new_compressed_len'] for x in results)}"
    )
    logger.info(f"Ref subproofs: {sum(x['ref_subproofs'] for x in results)}")
    logger.info(f"New subproofs: {sum(x['new_subproofs'] for x in results)}")
