# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import re
import logging
import argparse
import itertools
import shutil
import pdb
import io
import numpy as np
from collections import deque, Counter, OrderedDict
from engine import Frame, FrameStack, MetamathEnv, get_parser

E_HYP_NAMES = "E_HYP_%i"

logger = logging.getLogger()


class MetamathEnvStat(MetamathEnv):
    def process(self):
        """
        Process buffer.
        """
        self.fs.push()
        label = None
        last_comment = None
        started_process = self.args.start_label is None
        initial_buffer_length = len(self.buffer)
        log_interval = int(initial_buffer_length / 20)
        logged_up_to = 0
        self.proof_labels = []

        while len(self.buffer) > 0:

            progress = initial_buffer_length - len(self.buffer)
            # pdb.set_trace()
            if progress > logged_up_to + log_interval:
                percentage_complete = 100.0 * progress / initial_buffer_length
                logging.info(f"[{percentage_complete:1.1f}%/100%]")
                logged_up_to = logged_up_to + log_interval

            tok = self.buffer.pop()

            # comments are saved into last_comment to be pushed with corresponding later next
            if tok == "$(":
                last_comment = self.read_tokens_until("$)")
                last_comment_str = " ".join(last_comment)
                # remove HTML and <A> here to deal with unclosed <HTML> tags
                last_comment_str = re.sub(
                    r"<A(.*?)>(.*?)</A>", r"", last_comment_str, flags=re.DOTALL
                )
                last_comment_str = re.sub(
                    r"<HTML>(.*?)</HTML>", r"", last_comment_str, flags=re.DOTALL
                )  # remove HTML (which appears in comments and is annoying to parse)
                last_comment = last_comment_str.split()
                if "$t" in last_comment:
                    # parse this comment further to get metamath <-> latex mappings
                    self.parse_typesetting(last_comment)
                continue

            # open block
            if tok == "${":
                self.fs.push()
                self.log(f"========== New block (level {len(self.fs)})", 19)
                continue

            # close block
            if tok == "$}":
                self.fs.pop()
                self.log(f"========== End block (level {len(self.fs) + 1})", 19)
                continue

            # $c
            if tok == "$c":
                tokens = self.read_tokens_until()
                tokens = self.remove_comments(tokens)
                for tok in tokens:
                    self.fs.add_c(tok)
                self.log(f"$c: {', '.join(tokens)}", 18)
                continue

            # $v
            if tok == "$v":
                tokens = self.read_tokens_until()
                tokens = self.remove_comments(tokens)
                for tok in tokens:
                    self.fs.add_v(tok)
                self.log(f"$v: {', '.join(tokens)}", 18)
                continue

            # $d
            if tok == "$d":
                tokens = self.read_tokens_until()
                tokens = self.remove_comments(tokens)
                self.fs.add_d(tokens)
                self.log(f"$d: {', '.join(tokens)}", 18)
                continue

            # check label exists
            if tok in ["$f", "$e", "$a", "$p"]:
                if label is None:
                    raise Exception(f"{tok} must have a label")
                assert label not in self.labels

            # $f
            if tok == "$f":
                tokens = self.read_tokens_until()
                tokens = self.remove_comments(tokens)
                if len(tokens) != 2:
                    raise Exception("$f must contain a typecode and a variable")
                self.fs.add_f(label, tokens[0], tokens[1])
                self.log(f"$f ({label}): name={tokens[1]} type={tokens[0]}", 18)
                label = None
                continue

            # $e
            if tok == "$e":
                tokens = self.read_tokens_until()
                tokens = self.remove_comments(tokens)
                self.fs.add_e(label, tokens)
                self.log(f"$e ({label}): {' '.join(tokens)}", 18)
                label = None
                continue

            # $a
            if tok == "$a":
                assert label is not None and label not in self.labels
                tokens = self.read_tokens_until()
                tokens = self.remove_comments(tokens)
                assertion = self.fs.make_assertion(tokens)
                self.labels[label] = ("$a", assertion)
                assert last_comment is not None, "missing comment for assertion"
                self.comments[label] = last_comment
                self.log(
                    f"$a ({label}): {' '.join(tokens)} // {' '.join(last_comment)}", 19
                )
                label = None
                last_comment = None
                continue

            # $p
            if tok == "$p":
                tokens = self.read_tokens_until()
                tokens = self.remove_comments(tokens)
                if "$=" not in tokens:
                    raise Exception("$p must contain proof after $=")
                i = tokens.index("$=")
                statement = tokens[:i]
                comp_proof = tokens[i + 1 :]
                assertion = self.fs.make_assertion(statement)
                if self.args.rename_e_hyps:
                    comp_proof, assertion = self.rename_essential_hyps(
                        comp_proof, assertion
                    )
                self.log(f"$p ({label}): {' '.join(statement)}", 19)
                self.compressed_proofs[label] = comp_proof
                self.labels[label] = ("$p", assertion)
                assert (
                    last_comment is not None
                ), "didn't find comment for provable theorem"
                self.comments[label] = last_comment
                if started_process:
                    assert comp_proof[0] == "("
                    self.log(f'===== Decompressing "{label}" proof', 19)
                    decomp_proof = self.decompress_proof(assertion, comp_proof)
                    self.decompressed_proofs[label] = decomp_proof
                    self.proof_labels.append(label)
                    # pdb.set_trace()

                label = None
                last_comment = None
                continue

            # label
            assert tok[0] != "$", f"unexpected token {tok}"
            assert label is None
            assert tok not in self.labels
            label = tok

            # start label (optional)
            if self.args.start_label is not None and label == self.args.start_label:
                logger.info(
                    f"Start label: {self.args.start_label} - Starting processing..."
                )
                started_process = True

            # stop label (optional)
            if self.args.stop_label is not None and label == self.args.stop_label:
                logger.info(
                    f"Stop label: {self.args.stop_label} - Stopping processing..."
                )
                break

        # label stats
        logger.info(f"========== Read {len(self.labels)} labels ==========")
        label_types = Counter(k for (k, _) in self.labels.values())
        for k, v in label_types.items():
            logger.info(f"{k} labels: {v}")


if __name__ == "__main__":

    # parse arguments
    parser = get_parser()
    args = parser.parse_args()
    assert os.path.isfile(args.database_path)

    logging.basicConfig(level=logging.INFO)

    # create Metamath instance
    mm_env = MetamathEnvStat(filepath=args.database_path, args=args)

    mm_env.process()

    proof_lens = []
    proof_uniques = []
    proof_non_wff_uniques = []
    e_used = []
    f_used = []
    statement_lens = []
    statement_uniques = []

    for label in mm_env.proof_labels:
        assertion = mm_env.labels[label][1]

        proof = mm_env.decompressed_proofs[label]
        proof_lens.append(len(proof))
        proof_unique = (
            set(proof)
            - set(assertion["e_hyps_labels"])
            - set(assertion["f_hyps_labels"])
        )
        proof_uniques.append(len(proof_unique))

        proof_non_wff_unique = {p for p in proof_unique if p[0] != "w"}
        proof_non_wff_uniques.append(len(proof_non_wff_unique))

        e_used.append(len(assertion["e_hyps"]))
        f_used.append(len(assertion["f_hyps"]))
        statement_lens.append(len(assertion["tokens"]))
        statement_uniques.append(len(set(assertion["tokens"])))
