# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#

from evariste.envs.mm.env import MetamathEnv
import numpy as np
import logging


class ProofInteriorNode:
    def __init__(self, label, hypotheses, expression):
        self.label = label
        self.hypotheses = hypotheses
        self.expression = expression

    def proof(self, ftbl):
        seq = []
        for hypothesis in self.hypotheses:
            seq.extend(hypothesis.proof(ftbl))

        seq.append(self.label)
        return seq

    def __str__(self):
        s = f"{self.label}: {' '.join(self.expression)}\n"
        for hypothesis in self.hypotheses:
            s = s + hypothesis.__str__() + "\n"
        return s


class ProofLeafNode:
    def __init__(self, expression):
        self.expression = expression

    def proof(self, ftbl):
        new_label = ftbl[" ".join(self.expression)]
        return [new_label]

    def __str__(self):
        return " ".join(self.expression)


class SyntaxEngine(MetamathEnv):
    def __init__(self, filepath=None, buffer=None, args=None):
        super().__init__(
            filepath,
            buffer,
            start_label=args.start_label,
            stop_label=args.stop_label,
            rename_e_hyps=args.rename_e_hyps,
            decompress_proofs=args.decompress_proofs,
            verify_proofs=args.verify_proofs,
            log_level="debug",
        )
        self.args = args
        self.sample_wff_prob = 0.01
        self.wffs = []
        self.syntatic = set(["setvar", "wff", "class"])

    def _process_start(self):
        self.initial_buffer_length = len(self.buffer)
        self.log_interval = int(self.initial_buffer_length / 50)
        self.logged_up_to = 0

    def _process_token_start(self):
        progress = self.initial_buffer_length - len(self.buffer)
        # pdb.set_trace()
        if progress > self.logged_up_to + self.log_interval:
            percentage_complete = 100.0 * progress / self.initial_buffer_length
            logging.info(f"[{percentage_complete:1.1f}%/100%]")
            self.logged_up_to = self.logged_up_to + self.log_interval

    def _verify_start(self):
        self.node_stack = []

    def _verify_hypothesis_after(self, step_type, step_data):
        node = ProofLeafNode(expression=step_data)
        self.node_stack.append(node)

    def _verify_proposition_after(self, label, formula, n_pop):
        hypotheses = self.node_stack[len(self.node_stack) - n_pop :]

        # remove hypotheses from node stack
        del self.node_stack[len(self.node_stack) - n_pop :]

        node = ProofInteriorNode(label=label, hypotheses=hypotheses, expression=formula)
        self.node_stack.append(node)

        if len(formula) > 0 and formula[0] in self.syntatic:
            if np.random.uniform() < self.sample_wff_prob:
                assertion = self.fs.make_assertion(formula)
                assertion["proof_tree"] = node
                self.wffs.append(assertion)
