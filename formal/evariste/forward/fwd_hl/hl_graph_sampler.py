# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from numpy.random.mtrand import RandomState
import numpy as np

from evariste.model.data.dictionary import (
    B_NODE_WORD,
    E_NODE_WORD,
    Dictionary,
)
from evariste.backward.env.hol_light.graph import HLProofNode
from evariste.forward.training.helpers import postorder_traversal

logger = getLogger()

HL_FWD_TASK = "hl_fwd_seq2seq"
HL_GEN_TASK = "hl_gen_seq2seq"
HL_FWD_TASKS = [HL_FWD_TASK, HL_GEN_TASK]


Goal = HLProofNode
Target = HLProofNode
Label = str
Split = str  # "train", "valid" or "test"


class HLGraphSampler:
    def __init__(
        self,
        roots: Dict[str, List[HLProofNode]],
        names: Dict[str, List[str]],
        weights: Dict[str, np.array],
        dico: Dictionary,
        max_len: int,
    ):
        assert set(weights) == set(roots) == set(names)
        for split in names:
            assert len(roots[split]) == len(names[split]) == len(weights[split])
        self.cumulatives = {s: np.cumsum(w) for s, w in weights.items()}
        self.roots = roots
        self.names = names
        self.dico = dico
        self.max_len = max_len

    def _sample_graph(
        self, split: str, rng: RandomState
    ) -> Tuple[Label, Goal, Target, List[HLProofNode]]:
        cumulative = self.cumulatives[split]
        names = self.names[split]
        roots = self.roots[split]

        index = np.searchsorted(
            cumulative,  # a
            rng.random() * cumulative[-1],  # v
            side="right",  # a[i-1] <= v < a[i]
        )

        name = names[index]
        root = roots[index]

        order = postorder_traversal(root, rng=rng)
        # For the moment we only use the final goal as goal, not intermediate steps
        goal = order[-1]
        assert root == goal

        output_idx = rng.randint(0, len(order))
        target = order[output_idx]
        graph = order[:output_idx]

        return name, goal, target, graph

    def _make_sample(
        self,
        name: str,
        goal: HLProofNode,
        target: HLProofNode,
        graph: List[HLProofNode],
        include_goal: bool,
    ) -> Optional[Dict]:

        eos = self.dico.eos_word

        # encoder input
        # TODO: maybe create an other way to tokenize theorem when there are a node
        #  (and not a goal). Now the token <GOAL> is used in node.theorem.tokenize()
        #  which is probably bad
        #  + remove <HYP_NAME> token
        if include_goal:
            enc_inp = goal.theorem.tokenize()
        else:
            enc_inp = []
        for node in graph:
            enc_inp.append(B_NODE_WORD)
            enc_inp.extend(node.theorem.tokenize()[1:-1])
            enc_inp.append(E_NODE_WORD)

        # decoder output
        # TODO: try with tactic / theorem first
        next_node = [B_NODE_WORD, *target.theorem.tokenize()[1:-1], E_NODE_WORD]
        next_tactic = target.tactic.tokenize()
        dec_out = next_node + next_tactic

        # add sequence delimiters
        enc_inp = [eos, *enc_inp, eos]
        dec_out = [eos, *dec_out, eos]

        # skip too long sequences
        if max(len(enc_inp), len(dec_out)) > self.max_len:
            return None

        # index sequences
        enc_inp = [self.dico.index(t) for t in enc_inp]
        dec_out = [self.dico.index(t) for t in dec_out]

        return {"name": name, "x": enc_inp, "y": dec_out}

    def get_graph_sample(
        self, split: str, include_goal: bool, rng: RandomState
    ) -> Optional[Dict]:
        name, goal, target, order = self._sample_graph(split=split, rng=rng)
        return self._make_sample(name, goal, target, order, include_goal)

    @classmethod
    def from_proofs(
        cls,
        proofs: Dict[Label, HLProofNode],
        splits: Dict[Split, List[str]],  # {Split: List[Label]}
        dico: Dictionary,
        max_len: int,
    ) -> "HLGraphSampler":
        weights: Dict[Split, List[int]] = defaultdict(list)
        roots: Dict[Split, List[HLProofNode]] = defaultdict(list)
        names: Dict[Split, List[Label]] = defaultdict(list)
        for split, labels in splits.items():
            for name in labels:
                root = proofs[name]
                nodes = postorder_traversal(root, rng=None)
                weights[split].append(len(nodes))
                roots[split].append(root)
                names[split].append(name)
        weights = {s: np.array(w) for s, w in weights.items()}
        return cls(
            roots=roots, weights=weights, names=names, dico=dico, max_len=max_len
        )
