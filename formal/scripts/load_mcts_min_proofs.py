# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pickle
from collections import defaultdict
from numbers import Number
from pathlib import Path
from typing import List, Dict, Optional, Any
from logging import getLogger

from numpy.random import RandomState
import numpy as np

from evariste.comms.store import EmptyStore
from evariste.forward.fwd_lean.training.common import LeanProofNode
from evariste.forward.fwd_lean.training.lean_graph_sampler import LeanBaseGraphSampler
from evariste.forward.training.graph_sampler import (
    GraphTrainingDataset,
    GraphTrainingSample,
    SomeProofNode,
    GraphSampler,
)
from evariste.forward.training.helpers import count_unique_theorems, postorder_traversal
from evariste.forward.utils.stats_utils import prefix_dict

logger = getLogger()


class MinProofDataset(GraphTrainingDataset[LeanProofNode]):
    def __init__(self, mcts_dir: str):
        self.proof_dir = Path(mcts_dir)
        min_proof_path = self.proof_dir / "export_min_proofs" / "min_proofs.pkl"
        self.min_proof_path = min_proof_path
        assert min_proof_path.exists()
        self.proofs = self.load_proofs()
        self.labels: List[str] = list(self.proofs.keys())
        self.nodes = self.collect_nodes_for_noise()

    def get_graph_training_sample(
        self, task: str, split: str, rng: RandomState
    ) -> GraphTrainingSample[SomeProofNode]:
        assert split == "train"
        label_id = rng.randint(len(self.labels))
        label = self.labels[label_id]
        roots = self.proofs[label]
        root = roots[rng.randint(len(roots))]
        order = postorder_traversal(root, rng=None)
        node = order[rng.randint(len(order))]
        return GraphTrainingSample(label=label, root=node)

    def nodes_for_noise(self, split: str) -> List[SomeProofNode]:
        assert split == "train"
        return self.nodes

    def close(self):
        pass

    def load_proofs(self) -> Dict[str, List[LeanProofNode]]:
        with self.min_proof_path.open("rb") as fp:
            proof_list = pickle.load(fp)
        proofs = defaultdict(list)
        for label, root in proof_list:
            proofs[label].append(root)
        proofs = dict(proofs)
        logger.info(f"{self.__class__.__name__}: found proofs for {len(proofs)} goals")
        return proofs

    def collect_nodes_for_noise(self) -> List[LeanProofNode]:
        nodes: List[LeanProofNode] = []
        for proofs in self.proofs.values():
            for proof in proofs:
                order = postorder_traversal(proof, rng=None)
                nodes.extend(order)
        return nodes


class LeanSupervisedAndBackwardGraphSampler(GraphSampler):
    """
    GraphSampler which mixed human and 'parsable bwd' generated data with prob
    that depends on the number of nodes in each dataset.
    """

    def __init__(self, human: LeanBaseGraphSampler, generated: LeanBaseGraphSampler):
        self.human = human
        self.generated = generated
        n_thms_human = len(
            {node.theorem for node in self.human.dataset.nodes_for_noise("train")}
        )
        n_thms_generated = len(
            {node.theorem for node in self.generated.dataset.nodes_for_noise("train")}
        )

        self.generated_prob = n_thms_generated / (n_thms_human + n_thms_generated)
        logger.info(
            f"{self.__class__.__name__}: n_thms_human: {n_thms_human}, "
            f"n_thms_generated: {n_thms_generated} "
            f"- generated_prob: {self.generated_prob}"
        )

        self.n_empty_store = 0

    def get_graph_sample(
        self, split: str, task: str, rng: RandomState
    ) -> Optional[Dict[str, Any]]:

        if split == "train" and rng.random() < self.generated_prob:
            try:
                return self.generated.get_graph_sample(split=split, task=task, rng=rng)
            except EmptyStore:
                self.n_empty_store += 1
                # fallback on human data

        return self.human.get_graph_sample(split, task, rng)

    def get_stats(self) -> Dict[str, Number]:
        stats = {
            "empty_store": self.n_empty_store,
        }
        stats.update(prefix_dict(self.generated.get_stats(), "generated/"))
        stats.update(prefix_dict(self.human.get_stats(), "human/"))
        self.n_empty_store = 0
        return stats

    def close(self):
        self.human.close()
        self.generated.close()


if __name__ == "__main__":
    loader = MinProofDataset("")
    n_labels_with_long_proofs = 0
    for label, proofs in loader.proofs.items():
        sizes = [count_unique_theorems(p) for p in proofs]
        if np.min(sizes) > 1:
            n_labels_with_long_proofs += 1

        print(
            len(sizes),
            np.mean([sizes]),
            np.min([sizes]),
            np.max([sizes]),
            np.median([sizes]),
        )
    print(n_labels_with_long_proofs, len(loader.proofs))
    print(len({n.theorem for n in loader.nodes}))
