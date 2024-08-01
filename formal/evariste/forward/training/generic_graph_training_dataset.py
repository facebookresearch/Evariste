# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Generic, Dict, List, Tuple

import numpy
from numpy.random import RandomState

from evariste.forward.training.graph_sampler import (
    SomeProofNode,
    GraphTrainingSample,
    GraphTrainingDataset,
)
from evariste.forward.training.helpers import sample_from_cumulative


class GenericGraphTrainingDataset(
    Generic[SomeProofNode], GraphTrainingDataset[SomeProofNode]
):
    """
    Generic (= env agnostic) implementation of GraphTrainingDataset that sample
    for a list of ProofNode
    """

    def __init__(
        self,
        nodes: Dict[str, List[Tuple[str, SomeProofNode]]],
        weights: Dict[str, List[float]],
    ):
        self.nodes = nodes
        self.cumulatives = {s: numpy.cumsum(w) for s, w in weights.items()}
        assert set(self.nodes.keys()) == set(self.cumulatives.keys())
        assert all(
            len(self.nodes[s]) == len(self.cumulatives[s]) for s in self.nodes.keys()
        )

        self._nodes_for_noise = {s: [n for _, n in ns] for s, ns in nodes.items()}

    def get_graph_training_sample(
        self, task: str, split: str, rng: RandomState
    ) -> GraphTrainingSample[SomeProofNode]:
        cumulatives = self.cumulatives[split]
        label, root = sample_from_cumulative(cumulatives, self.nodes[split], rng)
        return GraphTrainingSample(label=label, root=root)

    def nodes_for_noise(self, split: str) -> List[SomeProofNode]:
        return self._nodes_for_noise[split]

    def close(self):
        pass
