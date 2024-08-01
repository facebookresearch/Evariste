# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from numbers import Number
from typing import Optional, Dict, Any

from numpy.random.mtrand import RandomState

from evariste.comms.store import EmptyStore
from evariste.forward.training.graph_sampler import GraphSampler
from evariste.forward.utils.stats_utils import prefix_dict


class LeanMixedGraphSampler(GraphSampler):
    """
    GraphSampler which mixed human and 'online' generated data with prob
    'generated_prob'

    Could be env agnostic if needed by HL
    (we need to remove the factory method in this case)
    """

    def __init__(
        self, human: GraphSampler, generated: GraphSampler, generated_prob: float
    ):
        self.human = human
        self.generated = generated
        self.generated_prob = generated_prob

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
