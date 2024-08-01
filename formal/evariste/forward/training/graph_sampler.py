# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Number
from typing import Generic, TypeVar, List, Any, Dict, Optional

from numpy.random import RandomState

from evariste.forward.common import ProofNode

SomeProofNode = TypeVar("SomeProofNode", bound=ProofNode)


class GraphSampler(ABC):
    """
    Abstract GraphSampler

    Implementations are supposed to:
    - Sample a graph for forward training
    - Apply proofs augmentation (shuffling order, noise)
    - Apply tokenization
    - Return a sample (dict)
    """

    @abstractmethod
    def get_graph_sample(
        self, split: str, task: str, rng: RandomState
    ) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Number]:
        pass

    @abstractmethod
    def close(self):
        pass


@dataclass
class GraphTrainingSample(Generic[SomeProofNode]):
    label: str
    root: SomeProofNode


class GraphTrainingDataset(ABC, Generic[SomeProofNode]):
    """
    Abstract GraphTrainingDataset

    Allow to sample a graph from a training dataset for fwd training.

    Supposed to be used in a GraphSampler, This allow to abstract a dataset and switch
    easily between human fixed dataset, online generation datasets, replay buffer.
    """

    @abstractmethod
    def get_graph_training_sample(
        self, task: str, split: str, rng: RandomState
    ) -> GraphTrainingSample[SomeProofNode]:
        pass

    @abstractmethod
    def nodes_for_noise(self, split: str) -> List[SomeProofNode]:
        """
        Collections of nodes that can be used for noise injection
        by GraphSampler
        """
        pass

    @abstractmethod
    def close(self):
        pass
