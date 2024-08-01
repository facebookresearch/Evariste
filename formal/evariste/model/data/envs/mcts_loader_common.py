# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import Optional, Dict

import numpy as np

# putting abstract class here since there is mcts_loader is importing TrainerArgs


class MCTSDataLoader(ABC):
    @abstractmethod
    def get_mcts_sample(
        self, split: str, index: Optional[int], rng: np.random.RandomState,
    ):
        pass

    @property
    @abstractmethod
    def mcts_fmt(self) -> Optional[str]:
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, float]:
        pass
