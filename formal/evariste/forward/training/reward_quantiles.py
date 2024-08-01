# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import deque
from typing import Tuple, Deque

import numpy as np


class RewardQuantiles:
    def __init__(self, size: int, n_quantiles: int):
        """
        quantiles: array_like of float
        Sequence of quantiles to compute, which must be between 0 and 1 inclusive.
        """
        self.size = size
        assert n_quantiles > 1
        self.n_quantiles = n_quantiles
        self.quantiles_to_compute = np.arange(n_quantiles)[1:] / n_quantiles
        assert len(self.quantiles_to_compute) == n_quantiles - 1

        self.deque: Deque[float] = deque(maxlen=size)
        self.cur_quantiles = np.array([])

    def update_with_reward(self, reward: float):
        self.deque.append(reward)
        self.cur_quantiles = np.quantile(self.deque, self.quantiles_to_compute)

    def get_quantile_idx(self, reward: float) -> int:
        assert len(self.deque) > 0
        assert len(self.cur_quantiles) == self.n_quantiles - 1
        idx = np.searchsorted(self.cur_quantiles, reward, side="right")
        assert 0 <= idx < self.n_quantiles
        return idx
