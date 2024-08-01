# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy

from evariste.forward.training.reward_quantiles import RewardQuantiles


def test_reward_quantiles_init():
    quantilizer = RewardQuantiles(size=10, n_quantiles=10)

    quantilizer.update_with_reward(0)
    quantile = quantilizer.get_quantile_idx(reward=-1)
    assert quantile == 0
    quantile = quantilizer.get_quantile_idx(reward=1)
    assert quantile == 9
    assert len(quantilizer.cur_quantiles) == 9


def test_reward_quantiles():
    quantilizer = RewardQuantiles(size=10, n_quantiles=10)

    quantilizer.update_with_reward(0)
    assert numpy.allclose(quantilizer.cur_quantiles, 9 * [0])

    for reward in range(0, 10):
        quantilizer.update_with_reward(reward)

    assert quantilizer.get_quantile_idx(reward=-1) == 0
    assert quantilizer.get_quantile_idx(reward=0) == 0

    assert quantilizer.get_quantile_idx(reward=10) == 9
    assert quantilizer.get_quantile_idx(reward=10.1) == 9
    assert quantilizer.get_quantile_idx(reward=9.1) == 9
    assert quantilizer.get_quantile_idx(reward=5.1) == 5
    assert quantilizer.get_quantile_idx(reward=0.1) == 0
