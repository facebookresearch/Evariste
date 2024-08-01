# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass

from params import Params, ConfStore
from evariste.model.data.envs.replay_buffer_loader import ReplayBufferArgs


@dataclass
class RLTrainArgs(Params):
    replay_buffer: ReplayBufferArgs
    negative_rewards: bool = False
    critic_weight: float = 1
    detach_critic: bool = False

    def _check_and_mutate_args(self):
        if self.negative_rewards:
            assert (
                self.replay_buffer.discount == 1
            ), "negative rewards => discount = 1 for now"
        assert self.critic_weight >= 0


ConfStore["default_rl"] = RLTrainArgs(
    replay_buffer=ReplayBufferArgs(), negative_rewards=False
)
ConfStore["neg_rl"] = RLTrainArgs(
    replay_buffer=ReplayBufferArgs(discount=1), negative_rewards=True
)
