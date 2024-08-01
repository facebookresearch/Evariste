# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from params import Params
from dataclasses import dataclass, field


@dataclass
class GANTrainingArgs(Params):

    smooth_eps: float = field(
        default=0, metadata={"help": "Epsilon smoothing for discriminator training."},
    )
    train_disc_freq: int = field(
        default=1,
        metadata={
            "help": (
                "Discriminator training frequency. N for N discriminator steps "
                "every generator step. The opposite if N negative."
            )
        },
    )
    train_real_freq: int = field(
        default=1,
        metadata={
            "help": (
                "Number of training steps on real data for "
                "each training step on generated sentences."
            )
        },
    )

    fixed_generator: bool = field(
        default=False,
        metadata={"help": ("Fixed generator to train only discriminator")},
    )

    def _check_and_mutate_args(self):
        assert 0 <= self.smooth_eps < 0.5
        assert self.train_disc_freq >= 1 or self.train_disc_freq <= -2
        assert self.train_real_freq >= 1
