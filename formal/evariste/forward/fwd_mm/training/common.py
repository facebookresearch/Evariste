# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from typing import Optional, List

from evariste.envs.mm.utils import Node


@dataclass
class MMFwdTrainingProof:
    """
    Class that represent the raw proof data. It can be obtained form supervised dataset
    or generated data (with online generation or adversarial training).

    It is transformed by the MMGraphSampler in MMFwdGraphData (by applying different
    data augmentation, goal and target sampling),
    which is then transformed in Dict[str, List[int]] by the MMGraphSampler
    """

    name: str
    root: Node
    generated: bool
    proved: Optional[bool] = None  # None if unk

    # in adversarial training, we don't have only a root node but a full trajectory
    traj: Optional[List[Node]] = None
    reward: Optional[float] = None
    reward_quantile: Optional[int] = None
