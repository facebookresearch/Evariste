# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import abc
from typing import List

from numpy.random.mtrand import RandomState

from evariste.forward.common import ForwardGoal


class ForwardGoalFactory(abc.ABC):
    @abc.abstractmethod
    def build_forward_goals(self, split: str, debug: bool = False) -> List[ForwardGoal]:
        pass

    @abc.abstractmethod
    def build_generation_goal(self, rng: RandomState, split: str) -> ForwardGoal:
        pass
