# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from typing import NamedTuple

from evariste.forward.common import ForwardGoal
from evariste.forward.env_specifics.generation_stats import GenerationStats


class AnnotatedGoal(NamedTuple):
    """
    Used to create an AnnotatedGeneration
    """

    selected_goal: ForwardGoal
    stats: GenerationStats
    generation_reward: float
