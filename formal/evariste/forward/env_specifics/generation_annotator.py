# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field
from typing import List
import abc

from params import Params
from evariste.forward.common import GenerationHistory
from evariste.forward.env_specifics.common import AnnotatedGoal


@dataclass
class NodeSelectionConfig(Params):

    # selection of generated goals
    n_send_to_provers: int = 1
    reward_type: str = field(
        default="size_by_depth",
        metadata={"help": "Reward type (size, depth, size_by_depth)."},
    )
    select_method: str = field(
        default="max",
        metadata={"help": "How to select goals from the generated graph (max, last)"},
    )

    def __post_init__(self):
        assert self.select_method in ["last", "max"]


class GenerationAnnotator(abc.ABC):
    """Class to
     1. select node that will be prover goal
     2. compute reward for this selected node
     3. compute stats for the generation and this selected node
    """

    @abc.abstractmethod
    def annotate_and_select_goals(
        self, history: GenerationHistory, select_cfg: NodeSelectionConfig
    ) -> List[AnnotatedGoal]:
        pass
