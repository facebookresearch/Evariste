# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import re
from typing import List

from numpy.random.mtrand import RandomState

from evariste.utils import find_descendents
from evariste.backward.env.hol_light.graph import HLTheorem
from evariste.envs.hl.utils import get_dag_and_th
from evariste.forward.common import ForwardGoal
from evariste.forward.env_specifics.fwd_goal_factory import ForwardGoalFactory
from evariste.trainer.args import TrainerArgs


class HLForwardGoalFactory(ForwardGoalFactory):
    def __init__(self, params: TrainerArgs):
        self.params = params

    def build_forward_goals(self, split: str, debug: bool = False) -> List[ForwardGoal]:
        if split.startswith("miniF2F"):
            from evariste.benchmark.miniF2F.hl_miniF2F import (
                get_miniF2F_goals_from_repo,
            )

            match = re.match(r"miniF2F-(?P<actual_split>.*)", split)
            goals = get_miniF2F_goals_from_repo(
                proving_mode="fwd", splitted=match is not None
            )
            if match:
                return goals[match.group("actual_split")]
            return goals

        available = []
        with open(
            os.path.join(self.params.hl.dataset.data_dir, f"split.{split}"), "r"
        ) as f:
            for i, label in enumerate(f.readlines()):
                available.append(label.strip())
        if debug:
            available = available[:100]
        dag, label_to_th, _ = get_dag_and_th(
            self.params.hl.dataset.data_dir,
            custom_dag=self.params.hl.dataset.custom_dag,
        )

        goals = []
        for name in available:
            th: HLTheorem = label_to_th[name]
            forbidden = set(find_descendents(dag, name))
            assert forbidden is not None and len(forbidden) > 0
            goal = ForwardGoal(
                thm=th, forbidden=forbidden, statement="", e_hyps=[], label=name
            )
            goals.append(goal)
        return goals

    def build_generation_goal(self, rng: RandomState, split: str) -> ForwardGoal:
        raise NotImplementedError
