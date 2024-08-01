# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from typing import List

from numpy.random import RandomState

from evariste.backward.env.equations import EQTheorem
from evariste.envs.eq.graph import Node
from evariste.forward.common import ForwardGoal
from evariste.forward.env_specifics.fwd_goal_factory import ForwardGoalFactory
from evariste.trainer.args import TrainerArgs

logger = getLogger()


def eq_forward_goal(
    thm: EQTheorem, name: str, is_generation: bool = False
) -> ForwardGoal[EQTheorem]:
    # global hyps are handled separatly
    return ForwardGoal(
        thm=thm if not is_generation else None,
        label=name,
        global_hyps=[EQTheorem(node=node, hyps=thm.eq_hyps) for node in thm.eq_hyps],
    )


def eq_generation_goal_from_hyps(
    eq_hyps: List[Node], name: str,
) -> ForwardGoal[EQTheorem]:
    # global hyps are handled separatly
    return ForwardGoal(
        thm=None,
        label=name,
        global_hyps=[EQTheorem(node=node, hyps=eq_hyps) for node in eq_hyps],
    )


class EQForwardGoalFactory(ForwardGoalFactory):
    def __init__(self, eq_data_env, params: TrainerArgs):

        from evariste.model.data.envs.equations import EquationsEnvironment

        assert isinstance(eq_data_env, EquationsEnvironment)
        self.eq_data_env: EquationsEnvironment = eq_data_env
        self.params = params

    def build_forward_goals(self, split: str, debug: bool = False) -> List[ForwardGoal]:
        eq_data_env = self.eq_data_env
        if split == "identities" and split not in eq_data_env.labels:
            logger.warning("Building identities dataset!")
            eq_data_env.create_identities_dataset()
        available = eq_data_env.labels[split]
        goals = []
        for name in available:
            th: EQTheorem = eq_data_env.label_to_eq[name]
            # global hyps are handled separatly
            goal = eq_forward_goal(thm=th, name=name)
            goals.append(goal)
            assert goal.use_global_hyps()
        return goals

    def build_generation_goal(self, rng: RandomState, split: str) -> ForwardGoal:
        assert split in ["train", "valid", "test"]

        n_init_hyps = rng.randint(self.params.eq.dataset.max_init_hyps + 1)
        init_hyps = self.eq_data_env.egg.create_init_hyps(n_init_hyps)  # type: ignore
        name = str(rng.randint(1 << 60))
        goal: ForwardGoal = eq_generation_goal_from_hyps(
            eq_hyps=init_hyps.nodes, name=name
        )
        assert goal.use_global_hyps()
        return goal
