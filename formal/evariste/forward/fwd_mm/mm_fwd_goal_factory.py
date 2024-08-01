# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List

from numpy.random import RandomState

from evariste.envs.mm.env import MetamathEnv
from evariste.envs.mm.utils import reward_quantile_tok
from evariste.forward.common import ForwardGoal
from evariste.forward.env_specifics.fwd_goal_factory import ForwardGoalFactory
from evariste.forward.fwd_mm.mm_fwd_tasks import MMFwdFormat
from evariste.forward.fwd_mm.mm_helpers import load_splits, build_forward_goals
from evariste.model.data.dictionary import UNPROVED_WORD
from evariste.trainer.args import TrainerArgs


class MMForwardGoalFactory(ForwardGoalFactory):
    def __init__(self, params: TrainerArgs, mm_env: MetamathEnv, fmt: MMFwdFormat):
        self.params = params
        self.mm_env = mm_env
        self.fmt = fmt

        self._splits = None

    def build_forward_goals(self, split: str, debug: bool = False) -> List[ForwardGoal]:
        return build_forward_goals(
            data_dir=self.params.mm.dataset.data_dir,
            split=split,
            mm_env=self.mm_env,
            debug=debug,
        )

    def build_generation_goal(self, rng: RandomState, split: str) -> ForwardGoal:
        if self._splits is None:
            self._splits = load_splits(self.params.mm.dataset.data_dir)
        names = self._splits[split]
        name = names[rng.randint(len(names))]
        assertion = self.mm_env.labels[name][1]
        e_hyps = [" ".join(h) for h in assertion.e_hyps]

        label_conditioning = None
        if self.fmt.label_conditioning:
            # We sample uniformly a train label to condition the generator with this
            # label. The generator is supposed to manage to generate a proof that will
            # use this label
            train_labels = self._splits["train"]
            label_conditioning = train_labels[rng.randint(len(train_labels))]

        reward_quantile_conditioning = None
        if self.fmt.reward_quantile_conditioning:
            best_quantile = self.params.rl_params.replay_buffer.n_reward_quantiles - 1
            assert best_quantile >= 0
            reward_quantile_conditioning = reward_quantile_tok(best_quantile)

        return ForwardGoal(
            statement="",
            e_hyps=e_hyps,
            forbidden=None,
            mand_disj=set(),
            label=assertion.label,
            label_conditioning=label_conditioning,
            proved_conditioning=UNPROVED_WORD if self.fmt.proved_conditioning else None,
            reward_quantile_conditioning=reward_quantile_conditioning,
        )
