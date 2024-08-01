# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple

import torch
from numpy.random.mtrand import RandomState
import numpy as np

from evariste.comms.store import AnnotatedGeneration
from evariste.forward.fwd_mm.mm_helpers import history_to_mm_nodes
from evariste.forward.fwd_mm.training.common import MMFwdTrainingProof
from evariste.model.data.envs.replay_buffer_loader import ReplayBuffer


def sample_node_sequence_from_rb(
    split: str,
    replay_buffer: ReplayBuffer[AnnotatedGeneration],
    block: bool,
    params,
    rng: RandomState,
) -> Tuple[MMFwdTrainingProof, np.array, AnnotatedGeneration]:
    from evariste.trainer.args import TrainerArgs

    assert isinstance(params, TrainerArgs)

    assert split == "train"
    if params.mm.cond_gen_with_proved:
        assert not replay_buffer.filter_if_rewards_zero
    annotated_generation, info = replay_buffer.get_sample(
        split=split, index=None, rng=rng, block=block
    )
    if len(annotated_generation.generation.forward_steps()) == 0:
        raise RuntimeError("This should have been filtered!")
    sequence = history_to_mm_nodes(annotated_generation.generation)
    annotated_generation.grabbed()

    # compute discounted returns
    # TODO: shouldn't we remove this?
    power = torch.FloatTensor(range(len(sequence))[::-1])  # [len(seq) - 1, ... , 0]
    discounts = params.rl_params.replay_buffer.discount ** power
    returns = discounts * annotated_generation.reward

    if replay_buffer.filter_if_rewards_zero:
        assert any(r != 0 for r in returns)
    assert len(sequence) > 0

    goal_statement = annotated_generation.prover_goal.statement
    candidates = [n for n in sequence if n.statement_str == goal_statement]

    assert len(candidates) == 1, (
        f"goal_statement: {goal_statement}\n"
        f"candidates: {candidates}\n"
        f"node statements: {[node.statement for node in sequence]}"
    )
    goal = candidates[0]

    assert goal.ltype != "$e", "e_hyps expected to be at the beginning"

    was_proved = annotated_generation.proved

    sample = MMFwdTrainingProof(
        name="from_rb",
        generated=True,
        proved=was_proved,
        root=goal,
        traj=sequence,
        reward=annotated_generation.reward,
        reward_quantile=info.reward_quantile,
    )
    return sample, returns, annotated_generation
