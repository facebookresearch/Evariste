# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest

from evariste.backward.env.metamath import MMTheorem
from evariste.forward.common import (
    ForwardGoal,
    GenerationHistory,
    MaybeForwardStep,
    ForwardStep,
)
from evariste.forward.fwd_mm.mm_helpers import MMForwardTactic, MMEnvInfo
from evariste.forward.online_generation.goal_selectors.goal_selector import (
    GeneratedGoalSelector,
    MixedGoalSelector,
    UniformGoalSelector,
)
import numpy as np


@pytest.mark.skip(reason="use old MM fmt")
def test_generated_goal_selector():
    selector = GeneratedGoalSelector(
        max_goals=5, seed=42, sampling_strategy="size", alpha=1
    )
    generation = build_generation()
    _ = selector.stats

    selector.update_with_generation(0, generation, solved=True)
    selector.update_state()
    assert len(selector.goals) == 3
    assert np.allclose(selector.weight_cumulative, [1, 3, 4])
    selector.update_with_generation(0, generation, solved=True)
    selector.update_state()
    assert len(selector.goals) == 3
    assert np.allclose(selector.weight_cumulative, [1, 3, 4])
    print(selector.goal_stats)
    assert selector.goal_stats[2].n_proved == 1
    for i in range(3):
        assert selector.goal_stats[i].n_generated == 2

    _ = selector.select_goal()
    assert max(s.n_selected for s in selector.goal_stats) == 1

    state = selector.state_dict()
    new_selector = GeneratedGoalSelector(
        max_goals=5, seed=42, sampling_strategy="size", alpha=1
    )
    new_selector.load_state_dict(state)
    assert np.allclose(new_selector.weight_cumulative, selector.weight_cumulative)
    assert selector.goal_stats == new_selector.goal_stats
    assert selector.forbiddens == new_selector.forbiddens
    print(new_selector.forbiddens)
    print(new_selector.select_goal())

    selector = GeneratedGoalSelector(
        max_goals=5, seed=42, sampling_strategy="visited", alpha=2
    )
    selector.update_with_generation(2, generation, solved=True)
    selector.update_with_generation(3, generation, solved=True)
    selector.update_state()
    assert np.allclose(
        selector.weight_cumulative, np.cumsum(np.array([0.5, 0.5, 0.5]) ** 2)
    )

    selector = GeneratedGoalSelector(
        max_goals=5, seed=42, sampling_strategy="proof_visited_score", alpha=1
    )
    selector.update_with_generation(2, generation, solved=True)
    selector.update_state()
    selector.update_with_generation(3, generation, solved=True)
    selector.update_state()
    print(selector.stats)
    assert np.allclose(selector.weight_cumulative, np.cumsum([0.5, 1.0, 0.5]))
    print(selector.forbiddens)

    selector = GeneratedGoalSelector(
        max_goals=1, seed=42, sampling_strategy="visited", alpha=1
    )
    selector.update_with_generation(0, generation, solved=True)
    selector.update_with_generation(1, generation, solved=True)
    selector.update_state()
    print(selector.goals)
    print(selector.goal_stats)
    assert np.allclose(selector.weight_cumulative, np.cumsum([0.5]))


@pytest.mark.skip(reason="use old MM fmt")
def test_mixed_goal_selector():
    generation = build_generation()
    supervised = UniformGoalSelector(seed=42, goals=[generation.goal])
    assert len(supervised.goals) == 1

    generated = GeneratedGoalSelector(
        max_goals=5, seed=42, sampling_strategy="size", alpha=1
    )

    selector = MixedGoalSelector(supervised=supervised, generated=generated, seed=42)
    _ = selector.select_goal()
    selector.update_with_generation(0, generation, solved=True)
    selector.update_state()

    assert len(generated.weight_cumulative) == 3

    new_generated = GeneratedGoalSelector(
        max_goals=5, seed=42, sampling_strategy="size", alpha=1
    )
    new_selector = MixedGoalSelector(
        supervised=supervised, generated=new_generated, seed=42
    )
    assert len(new_generated.weight_cumulative) == 0

    new_selector.load_state_dict(selector.state_dict())
    assert len(new_generated.weight_cumulative) == 3


def build_generation():
    fwd_goal = ForwardGoal(
        statement="|- ( 3 + 3 ) = 6",
        e_hyps=[],
        forbidden={"f1", "f2"},
        mand_disj=set(),
        label="unk",
        thm=None,
    )
    fwd_step_1 = ForwardStep(
        statement="|- 3 e. CC",
        children=[],
        tactic=MMForwardTactic(label="3cn", substitutions={}, children_ids=[]),
        score=-0.188421368598938,
        normalised_score=-0.037684273719787595,
        env_info=MMEnvInfo(new_disjoints=set()),
        generated=None,  # type: ignore
        fwd_tactic=MMForwardTactic(label="3cn", substitutions={}, children_ids=[]),
    )
    fwd_step_2 = ForwardStep(
        statement="|- ( 2 x. 3 ) = ( 3 + 3 )",
        children=[0],
        tactic=MMForwardTactic(
            label="2timesi", substitutions={"A": "3"}, children_ids=[0]
        ),
        score=-0.5372689962387085,
        normalised_score=-0.05372689962387085,
        env_info=MMEnvInfo(new_disjoints=set()),
        generated=None,  # type: ignore
        fwd_tactic=MMForwardTactic(label="3cn", substitutions={}, children_ids=[]),
    )
    fwd_step_3 = ForwardStep(
        statement="|- ( 3 + 3 ) = 6",
        children=[],
        tactic=MMForwardTactic(label="3p3e6", substitutions={}, children_ids=[]),
        score=-2.7461931705474854,
        normalised_score=-0.549238634109497,
        env_info=MMEnvInfo(new_disjoints=set()),
        generated=None,  # type: ignore
        fwd_tactic=MMForwardTactic(label="3p3e6", substitutions={}, children_ids=[]),
    )
    generation = GenerationHistory(
        goal=fwd_goal,
        stack=[
            MaybeForwardStep(step=fwd_step_1, err=None),
            MaybeForwardStep(step=fwd_step_2, err=None),
            MaybeForwardStep(step=fwd_step_3, err=None),
        ],
    )
    return generation
