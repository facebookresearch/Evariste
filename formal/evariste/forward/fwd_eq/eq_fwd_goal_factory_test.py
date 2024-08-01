# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from numpy.random import RandomState

from evariste.backward.env.equations import EQTheorem
from evariste.envs.eq.graph import eq_nodes_are_equal
from evariste.forward.common import ForwardGoal
from evariste.forward.fwd_eq.eq_fwd_goal_factory import EQForwardGoalFactory
from evariste.forward.fwd_eq.conftest import trainer_args

from evariste.model.data.envs.equations import EquationsEnvironment


def test_fwd_goal_factory(eq_env: EquationsEnvironment):
    goal_factory = EQForwardGoalFactory(eq_data_env=eq_env, params=trainer_args())
    fwd_goals = goal_factory.build_forward_goals("identities", debug=True)

    for fwd_goal in fwd_goals[:20]:
        assert isinstance(fwd_goal, ForwardGoal)
        assert isinstance(fwd_goal.thm, EQTheorem)
        assert isinstance(fwd_goal.global_hyps, list)
        assert len(fwd_goal.thm.eq_hyps) == len(fwd_goal.global_hyps)
        print(len(fwd_goal.global_hyps))
        for node_a, hyp_thm_b in zip(fwd_goal.thm.eq_hyps, fwd_goal.global_hyps):
            assert isinstance(hyp_thm_b, EQTheorem)
            assert node_a.eq(hyp_thm_b.eq_node)
            assert eq_nodes_are_equal(hyp_thm_b.eq_hyps, fwd_goal.thm.eq_hyps)
    # assert 0


def test_gen_goal_factory(eq_env: EquationsEnvironment):
    goal_factory = EQForwardGoalFactory(eq_data_env=eq_env, params=trainer_args())
    rng = RandomState(0)
    for _ in range(20):
        fwd_goal = goal_factory.build_generation_goal(rng, "train")
        assert isinstance(fwd_goal, ForwardGoal)
        assert fwd_goal.thm is None
        assert isinstance(fwd_goal.global_hyps, list)
        assert all(isinstance(hyp, EQTheorem) for hyp in fwd_goal.global_hyps)
        print(len(fwd_goal.global_hyps))
        eq_hyps = [hyp_thm.eq_node for hyp_thm in fwd_goal.global_hyps]
        for hyp_thm in fwd_goal.global_hyps:
            assert isinstance(hyp_thm, EQTheorem)
            assert eq_nodes_are_equal(hyp_thm.eq_hyps, eq_hyps)
    # assert 0
