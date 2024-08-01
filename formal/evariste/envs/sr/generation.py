# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple, List
import numpy as np

from params import ConfStore
from evariste.backward.env.sr.env import apply_bwd_tactic
from evariste.backward.env.sr.graph import SRTheorem, SRTactic
from evariste.envs.sr.env import SREnv, XYValues, Transition, SREnvArgs
from evariste.envs.eq.graph import Node, ZERO, infix_for_numexpr


def traj_to_goal_with_tactics(
    env: SREnv, trajectory: List[Transition], true_xy: XYValues
) -> List[Tuple[SRTheorem, SRTactic, Node]]:
    """
    Convert a list of transitions to a list of (goal, tactics).
    We may end before, if the last transitions are not required to solve the goal.
    """
    assert trajectory[0].eq.eq(ZERO)
    target_eq = trajectory[-1].next_eq
    result = []
    for step in trajectory:
        tactic = SRTactic(
            label=step.rule.name, prefix_pos=step.prefix_pos, to_fill=step.tgt_vars
        )
        curr_y = env.evaluate_at(step.eq, true_xy.x)
        curr_xy = XYValues(x=true_xy.x, y=curr_y)
        if env.is_identical(curr_xy, true_xy):
            break
        goal = SRTheorem(
            node=step.eq, true_xy=true_xy, curr_xy=curr_xy, target_node=target_eq
        )
        child = step.next_eq
        result.append((goal, tactic, child))

    return result


if __name__ == "__main__":

    # python -m evariste.envs.sr.generation

    import evariste.datasets

    sr_args = SREnvArgs(
        eq_env=ConfStore["sr_eq_env_default"], max_backward_steps=100, max_n_points=10
    )
    sr_env = SREnv.build(sr_args, seed=None)

    def test_trajectory_tactics(n_tests: int):

        print("===== TEST TRAJECTORY TACTICS")

        for _ in range(n_tests):

            n_init_ops = np.random.randint(1, 10 + 1)
            tgt_eq: Node = sr_env.eq_env.generate_expr(n_init_ops)
            true_xy = sr_env.sample_dataset(tgt_eq)
            trajectory = sr_env.sample_trajectory(tgt_eq)
            sr_env.check_trajectory(tgt_eq, trajectory)
            sr_env.print_trajectory(trajectory)

            goals_with_tactics = traj_to_goal_with_tactics(sr_env, trajectory, true_xy)

            print(f"{len(trajectory):>2} steps: {tgt_eq}")
            assert goals_with_tactics[0][0].curr_node.eq(ZERO)

            for i, (goal, tactic, next_eq) in enumerate(goals_with_tactics):

                # # check tokenization  # TODO: remove
                # goal_ = SRTheorem.from_tokens(goal.tokenize())
                # tactic_ = SRTactic.from_tokens(tactic.tokenize())
                # assert tactic_.is_valid

                # apply rule with env
                children = apply_bwd_tactic(sr_env, goal, tactic)
                assert children is not None

                # re-apply rule
                applied = sr_env.eq_env.apply_t_rule(
                    eq=goal.curr_node,
                    rule=tactic.rule,
                    fwd=True,
                    prefix_pos=tactic.prefix_pos,
                    to_fill=tactic.to_fill,
                )
                assert applied["eq"].eq(next_eq)
                assert len(applied["hyps"]) == 0

                # stop condition
                curr_xy = XYValues(true_xy.x, sr_env.evaluate_at(next_eq, true_xy.x))
                if len(children) == 0:
                    assert i == len(goals_with_tactics) - 1
                    assert sr_env.is_identical(curr_xy, true_xy)
                # not done yet
                else:
                    assert len(children) == 1
                    assert i < len(goals_with_tactics) - 1
                    assert not sr_env.is_identical(curr_xy, true_xy)
                    assert children.pop().curr_node.eq(next_eq)

        print("OK")

    test_trajectory_tactics(n_tests=20)
