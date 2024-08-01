# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Dict, Tuple, Optional
import math
from pathlib import Path
import pytest
import numpy as np

from evariste.backward.prover.mcts import MCTS, DeadRoot, EnvExpansion, FailedTactic
from evariste.backward.prover.nodes import MCTSNode
from evariste.backward.prover.policy import Policy
from evariste.backward.graph import Tactic, Theorem, BackwardGoal
from evariste.backward.prover.args import ConfStore


class MyTheorem(Theorem):
    def __init__(self, conclusion: str):
        super().__init__(conclusion, [])

    def tokenize(self):
        pass

    @classmethod
    def from_tokens(cls, tokens):
        raise NotImplementedError

    def to_dict(self, light=False):
        pass

    @classmethod
    def from_dict(cls, data):
        pass

    def __repr__(self):
        return self.conclusion


class MyTactic(Tactic):
    def __init__(self, unique_str: str):
        super().__init__(True, unique_str, "", False)

    def tokenize(self):
        pass

    @staticmethod
    def from_error(error_msg: str, tokens: Optional[List[str]] = None) -> "MyTactic":
        raise NotImplementedError

    @staticmethod
    def from_tokens(tokens):
        raise NotImplementedError

    def to_dict(self, light=False) -> Dict:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: Dict):
        pass


def approx(a, b):
    if isinstance(a, (int, float)):
        return abs(a - b) < 1e-9
    if isinstance(a, np.ndarray):
        return np.all(np.abs(a - b) < 1e-9)


def test_node():
    n = MCTSNode(
        theorem=MyTheorem(conclusion="root"),
        time_created=0,
        tactics=[MyTactic(unique_str="a"), MyTactic(unique_str="b")],
        log_critic=math.log(0.42),
        children_for_tactic=[[MyTheorem(conclusion="a")], [MyTheorem(conclusion="b")]],
        priors=[0.9, 0.1],
        exploration=0,  # ignore priors
        policy="other",
        error=None,
        effects=[],
        init_tactic_scores=0.5,
        q_value_solved=0,
    )
    assert n.value() == math.log(0.42)
    n.update(0, math.log(0.8))
    assert np.all(n.logW == np.array([math.log(0.8), -math.inf]))
    assert n.value() == math.log(0.8)

    # test init tactic scores
    assert approx(n.policy(), np.array([0.8, 0.5]) / 1.3)

    # test virtual counts
    n.virtual_counts = np.array([3, 2])
    q = np.array([0.8 / 4, 0.5 / 2])
    expected = q / q.sum()
    assert approx(n.policy(), expected)
    n.virtual_counts = np.array([0, 0])

    n.update(0, math.log(0.4))
    assert approx(n.logW[0], math.log(1.2))

    n.update(1, math.log(0.2))
    assert approx(n.logW[1], math.log(0.2))
    for ptype in ["other", "alpha_zero"]:
        for exploration in [0, 1e9]:
            n._policy = Policy(policy_type=ptype, exploration=exploration)
            if exploration == 0:
                # no priors. avg Q = [0.6, 0.2] -> [0.75, 0.25]
                assert approx(n.policy(), np.array([0.6, 0.2]) / 0.8)
            if exploration == 1e9:
                if ptype == "other":
                    # no Q, only priors
                    assert approx(n.policy(), np.array([0.9, 0.1]))
                else:
                    # priors * (np.sqrt(counts.sum()) / (1 + counts))
                    expected = (np.array([0.9, 0.1]) * np.sqrt(3)) / np.array([3, 2])
                    expected /= expected.sum()
                    assert approx(n.policy(), expected)

    n._policy = Policy(policy_type="other", exploration=0)

    n.solved = True
    n.solving_tactics.add(1)
    # q = [0.6 1] and exploration = 0 => no priors
    assert np.all(n.policy() == np.array([0.6, 1]) / 1.6)
    assert n.value() == 0

    n.solved = False
    n.solving_tactics.clear()

    n.kill_tactic(0, 0)
    assert np.all(n.policy() == np.array([0, 1]))
    assert n.kill_tactic(0, 1)
    with pytest.raises(RuntimeError):
        n.policy()  # no available tactics.


th = [MyTheorem(str(i)) for i in range(15)]
tac = [MyTactic(str(i)) for i in range(15)]


def find_expand_and_backup(mcts: MCTS, expansions: List[Tuple[int, List[List[int]]]]):
    simu_tree, _, to_ex = mcts.find_leaves_to_expand()
    for tex, e in zip(to_ex, expansions):
        assert tex.theorem == th[e[0]]
    to_ex = mcts.to_expand(simu_tree, to_ex)
    assert to_ex == [th[e[0]] for e in expansions]

    mcts.do_expand(
        [
            EnvExpansion(
                th[exp[0]],
                exp_duration=0.0,
                gpu_duration=0.0,
                env_durations=[0.0],
                effects=[],
                error=None,
                log_critic=math.log(0.5),
                tactics=[tac[i] for i in range(len(exp[1]))],
                child_for_tac=[
                    [th[child] for child in children] for children in exp[1]
                ],
                priors=[1.0 / len(exp[1]) for _ in range(len(exp[1]))],
            )
            for exp in expansions
        ]
    )
    mcts.do_backup()


def test_backup1():
    r"""
          0
          | 
          1
          |
          2
    """
    mcts = MCTS(
        goal=BackwardGoal(theorem=th[0], label="test"),
        mcts_params=ConfStore["mcts_very_fast"],
        log_path=Path("unused"),
        quiet=True,
    )
    find_expand_and_backup(mcts, [(0, [[1]])])
    assert mcts.nodes[th[0]].log_critic == math.log(0.5)

    find_expand_and_backup(mcts, [(1, [[2]])])
    assert mcts.nodes[th[0]].logW[0] == math.log(0.5)


def test_backup2():
    r"""
            0
            | 
          1 , 2
          |   |
          3   4
           \ /
            5
    """
    mcts = MCTS(
        goal=BackwardGoal(theorem=th[0], label="test"),
        mcts_params=ConfStore["mcts_very_fast"],
        log_path=Path("unused"),
        quiet=True,
    )
    find_expand_and_backup(mcts, [(0, [[1, 2]])])
    assert mcts.nodes[th[0]].log_critic == math.log(0.5)

    find_expand_and_backup(mcts, [(1, [[3]]), (2, [[4]])])
    assert mcts.nodes[th[0]].logW[0] == math.log(0.5 * 0.5)

    find_expand_and_backup(mcts, [(3, [[5]]), (4, [[5]])])
    assert mcts.nodes[th[0]].logW[0] == math.log(0.5 * 0.5 * 2)
    assert mcts.nodes[th[1]].logW[0] == math.log(0.5)
    assert mcts.nodes[th[2]].logW[0] == math.log(0.5)

    mcts.kill_tactic(mcts.nodes[th[4]], 0, 0)
    with pytest.raises(DeadRoot):
        mcts.find_leaves_to_expand()


def test_find_1():
    r"""
            0
           / \
          1   2
              |
              3 -> back to 0
    """
    mcts = MCTS(
        goal=BackwardGoal(theorem=th[0], label="test"),
        mcts_params=ConfStore["mcts_very_fast"],
        log_path=Path("unused"),
        quiet=True,
    )
    find_expand_and_backup(mcts, [(0, [[1], [2]])])
    assert mcts.nodes[th[0]].log_critic == math.log(0.5)
    mcts.kill_tactic(mcts.nodes[th[0]], 0, 0)

    find_expand_and_backup(mcts, [(2, [[3]])])
    assert mcts.nodes[th[0]].logW[1] == math.log(0.5)

    find_expand_and_backup(mcts, [(3, [[0]])])  # Cycle !
    assert mcts.nodes[th[0]].logW[1] == math.log(0.5 * 2)

    with pytest.raises(FailedTactic):
        mcts.find_leaves_to_expand_aux()
    with pytest.raises(DeadRoot):
        mcts.find_leaves_to_expand()


def test_find_2():
    r"""
             0
           /   \
          1     2
         / \   / \
        3   4 5   6
    """
    params = ConfStore["mcts_very_fast"]
    params.policy = "alpha_zero"

    mcts = MCTS(
        goal=BackwardGoal(theorem=th[0], label="test"),
        mcts_params=params,
        log_path=Path("unused"),
        quiet=True,
    )
    find_expand_and_backup(mcts, [(0, [[1], [2]])])
    find_expand_and_backup(mcts, [(1, [[3], [4]])])
    mcts.nodes[th[0]].priors = np.array([0.4, 0.6])  # force exploration of 2nd branch
    find_expand_and_backup(mcts, [(2, [[5], [6]])])

    # Note, this only works with alpha_zero policy
    for vl in [0, 3]:
        mcts.virtual_loss = vl
        all_seen = set()
        for i in range(4):
            root = mcts.nodes[th[0]]
            _, _, to_ex = mcts.find_leaves_to_expand()
            all_seen.add(to_ex[0].theorem.conclusion)
        # with vl, we see all 4 leaves
        # without we just see 1
        assert len(all_seen) == vl + 1
