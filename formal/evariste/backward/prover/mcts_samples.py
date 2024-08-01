# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Type, List
from dataclasses import dataclass

from evariste.backward.graph import Theorem, Tactic


ALL_MCTS_SUBTASKS = {"critic", "tactic", "subproof", "effect", "minproof"}
ONLINE_MCTS_SUBTASKS = ALL_MCTS_SUBTASKS - {"subproof"}


@dataclass
class MCTSSampleCritic:
    goal: Theorem
    q_estimate: float

    # not used in training but for PLS
    solved: bool
    bad: bool
    critic: float

    # extra info - not used in training
    label: Optional[str] = None
    visit_count: Optional[int] = None

    def __post_init__(self):
        assert 0 <= self.q_estimate <= 1 + 1e-4
        assert isinstance(self.visit_count, int) and self.visit_count >= 0

    def size_goal(self):
        return len(self.goal.tokenize())

    @staticmethod
    def from_json(
        sample, theorem_cls: Type[Theorem], tactic_cls: Type[Tactic]
    ) -> "MCTSSampleCritic":

        goal = theorem_cls.from_dict(sample["goal"])
        return MCTSSampleCritic(
            goal=goal,
            q_estimate=sample["q_estimate"],
            label=sample.get("label", None),
            visit_count=sample.get("visit_count", None),
            solved=sample["solved"],
            critic=sample["critic"],
            bad=sample["bad"],
        )

    def to_json(self):
        return {
            "goal": self.goal.to_dict(light=True),
            "q_estimate": self.q_estimate,
            "label": self.label,
            "visit_count": self.visit_count,
            "solved": self.solved,
            "critic": self.critic,
            "bad": self.bad,
        }


@dataclass
class MCTSSampleTactics:
    goal: Theorem
    tactics: List[Tactic]
    target_pi: List[float]
    # for learn_only_tactics_from
    inproof: float
    # for Q conditioning
    q_tactics: Optional[List[float]] = None
    # extra info - not used in training
    label: Optional[str] = None
    visit_count: int = -1

    def __post_init__(self):
        assert isinstance(self.visit_count, int) and self.visit_count >= 0
        assert len(self.tactics) > 0
        assert len(self.tactics) == len(self.target_pi)
        if self.q_tactics is not None:
            assert len(self.tactics) == len(self.q_tactics)
        assert 0 <= self.inproof <= 2 + 1e-4

    def size_goal(self):
        return len(self.goal.tokenize())

    def size_tactic(self):
        return sum(len(x.tokenize()) for x in self.tactics) / len(self.tactics)

    @staticmethod
    def from_json(
        sample, theorem_cls: Type[Theorem], tactic_cls: Type[Tactic]
    ) -> "MCTSSampleTactics":

        goal = theorem_cls.from_dict(sample["goal"])
        tactics = [tactic_cls.from_dict(tactic) for tactic in sample["tactics"]]
        return MCTSSampleTactics(
            goal=goal,
            tactics=tactics,
            target_pi=sample["target_pi"],
            inproof=sample["inproof"],
            q_tactics=sample.get("q_tactics", None),
            label=sample.get("label", None),
            visit_count=sample.get("visit_count", None),
        )

    def to_json(self):
        tactics = [tac.to_dict(light=True) for tac in self.tactics]
        return {
            "goal": self.goal.to_dict(light=True),
            "tactics": tactics,
            "target_pi": self.target_pi,
            "q_tactics": self.q_tactics,
            "inproof": self.inproof,
            "label": self.label,
            "visit_count": self.visit_count,
        }


@dataclass
class MCTSSampleEffect:
    goal: Theorem
    tactic: Tactic
    children: Optional[List[Theorem]]

    def __post_init__(self):
        assert (self.children is None) == self.tactic.is_error(), (
            self.children,
            self.tactic.error_msg,
        )

    def size_goal(self):
        return len(self.goal.tokenize())

    @staticmethod
    def from_json(
        sample, theorem_cls: Type[Theorem], tactic_cls: Type[Tactic]
    ) -> "MCTSSampleEffect":
        tactic = tactic_cls.from_dict(sample["tactic"])
        assert tactic.is_error() == ("children" not in sample)
        children = (
            None
            if "children" not in sample
            else [theorem_cls.from_dict(c) for c in sample["children"]]
        )
        return MCTSSampleEffect(
            goal=theorem_cls.from_dict(sample["goal"]),
            tactic=tactic,
            children=children,
        )

    def to_json(self):
        to_ret = {
            "goal": self.goal.to_dict(light=True),
            "tactic": self.tactic.to_dict(light=True),
        }
        if self.children is not None:
            to_ret["children"] = [c.to_dict(light=True) for c in self.children]
        return to_ret
