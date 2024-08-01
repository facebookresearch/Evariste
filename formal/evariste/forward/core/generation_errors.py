# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, fields
from typing import Optional, Any


@dataclass
class GenerationError(Exception):
    msg: str
    type: str = "generation_error"

    def __reduce__(self):
        # overwrite Exception's reduce to pickle all required args
        return type(self), tuple(self.__dict__[f.name] for f in fields(self))


@dataclass
class GoalError(GenerationError):
    type: str = "goal_error"
    goal: Any = None


@dataclass
class Duplicated(GenerationError):
    type: str = "duplicated_command"


@dataclass
class NotAllowedStop(GenerationError):
    type: str = "not_allowed_stop"


@dataclass
class InvalidLabel(GenerationError):
    type: str = "invalid_label"


@dataclass
class InvalidTactic(GenerationError):
    type: str = "invalid_tactic"
    fwd_tactic: Any = None  # circular imports
    parsed_goal: Any = None  # circular imports


@dataclass
class ParseGoalError(GenerationError):
    type: str = "parse_goal"
    fwd_tactic: Any = None  # circular imports


@dataclass
class ParseChildrenError(GenerationError):
    type: str = "parse_children"
    fwd_tactic: Any = None  # circular imports


@dataclass
class EnvTimeout(GenerationError):
    type: str = "env_timeout"
    fwd_tactic: Any = None  # circular imports


@dataclass
class ForbiddenLabel(GenerationError):
    type: str = "forbidden_label"


@dataclass
class EnvError(GenerationError):
    type: str = "env_error"


@dataclass
class NodeInGraph(GenerationError):
    type: str = "node_already_in_graph"
    fwd_tactic: Any = None  # circular imports


@dataclass
class WrongContext(GenerationError):
    type: str = "wrong_context"
    fwd_tactic: Any = None  # circular imports


@dataclass
class MissingHyp(GenerationError):
    type: str = "child_not_in_graph"
    fwd_tactic: Any = None  # circular imports
    missing: Optional[int] = None  # circular imports
    sub_goals: Any = None  # circular imports


@dataclass
class WrongSubsKeys(GenerationError):
    type: str = "wrong_subs_keys"


@dataclass
class NonExistentTheorem(GenerationError):
    type: str = "non_existent_theorem"


@dataclass
class MalformedCommand(GenerationError):
    type: str = "malformed_command"
    command: Optional[str] = None


@dataclass
class MalformedChildren(GenerationError):
    type: str = "malformed_children_command"


@dataclass
class DicoError(GenerationError):
    type: str = "dico_error"


@dataclass
class InputDicoError(GenerationError):
    type: str = "input_dico_error"


@dataclass
class SyntacticGenError(GenerationError):
    type: str = "syntactic_error"


@dataclass
class MaxLenReached(GenerationError):
    type: str = "max_len_reached"


@dataclass
class DisjointGenError(GenerationError):
    type: str = "disjoint_error"


@dataclass
class GoalDisjointError(GenerationError):
    type: str = "goal_mand_disj_not_respected"


@dataclass
class ChildrenMismatch(GenerationError):
    type: str = "children_id_mismatch"
