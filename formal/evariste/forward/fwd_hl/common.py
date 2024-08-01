# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from typing import List

from evariste.backward.env.hol_light.graph import HLTheorem, HLTactic
from evariste.forward.core.generation_errors import MissingHyp
from evariste.forward.env_specifics.prover_env_specifics import ChildrenIds
from evariste.forward.common import (
    EnvInfo,
    ForwardTactic,
    ForwardGraph,
    ForwardGoal,
)
from evariste.forward.common import SimpleFwdEnvOutput


@dataclass
class HLForwardTactic(ForwardTactic):
    next_node: HLTheorem
    bwd_tactic: HLTactic


class HLEnvInfo(EnvInfo):
    pass


HLFwdEnvOutput = SimpleFwdEnvOutput[HLTheorem, HLTactic]
HLForwardGraph = ForwardGraph[HLTheorem]
HLForwardGoal = ForwardGoal[HLTheorem]
HLSubGoals = List[HLTheorem]


def belongs_to_graph(goal: HLTheorem, graph: HLForwardGraph) -> bool:
    return any(goal == thm for thm in graph.generated_thms)


def search_sub_goals_in_graph(
    graph: HLForwardGraph, tactic: HLForwardTactic, subgoals: List[HLTheorem]
) -> ChildrenIds:
    # TODO: we check here that we have an exact match with a node in the graph,
    #  but we could check if present node hypotheses are a subset of expected
    #  node hypotheses.
    children_ids = []
    if len(graph.generated_thms) == 0 and len(subgoals) > 0:
        idx_sg = 0
        raise MissingHyp(
            str(subgoals[idx_sg]),
            fwd_tactic=tactic,
            missing=idx_sg,
            sub_goals=subgoals,
        )
    for idx_sg, subgoal in enumerate(subgoals):
        for child_id, thm in enumerate(graph.generated_thms):
            if thm == subgoal:
                children_ids.append(child_id)
            break
        else:
            raise MissingHyp(
                str(subgoal), fwd_tactic=tactic, missing=idx_sg, sub_goals=subgoals,
            )
    return children_ids
