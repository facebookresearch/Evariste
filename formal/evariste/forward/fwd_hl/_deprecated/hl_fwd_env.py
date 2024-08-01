# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from typing import Tuple

from evariste.forward.common import (
    ForwardGraph,
    ForwardTactic,
    EnvInfo,
)
from evariste.forward.core.generation_errors import EnvError, NodeInGraph, MissingHyp
from evariste.forward.env_specifics.prover_env_specifics import (
    Statement,
    ChildrenIds,
    ForwardEnv,
)
from evariste.backward.env.hol_light import HLTheorem, HLTactic
from evariste.backward.env.hol_light.env import HLBackwardEnv


@dataclass
class HLForwardTactic(ForwardTactic):
    next_node: HLTheorem
    bwd_tactic: HLTactic


class HLEnvInfo(EnvInfo):
    pass


class HLForwardEnv(ForwardEnv):
    def __init__(self, hl_bwd_env: HLBackwardEnv):
        self.bwd_env = hl_bwd_env

    def apply_tactic(
        self, graph: ForwardGraph, tactic: HLForwardTactic
    ) -> Tuple[Statement, ChildrenIds, HLEnvInfo]:

        env, (children, error) = self.bwd_env.execute(
            theorem=tactic.next_node, tactic=tactic.bwd_tactic
        )
        self.bwd_env = env
        if error:
            raise EnvError(error)

        statement2id = {s: i for i, s in enumerate(graph.nodes)}
        generated_goal = " ".join(tactic.next_node.tokenize())
        if generated_goal in statement2id:
            raise NodeInGraph(generated_goal)

        children_ids = []
        for child in children:
            assert isinstance(child, HLTheorem)
            statement = " ".join(child.tokenize())
            # TODO: we check here that we have an exact match with a node in the graph,
            #  but we could check if present node hypotheses are a subset of expected
            #  node hypotheses.
            if statement not in statement2id:
                raise MissingHyp(statement)
            children_ids.append(statement2id[statement])

        return generated_goal, children_ids, HLEnvInfo()
