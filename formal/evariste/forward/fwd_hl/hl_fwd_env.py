# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from evariste.backward.env.hol_light.env import HLEnvWorker
from evariste.forward.env_specifics.prover_env_specifics import ForwardEnv
from evariste.forward.core.generation_errors import EnvError, NodeInGraph
from evariste.forward.fwd_hl.common import (
    HLFwdEnvOutput,
    HLForwardTactic,
    HLForwardGraph,
    belongs_to_graph,
    search_sub_goals_in_graph,
)
from evariste.envs.hl.api import HOLLightAPI, HOLLightException


class HLForwardEnv(ForwardEnv):
    def __init__(self, hl_bwd_env: HOLLightAPI):
        self.bwd_env = hl_bwd_env

    def apply_tactic(
        self, graph: HLForwardGraph, tactic: HLForwardTactic
    ) -> HLFwdEnvOutput:
        tactic_tokens = tactic.bwd_tactic.tokenize()
        generated_goal = tactic.next_node
        if belongs_to_graph(generated_goal, graph):
            generated_goal_statement = " ".join(generated_goal.tokenize())
            raise NodeInGraph(generated_goal_statement)

        tokens_dict = HLEnvWorker.graph_to_api(generated_goal)
        try:
            sample = self.bwd_env.bwd_apply_tactic(
                tactic_tokens=tactic_tokens,
                concl_tokens=tokens_dict["concl_tokens"],
                hyps_tokens=tokens_dict["hyps_tokens"],
            )
        except HOLLightException as e:
            raise EnvError(f"{type(e).__name__}: {str(e)}")
        subgoals = [HLEnvWorker.api_to_graph(sg) for sg in sample.subgoals]
        children_ids = search_sub_goals_in_graph(
            graph=graph, tactic=tactic, subgoals=subgoals
        )

        return HLFwdEnvOutput(
            generated=generated_goal, tactic=tactic, children_ids=children_ids
        )
