# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from collections import deque, defaultdict
from typing import List
import numpy as np

from evariste.backward.env.equations import EQTheorem
from evariste.envs.eq.generation import GraphNode
from evariste.forward.common import ForwardGoal, GenerationHistory
from evariste.forward.env_specifics.common import AnnotatedGoal
from evariste.forward.env_specifics.generation_stats import GenerationStats
from evariste.forward.env_specifics.generation_annotator import (
    GenerationAnnotator,
    NodeSelectionConfig,
)
from evariste.forward.fwd_eq.eq_fwd_goal_factory import eq_forward_goal


@dataclass
class EQGenerationStats(GenerationStats):
    already_been_generated: int


class EQGenerationAnnotator(GenerationAnnotator):
    def __init__(self, params):
        from evariste.trainer.args import TrainerArgs

        assert isinstance(params, TrainerArgs)
        self.params = params

        # for generation stats
        self.recent_generated = deque(maxlen=10_000)
        self.recent_generated_count = defaultdict(int)

    def annotate_and_select_goals(
        self, history: GenerationHistory, select_cfg: NodeSelectionConfig
    ) -> List[AnnotatedGoal]:

        from evariste.forward.fwd_eq.history_to_eq_nodes import history_to_eq_nodes

        # convert history to list of nodes / score nodes
        nodes, _ = history_to_eq_nodes(history)
        scores = [self._get_node_score(node, select_cfg) for node in nodes]

        # select most promising node(s)
        n = select_cfg.n_send_to_provers
        assert n > 0
        if select_cfg.select_method == "last":
            selected_ids: List[int] = list(range(len(nodes)))[-n:]
        elif select_cfg.select_method == "max":
            selected_ids = np.argsort(scores)[-n:].tolist()
        else:
            raise RuntimeError(f"Unknown method: {select_cfg.select_method}")

        res: List[AnnotatedGoal] = []
        for i in selected_ids:
            selected_node = nodes[i]
            stats = self._gen_to_stats(history, selected_node)
            eq_thm = EQTheorem(
                node=selected_node.node, hyps=selected_node.used_hyps.nodes
            )
            forward_goal: ForwardGoal = eq_forward_goal(eq_thm, name=history.goal.label)
            goal = AnnotatedGoal(
                selected_goal=forward_goal, stats=stats, generation_reward=scores[i]
            )
            res.append(goal)
        return res

    def _gen_to_stats(
        self, gen: GenerationHistory, selected_node: GraphNode
    ) -> EQGenerationStats:

        this_str = selected_node.node.prefix()
        n_seen = self.recent_generated_count[this_str]
        self.recent_generated_count[this_str] += 1
        if len(self.recent_generated) == 10_000:
            self.recent_generated_count[self.recent_generated.popleft()] -= 1
        self.recent_generated.append(this_str)

        depth = selected_node.depth
        size = selected_node.size
        size_by_depth = 0.0 if depth == 0 else (size / depth)

        return EQGenerationStats(
            last_node_depth=depth,
            last_node_proof_size=size,
            last_node_size_by_depth=size_by_depth,
            statement_n_tok=selected_node.node.prefix_len(),
            n_forward_steps=len(gen.forward_steps()),
            n_forward_errors=len(gen.errors()),
            already_been_generated=n_seen,
        )

    def _get_node_score(
        self, node: GraphNode, select_cfg: NodeSelectionConfig
    ) -> float:
        """
        Select a goal from a generated graph.
        """
        heuristic = select_cfg.reward_type
        if heuristic == "depth":
            return float(node.depth)
        elif heuristic == "size":
            return float(node.size)
        elif heuristic == "size_by_depth":
            size = node.size
            depth = node.depth
            return 0.0 if depth == 0 else (size / depth)
        else:
            raise RuntimeError(f"Unknown heuristic: {heuristic}")
