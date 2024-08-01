# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import deque, defaultdict
from typing import Set, List, Dict
from dataclasses import dataclass
import math
import numpy as np

from evariste.adversarial.generator import AdvGeneratorConfig
from evariste.envs.mm.env import MetamathEnv
from evariste.forward.common import ForwardGoal, GenerationHistory
from evariste.forward.env_specifics.common import AnnotatedGoal
from evariste.forward.env_specifics.generation_stats import GenerationStats
from evariste.forward.env_specifics.generation_annotator import (
    GenerationAnnotator,
    NodeSelectionConfig,
)
from evariste.forward.fwd_mm.mm_helpers import history_to_mm_nodes
from evariste.envs.mm.utils import count_unique_nodes, Node_a_p
from evariste.model.data.dictionary import Dictionary


MM_HEURISTICS = {
    "size",
    "depth",
    "size_by_depth",
    "mean_tok_freq",
    "max_tok_freq",
    "n_tokens",
}


@dataclass
class MMGenerationStats(GenerationStats):
    is_in_train: int = None
    already_been_generated: int = None
    mean_tok_freq: float = None
    max_tok_freq: int = None
    relative_pos: float = None  # relative position of the selected node (between 0 and 1)


class MMGenerationAnnotator(GenerationAnnotator):
    def __init__(self, params, mm_env: MetamathEnv, dico: Dictionary):
        from evariste.trainer.args import TrainerArgs

        assert isinstance(params, TrainerArgs)
        self.params = params
        self.mm_env = mm_env
        self.dico = dico
        # for generation stats
        self.recent_generated = deque(maxlen=10_000)
        self.recent_generated_count = defaultdict(int)
        self._train_set = None
        assert all(k in MM_HEURISTICS for k in params.mm.graph.generation_reward.keys())

    def annotate_and_select_goals(
        self, history: GenerationHistory, select_cfg: NodeSelectionConfig
    ) -> List[AnnotatedGoal]:

        # build nodes / stats
        nodes = history_to_mm_nodes(history)
        node_stats = self._compute_node_stats(nodes)

        # select node
        if self.params.mm.stop_action:
            selected_id = len(nodes) - 1
        else:
            selected_id = max(node_stats, key=lambda x: x["reward"])["node_id"]
        node = nodes[selected_id]
        stats = node_stats[selected_id]

        # stats
        gen_stats = MMGenerationStats(
            # selected node stats
            last_node_depth=stats["depth"],
            last_node_proof_size=stats["size"],
            last_node_size_by_depth=stats["size_by_depth"],
            statement_n_tok=stats["statement_n_tok"],
            is_in_train=stats["is_in_train"],
            already_been_generated=stats["already_been_generated"],
            mean_tok_freq=stats["mean_tok_freq"],
            max_tok_freq=stats["max_tok_freq"],
            relative_pos=stats["relative_pos"],
            # generation stats
            n_forward_steps=len(history.forward_steps()),
            n_forward_errors=len(history.errors()),
        )

        # build goal
        forward_goal = ForwardGoal(
            statement=node.statement_str,
            e_hyps=list(node.e_hyps.values()),
            forbidden=history.goal.forbidden,
            label=history.goal.label,
            mand_disj=node.disjoint,
        )

        res = AnnotatedGoal(
            selected_goal=forward_goal,
            stats=gen_stats,
            generation_reward=stats["reward"],
        )
        return [res]

    def _compute_node_stats(self, nodes: List[Node_a_p]):

        node_stats = []

        for node_id, node in enumerate(nodes):

            # size / depth
            node.set_nodes_and_depth()
            depth = node.depth["no_syntactic"]
            size = count_unique_nodes(node, ignore_e_hyps=True)
            size_by_depth = size / max(node.depth["no_syntactic"], 1)

            # in train / already generated
            this_str = node.statement_str
            n_seen = self.recent_generated_count[this_str]
            self.recent_generated_count[this_str] += 1
            if len(self.recent_generated) == 10_000:
                self.recent_generated_count[self.recent_generated.popleft()] -= 1
            self.recent_generated.append(this_str)
            is_in_train = node.statement_str in self._get_train_set()

            # token frequencies
            token_indexes = [self.dico.index(tok) for tok in node.statement]
            mean_tok_freq = float(np.mean(token_indexes))
            max_tok_freq = max(token_indexes)

            # node stats
            stats = {
                "node_id": node_id,
                "relative_pos": node_id / len(nodes),
                "size": size,
                "depth": depth,
                "size_by_depth": size_by_depth,
                "already_been_generated": n_seen,
                "is_in_train": is_in_train,
                "statement_n_tok": len(node.statement),
                "mean_tok_freq": mean_tok_freq,
                "max_tok_freq": max_tok_freq,
            }

            # node reward -- only select $a or $p nodes
            if node.ltype in ["$a", "$p"]:
                stats["reward"] = self._compute_reward(stats)
            else:
                stats["reward"] = -math.inf

            node_stats.append(stats)

        return node_stats

    def _compute_reward(self, stats: Dict):
        reward = 0
        for heuristic, weight in self.params.mm.graph.generation_reward.items():
            if heuristic in [
                "size",
                "depth",
                "size_by_depth",
                "mean_tok_freq",
                "max_tok_freq",
            ]:
                reward += weight * stats[heuristic]
            elif heuristic == "n_tokens":
                reward += weight * stats["statement_n_tok"]
            else:
                raise NotImplementedError(f"Unknown heuristic: {heuristic}")
        return reward

    def _get_train_set(self) -> Set[str]:
        if self._train_set is not None:
            return self._train_set
        self._train_set = set()
        for x in self.mm_env.labels.values():
            self._train_set.add(x[1]["tokens_str"])
        return self._train_set
