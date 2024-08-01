# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from evariste import json as json
import pickle
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Dict, Tuple

from numpy.random.mtrand import RandomState

from evariste.envs.mm.utils import count_unique_nodes, get_canonical_order
from evariste.forward.fwd_mm.mm_helpers import (
    load_splitted_proof_trees,
    get_mand_disj,
)
from evariste.forward.common import (
    ForwardGoal,
    GenerationHistory,
)
from evariste.forward.online_generation.goal_selectors.goal_selector import (
    GoalMetadata,
    _Infos,
    EmptySelector,
)


@dataclass
class GoalSelectorConfig:
    rank: int
    world_size: int
    n_max_selection: int


@dataclass
class SelectionStats:
    n_selected: int = 0
    # once it is returned by prover we can increment this column
    n_updated: int = 0
    n_proved: int = 0
    n_generated: int = 0


class HardMiningGoalSelector:
    def __init__(self, cfg: GoalSelectorConfig):
        self.cfg = cfg
        self.goals: List[ForwardGoal] = []
        self.metadata: List[List[GoalMetadata]] = []
        self.stats: List[SelectionStats] = []

        self.reversed_index = {}

        self.scores: Dict[int, float] = {}
        self.rng = RandomState(42 + cfg.rank)
        self.scores_sum = 0

        self.waiting = {}
        self.n_selections = 0
        self.n_updates = 0

        self.interesting_goals = []

        self.infos = _Infos()

    def select_goal(self) -> Tuple[int, ForwardGoal]:
        if len(self.scores) == 0:
            raise EmptySelector
        # TODO: replace with heap if too slow
        local_id, score = max(self.scores.items(), key=lambda x: x[1])
        goal, stats = self.goals[local_id], self.stats[local_id]
        stats.n_selected += 1
        if stats.n_selected >= self.cfg.n_max_selection:
            self.scores.pop(local_id)
        generation_id = self.new_generation_id()
        self.waiting[generation_id] = local_id
        return generation_id, goal

    def new_generation_id(self) -> int:
        generation_id = self.n_selections * self.cfg.world_size + self.cfg.rank
        self.n_selections += 1
        return generation_id

    def update_with_generation(
        self, generation_id: int, generated: GenerationHistory, solved: bool
    ):
        received_goal = generated.goal
        self.n_updates += 1

        assert generation_id in self.waiting
        local_id = self.waiting.pop(generation_id)
        goal, met, stats = (
            self.goals[local_id],
            self.metadata[local_id],
            self.stats[local_id],
        )
        assert goal == received_goal, f"{generation_id}: {goal} != {received_goal}"
        stats.n_updated += 1
        stats.n_proved += int(solved)
        label = met[0].label
        # if first time we don't manage to prove this goal
        if not int(solved) and (stats.n_updated - stats.n_proved):
            # if this goal was generated and not from training set
            if min(m.generation_id for m in met) > -1:
                self.infos.n_interesting += 1
                self.interesting_goals.append((goal, met, stats))

        for node_id, node in enumerate(generated.nodes()):
            if node.ltype == "$e":
                continue
            node.set_nodes_and_depth()
            depth = node.depth["no_syntactic"]
            size = count_unique_nodes(node, ignore_e_hyps=True)
            e_hyps = list(node.e_hyps.values())

            new_goal = ForwardGoal(
                statement=node.statement_str,
                e_hyps=e_hyps,
                forbidden=received_goal.forbidden,
                mand_disj=get_mand_disj(
                    statement=node.statement_str,
                    e_hyps=node.e_hyps,
                    disjoints=node.disjoint,
                ),
            )
            new_metadata = GoalMetadata(
                generation_id=generation_id,
                node_id=node_id,
                depth=depth,
                size=size,
                label=label,
            )

            self._update_with_goal(new_goal, new_metadata, update_stats=True)

    @classmethod
    def init_with_dataset(
        cls, mm_data_dir: str, split: str, cfg: GoalSelectorConfig
    ) -> "HardMiningGoalSelector":
        data = load_splitted_proof_trees(mm_data_dir)[split]
        data = [s for i, s in enumerate(data) if i % cfg.world_size == cfg.rank]
        goals = []
        for trm_name, root in data:
            root.set_nodes_and_depth()
            for node_id, node in enumerate(get_canonical_order(root)):
                if node.ltype == "$e":
                    continue
                depth = node.depth["no_syntactic"]
                size = count_unique_nodes(node, ignore_e_hyps=True)
                e_hyps = list(node.e_hyps.values())
                new_goal = ForwardGoal(
                    statement=node.statement_str,
                    e_hyps=e_hyps,
                    forbidden=None,
                    mand_disj=get_mand_disj(
                        statement=node.statement_str,
                        e_hyps=node.e_hyps,
                        disjoints=node.disjoint,
                    ),
                )
                new_metadata = GoalMetadata(
                    generation_id=-1,
                    node_id=node_id,
                    depth=depth,
                    size=size,
                    label=trm_name,
                )

                goals.append((new_goal, new_metadata))
        selector = cls(cfg=cfg)
        for goal, metadata in goals:
            selector._update_with_goal(goal, metadata, update_stats=False)
        print(f"Selector init with {len(selector.goals)} goals")
        return selector

    def _update_with_goal(
        self, goal: ForwardGoal, metadata: GoalMetadata, update_stats: bool = True
    ):
        assert goal.forbidden is None
        key = self.goal_key(goal)
        if key in self.reversed_index:
            local_id = self.reversed_index[key]
            self.metadata[local_id].append(metadata)
            if local_id in self.scores and self.score(metadata) < self.scores[local_id]:
                self.scores[local_id] = self.score(metadata)
        else:
            local_id = len(self.goals)
            self.reversed_index[key] = local_id
            self.goals.append(goal)
            self.metadata.append([metadata])
            self.stats.append(SelectionStats())
            self.scores[local_id] = self.score(metadata)

        if update_stats:
            self.stats[local_id].n_generated += 1

    @staticmethod
    def goal_key(goal: ForwardGoal):
        return tuple(
            [goal.statement, tuple(sorted(goal.e_hyps)), tuple(sorted(goal.mand_disj))]
        )

    @staticmethod
    def score(metadata: GoalMetadata) -> float:
        return metadata.size

    def dump_state(self, dst: Path):
        dst.mkdir(exist_ok=True)
        state_path = dst / "state.pkl"
        goal_paths = dst / "interesting_goals.jsonl"
        start = time.time()
        with goal_paths.open("a") as fp:
            for goal, met, stats in self.interesting_goals:
                fp.write(
                    json.dumps(
                        {
                            "iteration": self.n_updates,
                            "goal": goal.to_dict(),
                            "metadata": [asdict(m) for m in met],
                            "stats": asdict(stats),
                        }
                    )
                    + "\n"
                )
        self.interesting_goals = []

        with TemporaryDirectory(prefix=str(dst.parent) + "/") as tmp_dir:
            tmp_path = Path(tmp_dir) / "stats.pkl"
            with tmp_path.open("wb") as fp:
                pickle.dump(
                    {
                        "goals": self.goals,
                        "metadata": self.metadata,
                        "stats": self.stats,
                        "scores": self.scores,
                    },
                    fp,
                )
            tmp_path.rename(state_path)
        self.infos.time_in_dumping += time.time() - start
        infos = asdict(self.infos)
        infos.update(
            {
                "n_updates": self.n_updates,
                "scores_lengths": len(self.scores),
                "scores_max": max(self.scores.values()),
            }
        )
        print(
            "GoalSelector infos", infos,
        )
