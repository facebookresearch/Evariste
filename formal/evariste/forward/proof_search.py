# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import abc
import copy
from dataclasses import dataclass
from enum import Enum, unique
from typing import List, Tuple, Dict

from evariste.forward.common import (
    ForwardGoal,
    ForwardGraph,
    GenerationHistory,
    ForwardStepBeam,
    ForwardStep,
    MaybeForwardStep,
    GenerationInfos,
)
from evariste.forward.core.generation_errors import MaxLenReached, InputDicoError
from params import Params


@unique
class SearchType(str, Enum):
    STD = "std"
    DFS = "dfs"
    BFS = "bfs"
    OPEN_AI = "open_ai"
    GO_EXPLORE = "go_explore"
    PROVER_LOSS = "prover_loss"


@dataclass
class StopConfig:
    max_nodes: int
    max_generations: int
    # If not None, the prover stop in n consecutive generation candidates are invalid
    max_cons_inv_allowed: int


class ForwardProofSearch(abc.ABC):
    @property
    @abc.abstractmethod
    def goal(self) -> ForwardGoal:
        pass

    @abc.abstractmethod
    def next_graphs(self) -> List[Tuple[int, ForwardGraph]]:
        pass

    @abc.abstractmethod
    def update_with_beam(self, id_in_search: int, step_beam: ForwardStepBeam):
        pass

    @abc.abstractmethod
    def should_stop(self) -> bool:
        pass

    @abc.abstractmethod
    def finish(self):
        pass


class StandardProofSearch(ForwardProofSearch):
    def __init__(
        self, stop_config: StopConfig, goal: ForwardGoal, is_generation: bool = False
    ):
        self.cfg = stop_config
        graph = ForwardGraph.from_goal(goal)

        ## During goal generation goal.thm is None
        # TODO: maybe create a new simple proof search for graph gen ?
        self.graph = graph
        self.info = GenerationInfos()
        self.generation = GenerationHistory(goal=goal, stack=[])
        self.finished = False
        self.is_generation = is_generation
        self.id_in_search = 0

    @property
    def goal(self) -> ForwardGoal:
        return self.graph.fwd_goal

    def next_graphs(self) -> List[Tuple[int, ForwardGraph]]:
        return [(self.id_in_search, self.graph)]

    def update_with_beam(self, id_in_search: int, step_beam: ForwardStepBeam):
        assert id_in_search == self.id_in_search  # sanity check
        self.id_in_search += 1
        if len(step_beam) == 1:
            (maybe_step,) = step_beam
        else:
            selected_in_beam, maybe_step = self.sample_valid_candidate(step_beam)
            self.generation.append_beam(step_beam, selected_in_beam)
        self.generation.append_step(maybe_step)

        is_stop_tactic = False
        if maybe_step.step and not maybe_step.step.is_stop():
            self.graph.update_with_step(maybe_step.step)
            self.info.n_invalid_consecutive = 0
        elif maybe_step.step and maybe_step.step.is_stop():
            is_stop_tactic = True
        else:
            self.info.n_invalid_consecutive += 1

        finished = True
        if (
            maybe_step.step
            and not self.is_generation
            and self.graph.fwd_goal.is_solved_by_step(maybe_step.step)
        ):
            self.info.solved = True
        elif is_stop_tactic:
            self.info.stopped = ("stop_tactic", self.graph.n_nodes())
        elif isinstance(maybe_step.err, MaxLenReached):
            self.info.stopped = ("max_len_reached", maybe_step.err.msg)
        elif isinstance(maybe_step.err, InputDicoError):
            self.info.stopped = ("input_dico_error", maybe_step.err.msg)
        elif self.graph.n_nodes() >= self.cfg.max_nodes:
            self.info.stopped = ("max_nodes_reached", self.graph.n_nodes())
        elif len(self.generation.stack) >= self.cfg.max_generations:
            self.info.stopped = ("max_trials_reached", len(self.generation.stack))
        elif self.info.n_invalid_consecutive >= self.cfg.max_cons_inv_allowed:
            self.info.stopped = (
                "max_cons_inv_allowed",
                self.info.n_invalid_consecutive,
            )
        else:
            finished = False

        self.finished = finished
        if self.finished:
            # convenient to store them in GenerationHistory
            self.generation.info = self.info

    def should_stop(self) -> bool:
        return self.finished

    def sample_valid_candidate(
        self, maybe_steps: ForwardStepBeam
    ) -> Tuple[int, MaybeForwardStep]:
        valid_steps: List[Tuple[int, MaybeForwardStep]] = [
            (i, s) for i, s in enumerate(maybe_steps) if s.step
        ]
        # TODO: try with normalised score ?
        def _key_fn(inp: Tuple[int, MaybeForwardStep]) -> float:
            assert isinstance(inp[1].step, ForwardStep)
            return -inp[1].step.score

        # valid_steps = list(sorted(valid_steps, key=lambda x: -x.step.normalized_score))
        valid_steps = list(sorted(valid_steps, key=_key_fn))

        for id_in_beam, valid_step in valid_steps:
            assert isinstance(valid_step.step, ForwardStep)
            # we choose the step that solve the goal if there
            if self.goal.is_solved_by_step(valid_step.step):
                # print("Found a step that was solving goal!")
                return id_in_beam, valid_step
        # print(f"Valid: {len(valid_steps)}/{len(maybe_steps)}")

        # # else: random valid step
        # random.shuffle(valid_steps)

        # else select best one

        if not valid_steps:
            return 0, maybe_steps[0]
        else:
            return valid_steps[0]

    def finish(self):
        pass


class TreeProofSearch(ForwardProofSearch):
    @property
    def goal(self) -> ForwardGoal:
        return self.fwd_goal

    def __init__(self, search_type: SearchType, goal: ForwardGoal):
        assert search_type in {SearchType.DFS, SearchType.BFS, SearchType.OPEN_AI}
        self.fwd_goal = goal
        self.proved = False
        self.search_type = search_type

        self.queue: List[Tuple[float, ForwardGraph]] = [
            (0.0, ForwardGraph.from_goal(goal))
        ]
        self.n_expansions = 0
        self.waiting_graphs: Dict[int, Tuple[float, ForwardGraph]] = {}

    def next_graphs(self) -> List[Tuple[int, ForwardGraph]]:
        if self.should_stop():
            raise RuntimeError("Should not be called")
        if not self.queue:
            return []
        if self.search_type == SearchType.BFS:
            score, graph = self.queue.pop(0)
        elif self.search_type == SearchType.DFS:
            score, graph = self.queue.pop(-1)
        elif self.search_type == SearchType.OPEN_AI:
            # best score at the end
            self.queue = sorted(self.queue)
            score, graph = self.queue.pop(-1)
        else:
            raise NotImplementedError

        graph_id = self.n_expansions
        self.n_expansions += 1
        assert graph_id not in self.waiting_graphs
        self.waiting_graphs[graph_id] = (score, graph)
        return [(graph_id, graph)]

    def update_with_beam(self, id_in_search: int, step_beam: ForwardStepBeam):
        score, graph = self.waiting_graphs.pop(id_in_search)
        for maybe_step in step_beam:
            if not maybe_step.step:
                continue
            step: ForwardStep = maybe_step.step
            new_graph = copy.deepcopy(graph)
            new_graph.update_with_step(step)
            if new_graph.fwd_goal.is_solved_by_step(step):
                self.proved = True
            new_score = score + step.score
            self.queue.append((new_score, new_graph))

    def should_stop(self) -> bool:
        if self.waiting_graphs:
            return False
        if not self.queue:
            return True
        if self.n_expansions >= 128:
            return True
        if self.proved:
            return True
        return False

    def finish(self):
        pass


State = Tuple[str, ...]


@dataclass
class GoExploreConfig(Params):
    n_trials_max: int
    max_cons_inv_allowed: int


#
# class GoExplore(ForwardProofSearch):
#     """Probably not exactly like go explore, but same idea.
#     Go: Sample an already generated state as starting point. Can be random, using some
#     heuristics score or critic score
#
#     Explore: Sample a trajectory starting from this point. Stop to generate when we
#     reach max_len, the goal or the maximum number of consecutive errors allowed
#
#     """
#
#     def __init__(self, goal: ForwardGoal, cfg: GoExploreConfig, seed: int):
#         self._goal = goal
#         self.proved = False
#         self.cfg = cfg
#
#         self.waiting_graphs: Dict[int, ForwardGraph] = {}
#         self.rng = RandomState(seed)
#
#         self.cur_trial_id = 0
#         self.n_steps = 0
#
#         # graph on which we are currently exploring
#         graph = ForwardGraph.from_goal(self.goal)
#         self.current_graph: Tuple[int, ForwardGraph] = (self.cur_trial_id, graph)
#         self.info = GenerationInfos()
#         self.trial_is_done: bool = False
#         first_cell = self.graph_to_cell(graph)
#         self.cells: List[State] = [first_cell]
#         self.cell2cell_id: Dict[State, int] = {first_cell: 0}
#
#     @staticmethod
#     def graph_to_cell(graph: ForwardGraph) -> State:
#         return tuple(graph.nodes)
#
#     def cell_to_graph(self, cell: State) -> ForwardGraph:
#         return ForwardGraph(fwd_goal=self.goal, nodes=list(cell))
#
#     def next_graphs(self) -> List[Tuple[int, ForwardGraph]]:
#         if self.should_stop():
#             raise RuntimeError("Should not be called")
#
#         if self.trial_is_done:
#             # //GO: select a starting "state"
#             self.trial_is_done = False
#             self.cur_trial_id += 1
#             assert self.cur_trial_id < self.cfg.n_trials_max
#
#             # TODO: change to something better than random sampling
#             cell_id = self.rng.randint(len(self.cells))
#             cell = self.cells[cell_id]
#
#             # building graph from cell
#             graph = self.cell_to_graph(cell)
#
#             graph_id = self.cur_trial_id
#             self.current_graph = (graph_id, graph)
#             self.info = GenerationInfos()
#         return [self.current_graph]
#
#     def update_with_beam(self, steps: List[Tuple[int, ForwardStepBeam]]):
#         assert len(steps) == 1
#         ((graph_id, maybe_steps),) = steps
#         assert len(maybe_steps) == 1, "Support only beam=1, check your config"
#         (maybe_step,) = maybe_steps
#         expected_graph_id, graph = self.current_graph
#         info = self.info
#         assert expected_graph_id == graph_id
#         if maybe_step.step:
#             graph.append(maybe_step.step)
#             info.n_invalid_consecutive = 0
#             # Note: we dont sort the nodes for the moment, we don't collapse graphs
#             # if they have the same nodes
#             cell = self.graph_to_cell(graph)
#             if cell in self.cell2cell_id:
#                 pass
#             else:
#                 cell_id = len(self.cells)
#                 self.cells.append(cell)
#                 self.cell2cell_id[cell] = cell_id
#         else:
#             info.n_invalid_consecutive += 1
#
#         solved, stopped = self.check_if_trial_is_done(graph, info, maybe_step)
#         info.solved = solved
#         self.proved = solved
#         info.stopped = stopped
#
#         # Update internal state
#         self.trial_is_done = (info.stopped is not None) | info.solved
#         self.current_graph = (graph_id, graph)
#         self.info = info
#
#     def check_if_trial_is_done(
#         self, graph: ForwardGraph, info: GenerationInfos, maybe_step: MaybeForwardStep
#     ) -> Tuple[bool, Optional[Tuple]]:
#         solved, stopped = False, None
#         if graph.fwd_goal.is_solved_by_step(maybe_step.step):
#             solved = True
#         elif isinstance(maybe_step.err, MaxLenReached):
#             stopped = ("max_len_reached", maybe_step.err.msg)
#         elif info.n_invalid_consecutive >= self.cfg.max_cons_inv_allowed:
#             stopped = (
#                 "max_cons_inv_allowed",
#                 info.n_invalid_consecutive,
#             )
#         return solved, stopped
#
#     def should_stop(self) -> bool:
#         n_trials = self.cur_trial_id + 1
#         if n_trials >= self.cfg.n_trials_max and self.trial_is_done:
#             return True
#         if self.proved:
#             return True
