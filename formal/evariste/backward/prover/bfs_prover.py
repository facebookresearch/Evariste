# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from dataclasses import dataclass, asdict
from time import time
from typing import List, Set, Optional, Dict, Union

from evariste.backward.env.core import EnvExpansion
from evariste.backward.prover.bfs_proof_graph import (
    ProofTheoremNode,
    ProofGraph,
)

from evariste.backward.env.core import BackwardGoal
from evariste.backward.graph import Theorem, UnMaterializedTheorem
from evariste.backward.prover.core import ProofHandler, ProofResult
from evariste.backward.prover.utils import Number


@dataclass
class HandlerStats:
    n_batches_of_expansions: int = 0
    n_expansions_sent: int = 0
    n_expansions_received: int = 0
    not_useful_to_expand: int = 0
    cum_time_in_graph_state_update: float = 0.0
    cum_time_in_get_thms_to_expand: float = 0.0
    cum_time_in_send_env_exps: float = 0.0
    stopped: Optional[str] = None  # proved, done, max_expansions


@dataclass
class BFSResult(ProofResult):
    stats: Dict[str, Union[Number, str]]


class BFSProofHandler(ProofHandler):
    """
    ProofHandler used for handling a Greedy proof strategy.
    """

    def __init__(
        self,
        goal: BackwardGoal,
        n_expansions_max: int,
        best_first_search: bool = False,
        n_concurrent_expansions_max: Optional[int] = None,
    ):
        super().__init__(goal)

        # params
        self.n_expansions_max = n_expansions_max
        self.best_first_search = best_first_search
        self.n_concurrent_expansions_max = n_concurrent_expansions_max

        # state
        self._init_state(root_thm=goal.theorem)

    def _maybe_put_in_queue(self, node: ProofTheoremNode):
        """put in queue and mark as visited if not already visited"""
        if node.thm in self.visited:
            return
        assert node.thm not in self.visited
        self.queue.append(node)
        self.visited.add(node.thm)

    def _init_state(self, root_thm: Theorem):
        self.proved: bool = False
        self.queue: List[ProofTheoremNode] = []
        self.visited: Set[Theorem] = set()
        self.waiting_for_expansion: Set[
            Theorem
        ] = set()  # not really needed, for checks
        self.scores: Dict[Theorem, float] = {}

        # stats
        self.stats = HandlerStats()

        self.graph = ProofGraph(root_thm=root_thm)
        self.graph.update_state()
        self.scores[self.graph.root.thm] = 0.0
        self._maybe_put_in_queue(self.graph.root)

    def is_ready(self):
        if isinstance(self.graph.root.thm, UnMaterializedTheorem):
            return False
        return True

    def send_materialized(self, th: Theorem):
        self._init_state(root_thm=th)

    def get_theorems_to_expand(self) -> Optional[List[Theorem]]:
        start = time()
        if not self.is_ready():
            return None
        assert self.waiting_for_expansion == set()

        # not using heap but sorting queue to always have up-to-date scores
        self.queue.sort(key=lambda x: self.scores[x.thm])

        to_expand = []
        while self.queue:
            if self.stats.n_expansions_sent == self.n_expansions_max:
                break

            if (
                self.n_concurrent_expansions_max is not None
                and len(self.waiting_for_expansion) == self.n_concurrent_expansions_max
            ):
                break

            node = self.queue.pop(0)
            assert node.thm in self.visited, node.thm
            assert not node.expanded
            assert self.graph.is_leaf_worth_to_expand(node.thm)

            assert (
                node.thm not in self.waiting_for_expansion
            ), "thm already sent to expansion, bug in implementation"
            self.waiting_for_expansion.add(node.thm)
            to_expand.append(node.thm)
            self.stats.n_expansions_sent += 1

        self.stats.cum_time_in_get_thms_to_expand += time() - start
        self.stats.n_batches_of_expansions += 1

        assert len(to_expand) > 0  # if not, infinite loop
        return to_expand

    def send_env_expansions(self, tactics: List[EnvExpansion]) -> None:
        start = time()
        assert len(tactics) == len(self.waiting_for_expansion), (
            len(tactics),
            len(self.waiting_for_expansion),
        )
        tactics = sorted(tactics, key=lambda e: self.graph.th_to_node[e.theorem].id)

        to_put_in_queue: List[ProofTheoremNode] = []
        for exp in tactics:
            self.stats.n_expansions_received += 1

            thm = exp.theorem
            assert thm in self.waiting_for_expansion
            self.waiting_for_expansion.remove(thm)

            if exp.is_error:
                assert exp.error is not None  # mypy
                expanded_node = self.graph.set_expansion_failure(thm, exp.error)
                assert expanded_node.is_failed_leaf
                continue

            assert exp.tactics is not None  # mypy
            assert exp.child_for_tac is not None  # mypy
            assert exp.priors is not None

            assert len(exp.tactics) == len(exp.child_for_tac) == len(exp.priors)
            assert len(exp.tactics) > 0

            # mypy
            def _key(tid: int) -> float:
                assert exp.priors is not None
                return -exp.priors[tid]

            sorted_tactic_ids = sorted(
                range(len(exp.tactics)), key=_key
            )  # best prior first

            expanded_node = self.graph.set_expansion(
                thm=thm,
                tactics=[exp.tactics[tid] for tid in sorted_tactic_ids],
                child_for_tac=[exp.child_for_tac[tid] for tid in sorted_tactic_ids],
            )
            assert not expanded_node.is_failed_leaf
            sorted_priors = [exp.priors[tid] for tid in sorted_tactic_ids]

            # put in queue
            assert expanded_node.tactics is not None
            for tid, tactic_node in enumerate(expanded_node.tactics):
                for sub_goal in tactic_node.sub_goals:
                    if self.best_first_search:
                        self.update_best_first_search_score(
                            thm=sub_goal.thm,
                            prior=sorted_priors[tid],
                            parent_thm=expanded_node.thm,
                        )
                    else:
                        self.update_breath_first_search_score(
                            thm=sub_goal.thm, node_id=sub_goal.id
                        )
                    to_put_in_queue.append(sub_goal)

        assert self.stats.n_expansions_received == self.stats.n_expansions_sent
        assert len(self.waiting_for_expansion) == 0

        self.update_graph_state()

        for sub_goal in to_put_in_queue:
            self._maybe_put_in_queue(sub_goal)

        new_queue = []
        for node in self.queue:
            if self.graph.is_leaf_worth_to_expand(node.thm):
                new_queue.append(node)
            else:
                self.visited.remove(node.thm)
                self.stats.not_useful_to_expand += 1
        self.queue = new_queue

        if self.graph.is_proved(self.graph.root.thm):
            self.done = True
            self.proved = True
            self.stats.stopped = "proved"
        elif self.graph.is_failed(self.graph.root.thm):
            self.done = True
            self.stats.stopped = "failed_root"
        elif len(self.queue) == 0:
            self.done = True
            self.stats.stopped = "empty_queue"
        elif self.stats.n_expansions_received == self.n_expansions_max:
            self.done = True
            self.stats.stopped = "n_expansions_max"
        self.stats.cum_time_in_send_env_exps += time() - start

    def update_graph_state(self):
        start = time()
        self.graph.update_state()
        self.stats.cum_time_in_graph_state_update += time() - start

    def get_result(self) -> ProofResult:
        print(
            f"Calling get_results: is proved: {self.proved}, stats:{self._get_stats()}"
        )
        if self.done and self.proved:
            proof = self.graph.proof(self.graph.root.thm)
            return BFSResult(
                proof=proof, goal=self.goal, exception=None, stats=self._get_stats()
            )
        else:
            return BFSResult(
                proof=None, goal=self.goal, exception=None, stats=self._get_stats()
            )

    def update_breath_first_search_score(self, thm: Theorem, node_id: int):
        self.scores[thm] = min(self.scores.get(thm, math.inf), float(node_id))

    def update_best_first_search_score(
        self, thm: Theorem, prior: float, parent_thm: Theorem
    ):
        assert parent_thm in self.scores
        self.scores[thm] = min(
            (-math.log(prior)) + self.scores[parent_thm], self.scores.get(thm, math.inf)
        )
        assert self.scores[thm] >= 0.0

    def _get_stats(self) -> Dict[str, Union[Number, str]]:
        stats = asdict(self.stats)
        stats.update({f"graph/{k}": v for k, v in self.graph.get_graph_stats().items()})
        stats["proved"] = self.proved
        if self.proved:
            stats["proof_size"] = self.graph.proof_size(self.graph.root.thm)
        return stats
