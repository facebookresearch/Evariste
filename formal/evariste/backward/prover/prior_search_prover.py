# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Union, Any

from numpy.random import RandomState

from evariste.backward.env.core import EnvExpansion
from evariste.backward.prover.bfs_proof_graph import (
    ProofTheoremNode,
    ProofTacticNode,
)
from evariste.backward.prover.nomerge_search_tree import (
    NoMergeSearchTree,
    NoMergeSearchTreeStats,
)

from evariste.backward.prover.prover import ProofResult
from evariste.backward.env.core import BackwardGoal
from evariste.backward.graph import Theorem, UnMaterializedTheorem, Proof
from evariste.backward.prover.prover import ProofHandler
from evariste.backward.prover.utils import Number

import numpy as np

from evariste.metrics import Timer, StatsCollection, DistribStats


@dataclass
class ProofResultWithStats(ProofResult):
    proof_stats: Dict[str, Any]


# do enum
N_EXPANSION_MAX = "n_expansions_max"
FAILED_ROOT = "failed_root"
PROVED = "proved"


@dataclass
class PriorSearchStats(StatsCollection):
    tree_stats: NoMergeSearchTreeStats
    time: float = -1
    stopped: str = "unstopped"
    expansions_batch: DistribStats = field(default_factory=DistribStats)
    distinct_thms_in_exp_batch: DistribStats = field(default_factory=DistribStats)
    time_in_node_selection: Timer = field(default_factory=Timer)
    time_in_node_expansion: Timer = field(default_factory=Timer)
    time_in_send_env_expansions: Timer = field(default_factory=Timer)
    time_in_get_theorems_to_expand: Timer = field(default_factory=Timer)


@dataclass
class PriorSearchResult(ProofResult):
    stats: Dict[str, Union[Number, str]]


class PriorSearchProofHandler(ProofHandler):
    """
    ProofHandler used for handling a Greedy proof strategy.
    """

    def __init__(
        self,
        goal: BackwardGoal,
        n_expansions_max: int,
        n_concurrent_expansions_max: int,
        policy_temperature: float = 1.0,
    ):
        super().__init__(goal)

        # params
        self.n_expansions_max = n_expansions_max
        self.n_concurrent_expansions_max = n_concurrent_expansions_max
        self.policy_temperature = (
            goal.params.policy_temperature
            if (goal.params is not None and goal.params.policy_temperature is not None)
            else policy_temperature
        )

        self.waiting_thm_to_node_ids: Dict[Theorem, List[int]] = {}
        self.proved: bool = False
        self.done: bool = False

        # state
        self._init_state(root_thm=goal.theorem)

        self.rng = RandomState(None)

        self.proof: Optional[Proof] = None
        self.proof_size: Optional[int] = None
        self.tree: NoMergeSearchTree

        self.n_expansions_sent: int = 0
        self.n_expansions_received: int = 0

        self.start = time.time()

    def _init_state(self, root_thm: Theorem):
        # stats
        self.tree = NoMergeSearchTree(root_thm=root_thm)
        self.stats = PriorSearchStats(tree_stats=self.tree.stats)

    def is_ready(self):
        if isinstance(self.tree.root.thm, UnMaterializedTheorem):
            return False
        return True

    def send_materialized(self, th: Theorem):
        self._init_state(root_thm=th)

    def get_theorems_to_expand(self) -> Optional[List[Theorem]]:
        with self.stats.time_in_get_theorems_to_expand.timeit():
            return self._get_theorems_to_expand()

    def _get_theorems_to_expand(self) -> Optional[List[Theorem]]:
        if not self.is_ready():
            return None

        if self.tree.is_solved_node(self.tree.root) or self.tree.is_failed_node(
            self.tree.root
        ):
            assert self.done
            return None

        to_expand: List[ProofTheoremNode] = []
        to_expand_ids: Set[int] = set()
        for _ in range(self.n_concurrent_expansions_max):
            with self.stats.time_in_node_selection.timeit():
                selected = self._select_node_to_expand(
                    self.tree, self.rng, self.policy_temperature
                )
                if selected.id in to_expand_ids:
                    continue
                to_expand_ids.add(selected.id)
                to_expand.append(selected)
                self.n_expansions_sent += 1
                if self.n_expansions_sent == self.n_expansions_max:
                    break

        assert len(self.waiting_thm_to_node_ids) == 0
        self.waiting_thm_to_node_ids = defaultdict(list)
        for node in to_expand:
            assert not node.expanded
            self.waiting_thm_to_node_ids[node.thm].append(node.id)

        self.stats.expansions_batch.update(len(to_expand))
        self.stats.distinct_thms_in_exp_batch.update(len(self.waiting_thm_to_node_ids))
        # print(
        #     f"to_expand: {len(to_expand)} ({len(self.waiting_thm_to_node_ids)} thms) {self.goal.name}"
        # )
        assert len(to_expand) == len({n.id for n in to_expand})
        return [node.thm for node in to_expand]

    @classmethod
    def _select_node_to_expand(
        cls, tree: NoMergeSearchTree, rng: RandomState, policy_temperature: float
    ) -> ProofTheoremNode:
        def _sample_accordingly_to_prior(node: ProofTheoremNode):
            assert node.expanded, node
            assert node.id not in tree.failed
            assert node.id not in tree.solved
            assert node.tactics is not None
            potential_tactics = [
                t for t in node.tactics if not tree.is_failed_tactic(t)
            ]
            assert len(potential_tactics) >= 1
            scores: np.ndarray = np.array([t.prior for t in potential_tactics])
            scores = scores ** (1.0 / policy_temperature)
            probs = scores / scores.sum()
            tid = rng.choice(range(len(potential_tactics)), p=probs)
            selected = potential_tactics[tid]
            assert isinstance(selected, ProofTacticNode)
            node = cls._first_child_not_failed_or_solved(tree, selected)
            return node

        this_node = tree.root
        while this_node.expanded:
            this_node = _sample_accordingly_to_prior(this_node)

        return this_node

    @staticmethod
    def _first_child_not_failed_or_solved(
        tree: NoMergeSearchTree, tactic_node: ProofTacticNode
    ) -> ProofTheoremNode:
        for child in tactic_node.sub_goals:
            if tree.is_failed_node(child) or tree.is_solved_node(child):
                continue
            return child
        raise ValueError("All child not failed or solved")

    def send_env_expansions(self, tactics: List[EnvExpansion]) -> None:
        with self.stats.time_in_send_env_expansions.timeit():
            self._send_env_expansions(tactics)

    def _send_env_expansions(self, tactics: List[EnvExpansion]) -> None:
        for exp in tactics:
            self.n_expansions_received += 1

            thm = exp.theorem
            node_id = self.waiting_thm_to_node_ids[thm].pop()
            if len(self.waiting_thm_to_node_ids[thm]) == 0:
                self.waiting_thm_to_node_ids.pop(thm)

            if exp.is_error:
                assert exp.error is not None  # mypy
                expanded_node = self.tree.set_expansion_failure(node_id, exp.error)
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

            with self.stats.time_in_node_expansion.timeit():
                _ = self.tree.expand_node(
                    node_id=node_id,
                    tactics=[exp.tactics[tid] for tid in sorted_tactic_ids],
                    child_for_tac=[exp.child_for_tac[tid] for tid in sorted_tactic_ids],
                    priors=[exp.priors[tid] for tid in sorted_tactic_ids],
                )

        assert self.n_expansions_received == self.n_expansions_sent, (
            self.n_expansions_sent,
            self.n_expansions_received,
        )
        assert len(self.waiting_thm_to_node_ids) == 0
        # print(
        #     f"name: {self.goal.name} --> received {self.n_expansions_received}/{self.n_expansions_max}"
        # )

        if 0 in self.tree.solved:
            self.done = True
            self.proved = True
            self.stats.stopped = PROVED
        elif 0 in self.tree.failed:
            self.done = True
            self.stats.stopped = FAILED_ROOT
        elif self.n_expansions_received >= self.n_expansions_max:
            self.done = True
            self.stats.stopped = N_EXPANSION_MAX

        if self.done:
            print(f"stopped: {self.goal.name} {self.stats.stopped}")

        if self.proved:
            proof, proof_size = self.tree.get_root_proof_and_proof_size()
            self.proof = proof
            self.proof_size = proof_size

    def get_result(self) -> ProofResultWithStats:
        stats = self._get_stats()
        print(f"Calling get_results: is proved: {self.proved}, stats:{stats}")
        if self.done and self.proved:
            assert self.proof is not None
        return ProofResultWithStats(
            self.proof, goal=self.goal, exception=None, proof_stats=stats
        )

    def _get_stats(self) -> Dict[str, Any]:
        self.tree.update_stats()
        self.stats.time = time.time() - self.start
        stats = self.stats.rate_and_reset()
        stats["proved"] = float(self.proved)
        stats["proof_size"] = (
            float(self.proof_size) if self.proof_size is not None else -1
        )
        return stats
