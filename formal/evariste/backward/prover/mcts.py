# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, List, Set, Dict, Any
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime
from logging import getLogger
from pathlib import Path
import os
import sys
import math
import time
import torch
import pickle
import random
import numpy as np

from evariste.backward.remote.prioritized_label_sampler import PLSStats
from evariste.model.data.envs.minproof_replay_buffer import MCTSSampleProof
from evariste.utils import get_tail_logger, wrap_timer, NoneLogger
from evariste.backward.env.core import BackwardGoal, EnvExpansion
from evariste.backward.env.equations.graph import EQRuleTactic
from evariste.backward.model.beam_search import FAILED_UNK, FAILED_BIG, FAILED_GPU_OOM
from evariste.backward.graph import (
    Theorem,
    Tactic,
    Proof,
    NonPicklableProof,
)
from evariste.backward.env.lean.graph import LeanTactic
from evariste.backward.prover.args import MCTSParams
from evariste.backward.prover.nodes import (
    SimuTree,
    MCTSNode,
    display_from_leaves,
)
from evariste.backward.prover.utils import (
    WeightedAvgStats,
    compute_usage_entropy,
    Number,
)
from evariste.backward.prover.core import ProofResult
from evariste.backward.prover.prover_args import ProverParams
from evariste.model.data.mcts_subproof import SimplifiedMCTSState
from evariste.metrics import ActionCounter
from evariste.backward.prover.mcts_samples import (
    MCTSSampleCritic,
    MCTSSampleEffect,
    MCTSSampleTactics,
)
from evariste.backward.prover.graph import Graph


class DeadRoot(Exception):
    pass


class FailedTactic(Exception):
    pass


class FailedFindToExpand(Exception):
    pass


@dataclass
class MCTSResult(ProofResult):
    mcts_samples_critic: Optional[List[MCTSSampleCritic]]
    mcts_samples_tactic: Optional[List[MCTSSampleTactics]]
    mcts_samples_effect: Optional[List[MCTSSampleEffect]]
    sample_proof: Optional[MCTSSampleProof]
    simplified_state: Optional[SimplifiedMCTSState]
    stats: Dict[str, Tuple[Number, Number]]
    hist_stats: Dict[str, List[int]]


class MCTS(Graph[MCTSNode]):
    """Holds the nodes and statistics for the MCTS. Handles finding the
    :param goal: the root :class:`BackwardGoal` for the search
    :param mcts_params: holds all parameters relevant for the MCTS exploration (exploration constant, policy, ...)
    :param log_path: where tail logs should be dumped
    :param quiet: whether we should write to tail logs
    :param virtual_loss: virtual_loss value (TODO(@tim/@mal): why isn't this an mcts_param ?)
    """

    def __init__(
        self,
        goal: BackwardGoal,
        mcts_params: MCTSParams,
        log_path: Path,
        quiet: bool = False,
        virtual_loss: float = 1,
    ):
        assert goal.materialized

        self.virtual_loss = virtual_loss
        self.log_path = log_path
        self.tail_logger = get_tail_logger(log_path) if not quiet else NoneLogger()

        self.timestep = 0

        super().__init__(root=goal.theorem)
        self.goal: BackwardGoal = goal
        self.params = mcts_params
        self.exploration = mcts_params.exploration
        self.policy = mcts_params.policy
        self.no_critic = mcts_params.no_critic
        self.logger = getLogger()
        self.start_time = time.time()

        # ID of the expansion that solved the root (if solved)
        self.expansion_count = 0
        self.solving_expansion: Optional[int] = None

        # Holds stats
        self.single_stats: Dict[str, float] = defaultdict(float)

        # Not a defaultdict, because 0 should always be counted.
        self.sum_stats: Dict[str, float] = {
            "dead_root": 0,
            "n_find_leaves_to_expand": 0,
            "n_find_leaves_to_expand_aux": 0,
            "failed_find_to_expand": 0,
            "n_to_expand_batch": 0,
            "backup_from_failed_no_tactic_after_env": 0,
            "backup_from_failed_no_tactics_after_filtering": 0,
            f"backup_from_{FAILED_BIG}": 0,
            f"backup_from_{FAILED_GPU_OOM}": 0,
            f"backup_from_{FAILED_UNK}": 0,
            "backup_from_solved": 0,
            "backup_from_solved_temp_leaf": 0,
            "backup_from_solved_internal": 0,
            "backup_from_critic": 0,
            "propagate_expandable": 0,
            "killed_tactics": 0,
        }

        self.weighted_stats = WeightedAvgStats()
        self.hist_stats: Dict[str, List[int]] = {}
        self.useful_updates = ActionCounter(
            name="useful_updates", is_rate=False, silent=True
        )

        self.all_simu_trees_depth: List[int] = []

        self.next_simutree_id = 0
        # Holds simulations that should be backed-up when all expansions are received
        self.simulations_to_backup: Dict[int, SimuTree] = {}
        # "in flight" means the theorem has been sent for expansion.
        self.in_flight: Set[Theorem] = set()
        # Maps a theorem to all simutree ids that use it
        self.theorem_to_simutrees: Dict[Theorem, List[int]] = defaultdict(list)
        # Maps a theorem to all leaves in simutrees
        self.theorem_to_leaves: Dict[Theorem, List[SimuTree]] = defaultdict(list)
        # This is updated when we receive an expansion.
        # When the number of expansions in flight drops to zero, backup the corresponding simutree.
        self.n_expansions_in_flight: Dict[int, int] = {}

        # Do not launch new simulations if we've already reached the limit of theorems in flight
        self.max_in_flight_reached = False

        self.train_samples_critic: Optional[List[MCTSSampleCritic]] = None
        self.train_samples_tactic: Optional[List[MCTSSampleTactics]] = None
        self.train_samples_effect: Optional[List[MCTSSampleEffect]] = None

        # This flag controls whether propagation does anything
        # It is reset to false before each batch of find_to_expand and set to true if one find_leaves_to_expand fails
        self.propagate_for_this_batch = False

        # MCTS move - map a top goal to its number of visits when it became a top goal
        self.top_goals: Optional[Dict[Theorem, int]] = None
        if mcts_params.move.enable:
            self.top_goals = {goal.theorem: 0}

        # time spent to compute the policy
        self.policy_time: float = 0.0

        self.backuped_simutrees: Set[int] = set()

    def get_train_samples(
        self,
    ) -> Tuple[
        List[MCTSSampleCritic],
        List[MCTSSampleTactics],
        List[MCTSSampleEffect],
        Optional[MCTSSampleProof],
    ]:
        """
        Returns an iterator over all nodes with counts.sum() > threshold
        formatted as training samples.
        """
        max_nodes = self.params.max_nodes_for_train
        balance_data = self.params.balance_critic_data
        nodes: List[Tuple[Theorem, MCTSNode]] = list(self.nodes.items())
        sent = 0

        critic_solved = []
        critic_unsolved = []
        tactic_samples = []
        effect_samples = []

        if max_nodes > 0 or balance_data:
            random.shuffle(nodes)
        for theorem, node in nodes:
            effect_samples.extend(node.get_train_sample_effect(self.params))
            critic_sample = node.get_train_sample_critic(self.params)

            if critic_sample is not None:
                assert node.counts is not None
                mcts_critic_sample = MCTSSampleCritic(
                    goal=theorem,
                    label=self.goal.label,
                    visit_count=int(node.counts.sum().item()),
                    solved=node.solved,
                    bad=node.is_bad(),
                    critic=math.exp(node.old_critic)
                    if node.old_critic is not None
                    else math.exp(node.log_critic),
                    **critic_sample,
                )
                if node.solved:
                    critic_solved.append(mcts_critic_sample)
                else:
                    critic_unsolved.append(mcts_critic_sample)
            if self.params.only_learn_tactics_from == "minproof-solving":
                node_mask = "minproof" if self.is_proved else "solving"
            else:
                node_mask = self.params.only_learn_tactics_from

            tactic_sample = self._get_train_sample_tactic_from_node(node, node_mask)

            if tactic_sample is not None:
                tactic_samples.append(tactic_sample)
                sent += 1
            if max_nodes == sent > 0:
                break
        if balance_data:
            min_count = min(len(critic_solved), len(critic_unsolved))
            critic_samples = critic_solved[:min_count] + critic_unsolved[:min_count]
        else:
            critic_samples = critic_solved + critic_unsolved

        proof_sample = self.get_proof_sample()
        return critic_samples, tactic_samples, effect_samples, proof_sample

    def _get_train_sample_tactic_from_node(
        self, node, node_mask
    ) -> Optional[MCTSSampleTactics]:
        tactic_sample = node.get_train_sample_tactic(self.params, node_mask)
        if tactic_sample is None:
            return tactic_sample
        assert node.counts is not None
        return MCTSSampleTactics(
            goal=node.theorem,
            label=self.goal.label,
            visit_count=int(node.counts.sum().item()),
            **tactic_sample,
        )

    def is_leaf_for_now(self, node: MCTSNode) -> bool:
        return (
            node.solved
            and not self.nodes[self.root].solved
            and self.params.early_stop_on_solved_if_root_unproved
        )

    def get_simplified_state(self) -> Optional[SimplifiedMCTSState]:
        """
        Return the whole mcts hypertree but without stats, only nodes edges and tactics.
        :return: SimplifiedMCTS state is only
        root: Theorem,
        nodes: Dict[Theorem, HyperTreeNode]
        but with JSON serialization
        """

        # Condition >= 3 so we keep only not too small hypertrees
        # Could be remove in the future
        # TODO: fix 3
        if len(self.nodes) < 3:
            return None
        return SimplifiedMCTSState.from_mcts_data(self.root, self.nodes)

    def get_proof_sample(self) -> Optional[MCTSSampleProof]:
        self.logger.info("get_proof_sample Looking for proof sample")
        if not self.is_proved:
            self.logger.info("get_proof_sample Not proved")
            return None
        self.logger.info("get_proof_sample Proved")
        samples = []
        for node in self.nodes.values():
            sample = self._get_train_sample_tactic_from_node(node, "minproof")
            if sample is not None:
                samples.append(sample)

        size = self.minproof_size[self.params.proof_stype]
        assert isinstance(size, (float, int)), size
        size = float(size)
        self.logger.info(
            f"get_proof_sample {self.params.proof_stype} "
            f"size: {size}, n_samples: {len(samples)}"
        )

        return MCTSSampleProof(
            label=self.goal.label,
            size=size,
            stype=self.params.proof_stype,
            samples=samples,
        )

    @wrap_timer()
    def find_unexplored_and_propagate_expandable(self):
        """If `propagate_for_this_batch` is set, find all unexplored nodes and use this info to set `expandable`"""
        if not self.propagate_for_this_batch:
            return
        self.sum_stats["propagate_expandable"] += 1
        ignore_solved = self.params.early_stop or (
            not self.nodes[self.root].solved
            and self.params.early_stop_on_solved_if_root_unproved
        )
        super().find_unexplored_and_propagate_expandable(ignore_solved)

    @property
    def dead_root(self) -> bool:
        return (self.propagate_for_this_batch and len(self.unexplored_th) == 0) or (
            self.root in self.nodes and self.nodes[self.root].is_bad()
        )

    @wrap_timer()
    def find_leaves_to_expand(self) -> Tuple[SimuTree, List[SimuTree], List[SimuTree]]:
        """Go down the tree once to find leaves that if solved, would lead to a proof.
        Aggregates a SimuTree representing the search to enable back-propagation.

        This is a wrapper around :meth:`find_leaves_to_expand_aux`
        """
        # Safe because FailedTactic will remove tactics and propagate
        self.sum_stats["n_find_leaves_to_expand"] += 1
        start = time.time()
        n_iter, n_killed = 0, 0
        times = []
        self.policy_time = 0
        while time.time() - start < 60 * 3:
            n_iter += 1
            s = time.time()
            try:
                if self.dead_root:
                    raise DeadRoot
                simutree, terminal, to_expand = self.find_leaves_to_expand_aux()
                if not self.propagate_for_this_batch and len(to_expand) == 0:
                    self.propagate_for_this_batch = True
                    self.find_unexplored_and_propagate_expandable()
                    self.clean_up(simutree)
                    continue
                elif self.propagate_for_this_batch and len(to_expand) == 0:
                    raise RuntimeError("shouldn't happen")

                assert len(to_expand) > 0
                return simutree, terminal, to_expand
            except FailedTactic:
                n_killed += 1
                self.sum_stats["killed_tactics"] += 1
            times.append(time.time() - s)
        # increasing virtual counts did not allow us to find a node to expand
        raise FailedFindToExpand(
            f"time: {time.time() - start:.3f}s "
            f"policy_time: {self.policy_time:.3f}s "
            f"n_killed: {n_killed} n_iter: {n_iter} "
            f"len(self.nodes): {len(self.nodes)} "
            f"avg_find_leaves_time: {np.mean(times):.6f}s "
            f"len(self.unexplored_th): {len(self.unexplored_th)} "
        )

    @wrap_timer()
    def find_leaves_to_expand_aux(
        self,
    ) -> Tuple[SimuTree, List[SimuTree], List[SimuTree]]:
        """
        Contrary to usual MCTS, one simulation in our case updates many leaves.
        We follow the best tactic until we reach leaves, building a tree that
        represents our path.
        We then expand some of these leaves and then backup the tree.

        If a loop is created by a tactic, we kill the tactic and update reachabilities. This is not optimal
        and may prevent finding proofs. But many attempts to make this better have failed...
        """
        self.sum_stats["n_find_leaves_to_expand_aux"] += 1
        simu_tree = SimuTree(self.root, depth=0, parent=None, message="root")
        to_expand = deque([simu_tree])
        leaves_terminal: List[SimuTree] = []
        leaves_to_expand: List[SimuTree] = []

        # sample the next policy. either have a sampling temperature:
        #     - constant for the entire MCTS (global)
        #     - randomly sampled for each simutree (simutree)
        #     - sampled every time we need to select a tactic (tactic)
        policy_temperature = self.params.policy_temperature
        if self.params.policy_temp_level == "simutree":
            policy_temperature = self.params.policy_temperature * np.random.rand()

        while to_expand:
            cur_node = to_expand.pop()

            # TODO: not compatible with propagate_expandable. Not trivial to make compatible since depth depends on path
            # for HOL-Light proof shortening
            # if self.goal.num_steps is not None and cur_node.depth > self.goal.num_steps:
            #     cur_node.message = "max_depth_exceeded"
            #     cur_node.value_to_backup = -math.inf
            #     leaves_terminal.append(cur_node)
            #     continue

            # node has not been expanded yet
            mcts_node = self.nodes.get(cur_node.theorem, None)
            if mcts_node is None:
                prev_depth = cur_node.theorem.info.depth
                if prev_depth is None:
                    cur_node.theorem.info.depth = cur_node.depth
                else:
                    cur_node.theorem.info.depth = min(prev_depth, cur_node.depth)
                leaves_to_expand.append(cur_node)
                continue

            # terminal node must be solved.
            # Otherwise its ancestors have been killed so its unreachable
            is_leaf_for_now = self.is_leaf_for_now(mcts_node)
            if mcts_node.is_terminal() or is_leaf_for_now:
                if not mcts_node.is_solved_leaf and not is_leaf_for_now:
                    raise RuntimeError(
                        f"huh {len(mcts_node.children_for_tactic)} {self.goal.label} "
                        f"{mcts_node.is_bad()} {cur_node.theorem == self.root}"
                    )
                cur_node.value_to_backup = mcts_node.value()
                cur_node.message = (
                    "solved" if not is_leaf_for_now else "solved_temp_leaf"
                )
                cur_node.solved = True
                leaves_terminal.append(cur_node)
                continue
            if self.params.early_stop and mcts_node.solved:
                cur_node.value_to_backup = 0
                cur_node.message = "solved_internal"
                cur_node.solved = True
                leaves_terminal.append(cur_node)
                continue

            # node has already been expanded and we're not at max depth
            start = time.time()
            policy = mcts_node.policy(expansion_time=True)
            self.policy_time += time.time() - start

            # Stores current statistics used for decision making. Useful for the visualizer
            cur_node.policy = policy.copy()
            assert (
                mcts_node.logW is not None
                and mcts_node.counts is not None
                and mcts_node.virtual_counts is not None
                and mcts_node.priors is not None
            )
            cur_node.logW = mcts_node.logW.copy()
            cur_node.counts = mcts_node.counts.copy()
            cur_node.virtual_counts = mcts_node.virtual_counts.copy()
            cur_node.priors = mcts_node.priors.copy()

            # select the next tactic
            if self.params.policy_temperature == 0:
                tactic_id = policy.argmax().item()
            else:
                if self.params.policy_temp_level == "node":
                    policy_temperature = (
                        self.params.policy_temperature * np.random.rand()
                    )
                policy_temperature = max(policy_temperature, 0.001)
                logits = torch.from_numpy(policy).log() / policy_temperature
                p = torch.softmax(logits, dim=-1).numpy()
                tactic_id = np.random.choice(len(policy), p=p)
            assert tactic_id not in mcts_node.killed_tactics
            tactic = mcts_node.tactics[tactic_id]
            cur_node.tactic_id = tactic_id
            cur_node.tactic_str = repr(tactic)
            children = mcts_node.children_for_tactic[tactic_id]

            # Check if this tactic creates a loop
            assert cur_node.theorem_seen is not None
            loop_created = any(child in cur_node.theorem_seen for child in children)
            if not loop_created:
                mcts_node.virtual_counts[tactic_id] += self.virtual_loss
                cur_node.virtual_count_added = True
                simu_childrens = [
                    SimuTree(theorem=child, depth=cur_node.depth + 1, parent=cur_node)
                    for child in children
                ]
                cur_node.theorem_seen = None  # O(n^2) memory
                to_expand.extendleft(simu_childrens)
            else:
                self.kill_tactic(
                    mcts_node,
                    tactic_id,
                    self.timestep,
                    add_to_history=self.params.save_history_updates,
                )
                self.clean_up(simu_tree)
                self.find_unexplored_and_propagate_expandable()  # something changed, recompute
                raise FailedTactic

        # sanity check
        all_leaves = leaves_terminal + leaves_to_expand
        counted = simu_tree.count_leaves()
        assert len(all_leaves) > 0
        assert all(leaf.theorem not in self.nodes for leaf in leaves_to_expand)
        assert counted == len(all_leaves), f"{counted} != {len(all_leaves)}"

        for leaf in all_leaves:
            leaf.theorem_seen = None  # Could be big and not needed anymore

        # update stats
        self.all_simu_trees_depth.append(max([leaf.depth for leaf in all_leaves]))

        # This works, but let's not go through this more than needed
        # for n in simu_tree.walk():
        #     assert n.theorem_seen is None

        return simu_tree, leaves_terminal, leaves_to_expand

    @wrap_timer()
    def to_expand(
        self, simu_tree: SimuTree, leaves_to_expand: List[SimuTree]
    ) -> List[Theorem]:
        """Book-keeping between :meth:`find_leaves_to_expand` and :meth:`receive_expansion`.
        Store the simu_tree, make leaves_to_expand unique (with other leaves_to_expand in the batch).

        When we receive expansions, we will be able to retrieve all simu-trees associated to leaves being expanded.
        """
        to_ret: List[Theorem] = []
        seen: Set[Theorem] = set()  # count each theorem only once
        self.simulations_to_backup[self.next_simutree_id] = simu_tree
        self.n_expansions_in_flight[self.next_simutree_id] = 0
        for leaf in leaves_to_expand:
            self.theorem_to_leaves[leaf.theorem].append(leaf)
            if leaf.theorem not in seen:
                self.theorem_to_simutrees[leaf.theorem].append(self.next_simutree_id)
                self.n_expansions_in_flight[self.next_simutree_id] += 1
                seen.add(leaf.theorem)
            if leaf.theorem not in self.in_flight:
                self.in_flight.add(leaf.theorem)
                to_ret.append(leaf.theorem)
                assert (
                    leaf.theorem.info.depth is not None
                    and leaf.theorem.info.depth <= leaf.depth
                ), (leaf.theorem.info.depth, leaf.depth)
        self.next_simutree_id += 1
        return to_ret

    @wrap_timer()
    def receive_expansion(self, theorem: Theorem, v: float, solved: bool, message: str):
        """
        Sets the value / solved / message received from an expansion in all corresponding simutrees.
        """
        assert v <= 0, v
        # set the expansion result in all simutrees
        for leaf in self.theorem_to_leaves[theorem]:
            leaf.value_to_backup = v
            leaf.solved = solved
            leaf.message = message
        del self.theorem_to_leaves[theorem]

        # then decrease the expansion in flight counter. if it reaches 0, we can backup this simutree
        for simutree_id in self.theorem_to_simutrees[theorem]:
            self.n_expansions_in_flight[simutree_id] -= 1
            assert self.n_expansions_in_flight[simutree_id] >= 0
        del self.theorem_to_simutrees[theorem]

        self.in_flight.remove(theorem)

    @wrap_timer(verbose=1)
    def do_expand(self, expansions: List[EnvExpansion]):
        """
        For all expansions in the list:

        * if error: create an MCTSNode with no tactics, critic 0 and error message

        * if solved: create a solved MCTSNode

        * otherwise : create an MCTSNode with corresponding tactics and children

        Then, update the underlying :class:`Graph` for unexplored / reachable / solved / invalid
        """
        new_nodes: List[MCTSNode] = []
        for expansion in expansions:
            assert expansion.theorem not in self.nodes
            if expansion.is_error:  # node is invalid
                assert expansion.error is not None
                assert expansion.error.startswith("failed")
                new_node = MCTSNode(
                    theorem=expansion.theorem,
                    time_created=self.timestep,
                    tactics=[],
                    log_critic=-math.inf,
                    children_for_tactic=[],
                    priors=[],
                    exploration=self.exploration,
                    policy=self.policy,
                    error=expansion.error,
                    effects=expansion.effects,
                    init_tactic_scores=self.params.init_tactic_scores,
                    q_value_solved=self.params.q_value_solved,
                )  # terminal node
                self.receive_expansion(
                    expansion.theorem,
                    v=-math.inf,
                    solved=False,
                    message=expansion.error,
                )
            else:
                assert (
                    expansion.log_critic is not None
                    and expansion.tactics is not None
                    and expansion.child_for_tac is not None
                    and expansion.priors is not None
                )

                if len(expansion.child_for_tac[0]) == 0:  # node is solved
                    assert all(len(sg) == 0 for sg in expansion.child_for_tac)
                    assert all(t.is_valid for t in expansion.tactics)
                    new_node = MCTSNode(
                        theorem=expansion.theorem,
                        time_created=self.timestep,
                        tactics=expansion.tactics,
                        log_critic=0.0,
                        children_for_tactic=expansion.child_for_tac,
                        priors=expansion.priors,
                        exploration=self.exploration,
                        policy=self.policy,
                        error=None,
                        effects=expansion.effects,
                        init_tactic_scores=self.params.init_tactic_scores,
                        q_value_solved=self.params.q_value_solved,
                    )  # terminal node
                    self.receive_expansion(
                        expansion.theorem, v=0, solved=True, message="solved"
                    )
                else:  # regular node
                    assert len(expansion.tactics) > 0

                    if self.no_critic:
                        log_critic = math.log(0.5)
                    else:
                        log_critic = expansion.log_critic

                    self.weighted_stats.update(
                        {"n_tac_per_node": (len(expansion.tactics), 1.0)}
                    )

                    new_node = MCTSNode(
                        theorem=expansion.theorem,
                        time_created=self.timestep,
                        tactics=expansion.tactics,
                        log_critic=log_critic,
                        children_for_tactic=expansion.child_for_tac,
                        priors=expansion.priors,
                        exploration=self.exploration,
                        policy=self.policy,
                        error=None,
                        effects=expansion.effects,
                        init_tactic_scores=self.params.init_tactic_scores,
                        q_value_solved=self.params.q_value_solved,
                    )
                    self.receive_expansion(
                        expansion.theorem, v=log_critic, solved=False, message="critic"
                    )

            new_nodes.append(new_node)
        self.add_nodes(
            new_nodes, self.timestep, add_to_history=self.params.save_history_updates
        )
        # log ID of the expansion that solved the root
        if self.solving_expansion is None and self.nodes[self.root].solved:
            self.solving_expansion = self.expansion_count
        self.expansion_count += 1

    @wrap_timer(verbose=1)
    def do_backup(self):
        """
        Backup all simutrees for received expansions
        """
        for simutree_id, n_expansions_in_flight in list(
            self.n_expansions_in_flight.items()
        ):
            if n_expansions_in_flight > 0:
                continue
            leaves = self.simulations_to_backup[simutree_id].leaves()
            for leaf in leaves:
                if leaf.message != "":
                    self.sum_stats[f"backup_from_{leaf.message}"] += 1
            only_vl = False
            if self.params.backup_once:
                hashed = self.simulations_to_backup[simutree_id].get_hash()
                only_vl = hashed in self.backuped_simutrees
                self.backuped_simutrees.add(hashed)
            self.backup_leaves(leaves, only_vl)
            self.simulations_to_backup[simutree_id].pre_delete()
            del self.n_expansions_in_flight[simutree_id]
            del self.simulations_to_backup[simutree_id]

    @wrap_timer()
    def backup_leaves(self, leaves: List[SimuTree], only_vl: bool = False) -> None:
        """
        Goes backward through the Simulation tree in topological order, propagating the values.
        :param leaves: the leaves of the simutree to start backpropagation from
        :param only_vl:  used to not backup multiples times the same thing when there are duplicated simutrees
        """
        root_updated = False
        to_backup = []
        for leaf in leaves:
            mcts_node = self.nodes[leaf.theorem]
            if mcts_node.virtual_counts is not None and leaf.virtual_count_added:
                mcts_node.virtual_counts[leaf.tactic_id] -= self.virtual_loss
                assert mcts_node.virtual_counts[leaf.tactic_id] >= 0
            assert leaf.value_to_backup <= 0
            if leaf.parent is None:
                root_updated = True
                continue
            leaf.parent.children_propagated += 1
            if leaf.parent.children_propagated == len(leaf.parent.children):
                to_backup.append(leaf.parent)

        while to_backup:
            cur = to_backup.pop()
            v_to_b = [x.value_to_backup for x in cur.children]
            assert all(x <= 0 for x in v_to_b), v_to_b
            to_update = sum(v_to_b)  # sum logprobs = proba product
            mcts_node = self.nodes[cur.theorem]
            # If we know that this node is solved, then we should backprop this and not the potentially
            # pessimistic value we just explored. This means that further exploring already solved branches
            # won't affect their values.
            if mcts_node.solved and self.params.backup_one_for_solved:
                to_update = 0.0  # log(1)

            # backup depth. WARNING: not totally accurate for solved nodes
            if self.params.depth_penalty < 1:
                to_update += math.log(self.params.depth_penalty)

            cur.value_to_backup = to_update

            if cur.virtual_count_added:
                assert mcts_node.virtual_counts is not None
                mcts_node.virtual_counts[cur.tactic_id] -= self.virtual_loss
                assert mcts_node.virtual_counts[cur.tactic_id] >= 0
            if not only_vl:
                assert isinstance(to_update, float) and to_update <= 0, to_update
                assert cur.tactic_id is not None
                mcts_node.update(cur.tactic_id, to_update)
                if self.params.save_history_updates:
                    mcts_node.history.update(self.timestep, cur.tactic_id, to_update)
            if cur.parent is None:
                root_updated = True
                continue
            cur.parent.children_propagated += 1
            if cur.parent.children_propagated == len(cur.parent.children):
                to_backup.append(cur.parent)

        if not root_updated:
            display_from_leaves(leaves)
            raise Exception("Somehow we didn't back-up to the root!")

        # # check propagation
        # to_backup = [leaf.parent for leaf in leaves if leaf.parent is not None]
        # while to_backup:
        #     cur = to_backup.pop()
        #     assert cur.children_propagated == len(cur.children)
        #     if cur.parent is not None:
        #         to_backup.append(cur.parent)

    @wrap_timer()
    def clean_up(self, to_clean: SimuTree) -> None:
        """
        If find_leaves_to_expand was interrupted, clean up virtual loss
        """
        for cur_node in to_clean.walk():
            mcts_node = self.nodes.get(cur_node.theorem, None)
            if mcts_node is None:
                continue
            if mcts_node.virtual_counts is not None and cur_node.virtual_count_added:
                mcts_node.virtual_counts[cur_node.tactic_id] -= self.virtual_loss
                assert mcts_node.virtual_counts[cur_node.tactic_id] >= 0

    def mcts_move(self):
        """
        Move in the MCTS, e.g. commit to the most promising tactics.
        """
        move_params = self.params.move
        if not move_params.enable:
            return

        assert self.top_goals is not None and len(self.top_goals) > 0
        old_top_goals: Set[Theorem] = set()
        new_top_goals: Set[Theorem] = set()

        for goal, visits_before_top in self.top_goals.items():

            node = self.nodes[goal]
            if node.solved:  # solved nodes must be removed from top nodes
                old_top_goals.add(goal)
                continue
            if node.all_tactics_killed:
                return  # node is dead. the search will end here
            if node.counts is None:
                continue  # nothing tried so far
            n_visits = node.counts.sum()
            assert n_visits >= visits_before_top
            if n_visits - visits_before_top < move_params.budget:
                continue  # not enough visits as a top goal

            # the node is not solved and has used all its budget.
            # commit to the most promisting tactic by killing the others
            policy = node.policy(expansion_time=True)
            selected_tid = policy.argmax()
            for tid in range(node.n_tactics):
                if tid != selected_tid and tid not in node.killed_tactics:
                    self.kill_tactic(node, tactic_id=tid, timestep=self.timestep)

            # compute old / new top goals
            old_top_goals.add(goal)
            new_top_goals.update(set(node.children_for_tactic[selected_tid]))

        # set the number of counts when a goal becomes a top node
        for goal in new_top_goals:
            if goal not in self.top_goals:
                self.top_goals[goal] = (
                    0
                    if goal not in self.nodes or self.nodes[goal].counts is None
                    else self.nodes[goal].counts.sum()
                )
        for goal in old_top_goals:
            self.top_goals.pop(goal)

        assert len(self.top_goals) > 0

    @wrap_timer()
    def get_stats(self) -> Dict[str, Tuple[Number, Number]]:
        """
        MCTS search tree stats such as depth, stats on tactics, number of solved nodes.
        """

        def get_solved_ratio(nodes: List[MCTSNode]) -> float:
            if len(nodes) == 0:
                return 0.0
            return len([node for node in nodes if node.solved]) / len(nodes)

        self.get_tactics_stats()
        self.get_depth_stats()
        self.get_proof_size_stats()
        self.get_critic_and_q_stats()
        assert self.train_samples_critic is not None
        assert self.train_samples_tactic is not None
        assert self.train_samples_effect is not None

        mcts_stats_: Dict[str, Number] = {
            "nsteps": self.timestep,
            "n_nodes": len(self.nodes),
            "train_samples_n_nodes_critic": len(self.train_samples_critic),
            "train_samples_n_nodes_tactic": len(self.train_samples_tactic),
            "train_samples_n_nodes_effect": len(self.train_samples_effect),
            "train_samples_critic_solved_ratio": get_solved_ratio(
                [self.nodes[s.goal] for s in self.train_samples_critic]
            ),
            "useful_updates": self.useful_updates.rate_and_reset(),
        }

        assert mcts_stats_.keys().isdisjoint(self.sum_stats.keys())
        assert mcts_stats_.keys().isdisjoint(self.single_stats.keys())
        mcts_stats_.update(self.sum_stats)
        mcts_stats_.update(self.single_stats)

        weighted_mcts_stats: Dict[str, Tuple[Number, Number]] = {
            k: (v, 1.0) for k, v in mcts_stats_.items()
        }
        ratio_solved = get_solved_ratio(list(self.nodes.values()))
        weighted_mcts_stats["solved_ratio_total"] = (ratio_solved, len(self.nodes))
        weighted_mcts_stats["solved_ratio_per_th"] = (ratio_solved, 1.0)
        weighted_mcts_stats.update(self.weighted_stats.tuple_stats)

        return weighted_mcts_stats

    @wrap_timer()
    def get_depth_stats(self):
        """
        Return stats on depths of the MCTS search tree nodes
        i.e the minimal depth to reach search tree leaf node from root,
        and the depth of simutrees.
        """
        nodes_to_depth = {}
        to_visit = deque([(self.root, 0)])
        while to_visit:
            cur, d = to_visit.pop()  # pop right
            if cur in nodes_to_depth or cur not in self.nodes:
                continue
            nodes_to_depth[cur] = d
            for children in self.nodes[cur].children_for_tactic:
                for c in children:
                    to_visit.appendleft((c, d + 1))
        depths = np.fromiter(nodes_to_depth.values(), dtype=np.float64)

        self.single_stats.update(
            {
                "depth_mean": np.mean(depths),
                "depth_max": np.max(depths),
                "simu_depth_max": max(self.all_simu_trees_depth),
                "simu_depth_mean": float(np.mean(self.all_simu_trees_depth)),
            }
        )

    @wrap_timer()
    def get_proof_size_stats(self):
        """
        Return stats on proof sizes.
        """
        for stype in self.stypes:
            proof_stats = {
                "proof_in_proof": sum([node.in_proof for node in self.nodes.values()])
                / len(self.nodes),
                f"proof_{stype}_in_miniproof": sum(
                    [node.in_minproof[stype] for node in self.nodes.values()]
                )
                / len(self.nodes),
            }
            assert (
                (proof_stats["proof_in_proof"] == 0)
                == (proof_stats[f"proof_{stype}_in_miniproof"] == 0)
                == (not self.is_proved)
            )
            self.single_stats.update(proof_stats)
            if self.is_proved:
                self.single_stats[f"minproof_{stype}"] = self.minproof_size[stype]

    @wrap_timer()
    def get_tactics_stats(self):

        # usage / entropy
        usage, entropy, hist = self.get_tactics_usage_entropy(
            [t for node in self.nodes.values() for t in node.tactics]
        )
        self.single_stats["tactics_usage"] = usage
        self.single_stats["tactics_entropy"] = entropy
        self.hist_stats["tactics_global"] = hist
        usage, entropy, hist = self.get_tactics_usage_entropy(
            [t for sample in self.train_samples_tactic for t in sample.tactics]
        )
        self.single_stats["train_samples_tactics_usage"] = usage
        self.single_stats["train_samples_tactics_entropy"] = entropy
        self.hist_stats["train_samples_tactics_global"] = hist

        for sample in self.train_samples_tactic:
            node = self.nodes[sample.goal]
            if node.is_terminal():
                continue
            s = "solved" if node.solved else "unsolved"
            self.weighted_stats.update(
                {
                    f"train_samples_n_tac_per_{s}_filtered": (len(sample.tactics), 1.0),
                    f"train_samples_n_tac_per_{s}": (len(node.tactics), 1.0),
                }
            )

    @wrap_timer()
    def get_tactics_usage_entropy(
        self, tactics: List[Tactic]
    ) -> Tuple[float, float, List[int]]:
        # TODO implement for other envs
        tactics = [tac for tac in tactics if tac.is_valid]  # remove "fake" tactics
        if len(tactics) == 0 or not isinstance(tactics[0], (EQRuleTactic, LeanTactic)):
            return -1.0, -1.0, []
        if isinstance(tactics[0], EQRuleTactic):
            # compute histogram
            counts = np.zeros(len(EQRuleTactic.LABEL_TO_ID), dtype=np.int64)
            for tactic in tactics:
                assert isinstance(tactic, EQRuleTactic)
                counts[EQRuleTactic.LABEL_TO_ID[tactic.label]] += 1
        elif isinstance(tactics[0], LeanTactic):
            counts = np.zeros(tactics[0].n_theorems, dtype=np.int64)
            for tactic in tactics:
                assert isinstance(tactic, LeanTactic)
                assert tactic.n_theorems == counts.shape[0]
                for th in tactic.uses_theorems:
                    counts[th] += 1
        usage, entropy = compute_usage_entropy(counts)
        return usage, entropy, counts.tolist()

    def get_critic_and_q_stats(self):

        # histograms q value for trainer
        self.hist_stats["train_samples_q_nodes"] = np.histogram(
            [ts.q_estimate for ts in self.train_samples_critic], bins=10, range=(0, 1),
        )[0].tolist()
        if self.params.train_sample_for_q_conditioning:
            all_q_tactics: List[float] = []
            for s in self.train_samples_tactic:
                assert s.q_tactics is not None
                all_q_tactics += s.q_tactics
            self.hist_stats["train_samples_q_tactics"] = np.histogram(
                all_q_tactics, bins=10, range=(0, 1)
            )[0].tolist()

        # av critic and q value
        def status(node):
            if node.solved:
                return ["solved"]
            elif node.is_terminal():
                if node.should_send(self.params):
                    return ["training_terminal", "terminal"]
                return ["terminal"]
            elif node.should_send(self.params):
                return ["training_unsolved", "unsolved"]
            else:
                return ["unsolved"]

        def add_stats(values, name):
            self.weighted_stats.update(
                {f"av_{name}": (np.mean(values), self.single_stats[f"n_nodes_{type}"],)}
            )
            self.single_stats[f"std_{name}"] = np.std(values)
            self.hist_stats[f"hist_{name}"] = np.histogram(
                values, bins=10, range=(0, 1),
            )[0].tolist()

        nodes_by_type = defaultdict(list)
        for node in self.nodes.values():
            for s in status(node):
                nodes_by_type[s].append(node)
        # critic
        for type, nodes in nodes_by_type.items():
            self.single_stats[f"n_nodes_{type}"] = len(nodes)
            if self.single_stats[f"n_nodes_{type}"] > 0:
                critics = [
                    math.exp(node.old_critic)
                    if node.old_critic is not None
                    else math.exp(node.log_critic)
                    for node in nodes
                ]
                add_stats(critics, f"critic_{type}")
        # q
        for type in ["training_unsolved", "unsolved"]:
            if self.single_stats[f"n_nodes_{type}"] > 0:
                q_estimates = [math.exp(node.value()) for node in nodes_by_type[type]]
                add_stats(q_estimates, f"q_{type}")


class MCTSHandler:
    """
    Thin wrapper around :class:`MCTS`. Was created to handle two modes :
    * `batch` (current one) where we select a batch of nodes to expand and receive them all at once
    * `fluid` where we have a maximal amount of `hanging` expansion requests, but produce and ingest them smoothly otherwise.
    This was supposed to help GPU usage but was never useful and made code more complex
    """

    def __init__(
        self, goal: BackwardGoal, prover_params: ProverParams, process_id: int,
    ):
        self.goal = goal
        new_params = prover_params.from_goal_params(goal.params)

        self.mcts = MCTS(
            goal=goal,
            mcts_params=new_params.mcts,
            log_path=prover_params.dump_path / "mcts_logs" / f"MCTS.{process_id}.log",
            quiet=prover_params.quiet,
        )
        self.done = False

        self.dump_mcts = prover_params.dump_mcts
        self.dump_path = prover_params.dump_path
        assert self.dump_path.exists()
        os.makedirs(self.dump_path / "mcts", exist_ok=True)

        self.leaves_to_expand: List[SimuTree] = []
        self.n_expansions = 0
        self.mcts_subproofs_online_training = (
            prover_params.mcts_subproofs_online_training
        )

        self._timings: Dict[str, float] = defaultdict(float)

        # log path / tail logger
        log_path = self.dump_path / "mcts_logs" / f"MCTSHandler.{process_id}.log"
        self.tail_logger = (
            get_tail_logger(log_path) if not prover_params.quiet else NoneLogger()
        )

    @wrap_timer(verbose=1)
    def expand_and_backup(self, expansions: List[EnvExpansion]):
        """Add expansions the the MCTS and backup new critic estimates."""
        start_t = time.time()
        self.mcts.do_expand(expansions)
        self.mcts.do_backup()

        # all virtual counts should be back to zero
        for node in self.mcts.nodes.values():
            assert node.virtual_counts is None or node.virtual_counts.sum() == 0

        # if solved for the first time, log the minproof size
        if self.mcts.is_proved and self.mcts.init_minproof_size is None:
            self.mcts.get_inproof_nodes()
            self.mcts.get_node_proof_sizes_and_depths()
            self.mcts.init_minproof_size = {
                stype: self.mcts.nodes[self.mcts.root].my_minproof_size[stype]
                for stype in self.mcts.stypes
            }
            assert all(v < math.inf for v in self.mcts.init_minproof_size.values())
            self.mcts.reset_minproof_stats()  # reset stats for final minproof computation

        if self.mcts.is_proved:
            self.done |= self.mcts.params.early_stop
        self.mcts.timestep += 1
        assert len(self.mcts.nodes) == self.n_expansions
        self.done |= self.n_expansions >= self.mcts.params.n_expansions
        if not self.done:
            self.mcts.mcts_move()
        self._timings["expand_and_backup"] += time.time() - start_t

    def get_theorems_to_expand(self) -> List[Theorem]:
        """
        Get theorems to expand.
        In batch mode, run succ_expansions simulations.
        In fluid mode, run one if not enough theorems are 'in flight'.
        """
        start_t = time.time()
        try:
            to_ret = self.to_expand_batch()
        except RuntimeError as e:
            path = self.dump()
            print(f"RUNTIME ERROR -- {path} -- {e}", file=sys.stderr, flush=True)
            raise
        assert len(to_ret) > 0 or self.done, f"WHAT {self.mcts.goal.label}"
        self._timings["get_theorems_to_expand"] += time.time() - start_t
        return list(to_ret)

    @wrap_timer(verbose=1)
    def to_expand_batch(self) -> Set[Theorem]:
        """Find nodes to expand by running several :meth:`MCTS.find_leaves_to_expand`.

        * If we get stuck, raise :exc:`FailedFindToExpand`. This should not happen anymore.

        * If there are no more valid tactics at the root, raise :exc:`DeadRoot`

        * Stop searching early if we found enough nodes to expand
        """
        start_t = time.time()
        self.mcts.sum_stats["n_to_expand_batch"] += 1
        useful_updates = 0
        res: Set[Theorem] = set()
        start = time.time()
        try:
            self.mcts.propagate_for_this_batch = False
            for i in range(self.mcts.params.succ_expansions):
                start_ = time.time()
                simu_tree, term, to_ex = self.mcts.find_leaves_to_expand()
                if time.time() - start > 1000:
                    self.mcts.clean_up(simu_tree)
                    raise FailedFindToExpand(
                        f"Fail to expand Batch - Too Long"
                        f"time: {time.time() - start:.3f}s "
                        f"n_killed: {self.mcts.sum_stats['killed_tactics']} "
                        f"iter: {i} "
                        f"time find_leaves_to_exp_aux: {time.time() - start_:.3f}s "
                        f"len(self.nodes): {len(self.mcts.nodes)} "
                        f"len(self.unexplored_th): {len(self.mcts.unexplored_th)} "
                    )
                before_len = len(res)
                res.update(self.mcts.to_expand(simu_tree, to_ex))
                if len(res) != before_len:
                    useful_updates += 1
                # avoid having too many nodes to expand at once
                if (
                    self.mcts.params.max_nodes_in_expansion is not None
                    and len(res) >= self.mcts.params.max_nodes_in_expansion
                ):
                    break
            self.mcts.useful_updates.act(useful_updates)
        except DeadRoot:
            self.mcts.sum_stats["dead_root"] += 1
        except FailedFindToExpand as e:
            print(
                f'FailedFindToExpand {datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}: '
                f"{self.mcts.goal.label[:40]:<40} {e}",
                file=sys.stderr,
                flush=True,
            )
            self.dump(force_dump=True)
            self.mcts.sum_stats["failed_find_to_expand"] += 1

        if len(res) == 0:
            self.done = True
            self._timings["to_expand_batch"] += time.time() - start_t
            return set()

        assert len(set(res)) == len(res)
        self.n_expansions += len(res)
        available_exp = len(self.mcts.unexplored_th)
        self.mcts.weighted_stats.update(
            {
                "avg_expansion_size": (len(res), 1.0),
                "avg_available_exp_size": (available_exp, 1.0),
                "avg_expansion_ratio": (
                    len(res) / available_exp if available_exp > 0 else 1,
                    1.0,
                ),
            }
        )
        self._timings["to_expand_batch"] += time.time() - start_t
        return res

    @wrap_timer(verbose=1)
    def result(self) -> MCTSResult:
        """
        Aggregates all stats and training samples to create the MCTSResult.
        """

        start_t = time.time()

        res_stats: Dict[str, Tuple[Number, Number]] = {}

        try:
            self.mcts.check_solved_ok()
        except Exception as e:
            path = self.dump(force_dump=True)
            raise RuntimeError(f"Check solved error! DUMP IN {path} {e}")
        for node in self.mcts.nodes.values():
            if node.virtual_counts is not None and not np.all(node.virtual_counts == 0):
                path = self.dump(force_dump=True)
                raise RuntimeError(f"Virtual Counts not zero! Dump in {path}")
        self._timings["result__check"] = time.time() - start_t

        # compute inproof / minproof stats
        t = time.time()
        self.mcts.get_inproof_nodes()
        self.mcts.get_node_proof_sizes_and_depths()
        self._timings["result__inproof_sizes"] = time.time() - t

        # if solved, compute how much we reduced the minproof size since the initial solving
        if self.mcts.is_proved:
            assert self.mcts.init_minproof_size is not None
            assert self.mcts.init_minproof_size.keys() == self.mcts.minproof_size.keys()
            for stype, init_v in self.mcts.init_minproof_size.items():
                final_v = self.mcts.minproof_size[stype]
                assert final_v is not None, final_v
                assert final_v <= init_v < math.inf, (stype, init_v, final_v)
                res_stats[f"minproof_reduced__{stype}"] = (init_v - final_v, 1.0)

        if self.dump_mcts:
            self.dump()

        t = time.time()
        if self.mcts_subproofs_online_training:
            self.mcts.train_samples_critic = None
            self.mcts.train_samples_tactic = None
            self.mcts.train_samples_effect = None
            sample_proof = None
            mcts_state = self.mcts.get_simplified_state()
        else:
            (
                self.mcts.train_samples_critic,
                self.mcts.train_samples_tactic,
                self.mcts.train_samples_effect,
                sample_proof,
            ) = self.mcts.get_train_samples()
            mcts_state = None
            assert (sample_proof is not None) == self.mcts.is_proved
        self._timings["result__get_samples"] = time.time() - t

        res_stats.update(self.mcts.get_stats())
        res_stats["proved"] = (int(self.mcts.is_proved), 1.0)

        pls_stats = PLSStats.from_mcts(
            samples=self.mcts.train_samples_critic,
            proved=self.mcts.is_proved,
            minproof_size=self.mcts.minproof_size["size"],
        )
        res_stats.update(pls_stats.to_stats())

        # dump time
        if time.time() - self.mcts.start_time > 3600:
            self.mcts.logger.warning(
                f"MCTS TOOK TOO LONG: {self.mcts.goal.label[:40]:<40} - "
                f"{self.mcts.timestep} steps - "
                f"Since start time: {time.time() - self.mcts.start_time:.3f}",
            )

        # extract one proof
        t = time.time()
        proof: Optional[Proof] = None
        if self.mcts.is_proved:
            try:
                stype = self.mcts.params.proof_stype
                proof = self.mcts.get_minproof(self.mcts.root, stype=stype)
                res_stats["get_minproof_recursion_error"] = (0, 1.0)
            except RecursionError:
                print(
                    f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")} -- '
                    f"RecursionError on {self.goal.name} while calling get_minproof.",
                )
                proof = NonPicklableProof()
                res_stats["get_minproof_recursion_error"] = (1, 1.0)
                dump_path = self.dump()
                self.mcts.logger.warning(
                    f"get_minproof error on {self.goal.label} -- dumping in {dump_path}"
                )
        self._timings["result__extract_one"] = time.time() - t

        # do not dump proof if pickling fails
        t = time.time()
        init_rec_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(init_rec_limit - 10)
        try:
            pickle.dumps(proof)
            res_stats["non_picklable_proof"] = (0, 1.0)
        except RecursionError:
            print(
                f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")} -- '
                f"RecursionError on {self.goal.name} while pickling the proof.",
            )
            proof = NonPicklableProof()
            res_stats["non_picklable_proof"] = (1, 1.0)
            self.mcts.logger.warning(f"pickle.dumps(proof) error on {self.goal.label}")
        sys.setrecursionlimit(init_rec_limit)
        self._timings["result__pickle"] = time.time() - t

        mcts_samples_tactic = self.mcts.train_samples_tactic

        # total proof time / result time
        self._timings["result__total"] = time.time() - start_t
        self._timings["total"] = time.time() - self.mcts.start_time
        res_stats["time"] = (time.time() - self.mcts.start_time, 1.0)

        # add times to stats
        for k, v in self._timings.items():
            res_stats[f"MCTS_H_timings__{k}"] = (v, 1.0)

        return MCTSResult(
            mcts_samples_critic=self.mcts.train_samples_critic,
            mcts_samples_tactic=mcts_samples_tactic,
            mcts_samples_effect=self.mcts.train_samples_effect,
            sample_proof=sample_proof,
            simplified_state=mcts_state,
            stats=res_stats,
            hist_stats=self.mcts.hist_stats,
            proof=proof,
            goal=self.mcts.goal,
            exception=None,
        )

    @wrap_timer(verbose=1)
    def dump(self, dump_self: bool = False, force_dump: bool = False) -> Path:
        start_t = time.time()
        path = self.dump_path / "mcts" / f"{self.goal.name}.pkl"
        print(
            f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")} -- '
            f"Entering MCTSHandler.dump ({self.goal.label}) -- dump_path: {path} -- "
            f"dump_self: {dump_self} -- force_dump: {force_dump} -- "
            f"n_nodes: {len(self.mcts.nodes)} -- "
            f"n_simutrees: {len(self.mcts.theorem_to_simutrees)}",
            file=sys.stderr,
            flush=True,
        )

        # data to dump
        if dump_self:
            mcts_data: Dict[str, Any] = {"self": self}
        else:
            mcts_data = {
                "mcts_root": self.mcts.root,
                "mcts_nodes": self.mcts.nodes,
                "timesteps": self.mcts.timestep,
                "permanent_ancestors": self.mcts.permanent_ancestors,
                "ancestors": self.mcts.ancestors,
                "history": self.mcts.history,
                "label": self.goal.label,
                "time": time.time() - self.mcts.start_time,
            }

        with open(path, "wb") as f:
            try:
                pickle.dump(mcts_data, f)
            except RecursionError:
                print(
                    f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")} -- '
                    f"RecursionError on {self.goal.name} while dumping MCTS",
                    file=sys.stderr,
                    flush=True,
                )
                if force_dump:
                    print("force_dump", file=sys.stderr, flush=True)
                    try:
                        mcts_data["recursion_error"] = 1
                        old_rc = sys.getrecursionlimit()
                        sys.setrecursionlimit(1_000_000)
                        pickle.dump(mcts_data, f)
                        sys.setrecursionlimit(old_rc)
                    except RecursionError:
                        print("RecursionError again", file=sys.stderr, flush=True)
        self._timings["dump"] += time.time() - start_t
        return path
