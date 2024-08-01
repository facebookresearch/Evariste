# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field
from enum import unique, Enum
from typing import Optional

from params import Params, ConfStore


# depth is the length of the longest branch in the proof.
# size is the total number of nodes in the proof.
STYPES = ["depth", "size", "time"]


@unique
class CycleHandling(str, Enum):
    # Exponential = "exponential"
    # Linear = "linear"
    Kill = "kill"  # old version. 50% perf on eval_async, 52.6
    Resuscitate = "resuscitate"


@dataclass
class MCTSMoveParams(Params):

    enable: bool = False

    # number of trials on a node before we select its most promising tactic
    budget: int = 100

    def __post_init__(self):
        assert self.budget > 0


@dataclass()
class ExpanderParams(Params):
    sorting_strategy: str = "chunk_slen"
    tokens_per_batch: int = 5000
    min_tpb: int = 3000
    max_tpb: int = 30000
    resize_with_oom: bool = True
    chunk_size: int = 1024
    allow_mix: bool = True
    max_batches_in_queue: int = 2
    max_input_len: int = field(
        default=1024, metadata={"help": "Max encoder input length"}
    )

    def __post_init__(self):
        assert self.sorting_strategy in ["none", "pid_slen", "chunk_slen"]
        assert self.allow_mix or self.sorting_strategy != "none"
        assert self.min_tpb <= self.tokens_per_batch <= self.max_tpb
        assert self.max_input_len <= self.min_tpb
        assert self.max_batches_in_queue >= 2


@dataclass
class MCTSParams(Params):

    # max_depth: int = 30  # this shouldn't matter.
    exploration: float = 5
    policy: str = "other"
    n_expansions: int = 1000
    expander: ExpanderParams = field(default_factory=lambda: ExpanderParams())
    succ_expansions: int = 50
    max_in_flight: int = 128  # maximum number of theorems being processed by gpu + cpu
    early_stop: bool = False
    no_critic: bool = False

    max_nodes_in_expansion: Optional[int] = None

    cycle_handling: CycleHandling = CycleHandling.Kill  # old behavior

    # filter nodes with a minimum number of visits. for online training only,
    # see MCTSTrainArgs.count_threshold_offline for offline training
    count_threshold: int = 50
    count_tactic_threshold: int = 0

    # maximum number of nodes to send for MCTS training
    max_nodes_for_train: int = -1

    # make sure the ratio of nodes solved vs unsolved we sent to critic trainer is balanced
    balance_critic_data: bool = False

    # if False we send to training all solved nodes, even if ncounts < count_threshold
    use_count_threshold_for_solved: bool = False

    # filter out tactics with a too small estimated policy target
    tactic_p_threshold: float = 0.05

    # set to False for old behaviour of using critic instead of 1 if internal node
    backup_one_for_solved: bool = True

    # depth penalty, as a discount factor, in ]0, 1] (no effect with 1)
    depth_penalty: float = 1

    # if we train our model with q conditioning, the tactic filtering done to extract train sample, is not the same
    train_sample_for_q_conditioning: bool = False

    # initial tactic scores
    init_tactic_scores: float = 0.5

    # use mu_fpu rather than constant fpu
    mu_fpu: bool = False
    # use mu_fpu at and node. if False only use mu_fpu at Or nodes.
    mu_fpu_at_and: bool = True

    # 7 options [0:6] for q value used in policy for solved nodes
    q_value_solved: int = 5

    # if root is not solved, do not try to explore paths that are already solved
    early_stop_on_solved_if_root_unproved: bool = False

    # Keep a set of hashed, backed up simutrees to avoid backuping more than once the same simutree
    backup_once: bool = True

    # save history updates (can significantly increase memory usage)
    save_history_updates: bool = False

    only_learn_tactics_from: str = ""
    only_learn_best_tactics: bool = False
    proof_size_alpha: float = 1
    proof_stype: str = "depth"

    # sample the tactic to follow in the policy. if 0, take the argmax
    policy_temperature: float = 0
    policy_temp_level: str = "global"

    move: MCTSMoveParams = field(default_factory=lambda: MCTSMoveParams())

    effect_subsampling: float = 1
    critic_subsampling: float = 1

    dump_tactic_sample_pairs: bool = False

    def __post_init__(self):
        assert self.policy_temperature >= 0
        assert self.policy_temp_level in ["global", "simutree", "node"]
        # assert self.policy_temp_level == "global" or self.policy_temperature > 0
        assert 0 < self.depth_penalty <= 1
        if self.depth_penalty < 1 and self.backup_one_for_solved:
            print(
                "WARNING: depth_penalty < 1 and backup_one_for_solved = True, "
                "depth will not be properly backed-up for solved nodes"
            )
        assert self.policy in ["alpha_zero", "other"]
        assert 0 <= self.proof_size_alpha <= 1
        assert self.proof_stype in STYPES
        assert self.init_tactic_scores > 0

        if self.policy == "other":
            assert self.mu_fpu is False

    def _check_and_mutate_args(self):
        assert self.only_learn_tactics_from in [
            "",
            "solving",
            "proof",
            "minproof",
            "minproof-solving",
        ]
        if (
            self.only_learn_tactics_from == "minproof"
            and not self.only_learn_best_tactics
        ):
            print(
                "WARNING: when we learn tactics from minproof by default we always take the best tactics."
            )
        assert not self.only_learn_best_tactics or self.only_learn_tactics_from != ""


ConfStore["mcts_very_fast"] = MCTSParams(n_expansions=1, early_stop=True)
ConfStore["mcts_fast"] = MCTSParams(n_expansions=1000, early_stop=True)
ConfStore["mcts_slow"] = MCTSParams(n_expansions=10000, early_stop=True)
