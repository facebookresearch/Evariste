# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Dict
import functools

from params import Params, ConfStore
from evariste.forward.fwd_mm.training.curriculum import (
    CurriculumDataset,
    MaxLenSchedule,
)
from evariste.forward.online_generation.goal_selectors.goal_selector import (
    GoalSelectionConfig,
)

# should all of this be moved here actually ?
from evariste.datasets.metamath import MetamathDatasetConf
from evariste.model.data.subproof_args import MCTSSubProofArgs


@unique
class WeightsStrategies(str, Enum):
    # TODO: Safely delete this class (see refac)
    SIZE = "size"
    DEPTH = "depth"
    MIN_LP = "min_log_probs"
    MEAN_LP = "mean_log_probs"
    MIN_NORM_LP = "min_norm_log_probs"
    MEAN_NORM_LP = "mean_norm_log_probs"
    SUM_NORM_LP = "sum_norm_log_probs"


@dataclass
class MetamathGraphArgs(Params):
    # If True, sample goal instead of choosing root node
    sample_goal: bool = False
    # In forward, at each step, sample a noisy node with proba mm_graph_insert_noise_prob
    insert_noise_prob: float = 0
    # Randomly drop nodes from input graphs
    dropout: float = 0
    # To test slightly different versions of random topological sort to force the
    # generations to be less flat
    topo_sort_version: int = 0
    # If True, in forward, drop not useful nodes if max_len is reached
    drop_if_too_long: bool = False

    # path toward generated data pkl, can be sharded data
    generated_proof_path: str = ""

    # probablity to sample a generated sample while sampling graphs
    # generated_proof_path should point to generated proofs pkl in this case
    generated_prob: float = 0.0

    # while loading proof trees, remap labels if True
    remap_label: bool = False

    # curriculum on proof size
    curriculum_str: str = ""
    # curriculum on max_len
    max_len_schedule_str: str = ""
    # test: HACK, no curriculum learning for generated data, since the model
    # was already able to generate it by itself.
    no_curriculum_for_generated_data: bool = False

    # condition the generation with a "label" token from the proof to favor diversity
    # this token is placed in the encoder
    label_conditioning: bool = False

    # condition the generation with a "proved"/"unproved" token
    # this token is placed in the encoder
    proved_conditioning: bool = False

    # reward quantile conditioning: we condition the generator with a
    # token that representing the reward quantile of the generation
    # this token is placed in the encoder
    reward_quantile_conditioning: bool = False

    # in adversarial setup wait for generated data to start
    wait_generated: bool = False

    # heuristic to use as generation reward
    # "size" / "size_by_depth"
    generation_reward_str: str = "size:1"

    # TODO: move this in Metamath Args
    axioms_only: bool = field(
        default=False,
        metadata={"help": "Do not include theorems as valid tatics, use axioms only"},
    )

    @functools.lru_cache
    def generation_reward(self) -> Dict[str, float]:

        from evariste.forward.fwd_mm.mm_generation_annotator import MM_HEURISTICS

        s = [x.split(":") for x in self.generation_reward_str.split(",")]
        assert all(len(x) == 2 for x in s), s
        assert len(s) == len(set([x[0] for x in s])) > 0, s
        assert all(x in MM_HEURISTICS and float(y) != 0 for x, y in s), s
        return {x: float(y) for x, y in s}

    def _check_and_mutate_args(self):
        if self.generated_proof_path != "":
            assert self.generated_prob > 0
        if self.curriculum_str:
            CurriculumDataset.parse_curriculum_str(self.curriculum_str)
        if self.max_len_schedule_str:
            MaxLenSchedule.parse_max_len_schedule_str(self.max_len_schedule_str)
        assert not (self.curriculum_str and self.max_len_schedule_str)


@dataclass()
class MetamathNegArgs(Params):
    sample_size: int = field(
        default=128,
        metadata={
            "help": "Total size of the softmax when we train with negative sampling"
        },
    )
    worst_offenders: float = field(
        default=0,
        metadata={"help": "Fraction of worst offenders in negative sampling."},
    )
    update_frequency: int = field(
        default=-1,
        metadata={"help": "Update statement embeddings periodically (-1 to disable)"},
    )
    softmax_all: bool = field(
        default=False,
        metadata={"help": "Perform the softmax on all theorem embeddings"},
    )


@dataclass
class MetamathArgs(Params):
    dataset: MetamathDatasetConf
    graph: MetamathGraphArgs
    neg: MetamathNegArgs
    mcts_subproof: MCTSSubProofArgs
    critic_no_syntactic: bool = field(
        default=True,
        metadata={
            "help": "Do not train the critic on nodes that can be handled by the parser"
        },
    )
    additional_proofs_path: str = field(
        default="", metadata={"help": "Additional proofs (generated by forward) path"},
    )
    critic_max_neg: int = field(
        default=100,
        metadata={"help": "Maximum number of negative examples to train the critic"},
    )
    # help: use stop action for generation
    stop_action: bool = False

    # help: when using a generation task, this will condition the generation decoding
    # to the token 'proved' or 'unproved' by putting it in the encoder
    # WARNING this is deprecated and kept to be able to evaluate old models trained with
    # this parameter. Please use mm.graph.proved_conditioning (that will put
    # a token "proved"/"unproved" in the encoder (and not the decoder).
    cond_gen_with_proved: bool = False

    def _check_and_mutate_args(self):
        if self.cond_gen_with_proved:
            raise ValueError(
                "Please use mm.graph.proved_conditioning instead"
                "this argument is not supported anymore for training"
                "(kept for the moment for bwd compatibility"
                "to evaluate old models)"
            )


ConfStore["default_mm"] = MetamathArgs(
    dataset=ConfStore["new3"],
    graph=MetamathGraphArgs(
        sample_goal=True, remap_label=True, insert_noise_prob=0.2, topo_sort_version=0,
    ),
    neg=MetamathNegArgs(),
    mcts_subproof=MCTSSubProofArgs(),
)


# TODO: move this stuff annd update "refac" for bwd comp.
@dataclass
class OnlineGenerationArgs(Params):
    refresh_every: int = 100_000
    n_max_proofs: int = 1_000_000
    splay_wait_time: int = 0
    goal_selection: GoalSelectionConfig = field(
        default_factory=lambda: GoalSelectionConfig()
    )
    n_max_samples: int = 200_000  # used when doing error conditioning
    prover_temperature: float = 1.0
    prover_type: str = "sampling"

    zip_chunk_size: int = 2048


ConfStore["default_online_gen_cfg"] = OnlineGenerationArgs()
