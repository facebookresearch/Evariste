# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field
from typing import Optional
import os

from params import Params


@dataclass
class MCTSSubProofArgs(Params):

    p_hyp: float = field(
        default=-1,
        metadata={
            "help": (
                "Probability to stop at a given node and "
                "add it as an hypothesis. -1 to disable"
            )
        },
    )
    n_sample_per_proofs: int = field(
        default=100,
        metadata={"help": "Number of subproofs to sample from each SimuTree."},
    )
    max_depth: int = field(
        default=10,
        metadata={
            "help": (
                "If set to N, uniform probability to stop at "
                "each depth between 1 and N. -1 to disable"
            )
        },
    )
    solved_as_hyp: bool = field(
        default=True,
        metadata={
            "help": "If False, nodes that were solved will not be added as hypotheses"
        },
    )
    weight_by_subgoals: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, tactics will be sampled according to the number of "
                f"subnodes they generate. If False, they will be sampled uniformly."
            )
        },
    )
    min_nodes: int = field(
        default=2, metadata={"help": "Minimum number of MCTS nodes. Must be >= 2"}
    )
    weight_samplers_alpha: float = field(
        default=0,
        metadata={
            "help": (
                "Weight proof samplers according to the number of MCTS nodes "
                "(n_nodes ** alpha, alpha=0 for uniform sampling)"
            )
        },
    )
    internal_nodes: bool = field(
        default=True,
        metadata={"help": "Sample goals from internal nodes of the proof."},
    )

    max_noise: int = field(
        default=10,
        metadata={"help": "Maximum number of useless hyps added to each theorems."},
    )
    dump_path: Optional[str] = field(
        default=None, metadata={"help": "MCTS dump path"},
    )
    max_files: int = field(
        default=-1,
        metadata={"help": "If > 0, maximum number of .pkl files loaded by worker"},
    )

    def __post_init__(self):
        if self.dump_path is not None:
            assert os.path.isdir(self.dump_path), self.dump_path
        if self.p_hyp == -1:
            assert self.max_depth >= 1
        else:
            assert self.max_depth == -1
            assert 0 <= self.p_hyp <= 1
        assert self.min_nodes >= 2
