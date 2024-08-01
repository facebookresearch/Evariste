# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from params import Params, ConfStore
from dataclasses import dataclass, field

from evariste.datasets import LeanDatasetConf
from evariste.forward.fwd_lean.training.lean_graph_sampler import LeanGraphConfig
from evariste.model.data.subproof_args import MCTSSubProofArgs


@dataclass
class LeanArgs(Params):

    dataset: LeanDatasetConf
    mcts_subproof: MCTSSubProofArgs
    graph: LeanGraphConfig = field(default_factory=lambda: LeanGraphConfig())
    additional_training_proofs: str = ""


ConfStore["default_lean"] = LeanArgs(
    dataset=ConfStore["lean_latest"], mcts_subproof=MCTSSubProofArgs()
)
