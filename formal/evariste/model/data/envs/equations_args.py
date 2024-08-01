# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field

from params import Params, ConfStore
from evariste.datasets import EquationsDatasetConf
from evariste.model.data.subproof_args import MCTSSubProofArgs


@dataclass
class EquationArgs(Params):
    dataset: EquationsDatasetConf
    mcts_subproof: MCTSSubProofArgs = field(default_factory=lambda: MCTSSubProofArgs())
    stop_action: bool = False  # Used in fwd mode if the model has to decide when to emit a special STOP tactic
    proved_conditioning: bool = False  # for eq_gen_graph_offline_seq2seq, start decoding with PROVED or UNPROVED word


ConfStore["default_eq"] = EquationArgs(dataset=ConfStore["eq_dataset_exp_trigo_hyper"])
