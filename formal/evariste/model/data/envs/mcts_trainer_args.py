# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field
import os

from params import Params
from evariste.model.data.envs.minproof_replay_buffer import MinProofReplayBufferCfg
from evariste.model.data.envs.replay_buffer_loader import ReplayBufferArgs


@dataclass
class MCTSTrainArgs(Params):
    replay_buffer: ReplayBufferArgs
    minproof_rb: MinProofReplayBufferCfg = field(
        default_factory=lambda: MinProofReplayBufferCfg()
    )
    jsonl_data_dir: str = field(
        default="",
        metadata={
            "help": "MCTS .jsonl files location. If empty, use provers' on-the-fly generated data."
        },
    )
    count_threshold_offline: int = 0  # used for debug only

    # filter the tactics to use for the MCTS tactic task
    only_learn_tactics_from: str = ""

    # # The below should be used in combination with zmq_prover.prover.start_with_token = <PROVED_WORD>
    # decode_critic_then_tactic: bool = False

    # if True, do not train the critic. `no_critic`
    train_critic: bool = True

    # train the critic with 0 / 1, and not with the estimates (similarly to OpenAI)
    hard_critic_estimates: bool = False

    q_conditioning: str = field(
        default="",
        metadata={
            "help": "When training, condition tactic generation on Q value. At inference time, decode with high value of Q. Choices of mode, either sum or prefix."
        },
    )

    def __post_init__(self):
        if self.hard_critic_estimates:
            assert self.train_critic
        assert self.q_conditioning in ["", "sum", "prefix"]
        assert self.only_learn_tactics_from in [
            "",
            "solving",
            "proof",
            "minproof",
            "minproof-solving",
        ]
        if self.jsonl_data_dir != "":
            assert os.path.isdir(self.jsonl_data_dir), self.jsonl_data_dir
            # proof_size only works for online MCTS training
            assert self.only_learn_tactics_from != "minproof-solving"
