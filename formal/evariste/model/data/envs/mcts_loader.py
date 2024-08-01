# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Type, Union, Dict
from abc import abstractmethod
from logging import getLogger
from pathlib import Path
import os
import numpy as np

from evariste import json as json
from evariste.comms.zmq import ZMQReceiver
from evariste.backward.graph import Theorem, Tactic
from evariste.backward.prover.mcts_samples import (
    MCTSSampleCritic,
    MCTSSampleEffect,
    MCTSSampleTactics,
    ONLINE_MCTS_SUBTASKS,
    ALL_MCTS_SUBTASKS,
)
from evariste.model.data.envs.mcts_loader_common import MCTSDataLoader
from evariste.model.data.envs.minproof_replay_buffer import (
    MinProofMCTSReplayBuffer,
    MCTSSampleProof,
)
from evariste.model.data.envs.replay_buffer_loader import ReplayBuffer, ReplayBufferArgs
from evariste.model.data.mcts_subproof import ProofStepSample
from evariste.trainer.args import TrainerArgs


logger = getLogger()


def get_mcts_comms(path: Path) -> Path:
    # used to be path/mcts/
    os.makedirs(path, exist_ok=True)
    return path


def mcts_rb_factory(
    args: TrainerArgs, mcts_subtask: str
) -> Union[
    ReplayBuffer[MCTSSampleCritic],
    ReplayBuffer[MCTSSampleTactics],
    ReplayBuffer[MCTSSampleEffect],
    MinProofMCTSReplayBuffer,
]:
    receiver = mcts_subtask_receiver_factory(args, mcts_subtask)
    rb_params = ReplayBufferArgs(
        min_len=1 if args.debug.debug else args.mcts_train.replay_buffer.min_len,
        max_len=args.mcts_train.replay_buffer.max_len,
    )
    logger.info(f"Creating replay buffer for {mcts_subtask} with args: {rb_params}")
    return ReplayBuffer(receiver=receiver, rb_params=rb_params)


def mcts_subtask_receiver_factory(
    args: TrainerArgs, mcts_subtask: str
) -> Union[
    ZMQReceiver[MCTSSampleCritic],
    ZMQReceiver[MCTSSampleTactics],
    ZMQReceiver[MCTSSampleEffect],
    ZMQReceiver[MCTSSampleProof],
]:
    MCTSSample = {
        "critic": MCTSSampleCritic,
        "tactic": MCTSSampleTactics,
        "effect": MCTSSampleEffect,
        "minproof": MCTSSampleProof,
    }[mcts_subtask]
    receiver = ZMQReceiver[MCTSSample](
        dump_path=get_mcts_comms(Path(args.dump_path)),
        global_rank=args.slurm_conf.global_rank,
        name=f"mcts_{mcts_subtask}_sample_store",
        heartbeat_freq=10.0,
    )
    return receiver


def subproof_rb_factory(args: TrainerArgs) -> ReplayBuffer[ProofStepSample]:
    """
    Create a replay buffer to store the hypertrees from the MCTS
    @param args:
    @return:
    """
    receiver = ZMQReceiver[ProofStepSample](
        dump_path=get_mcts_comms(Path(args.dump_path)),
        global_rank=args.slurm_conf.global_rank,
        name=f"mcts_subproof_sample_store",
        heartbeat_freq=10.0,
    )
    rb_params = ReplayBufferArgs(min_len=1 if args.debug else 1_000, max_len=50_000)
    logger.info(f"Creating subproof replay buffer with args: {rb_params}")
    return ReplayBuffer(receiver=receiver, rb_params=rb_params)


class ZMQDataLoader(MCTSDataLoader):
    def __init__(self, params: TrainerArgs, mcts_subtask: str):
        self.params: TrainerArgs = params
        self.replay_buffer: Optional[ReplayBuffer] = None
        self.first_sample = True
        self.n_gen_proofs = 0
        self.should_dump = False
        self.mcts_subtask = mcts_subtask
        assert mcts_subtask in ALL_MCTS_SUBTASKS, mcts_subtask

    def mcts_fmt(self) -> Optional[str]:
        return None

    def get_mcts_sample(
        self, split: str, index: Optional[int], rng: np.random.RandomState,
    ):
        if self.first_sample:
            if not self.params.online_bwd_gen:
                pass
            self.replay_buffer.fill_store()  # initialize ZMQ comms
            self.first_sample = False
            self.load_mcts_data()
            self.maybe_reload()
            self.should_dump = True
        self.n_gen_proofs += 1
        return self.replay_buffer.get_sample(split, index, rng, block=True)[0]

    @abstractmethod
    def parse_mcts_sample(self, json):
        raise NotImplementedError

    def maybe_reload(self) -> None:
        """If a dataset_mctssubtask.json exists, we reload lines % rank"""
        fpath = os.path.join(
            self.params.dump_path, f"dataset_{self.mcts_subtask}.jsonl"
        )
        if not os.path.exists(fpath):
            return
        total, selected, discarded, added = 0, 0, 0, 0
        rank = self.params.slurm_conf.global_rank
        world_size = self.params.slurm_conf.world_size
        logger.info(f"Reloading JSON data from {fpath} ({rank}/{world_size}) ...")
        to_add = []
        with open(fpath, "r") as f:
            for i, line in enumerate(f):
                total += 1
                if i % world_size != rank:
                    continue
                if (i + 1) % 1_000_000 == 0:
                    logger.info(f"Reloaded {i + 1} lines from {fpath}")
                selected += 1
                sample = self.parse_mcts_sample(json.loads(line))
                if sample is None:
                    discarded += 1
                    continue
                added += 1
                to_add.append(sample)
        logger.info(
            f"maybe_reload ({rank}/{world_size}) -- "
            f"Found {total} lines in {fpath} -- "
            f"Selected {selected} -- "
            f"Discarded {discarded} -- "
            f"Added {added}"
        )
        self.replay_buffer.fill_store(to_add)

    def load_mcts_data(self):
        if self.params.mcts_train.jsonl_data_dir == "":
            return
        path = Path(self.params.mcts_train.jsonl_data_dir)
        to_add = []
        local_rank = self.params.slurm_conf.local_rank
        fpath = path / f"dataset_{self.mcts_subtask}.jsonl"
        chunk = path / f"dataset_{self.mcts_subtask}.jsonl.{local_rank}"
        if chunk.is_file():
            logger.info(f"Chunk {chunk} is available ...")
            fpath = chunk
        else:
            logger.info(f"Chunk {chunk} is not available ...")
        logger.info(f"Loading train {self.mcts_subtask} MCTS samples from {fpath} ...")
        with open(fpath) as f:
            n_lines = 0
            for i, line in enumerate(f):
                n_lines += 1
                if i % 100000 == 0:
                    logger.debug(f"Loaded {i} lines (kept {len(to_add)} samples) ...")
                sample = json.loads(line.rstrip())
                sample = self.parse_mcts_sample(sample)
                if sample is None:
                    continue
                to_add.append(sample)
                if self.params.debug.train and len(to_add) >= 500:
                    break
            logger.info(
                f"Loaded {len(to_add)} train {self.mcts_subtask} "
                f"MCTS samples from {n_lines} lines."
            )
        self.replay_buffer.fill_store(to_add)

    def get_stats(self) -> Dict[str, float]:
        if self.should_dump and self.replay_buffer is not None:
            return self.replay_buffer.store_stats_and_reset()
        return {}


class BaseMCTSDataLoader(ZMQDataLoader):
    def __init__(
        self,
        env_name: str,
        mcts_subtask: str,
        params: TrainerArgs,
        tactic_cls: Type[Tactic],
        theorem_cls: Type[Theorem],
    ):
        assert mcts_subtask in ONLINE_MCTS_SUBTASKS, mcts_subtask
        super().__init__(params, mcts_subtask=mcts_subtask)
        mcts_params = params.mcts_train
        mcts_tasks = params.parsed_tasks(f"{env_name}_mcts_{self.mcts_subtask}")
        if len(mcts_tasks) == 0:
            return
        assert len(mcts_tasks) == 1, "MCTS Data Loader only supports one MCTS task"
        self.mcts_task = mcts_tasks[0]
        if self.mcts_subtask == "tactic":
            s = self.mcts_task.split("_")  # lang, mcts, tactic, str_fmt
            assert len(s) == 4, self.mcts_task
            self._mcts_fmt = s[3]
        else:
            self._mcts_fmt = None

        logger.info(f"{env_name} TASKS {params.parsed_tasks(env_name)}")

        self.tactic_cls = tactic_cls
        self.theorem_cls = theorem_cls
        self.env_name = env_name
        self.mcts_params = mcts_params
        self.replay_buffer: ReplayBuffer = mcts_rb_factory(
            self.params, mcts_subtask=self.mcts_subtask
        )

    @property
    def mcts_fmt(self) -> Optional[str]:
        return self._mcts_fmt

    def parse_mcts_sample(
        self, sample
    ) -> Optional[Union[MCTSSampleCritic, MCTSSampleTactics]]:
        """
        Parse a JSON MCTS sample. Used for debug only.
        """
        if self.mcts_params.count_threshold_offline > 0:
            if self.mcts_params.count_threshold_offline > sample["visit_count"]:
                return None
        if self.mcts_subtask == "critic":
            assert "q_estimate" in sample
            MCTSSample = MCTSSampleCritic
        elif self.mcts_subtask == "tactic":
            assert len(sample["tactics"]) == len(sample["target_pi"])
            if sample["q_tactics"] is not None:
                assert len(sample["tactics"]) == len(sample["q_tactics"])
            if len(sample["tactics"]) == 0:
                return None
            MCTSSample = MCTSSampleTactics
        elif self.mcts_subtask == "effect":
            MCTSSample = MCTSSampleEffect
        else:
            raise NotImplementedError

        return MCTSSample.from_json(sample, self.theorem_cls, self.tactic_cls)


class MCTSSubProofDataLoader(ZMQDataLoader):
    """
    This class manage the generation of subproof from the hypertree it retrieves.
    The proofs should be actually built in a subclass of this one, specialized
    for each environnement.
    """

    def __init__(
        self, params: TrainerArgs, tactic_cls: Type[Tactic], theorem_cls: Type[Theorem]
    ):

        super().__init__(params, mcts_subtask="subproof")
        self.params = params
        self.tactic_cls = tactic_cls
        self.theorem_cls = theorem_cls
        self.replay_buffer: ReplayBuffer = subproof_rb_factory(self.params)

    def parse_mcts_sample(self, sample) -> Optional[ProofStepSample]:
        """
        Parse a JSON MCTS SimplifiedMCTSState.
        """
        return ProofStepSample.from_json(sample, self.theorem_cls, self.tactic_cls)
