# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import itertools
import time
from typing import Optional, Union, List, Dict
from torch.utils.data import DataLoader
from collections import defaultdict
from logging import getLogger
import math
import torch
import itertools
import multiprocessing as mp

from evariste.datasets import DatasetConf
from evariste.utils import timeout, MyTimeoutError
from evariste.model.data.dictionary import Dictionary, SQUID_TOK
from evariste.model.data.envs.batch_iterator import BatchIterator
from evariste.model.data.envs.data_generator import DataGenerator
from evariste.model.data.envs.minproof_replay_buffer import MinProofMCTSReplayBuffer
from evariste.model.data.envs.mcts_loader import BaseMCTSDataLoader
from evariste.model.data.envs.mcts_loader_common import MCTSDataLoader
from evariste.trainer.args import TrainerArgs
from evariste.backward.prover.mcts_samples import (
    ONLINE_MCTS_SUBTASKS,
    MCTSSampleCritic,
    MCTSSampleTactics,
    MCTSSampleEffect,
)


logger = getLogger()


class DataEnvironment(DataGenerator):
    def __init__(
        self,
        dico: Dictionary,
        params: TrainerArgs,
        env_name: "str",
        tactic_cls,
        theorem_cls,
        dataset: DatasetConf,
    ):
        """
        Initialize environment.
        """
        self.params = params
        self.dico = dico
        self._cache: Dict[str, List] = {}
        self.env_name = env_name
        assert self.env_name in ["eq", "lean", "mm", "sr", "isabelle"], self.env_name
        self.tactic_cls = tactic_cls
        self.theorem_cls = theorem_cls

        self.dataset = dataset
        self._data = {}

        # MCTS data
        self.mcts: Dict[str, MCTSDataLoader] = {}
        self.mcts_stats: Dict[str, Dict[str, int]] = {}
        my_tasks = params.parsed_tasks(env_name)
        for task in my_tasks:
            if task.startswith(f"{env_name}_mcts_"):
                subtask = task.split("_")[2]
                assert subtask in ONLINE_MCTS_SUBTASKS, (
                    task,
                    subtask,
                    ONLINE_MCTS_SUBTASKS,
                )
                if subtask == "minproof":
                    self.mcts[subtask] = MinProofMCTSReplayBuffer.from_trainer_args(
                        params, mcts_task=task, dataset=dataset
                    )
                else:
                    self.mcts[subtask] = BaseMCTSDataLoader(
                        env_name=env_name,
                        mcts_subtask=subtask,
                        params=params,
                        tactic_cls=self.tactic_cls,
                        theorem_cls=self.theorem_cls,
                    )
                if subtask == "critic":
                    self.mcts_stats["mcts_critic_n_sampled_goal"] = defaultdict(int)
                if subtask in {"tactic", "minproof"}:
                    self.mcts_stats["mcts_tactic_n_sampled_goal"] = defaultdict(int)
                    self.mcts_stats["mcts_tactic_n_sampled_goal_tactic"] = defaultdict(
                        int
                    )

        self.first_mcts_sampled = True

    @property
    def data(self):
        return self._data

    def expand_mcts_critic_sample(
        self, sample: MCTSSampleCritic
    ) -> Optional[Dict[str, Union[float, List[str]]]]:
        eos = self.dico.eos_word
        # build x and y sequences
        x = [eos, *sample.goal.tokenize(), eos]
        x = [self.dico.index(tok) for tok in x]
        if len(x) > self.params.batch.max_len:
            return None
        return {"x": x, "q_estimate": sample.q_estimate}

    def expand_mcts_effect_sample(self, sample: MCTSSampleEffect):
        eos = self.dico.eos_word
        # build x and y sequences
        x = [eos, *sample.goal.tokenize(), SQUID_TOK, *sample.tactic.tokenize(), eos]
        assert sample.tactic.is_error() == (sample.children is None)
        if sample.tactic.is_error():
            y = [eos, *sample.tactic.tokenize_error(), eos]
        else:
            assert sample.children is not None
            y = [eos, *sum([c.tokenize() for c in sample.children], []), eos]
        if max(len(x), len(y)) > self.params.batch.max_len:
            return None
        x = [self.dico.index(tok) for tok in x]
        y = [self.dico.index(tok) for tok in y]
        return {"x": x, "y": y}

    def get_mcts_y_fmt(self, sample):
        raise NotImplementedError

    def expand_mcts_tactic_sample(self, sample: MCTSSampleTactics):
        # build x and y sequences

        y = self.get_mcts_y_fmt(sample)
        eos = self.dico.eos_word
        assert len(y) == len(sample.target_pi)

        x = [eos, *sample.goal.tokenize(), eos]
        x = [self.dico.index(tok) for tok in x]
        if len(x) > self.params.batch.max_len:
            return None

        y_filtered, target_pi, q_tactics = [], [], []
        for i, this_y in enumerate(y):
            if len(this_y) > self.params.batch.max_len:
                continue
            y_filtered.append(this_y)
            target_pi.append(sample.target_pi[i])
            if sample.q_tactics is not None:
                q_tactics.append(sample.q_tactics[i])
        if sample.q_tactics is None:
            q_tactics = None

        if len(y_filtered) == 0:
            return None

        target_pi = torch.FloatTensor(target_pi)
        target_pi /= target_pi.sum()

        return {
            "x": x,
            "y": y_filtered,
            "target_pi": target_pi,
            "q_tactics": q_tactics,
            "inproof": sample.inproof,
        }

    def get_mcts_sample(
        self, task: str, split: str, index: Optional[int]
    ) -> Optional[Dict]:
        if task == f"{self.env_name}_mcts_critic":
            pre_sample = self.mcts["critic"].get_mcts_sample(split, index, self.rng)
            if pre_sample is None:
                return None
            sample = self.expand_mcts_critic_sample(pre_sample)
            if sample is None:  # thrown away because of max_len
                return None
            self.mcts_stats["mcts_critic_n_sampled_goal"][pre_sample.goal.hash] += 1
            return {"x": sample["x"], "q_estimate": sample["q_estimate"]}
        elif task == f"{self.env_name}_mcts_effect":
            pre_sample = self.mcts["effect"].get_mcts_sample(split, index, self.rng)
            if pre_sample is None:
                return None
            return self.expand_mcts_effect_sample(pre_sample)
        elif task.startswith(f"{self.env_name}_mcts_tactic") or task.startswith(
            f"{self.env_name}_mcts_minproof"
        ):
            mcts_subtask = "tactic" if "tactic" in task else "minproof"
            pre_sample = self.mcts[mcts_subtask].get_mcts_sample(split, index, self.rng)
            if pre_sample is None:
                return None
            sample = self.expand_mcts_tactic_sample(pre_sample)
            if sample is None:  # thrown away because of max_len
                return None
            # if we do q_conditioning, we sample tactics uniformly,
            # otherwise following policy distribution
            if self.params.mcts_train.q_conditioning:
                tactic_id = torch.randint(len(sample["y"]), (1,)).item()
            else:
                tactic_id = torch.multinomial(sample["target_pi"], 1).item()
            self.mcts_stats["mcts_tactic_n_sampled_goal"][pre_sample.goal.hash] += 1
            self.mcts_stats["mcts_tactic_n_sampled_goal_tactic"][
                (pre_sample.goal.hash, pre_sample.tactics[tactic_id].hash)
            ] += 1
            if self.first_mcts_sampled:
                str_x = " ".join(self.dico.id2word[x] for x in sample["x"])
                str_y = " ".join(self.dico.id2word[x] for x in sample["y"][tactic_id])
                logger.info(f"MCTS SAMPLE x: {str_x}")
                logger.info(f"MCTS SAMPLE y: {str_y}")
                self.first_mcts_sampled = False
            returned_sample = {
                "x": sample["x"],
                "y": sample["y"][tactic_id],
                "inproof": sample["inproof"],
            }
            assert len(returned_sample["x"]) <= self.params.batch.max_len
            assert len(returned_sample["y"]) <= self.params.batch.max_len
            if sample["q_tactics"] is not None:
                returned_sample["q_tactics"] = sample["q_tactics"][tactic_id]
            return returned_sample
        else:
            raise RuntimeError(task)

    def quadratic_size(self, batch: List[Dict]):
        if len(batch) == 0:
            return 0
        max_x = max(len(x["x"]) for x in batch)
        max_y = 0
        if "y" in batch[0]:
            max_y = max(len(x["y"]) for x in batch)
        # if only quadratic cost, I have OOM on short sequences
        if self.params.batch.tokens > -1:
            n_toks = len(batch) * (max_x + max_y)
            if n_toks > self.params.batch.tokens:
                return math.inf
        return len(batch) * (max_x ** 2 + max_y ** 2)

    def create_data_loader(self, split: str, task: str):
        """
        Create a data loader for this environment.
        """
        assert split in ["train", "valid", "test"]
        logger.info(f"Creating {split} iterator for {task} ...")
        batch_iterator = BatchIterator(
            self, split, task, params=self.params, pad_index=self.dico.pad_index
        )

        # number of workers -- tasks that receive data from
        # other processes should only have 1 worker
        if "mcts" in task or "pact_seq2seq" in task or split != "train":
            num_workers = 1
        else:
            num_workers = self.params.num_workers
        logger.info(f"Setting num_workers to {num_workers} for task {task}")

        data_loader = DataLoader(
            batch_iterator, batch_size=None, num_workers=num_workers, shuffle=False,
        )
        return data_loader

    def create_data_iterator(
        self,
        split: str,
        task: str,
        try_fetch: bool = False,
        max_retry: int = 10,
        seconds: int = 300,
    ):
        if not try_fetch:
            return iter(self.create_data_loader(split, task))

        assert max_retry > 0
        assert seconds > 0

        @timeout(seconds=seconds)
        def get_next(iterator):
            start = time.time()
            res = next(iterator)
            logger.info(
                f"Fetching {task} ({split}) batch took {time.time() - start} sec."
            )
            return res

        n_retry = 0
        while True:
            try:
                logger.info(
                    f"Creating {task} ({split}) dataloader "
                    f"({n_retry + 1}/{max_retry}) ..."
                )
                batch_iterator = iter(self.create_data_loader(split, task))
                logger.info(f"Fetching {task} ({split}) batch ...")
                batch = get_next(batch_iterator)
                break
            except MyTimeoutError:
                logger.warning(
                    f"Timeout fetching {task} ({split}) batch ({n_retry + 1}/{max_retry}): "
                    f"recreating dataset and try again..."
                )
                if self.params.num_workers > 0:
                    import shutil

                    tmpdir = mp.util.get_temp_dir()
                    _, _, free = shutil.disk_usage(tmpdir)
                    logger.info(f"TMPDIR: {tmpdir} // disk_usage, free {free// 2**30}")
                n_retry += 1
                if n_retry == max_retry:
                    raise
        return itertools.chain([batch], batch_iterator)

    def store_mcts_stats(self) -> Dict[str, float]:
        return {
            k: sum([count for count in counts.values()]) / max(len(counts), 1)
            for k, counts in self.mcts_stats.items()
        }

    def get_stats(self) -> Dict[str, float]:
        stats = {}
        # MCTS stats
        for k, v in self.mcts.items():
            stats.update({f"mcts_{k}_{x}": y for x, y in v.get_stats().items()})
        stats.update(self.store_mcts_stats())
        return stats

    def close(self):
        logger.info(f"Closing DataEnvironment ({self.env_name}) ...")
        for subtask, mcts_data_loader in self.mcts.items():
            if (
                mcts_data_loader is not None
                and isinstance(mcts_data_loader, BaseMCTSDataLoader)
                and mcts_data_loader.replay_buffer is not None
            ):
                logger.info(f"Calling close on BaseMCTSDataLoader (subtask={subtask})")
                mcts_data_loader.replay_buffer.close()
            else:
                assert isinstance(mcts_data_loader, MinProofMCTSReplayBuffer)
                logger.info(
                    f"Calling close on MinProofMCTSReplayBuffer (subtask={subtask})"
                )
                mcts_data_loader.close()
        logger.info(f"Closed DataEnvironment ({self.env_name})")
