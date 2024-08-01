# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from logging import getLogger
from typing import Dict, List, Tuple, NamedTuple, Union

from numpy.random import RandomState
import numpy as np

from evariste.envs.mm.utils import count_unique_nodes, Node
from evariste.forward.fwd_mm.training.mm_training_helpers import (
    MMFwdTrainingDataset,
    sample_from_cumulative_mm,
    log_sizes,
)
from evariste.forward.fwd_mm.training.common import MMFwdTrainingProof
from evariste.model.checkpoints import get_latest_checkpoint

logger = getLogger()


class Schedule(NamedTuple):
    """
    If 'step', size=start_size until epoch reaches end_epoch, then max_size
    str: 'step,end_epoch10,start_size256'
    If 'linear', start with 'start_size' and
    reach the max_size in 'end_epoch' by linear increase
    str: 'linear,end_epoch10,start_size256'
    If 'exponential', starts with start_size and increase wth exponentially until reaching
    max_size in 'end_epoch'
    str: 'exponential,end_epoch10,start_size256'
    """

    end_epoch: int
    start_size: int
    type: str

    _types = {"linear", "step", "exponential"}

    @classmethod
    def from_str(cls, step_str: str) -> "Schedule":
        name, *others = step_str.split(",")
        assert name in cls._types, name
        assert len(others) == 2
        epoch, size = others
        assert epoch.startswith("end_epoch"), epoch
        assert size.startswith("start_size"), size
        return cls(
            start_size=int(size[len("start_size") :]),
            end_epoch=int(epoch[len("end_epoch") :]),
            type=name,
        )

    def get_size(self, epoch: int, max_size: int) -> int:
        start = self.start_size
        end_epoch = self.end_epoch
        if self.type == "step":
            if epoch < self.end_epoch:
                new_max_len = self.start_size
            else:
                new_max_len = max_size
        elif self.type == "linear":
            new_max_len = min(
                max_size, int(start + (max_size - start) * (epoch / self.end_epoch)),
            )
        elif self.type == "exponential":
            assert start <= max_size
            # we want start_size * 2 ** (gamma * end_epoch) = max_len
            gamma = math.log(max_size / start) / (math.log(2) * end_epoch)
            assert np.isclose(max_size, start * 2 ** (gamma * self.end_epoch))
            new_max_len = min(max_size, int(start * 2 ** (gamma * epoch)))
        else:
            raise NotImplementedError(self.type)
        return new_max_len


class CurriculumDataset(MMFwdTrainingDataset):
    """
    Curriculum dataset for train dataset.
    Some details:
     - we do a schedule on proof size (not directly on input size). Proof size is
     measured by possible output nodes (so e_hyps are not take into account).
     Maybe we should do a curriculum on input_size ?
     - right now schedule is hardcoded to max_size = epoch + 1 (
     so we will not see all proofs).
     - we discard all proofs that have a size > max_size, but we could subsample
      from them subproofs of size max_size to have more easy data.

    """

    def __init__(
        self,
        proof_trees: Dict[str, List[Tuple[str, Node]]],
        dump_path: str,  # hacky: to check for epoch by watching ckpts
        curriculum_str: str,  # 'default', 'step,epoch20,size10'
        refresh_every: int = 1_000,
    ):
        assert "train" in proof_trees
        sizes = [
            count_unique_nodes(proof_tree, ignore_e_hyps=True)
            for _, proof_tree in proof_trees["train"]
        ]
        self.max_size = max(sizes)
        self.refresh_every = refresh_every
        assert self.refresh_every > 0
        self.i = 0

        self.cur_epoch = -1
        self.dump_path = dump_path

        self.waiting_proofs: List[Tuple[Tuple[str, Node], int]] = []
        for proof, size in sorted(zip(proof_trees["train"], sizes), key=lambda x: x[1]):
            self.waiting_proofs.append((proof, size))
        assert len(self.waiting_proofs) == len(proof_trees["train"])

        self.schedule = self.parse_curriculum_str(curriculum_str)
        logger.info(f"Curriculum: {self.schedule}")

        self.cur_max_size = -1

        self.cur_proofs: List[Tuple[str, Node]] = []
        self.cur_sizes: List[float] = []
        self.cur_cumulative: np.array = []

    def refresh_data(self):
        """
        We look at the file system to parse the checkpoint to be
         able to know the epoch. This is due to the fact that this object is called
         on a background process of the DataLoader and don't have access to
         the trainer.epoch, trainer.n_iterations
        """
        _, latest_epoch = get_latest_checkpoint(self.dump_path, should_exist=True)
        cur_epoch = max(latest_epoch + 1, 0)
        changed = cur_epoch > self.cur_epoch
        self.cur_epoch = max(self.cur_epoch, cur_epoch)
        if changed and self.waiting_proofs:
            cur_max_size = self.schedule.get_size(self.cur_epoch, self.max_size)
            if cur_max_size == self.cur_max_size:
                return
            self.cur_max_size = cur_max_size
            last_idx = len(self.waiting_proofs)
            for idx, (_, size) in enumerate(self.waiting_proofs):
                if size > self.cur_max_size:
                    last_idx = idx
                    break
            self.cur_proofs.extend(
                (proof for proof, _ in self.waiting_proofs[:last_idx])
            )
            self.cur_sizes.extend((size for _, size in self.waiting_proofs[:last_idx]))
            self.cur_cumulative = np.cumsum(self.cur_sizes)

            assert len(self.cur_proofs) == len(self.cur_sizes)

            self.waiting_proofs = self.waiting_proofs[last_idx:]

            logger.info(
                f"Curriculum (epoch {self.cur_epoch}), "
                f"n_added proofs: {last_idx}, max_size {cur_max_size}, "
                f"n_waiting_proofs {len(self.waiting_proofs)}"
            )
            log_sizes(logger, sizes=self.cur_sizes, split="train")

    def sample_training_graph(self, rng: RandomState, split: str) -> MMFwdTrainingProof:
        assert split == "train"
        if self.i % self.refresh_every == 0:
            self.refresh_data()
        self.i += 1
        data = self.cur_proofs
        cumulative = self.cur_cumulative
        name, root = sample_from_cumulative_mm(cumulative, data, rng=rng)
        return MMFwdTrainingProof(name=name, generated=True, proved=None, root=root)

    @classmethod
    def parse_curriculum_str(cls, schedule_str: str) -> Schedule:
        return Schedule.from_str(schedule_str)


class MaxLenSchedule:
    """
    Class to create a schedule on max_len instead of filtering dataset
    on proof_size.
    """

    def __init__(
        self,
        max_len_schedule_str: str,
        max_len: int,
        dump_path: str,  # hacky: to check for epoch by watching ckpts,
        refresh_every: int = 1000,
    ):
        self.schedule: Schedule = self.parse_max_len_schedule_str(max_len_schedule_str)
        logger.info(f"MaxLenScheduler with schedule: {self.schedule}")
        self.refresh_every = refresh_every
        self.i = 0
        self.dump_path = dump_path
        self.max_len = max_len
        self.cur_max_len = -1

        self.cur_epoch = -2

    def get_max_len(self):
        """
        We look at the file system to parse the checkpoint to be
         able to know the epoch. This is due to the fact that this object is called
         on a background process of the DataLoader and don't have access to
         the trainer.epoch, trainer.n_iterations
        """
        if self.i % self.refresh_every == 0:
            _, latest_epoch = get_latest_checkpoint(self.dump_path, should_exist=True)
            epoch = max(0, latest_epoch + 1)
            if epoch != self.cur_epoch:
                self.cur_epoch = epoch
                new_max_len = self.schedule.get_size(epoch, max_size=self.max_len)
                if new_max_len != self.cur_max_len:
                    self.cur_max_len = new_max_len
                    logger.info(
                        f"MaxLenScheduler epoch {self.cur_epoch}, "
                        f"setting max_len to {self.cur_max_len}"
                    )

        self.i += 1
        assert self.cur_max_len > 0
        return self.cur_max_len

    @classmethod
    def parse_max_len_schedule_str(cls, max_len_schedule_str: str) -> Union[Schedule]:
        return Schedule.from_str(max_len_schedule_str)
