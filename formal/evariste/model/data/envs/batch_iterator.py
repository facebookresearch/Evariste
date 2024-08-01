# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from torch.utils.data.dataset import IterableDataset
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import math
import time
import numpy as np
import torch

from evariste.comms.zmq import ZMQNotReady, ZMQNotReadySample
from evariste import json
from evariste.logger import create_logger
from evariste.utils import COND_TOK, print_memory
from evariste.trainer.args import TrainerArgs
from evariste.metrics import Logger, ActionCounter, StatsCollection, Timer
from evariste.model.data.envs.data_generator import DataGenerator


SKIP_ITEM = "SKIP_ITEM"

logger = create_logger(None)


def key_fn_len(sample):
    if "y" in sample:
        return len(sample["x"]), len(sample["y"])
    else:
        return len(sample["x"])


def size_with_pad(batch: List[Dict]) -> int:
    if len(batch) == 0:
        return 0
    size = len(batch) * max(len(x["x"]) for x in batch)
    if "y" in batch[0]:
        size += len(batch) * max(len(x["y"]) for x in batch)
    return size


@dataclass
class WrapperTimers(StatsCollection):
    next_batch: Timer = field(default_factory=lambda: Timer())
    fill_queue: Timer = field(default_factory=lambda: Timer())
    update_stats: Timer = field(default_factory=lambda: Timer())
    sort: Timer = field(default_factory=lambda: Timer())
    generate_sample: Timer = field(default_factory=lambda: Timer())
    generate_none_sample: Timer = field(default_factory=lambda: Timer())


class BatchIterator(IterableDataset):
    """
    This class allows to create a pytorch `IterableDataset` from an env which is a
    `DataGenerator` (an object that has mainly  a `get_sample(self, task, split, index)` method,
    like a classic data environment. Samples return by this
    data_generator are dict, and needs to have at least a "x" key ("y" key Optional).

    This class allows to reduce create queues with the created samples and batch them by reducing the padding.


    If the env raise a ZMQNotReady (for the first sample for instance), the error is catched and a ZMQNotReadySample
    is yield instead of a batch.



    :param env: DataGenerator
    :param split: split for the dataset
    :param task: task of the dataset
    :param params: TrainerArgs. The collate queue type is defined by `params.batch.queue_strategy`
    :param pad_index: index of the pad token
    """

    def __init__(
        self,
        env: DataGenerator,
        split: str,
        task: str,
        params: TrainerArgs,
        pad_index: int,
    ):

        super().__init__()

        # environment
        self.env = env
        self.task = task
        self.params = params
        self.pad_index = pad_index
        self.global_rank = params.slurm_conf.global_rank
        self.env_base_seed = params.env_base_seed
        assert not hasattr(self.env, "rng"), "environment already has a rng"

        # data type
        assert split in ["train", "valid", "test"]
        self.split = split
        self.train = split == "train"
        self.index = 0

        # batching
        self.num_workers = params.num_workers
        self.batch_size = params.batch.size
        self.tokens_per_batch = params.batch.tokens
        assert self.tokens_per_batch > 0
        self.collate_queue_update_freq = params.batch.collate_queue_update_freq

        self.collate_queue: Optional[List] = [] if self.train else None
        self.deque = None
        self.n_batches = 0

        self.metrics: Optional[Logger] = None
        self.last_log = time.time()

        self.stats: Dict[str, ActionCounter] = defaultdict(
            lambda: ActionCounter("unused", is_rate=False, silent=True)
        )

        self.timers = WrapperTimers()

        # caching strategy
        self.queue_strategy = params.batch.queue_strategy
        if self.train:
            assert self.queue_strategy in [
                "uniform_sampling",
                "uniform_sampling_replacement",
            ]
            if self.queue_strategy == "uniform_sampling_replacement":
                self.batch_sizes = deque([64 for _ in range(100)], maxlen=100)

    def __iter__(self):
        try:
            self.env.maybe_load(self.task, self.split)
            self.init_rng()
            if self.train:
                yield from self._iter_train()
            else:
                yield from self._iter_valid_test()
        except Exception:
            if self.metrics is not None:
                self.metrics.close()
            raise

    def _iter_valid_test(self):
        assert not self.train
        assert self.env.data is not None
        size = len(self.env.data[self.task.replace(COND_TOK, "")][self.split])

        while True:
            before = self.index
            after = min(self.index + self.batch_size, size)
            if after - before == 0:
                break
            samples = [self.generate_sample(i) for i in range(before, after)]
            self.index = after
            batch = self.collate_fn(samples)
            yield batch

    def _iter_train(self):
        while True:
            with self.timers.next_batch.timeit():
                with self.timers.fill_queue.timeit():
                    try:
                        self.update_queue()
                    except ZMQNotReady:
                        yield ZMQNotReadySample()
                        continue
                assert len(self.collate_queue) == self.params.batch.collate_queue_size
                batch = self.get_batch_in_queue()
                yield batch

    def update_queue(self):
        assert self.queue_strategy in [
            "uniform_sampling",
            "uniform_sampling_replacement",
        ]
        to_be_sorted = False

        # In uniform sampling, we fill the queue with the number of elements
        # rmv from the queue when sampling batch
        if self.queue_strategy == "uniform_sampling":
            # number of elements to add
            n = self.params.batch.collate_queue_size - len(self.collate_queue)
            assert n > 0
            start = time.time()
            for i in range(n):
                with self.timers.generate_sample.timeit():
                    sample = self.generate_sample(index=None)
                self.collate_queue.append(sample)
                if "x" in sample:
                    self.stats["queue_last_x_slen"].act(len(sample["x"]))
                if "y" in sample:
                    self.stats["queue_last_y_slen"].act(len(sample["y"]))
                if (i + 1) % 1000 == 0:
                    print(f"FILLING QUEUE {i + 1} {time.time() - start}", flush=True)
            to_be_sorted = True
        # In unform sampling wih replacement,
        # we fill the deque and replace the queue by the deque every N batches
        else:
            # first time
            if self.deque is None:
                self.deque = deque(maxlen=self.params.batch.collate_queue_size)
                while len(self.deque) < self.params.batch.collate_queue_size:
                    self.deque.append(self.generate_sample(index=None))
                self.collate_queue = list(self.deque)
                to_be_sorted = True
            # add N sequences to deque, where N is the average number of
            # sequences per batch
            n_seqs = math.ceil(np.mean(self.batch_sizes))
            self.stats["n_seqs_to_add"].act(n_seqs)
            for _ in range(n_seqs):
                self.deque.append(self.generate_sample(index=None))
            # recreate collate queue with deque elements
            if self.n_batches >= self.collate_queue_update_freq:
                to_be_sorted = True
                self.collate_queue = list(self.deque)
                self.n_batches = 0

        if to_be_sorted:
            with self.timers.sort.timeit():
                self.collate_queue.sort(key=key_fn_len)

        with self.timers.update_stats.timeit():
            # update statistics
            self.update_stats()

    def get_batch_in_queue(self):
        assert self.queue_strategy in [
            "uniform_sampling",
            "uniform_sampling_replacement",
        ]
        # select random index
        # before = self.env.rng.randint(0, len(self.collate_queue))
        before = self.env.rng.randint(-self.batch_size, len(self.collate_queue))
        before = max(min(before, len(self.collate_queue) - self.batch_size), 0)
        after = self.get_last_seq_id(before)

        # create batch / remove sampled sequences from the queue
        batch = self.collate_fn(self.collate_queue[before:after])
        if self.queue_strategy == "uniform_sampling":
            self.collate_queue = (
                self.collate_queue[:before] + self.collate_queue[after:]
            )
        else:
            self.batch_sizes.append(after - before)
            self.n_batches += 1
        return batch

    def update_stats(self):
        self.stats["queue_size"].act(len(self.collate_queue))
        if time.time() - self.last_log > self.params.log_freq:
            to_log = {}

            def update(d: Dict):
                assert all(k not in to_log for k in d.keys())
                to_log.update(d)

            wrapper_timers = self.timers.rate_and_reset()
            wrapper_timers = {
                f"wrapper_timers/{k}": v for k, v in wrapper_timers.items()
            }
            update(wrapper_timers)

            # queue / batch stats
            for name, counter in self.stats.items():
                if counter.cum_count > 0:
                    to_log[name] = counter.rate_and_reset()

            # average length of sequences in the queue
            if "x" in self.collate_queue[0]:
                seq_lengths = [len(x["x"]) for x in self.collate_queue]
                to_log["queue_avg_x_slen"] = float(np.mean(seq_lengths))
            if "y" in self.collate_queue[0]:
                seq_lengths = [len(x["y"]) for x in self.collate_queue]
                to_log["queue_avg_y_slen"] = float(np.mean(seq_lengths))

            update(self.env.get_stats())
            self.log(to_log)

    def log(self, stats: Dict):
        if self.metrics is None:
            params: TrainerArgs = self.params
            self.metrics = Logger(
                outdir=params.dump_path,
                tag=f"data_loader_{self.task}",
                quiet=not (
                    params.slurm_conf.global_rank == 0 and self.get_worker_id() == 0
                ),
            )
        # logger.info(f"__log_data_loader_{self.task}__: {json.dumps(stats)}")
        print_memory(logger, f"data_loader_{self.task}")
        self.metrics.log_metrics(stats)
        self.last_log = time.time()

    def batch_sequences(self, x):
        """
        Create a batch of padded sequences.
        """
        assert type(x) is list
        assert all(type(xs) is list for xs in x)

        # sequence lengths
        bs = len(x)
        xlen = torch.LongTensor([len(s) for s in x])
        assert max(xlen) <= self.params.batch.max_len

        # merge sequences into a batch
        x_batch = torch.full((bs, max(xlen)), self.pad_index, dtype=torch.long)
        for sid, (xl, xs) in enumerate(zip(xlen, x)):
            assert len(xs) == xl
            x_batch[sid, :xl] = torch.LongTensor(xs)

        return x_batch, xlen

    def get_last_seq_id(self, before: int) -> int:
        """
        Return the last sequence ID that would allow to fit according to `size_fn`.
        """
        after = before
        while (
            after < len(self.collate_queue)
            and size_with_pad(self.collate_queue[before:after]) < self.tokens_per_batch
        ):
            after += 1
        # if we exceed `tokens_per_batch`, remove the last element
        size = size_with_pad(self.collate_queue[before:after])
        if size > self.tokens_per_batch:
            if after > before + 1:
                after -= 1
            else:
                logger.warning(
                    f"Exceeding tokens_per_batch: {size} "
                    f"({after - before} sequences)"
                )
        return after

    def collate_fn(self, seqs: List[Dict]):
        """
        Collate sequences into a batch.
        """
        assert all(seq.keys() == seqs[0].keys() for seq in seqs)

        batch = {}

        # input sequences
        if "x" in seqs[0]:
            x = [seq["x"] for seq in seqs]
            batch["x"], batch["xlen"] = self.batch_sequences(x)

        # output sequences
        if "y" in seqs[0]:
            y = [seq["y"] for seq in seqs]
            batch["y"], batch["ylen"] = self.batch_sequences(y)

        # sub-sequences
        if "x_subseq_pos" in seqs[0]:
            batch["x_subseq_pos"] = [x["x_subseq_pos"] for x in seqs]
            batch["y_subseq_pos"] = [x["y_subseq_pos"] for x in seqs]

        # return (for RL)
        if "return" in seqs[0]:
            batch["return"] = torch.tensor([x["return"] for x in seqs])

        # Q estimates (for MCTS)
        if "q_estimate" in seqs[0]:
            batch["q"] = torch.FloatTensor([x["q_estimate"] for x in seqs])
            assert batch["q"].min().item() >= 0, batch["q"]
            assert batch["q"].max().item() <= 1 + 1e-4, batch["q"]

        # Q tactic estimates (for MCTS)
        if "q_tactics" in seqs[0]:
            batch["q_tactics"] = torch.FloatTensor([x["q_tactics"] for x in seqs])
            assert batch["q_tactics"].min().item() >= 0, batch["q_tactics"]
            assert batch["q_tactics"].max().item() <= 1 + 1e-4, batch["q_tactics"]

        # Inproof
        if "inproof" in seqs[0]:
            batch["inproof"] = torch.FloatTensor([x["inproof"] for x in seqs])
            assert batch["inproof"].min().item() >= 0, batch["inproof"]
            assert batch["inproof"].max().item() <= 2 + 1e-4, batch["inproof"]

        # discriminator target
        if "disc_tgt" in seqs[0]:
            batch["disc_tgt"] = torch.FloatTensor([x["disc_tgt"] for x in seqs])
            batch["disc_inp"], batch["disc_inp_len"] = self.batch_sequences(
                [x["disc_inp"] for x in seqs]
            )

        # input_conditioning
        if "input_conditioning" in seqs[0]:
            batch["input_conditioning"] = torch.from_numpy(
                np.vstack([x["input_conditioning"] for x in seqs])
            )

        # theorem names
        if "name" in seqs[0]:
            batch["names"] = [x["name"] for x in seqs]

        # languages
        if "lang" in seqs[0]:
            batch["langs"] = torch.LongTensor([x["lang"] for x in seqs])
        if "lang1" in seqs[0]:
            batch["langs1"] = torch.LongTensor([x["lang1"] for x in seqs])
        if "lang2" in seqs[0]:
            batch["langs2"] = torch.LongTensor([x["lang2"] for x in seqs])

        # update stats
        if "x" in batch and "xlen" in batch:

            # batch / sequence sizes
            self.stats["batch_size"].act(len(batch["xlen"]))
            self.stats["batch_min_len"].act(batch["xlen"].min().item())
            self.stats["batch_max_len"].act(batch["xlen"].max().item())
            self.stats["batch_mean_len"].act(batch["xlen"].float().mean().item())

            # padding
            n_tok = batch["x"].nelement()
            n_pad = batch["x"].nelement() - batch["xlen"].sum().item()
            if "y" in batch and "ylen" in batch:
                n_tok += batch["y"].nelement()
                n_pad += batch["y"].nelement() - batch["ylen"].sum().item()
            self.stats["batch_padding_ratio"].act(n_pad / n_tok)

        return batch

    def get_worker_id(self):
        """
        Get worker ID.
        """
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0)
        worker_id = 0 if worker_info is None else worker_info.id
        assert self.train or worker_id == 0
        return worker_id

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if hasattr(self.env, "rng"):
            return
        logger.info(
            f"Initializing random generator for task={self.task}, split={self.split} ..."
        )
        if self.train:
            assert self.env_base_seed >= 0
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            if hasattr(self.env, "mcts"):
                for v in self.env.mcts.values():
                    v.worker_id = worker_id
            seed = [worker_id, self.global_rank, self.env_base_seed]
            self.env.set_rng(np.random.RandomState(seed))
            logger.info(
                f"Initialized random generator for worker {worker_id} "
                f"(task={self.task}, split={self.split}), "
                f"with seed {seed} (base seed={self.env_base_seed})."
            )
        else:
            self.env.set_rng(np.random.RandomState(0 if self.split == "valid" else 1))

    def generate_sample(self, index: Optional[int]):
        """
        Generate a sample. `None` corresponds to a generation failure,
        typically because of a too large number of tokens.
        """
        assert self.train == (index is None)
        while True:
            start = time.time()
            sample = self.env.get_sample(self.task, self.split, index=index)
            if sample is None:
                self.timers.generate_none_sample.add_interval(time.time() - start)
                assert self.train
                continue
            return sample
