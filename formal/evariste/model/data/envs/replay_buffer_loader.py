# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, TypeVar, Generic, List, Dict, Deque, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from logging import getLogger
import os
import time
import psutil
import numpy as np

from params import Params
from evariste.comms.zip import ZipReceiver
from evariste.comms.zmq import ZMQReceiver
from evariste.comms.store import EmptyStore, Receiver
from evariste.forward.training.reward_quantiles import RewardQuantiles
from evariste.metrics import ActionCounter, StatsCollection, Timer


@dataclass
class ReplayBufferArgs(Params):
    min_len: int = 1_000  # min number of sequences for rb to return something
    max_len: int = 100_000
    discard_first: int = 0
    discount: float = 0.99
    refresh_every: int = 1
    weighted_sampling: bool = False
    filter_if_rewards_zero: bool = False
    # if > 0, compute reward quantiles and return it in reward infos
    n_reward_quantiles: int = -1

    def __post_init__(self):
        assert self.min_len > 0
        assert self.min_len < self.max_len, (self.min_len, self.max_len)


logger = getLogger()
LOG_INTERVAL = 100_000


T = TypeVar("T")


@dataclass
class SampleInfos:
    reward_quantile: Optional[int] = None
    n_sampled: Optional[int] = None


@dataclass
class RBTimers(StatsCollection):
    receive_batch: Timer = field(default_factory=lambda: Timer())
    fill_store: Timer = field(default_factory=lambda: Timer())
    ps_memory: Timer = field(default_factory=lambda: Timer())
    sampling: Timer = field(default_factory=lambda: Timer())


class ReplayBuffer(Generic[T]):
    def __init__(self, receiver: Receiver[T], rb_params: ReplayBufferArgs):
        self.receiver = receiver
        self.rb_params = rb_params
        self.stored: Deque[T] = deque(maxlen=rb_params.max_len)
        self.n_sampled: Deque[int] = deque(maxlen=rb_params.max_len)
        self.weights: Deque[float] = deque(maxlen=rb_params.max_len)
        self.weight_probs: Optional[np.ndarray] = None
        self.ingress = ActionCounter("ingress", is_rate=True, silent=True)
        self.sampling = ActionCounter("sampling", is_rate=True, silent=True)
        self.memory = ActionCounter("rb_memory", is_rate=False, silent=True)
        self.discarded = 0

        # Sample size goal/tactic removed in #1063
        # took up ~90% of the time in fill_store to call tokenize()

        self.samples_filtered = ActionCounter(
            "samples_filtered", is_rate=False, silent=True
        )
        self.timers = RBTimers()

        self.min_seq = rb_params.min_len

        self.success_cnt = 0
        self.total_cnt = 0
        self.received_seqs: bool = False
        self.duration = 1
        self.weighted_sampling = rb_params.weighted_sampling
        self.filter_if_rewards_zero = rb_params.filter_if_rewards_zero
        self.last_empty_log = time.time()

        self.reward_quantiles: Optional[RewardQuantiles] = None
        if self.rb_params.n_reward_quantiles > 0:
            self.reward_quantiles = RewardQuantiles(
                size=self.rb_params.max_len,
                n_quantiles=self.rb_params.n_reward_quantiles,
            )

        self.receiver_filters: List[Callable[[T], bool]] = []

    def add_receiver_filter(self, fn: Callable[[T], bool]):
        self.receiver_filters.append(fn)

    def get_sample(
        self, split: str, index: Optional[int], rng: np.random.RandomState, block: bool
    ) -> Tuple[T, SampleInfos]:
        self.timers.sampling.start()
        try:
            return self._get_sample(split=split, index=index, rng=rng, block=block)
        finally:
            # don't know if it can raise error caught somewhere
            self.timers.sampling.stop()

    def _get_sample(
        self, split: str, index: Optional[int], rng: np.random.RandomState, block: bool
    ) -> Tuple[T, SampleInfos]:
        assert split == "train"

        if self.total_cnt == 0 and isinstance(self.receiver, ZipReceiver):
            # only used when receiver is a ZipReceiver
            # indeed we reload past sequences already in zip. This
            # accelerate dev process (no need to wait for other actors to reconnect)
            # + good when checkpointing for warming up replay buffer
            logger.info("RB: ZipReceiver detected, trying to reload existing zips")
            seqs = self.receiver.reload_last_chunks(self.rb_params.max_len)
            self.fill_store(seqs)

        if self.success_cnt % self.rb_params.refresh_every == 0:
            self.fill_store()
        if self.total_cnt % LOG_INTERVAL == 0:
            logger.info(f"RB: {len(self.stored)} seqs in store (PID: {os.getpid()})")
            if self.reward_quantiles:
                logger.info(
                    f"RB: reward_quantiles: {self.reward_quantiles.cur_quantiles}"
                )
        self.total_cnt += 1

        sample: Optional[T] = None
        while sample is None:
            try:
                if len(self.stored) < self.min_seq:
                    raise EmptyStore
                if index is None:
                    if self.weighted_sampling:
                        index = rng.choice(len(self.stored), p=self.weight_probs)
                    else:
                        index = rng.randint(0, len(self.stored))
                sample = self.stored[index]
                self.n_sampled[index] += 1
            except EmptyStore:
                if not block:
                    assert self.success_cnt == 0
                    raise
                self.fill_store()
                if len(self.stored) == 0:
                    logger.info(
                        f"RB: waiting {self.duration}s to receive first samples"
                        f" (PID: {os.getpid()})"
                    )
                    time.sleep(self.duration)
                    self.duration = min(60, self.duration * 2)
                elif time.time() - self.last_empty_log > 60:
                    self.last_empty_log = time.time()
                    logger.info(
                        f"RB: Has {len(self.stored)} waiting for {self.min_seq}"
                        f" (PID: {os.getpid()})"
                    )

        if self.success_cnt == 0:
            logger.info(f"RB: sampled first sample! (PID: {os.getpid()})")

        self.success_cnt += 1

        quantile = (
            self.reward_quantiles.get_quantile_idx(sample.get_reward())
            if self.reward_quantiles
            else None
        )
        assert index is not None
        infos = SampleInfos(reward_quantile=quantile, n_sampled=self.n_sampled[index])
        self.sampling.act()
        return sample, infos

    def fill_store(self, sequences: Optional[List[T]] = None):
        self.timers.fill_store.start()
        try:
            self._fill_store(sequences=sequences)
        finally:
            # don't know if it can raise error caught somewhere
            self.timers.fill_store.stop()

    def _fill_store(self, sequences: Optional[List[T]] = None):
        if sequences is None:
            self.timers.receive_batch.start()
            try:
                samples = self.receiver.receive_batch()
            finally:
                # don't know if it can raise error caught somewhere
                self.timers.receive_batch.stop()
            self.timers.ps_memory.start()
            self.memory.act(psutil.Process().memory_info().rss / 1024 ** 3)
            self.timers.ps_memory.stop()

        else:
            samples = sequences
        n_seqs = len(samples)
        if n_seqs > 0 and len(self.stored) == 0:
            logger.info(
                f"RB: received first samples: {n_seqs} seqs! (PID: {os.getpid()})"
            )
        n_samples_before_filter = len(samples)

        if self.filter_if_rewards_zero:
            samples = [s for s in samples if s.reward > 0]
        samples = [s for s in samples if all(fn(s) for fn in self.receiver_filters)]
        for sample in samples:
            if self.discarded < self.rb_params.discard_first:
                self.discarded += 1
                continue
            if self.reward_quantiles:
                self.reward_quantiles.update_with_reward(reward=sample.get_reward())
            self.stored.append(sample)
            self.n_sampled.append(0)
            if self.weighted_sampling:
                self.weights.append(sample.get_weight())
                assert self.weights[-1] > 0
                p = np.array(self.weights, dtype=np.float64)
                self.weight_probs = p / p.sum()
            self.ingress.act()
        n_samples_after_filter = len(samples)
        self.samples_filtered.act(n_samples_before_filter - n_samples_after_filter)
        return len(samples)

    def close(self):
        logger.info("Closing ReplayBuffer ...")
        self.receiver.close()
        logger.info("Closed ReplayBuffer")

    def store_stats_and_reset(self) -> Dict[str, float]:
        assert len(self.stored) == len(self.n_sampled)
        assert len(self.weights) == 0 or len(self.weights) == len(self.stored)
        res = {
            "ingress": self.ingress.rate_and_reset(),
            "rb_sample": self.sampling.rate_and_reset(),
            "rb_memory": self.memory.rate_and_reset(),
            "rb_stored": len(self.stored),
            "n_filtered_samples": self.samples_filtered.rate_and_reset(),
        }
        res.update(self.timers.rate_and_reset())
        if len(self.n_sampled) > 0:
            res["rb_avg_n_sampled"] = float(np.mean(self.n_sampled))
        if isinstance(self.receiver, ZMQReceiver):
            res["ingress_mb"] = self.receiver.recv_mb.rate_and_reset()
            res["bad_recv"] = self.receiver.bad_recv
        if len(self.weights) > 0:
            n_sampled_per_weight = [n / w for n, w in zip(self.n_sampled, self.weights)]
            res["rb_n_sampled_per_weight"] = float(np.mean(n_sampled_per_weight))
            res["rb_avg_weight"] = float(np.mean(self.weights))
        if len(self.stored) > 0 and hasattr(self.stored[0], "get_reward"):
            rewards = [sample.get_reward() for sample in self.stored]
            res["rb_avg_reward"] = float(np.mean(rewards))

        return res
