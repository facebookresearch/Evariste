# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import random
from typing import List, Generic, TypeVar, Optional, Sequence
from dataclasses import dataclass, field
from collections import Counter
from logging import getLogger
import abc

from evariste.forward.common import GenerationHistory, ForwardGoal
from evariste.forward.env_specifics.generation_stats import GenerationStats

logger = getLogger()


class EmptyStore(Exception):
    pass


@dataclass
class StorableStats:
    n_sample: int
    avg_sample_elem: float
    max_sample_elem: float
    seq_sample_cover: float


@dataclass
class AnnotatedGeneration:
    generation: GenerationHistory
    src: str
    prover_goal: ForwardGoal
    gen_ckpt: int
    generation_stats: GenerationStats
    generation_reward: float  # score given by heuristic during generation
    _reward: Optional[float] = None
    _proved: Optional[bool] = None

    # needed for store stats on the usage of these generations
    n_sample: int = 0
    elem_sampled_counter: Counter = field(default_factory=lambda: Counter())

    @property
    def reward(self) -> float:
        assert self._reward is not None
        return self._reward

    @property
    def proved(self) -> bool:
        assert self._proved is not None
        return self._proved

    def grabbed(self):
        self.n_sample += 1

    def grabbed_elem(self, i):
        self.elem_sampled_counter[i] += 1

    def get_stats_before_del(self) -> StorableStats:
        """not used anymore"""
        max_sampled = (
            0
            if not len(self.elem_sampled_counter)
            else max(self.elem_sampled_counter.values())
        )
        data_len = len(self.generation.forward_steps())
        return StorableStats(
            n_sample=self.n_sample,
            avg_sample_elem=self.n_sample / data_len,
            max_sample_elem=max_sampled,
            seq_sample_cover=len(self.elem_sampled_counter) / data_len,
        )

    def get_reward(self) -> float:
        assert self.reward is not None
        return float(self.reward)

    def get_weight(self) -> float:
        return float(self.generation_stats.last_node_proof_size)

    def __post_init__(self):
        assert self.generation_reward > 0


S = TypeVar("S")


class Sender(abc.ABC, Generic[S]):
    @abc.abstractmethod
    def store(self, seq: S) -> None:
        pass

    @abc.abstractmethod
    def rate_and_reset(self) -> float:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass


class Receiver(abc.ABC, Generic[S]):
    @abc.abstractmethod
    def receive_batch(self) -> List[S]:
        pass

    @abc.abstractmethod
    def rate_and_reset(self) -> float:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass


class MultiReceiver(Receiver[S]):
    def __init__(self, receivers: Sequence[Receiver[S]]):
        self.receivers = receivers

    def receive_batch(self) -> List[S]:
        objs = []
        for receiver in self.receivers:
            objs.extend(receiver.receive_batch())
        return objs

    def rate_and_reset(self) -> float:
        return sum(
            [receiver.rate_and_reset() for receiver in self.receivers], 0
        )  # assumes all _rate_avg_time are equal...

    def close(self) -> None:
        logger.info("Closing MultiReceiver ...")
        for receiver in self.receivers:
            receiver.close()
        logger.info("Closed MultiReceiver")


class MultiSender(Generic[S], Sender[S]):
    def __init__(self, senders: Sequence[Sender[S]], send_to_all: bool):
        self.senders = senders
        self.send_to_all = send_to_all
        # to avoid sending all provers to send first to the same trainer
        self.next_sender = int(random.randint(0, len(senders)))

    def store(self, obj: S) -> None:
        if self.send_to_all:
            for sender in self.senders:
                sender.store(obj)
        else:
            self.senders[self.next_sender % len(self.senders)].store(obj)
            self.next_sender += 1

    def rate_and_reset(self) -> float:
        return sum(
            [sender.rate_and_reset() for sender in self.senders], 0
        )  # assumes all _rate_avg_time are equal...

    def close(self) -> None:
        logger.info("Closing MultiSender ...")
        for sender in self.senders:
            sender.close()
        logger.info("Closed MultiSender")
