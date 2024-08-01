# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple, List

from evariste.backward.remote.pool_training import PoolTrainingCfg, PoolTrainingSender
from evariste.comms.store import Sender

N_MODELS = 10
N = 10_000


class DummySender(Sender[Tuple[str, int]]):
    def store(self, seq: Tuple[str, int]) -> None:
        self.sent.append(seq)

    def rate_and_reset(self) -> float:
        return len(self.sent)

    def close(self) -> None:
        pass

    def __init__(self):
        self.sent = []


def test_pool_training_sender_label_masking():
    _test_pool_training_sender_label_masking([f"{i}" for i in range(N)])


def test_pool_training_sender_eq_rwalk():
    _test_pool_training_sender_label_masking(
        [f"eq_bwd_rwalk_seq2seq" for _ in range(N)]
    )


def _test_pool_training_sender_label_masking(labels: List[str]):
    cfg = PoolTrainingCfg(
        is_a_pool_training=True, mask_some_labels=True, n_models=N_MODELS
    )
    inner_senders: List[DummySender] = [DummySender() for _ in range(N_MODELS)]
    sender = PoolTrainingSender(
        models=[f"model_{i}" for i in range(N_MODELS)], senders=inner_senders, cfg=cfg,
    )
    for i, label in enumerate(labels):
        sender.store((label, i))
    _ = sender.rate_and_reset()  # check that it doesn't crash

    for inner_sender in inner_senders:
        expected = len(labels) * (N_MODELS - 1) / N_MODELS
        assert 0.9 * expected < len(inner_sender.sent) < 1.1 * expected
