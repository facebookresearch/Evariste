# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import List, Any, Generic, Sequence, Tuple

from evariste.backward.graph import HashByString
from evariste.comms.store import Sender, S
from params import Params


logger = getLogger(name=__name__)


def _log_info(msg: str):
    logger.info(f"[{__name__}] {msg}")


Label = str


@dataclass
class PoolTrainingCfg(Params):
    is_a_pool_training: bool = False
    n_models: int = -1
    # if mask_some_labels each model see n_labels * (model - 1) / n_models labels
    mask_some_labels: bool = False

    def _check_and_mutate_args(self):
        assert self.is_a_pool_training == (self.n_models > 1)


class PoolTrainingSender(Generic[S], Sender[Tuple[Label, S]]):
    # Put it here (and not with senders) to be with all pool training code

    def __init__(
        self,
        models: Sequence[str],
        senders: Sequence[Sender[Tuple[Label, S]]],
        cfg: PoolTrainingCfg,
    ):
        """Simpler init for unitesting"""
        self.models = models
        assert sorted(self.models) == self.models
        self.senders = senders
        self.cfg = cfg
        assert len(self.models) == len(self.senders) == self.cfg.n_models
        assert self.cfg.is_a_pool_training

        self.sent: int = 0

    def store(self, seq: Tuple[Label, S]) -> None:
        label, sample = seq
        # not filtering by labels for the moment
        for i, sender in enumerate(self.senders):
            if self.cfg.mask_some_labels:
                if label in {"eq_bwd_rwalk_seq2seq", "eq_bwd_graph_seq2seq"}:
                    hash_ = self.sent
                else:
                    assert "eq_bwd" not in label
                    assert "seq2seq" not in label
                    # TODO: cache this hashing ?
                    hash_ = HashByString(fingerprint=label).hash
                if hash_ % len(self.models) == i:
                    continue
            sender.store((label, sample))
        self.sent += 1

    def rate_and_reset(self) -> float:
        return sum(s.rate_and_reset() for s in self.senders)

    def close(self) -> None:
        for sender in self.senders:
            sender.close()

    @classmethod
    def from_zmq_prover_params(cls, params: Any, client_id: str, mcts_subtask: str):
        # circular imports
        from evariste.backward.prover.zmq_prover import (
            ZMQProverParams,
            MCTSSubTaskSender,
        )

        assert isinstance(params, ZMQProverParams)
        assert params.pool_training.is_a_pool_training
        model_dirs = _get_and_wait_pool_trainers_dirs(
            prover_root_dir=params.root_dir,
            n_expected_handlers=params.pool_training.n_models,
        )

        _log_info(f"Starting: {mcts_subtask}, connecting to {len(model_dirs)} models")
        models: List[str] = []
        senders: List[MCTSSubTaskSender] = []
        for model_dir in model_dirs:
            models.append(model_dir.name)
            _log_info(f"connecting to {model_dir}...")
            senders.append(
                MCTSSubTaskSender(
                    client_id=client_id,
                    root_dir=model_dir,
                    mcts_subtask=mcts_subtask,
                    params=params,
                )
            )
            _log_info(f"connected to {model_dir}!")

        return cls(senders=senders, models=models, cfg=params.pool_training)


def _get_and_wait_pool_trainers_dirs(
    prover_root_dir: Path, n_expected_handlers: int, timeout=600
) -> List[Path]:
    start = time.time()
    # TODO: presents is maybe not correct in english ?
    presents = _get_pool_trainers_dirs(prover_root_dir)

    while len(presents) < n_expected_handlers:
        _log_info(
            f"waiting for {n_expected_handlers} handlers to start,"
            f" found {len(presents)} so far (ids: {[p.name for p in presents]})"
        )
        if time.time() - start > timeout:
            raise RuntimeError(
                f"expecting {n_expected_handlers} "
                f"handlers, found only {len(presents)} "
                f"(ids: {[p.name for p in presents]})"
            )
        time.sleep(10)
        presents = _get_pool_trainers_dirs(prover_root_dir)

    assert len(presents) == n_expected_handlers, (
        f"found {len(presents)}, expected {n_expected_handlers} "
        f"{[p.name for p in presents]}"
    )
    _log_info(f"Found {len(presents)} handlers (ids: {[p.name for p in presents]})")
    return presents


def _get_pool_trainers_dirs(prover_root_dir: Path):
    pool_dir = prover_root_dir.parent
    # warning this will not work if not slurm job
    # TODO: fix for local

    return sorted(
        [p for p in pool_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: p.name,
    )
