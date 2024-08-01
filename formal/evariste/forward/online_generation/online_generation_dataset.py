# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import os
from collections import deque
from typing import List, Tuple, Any, TypeVar, Generic

import numpy as np
from numpy.random.mtrand import RandomState

from evariste.comms.store import Receiver
from evariste.comms.zip import ZipReceiver
from evariste.forward.online_generation.worker_type import WorkerType
from evariste.forward.online_generation.online_generation_common import FwdTrajectory
from evariste.forward.common import ProofNode
from evariste.forward.training.graph_sampler import (
    GraphTrainingDataset,
    GraphTrainingSample,
)
from evariste.forward.training.helpers import (
    count_unique_theorems,
    sample_from_cumulative,
)

logger = logging.getLogger()

SomeProofNode = TypeVar("SomeProofNode", bound=ProofNode)


class OnlineGenerationDataset(
    Generic[SomeProofNode], GraphTrainingDataset[SomeProofNode]
):
    """Env agnostic class that allow to receive, store and sample online generations

    TODO: use internally a ReplayBuffer?
    """

    def __init__(
        self, receiver: Receiver[FwdTrajectory], n_max_nodes: int, refresh_every: int
    ):
        self.receiver = receiver

        self.n_max_nodes = n_max_nodes

        self._nodes_deque = deque(maxlen=n_max_nodes)
        self._weights_deque = deque(maxlen=n_max_nodes)
        self._cumulative: List[float] = []
        self._nodes: List[Tuple[str, SomeProofNode]] = []
        self._iteration = 0
        self._refresh_every = refresh_every

        self._nodes_for_noise: List[SomeProofNode] = []

        self.cur_traj_id = 0

    def _refresh_data(self):
        name = f"{self.__class__.__name__}"
        logger.info(f"(PID: {os.getpid()}) - {name}: start refreshing data")
        if self._iteration == 0 and isinstance(self.receiver, ZipReceiver):
            # in case of checkpointing we reload last data stored in zip
            trajs = self.receiver.reload_last_chunks(n_sequences=self.n_max_nodes)
        else:
            trajs = self.receiver.receive_batch()
        for traj in trajs:
            name = f"{self.cur_traj_id}"
            self.cur_traj_id += 1
            if traj.history.beams is not None:
                nodes = traj.history.beam_proof_nodes()
            else:
                nodes = traj.history.proof_nodes()
            for node in nodes:
                self._nodes_deque.append((name, node))
                self._weights_deque.append(count_unique_theorems(node))

        weights = np.array(self._weights_deque)
        self._cumulative = np.cumsum(weights)
        self._nodes: List[Tuple[str, SomeProofNode]] = list(self._nodes_deque)
        self._nodes_for_noise = [n for _, n in self._nodes]
        assert len(self._nodes) == len(self._cumulative)
        if self._nodes:
            n_theorems = len({n.theorem for _, n in self._nodes})
            n_trajs_in_rb = len({name for name, _ in self._nodes})
            logger.info(
                f"(PID: {os.getpid()}) - {name}: refreshed data"
                f" loaded {len(trajs)} new trajs,"
                f" n nodes: {len(self._nodes)},"
                f" n different theorems: {n_theorems}"
                f" n trajs inside rb: {n_trajs_in_rb}"
                f" avg proof size: {np.mean(weights):.02f} "
                f" max proof size: {np.max(weights):.02f}"
            )
        else:
            logger.info(f"(PID: {os.getpid()}) - {name}: no data yet")

    def get_graph_training_sample(
        self, task: str, split: str, rng: RandomState
    ) -> GraphTrainingSample[SomeProofNode]:
        assert split == "train"
        if self._iteration % self._refresh_every == 0:
            self._refresh_data()
        self._iteration += 1
        name, root = sample_from_cumulative(self._cumulative, self._nodes, rng=rng)
        return GraphTrainingSample(label=name, root=root)

    def nodes_for_noise(self, split: str) -> List[SomeProofNode]:
        assert split == "train"
        return self._nodes_for_noise

    @classmethod
    def from_trainer_args(cls, params: Any) -> "OnlineGenerationDataset":
        from evariste.trainer.args import TrainerArgs
        from evariste.comms.comms import make_receiver
        from evariste.comms.rl_distributed_config import RLDistributedConfig

        # for circular imports
        assert isinstance(params, TrainerArgs)
        rl_dist_cfg = RLDistributedConfig.new_fwd_distributed_rl_cfg(params)
        logger.info(
            f"Creating new RLDistributedConfig for this fwd rl setup: {rl_dist_cfg}"
        )
        receiver = make_receiver(
            rank=params.slurm_conf.global_rank,
            receiver_type=WorkerType.PROVER_TRAINER,
            cfg=rl_dist_cfg,
        )
        return cls(
            receiver=receiver,
            n_max_nodes=params.online_gen_cfg.n_max_proofs,
            refresh_every=params.online_gen_cfg.refresh_every,
        )

    def close(self):
        self.receiver.close()
