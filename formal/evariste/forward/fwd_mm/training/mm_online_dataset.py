# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from collections import deque
from typing import List, Tuple, Any

import numpy as np
from numpy.random.mtrand import RandomState

from evariste.comms.store import Receiver
from evariste.comms.zip import ZipReceiver
from evariste.envs.mm.utils import Node, count_unique_nodes
from evariste.forward.online_generation.worker_type import WorkerType
from evariste.forward.fwd_mm.training.mm_training_helpers import (
    MMFwdTrainingDataset,
    sample_from_cumulative_mm,
)
from evariste.forward.fwd_mm.training.common import MMFwdTrainingProof
from evariste.forward.fwd_mm.mm_helpers import history_to_mm_head_nodes
from evariste.forward.online_generation.online_generation_common import FwdTrajectory

logger = logging.getLogger()


class MMOnlineDataset(MMFwdTrainingDataset):
    def __init__(
        self,
        receiver: Receiver[FwdTrajectory],
        n_max_proofs: int,
        refresh_every: int,
        n_max_nodes: int,
    ):
        self.receiver = receiver

        self.n_max_proofs = n_max_proofs

        self._proof_deque = deque(maxlen=n_max_proofs)
        self._weights_deque = deque(maxlen=n_max_proofs)
        self._nodes_deque = deque(maxlen=n_max_nodes)
        self._cumulative = []
        self._proofs = []
        self._nodes = []
        self._iteration = 0
        self._refresh_every = refresh_every

        self.cur_proof_id = 0

    def _refresh_data(self):
        logger.info("OnlineGenerationDataset: start refreshing data")
        if self._iteration == 0 and isinstance(self.receiver, ZipReceiver):
            # in case of checkpointing we reload last data stored in zip
            trajs = self.receiver.reload_last_chunks(n_sequences=self.n_max_proofs)
        else:
            trajs = self.receiver.receive_batch()
        for traj in trajs:
            proofs = history_to_mm_head_nodes(traj.history)
            for root_node in proofs:
                name = f"{self.cur_proof_id}"
                self.cur_proof_id += 1
                self._proof_deque.append((name, root_node))
                self._weights_deque.append(
                    count_unique_nodes(root_node, ignore_e_hyps=True)
                )
                self._nodes_deque.append(root_node)

        weights = np.array(self._weights_deque)
        self._cumulative = np.cumsum(weights)
        self._proofs = list(self._proof_deque)
        assert len(self._proofs) == len(self._cumulative)
        self._nodes = list(self._nodes_deque)
        if self._proofs:
            n_statements = len({n.statement_str for n in self._nodes})
            logger.info(
                f"OnlineGenerationDataset: refreshed data"
                f" loaded {len(trajs)} new trajs,"
                f" n proofs: {len(self._proof_deque)},"
                f" n proofs statements: {n_statements}"
                f" avg proof size: {np.mean(weights):.02f} "
                f" max proof size: {np.max(weights):.02f}"
            )
        else:
            logger.info("OnlineGenerationDataset: no data yet")

    def proofs_and_cumulative(self) -> Tuple[List[Tuple[str, Node]], np.array]:
        if self._iteration % self._refresh_every == 0:
            self._refresh_data()
        self._iteration += 1
        return self._proofs, self._cumulative

    def nodes(self) -> List[Node]:
        return self._nodes

    def sample_training_graph(self, rng: RandomState, split: str) -> MMFwdTrainingProof:
        assert split == "train"
        data, cumulative = self.proofs_and_cumulative()
        name, root = sample_from_cumulative_mm(cumulative, data, rng=rng)
        return MMFwdTrainingProof(name=name, generated=True, proved=None, root=root)

    @classmethod
    def from_trainer_args(cls, args: Any) -> "MMOnlineDataset":
        from evariste.trainer.args import TrainerArgs
        from evariste.comms.comms import make_receiver
        from evariste.comms.rl_distributed_config import RLDistributedConfig

        # for circular imports
        assert isinstance(args, TrainerArgs)
        rl_dist_cfg = RLDistributedConfig.new_fwd_distributed_rl_cfg(args)
        logger.info(
            f"Creating new RLDistributedConfig for this fwd rl setup: {rl_dist_cfg}"
        )
        receiver = make_receiver(
            rank=args.slurm_conf.global_rank,
            receiver_type=WorkerType.PROVER_TRAINER,
            cfg=rl_dist_cfg,
        )
        return cls(
            receiver=receiver,
            n_max_proofs=args.online_gen_cfg.n_max_proofs,
            refresh_every=args.online_gen_cfg.refresh_every,
            n_max_nodes=args.online_gen_cfg.n_max_proofs,
        )

    def close(self):
        self.receiver.close()

    def __del__(self):
        self.close()
