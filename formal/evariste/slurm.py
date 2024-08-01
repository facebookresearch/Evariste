# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from dataclasses import dataclass
from logging import getLogger
import os
import sys
import torch
import socket
import signal
from evariste.utils import timeout, MyTimeoutError

# from datetime import timedelta

from params import Params


logger = getLogger()


@dataclass
class SlurmConf(Params):
    is_slurm_job: bool = False

    global_rank: int = -1  # TODO: rename global_rank in rank
    local_rank: int = -1
    world_size: int = -1
    master_addr: str = ""
    # Master port (for multi-node SLURM jobs)
    master_port: int = -1

    # if different of -1, we use this world size to initialize torch distributed
    # instead of world size. To be deprecated in the future
    torch_world_size: int = -1

    # flag for debugging purposes, to be sure that the slurm is finalized
    finalized: bool = False

    @property
    def multi_gpu(self) -> bool:
        if self.torch_world_size > -1:
            return self.torch_world_size > 1
        return self.world_size > 1

    @property
    def is_master(self):
        return self.global_rank == 0

    def _check_and_mutate_args(self):
        assert self.finalized  # check that the config is done when we check it
        assert 0 <= self.global_rank < self.world_size, (
            self.global_rank,
            self.world_size,
        )
        assert self.torch_world_size <= self.world_size


def init_torch_distributed(cfg: SlurmConf, max_retry: int = 10, seconds: int = 300):
    assert max_retry > 0
    assert seconds > 0

    @timeout(seconds=seconds)
    def init_process_group():
        torch.distributed.init_process_group(init_method="env://", backend="nccl")

    if not cfg.multi_gpu:
        logger.warning("No multi gpu detected, don't initialize torch distributed")
        return

    logger.info(f"Initializing torch distributed with config: {cfg}")

    if cfg.is_slurm_job:
        # set environment variables for 'env://'
        os.environ["MASTER_PORT"] = str(cfg.master_port)
        os.environ["MASTER_ADDR"] = cfg.master_addr
        os.environ["RANK"] = str(cfg.global_rank)

        if cfg.torch_world_size > 0:
            assert cfg.torch_world_size < cfg.world_size
            logger.warning(
                f"Using torch world world size of {cfg.torch_world_size} "
                f"instead of {cfg.world_size} for torch distributed"
            )
            os.environ["WORLD_SIZE"] = str(cfg.torch_world_size)
        else:
            os.environ["WORLD_SIZE"] = str(cfg.world_size)

        assert 10001 <= cfg.master_port <= 20000 or cfg.world_size == 1

    if cfg.global_rank >= cfg.torch_world_size > 0:
        logger.warning(
            "Not initializing torch distributed for this worker :"
            f"{cfg.global_rank} >= {cfg.torch_world_size}"
        )
        raise RuntimeError("Should not be called for this global rank")

    # http://pytorch.apachecn.org/en/0.3.0/distributed.html#environment-variable-initialization
    # 'env://' will read these environment variables:
    # MASTER_PORT - required; has to be a free port on machine with rank 0
    # MASTER_ADDR - required (except for rank 0); address of rank 0 node
    # WORLD_SIZE - required; can be set either here, or in a call to init function
    # RANK - required; can be set either here, or in a call to init function

    NCCL_SOCKET_IFNAME = os.environ.get("NCCL_SOCKET_IFNAME", None)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"
    print(
        f"Initializing PyTorch distributed ... NCCL_SOCKET_IFNAME={NCCL_SOCKET_IFNAME}"
    )

    n_retry = 0
    while True:
        try:
            init_process_group()
            break
        except MyTimeoutError:
            logger.warning("torch.distributed.init_process_group timeout: retrying...")
            n_retry += 1
            if n_retry >= max_retry:
                raise
            else:
                continue


def sig_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    host = socket.gethostname()
    if "SLURM_PROCID" not in os.environ:
        logger.warning(f"Host: {host} - Not a SLURM process, cannot requeue")
        sys.exit(-1)
    prod_id = int(os.environ["SLURM_PROCID"])
    logger.warning(f"Host: {host} - Global rank: {prod_id}")
    if prod_id == 0:
        logger.warning("Requeuing job " + os.environ["SLURM_JOB_ID"])
        os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
        sys.exit(-1)
    else:
        logger.warning("Not the master process, no need to requeue.")
        # do not sys.exit, because it might kill the master task


def term_handler(signum, frame):
    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Bypassing SIGTERM.")


def init_signal_handler():
    """
    Handle signals sent by SLURM for time limit / pre-emption.
    """
    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, term_handler)
    logger.warning("Signal handler installed.")
