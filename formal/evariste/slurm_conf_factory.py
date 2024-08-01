# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import subprocess
import os
from logging import getLogger
from typing import Optional

from evariste.trainer.args import TrainerArgs
from evariste.slurm import SlurmConf

logger = getLogger()


def from_trainer_args(params: TrainerArgs) -> SlurmConf:
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - global_rank
        - world_size
        - master_address
        - master_port
    """
    is_slurm_job = "SLURM_JOB_ID" in os.environ and not params.debug.slurm
    logger.info(f"is_slurm_job: {is_slurm_job}")
    logger.info(f"params.debug.slurm: {params.debug.slurm}")

    if is_slurm_job:
        assert (
            params.slurm_conf.local_rank == -1
        )  # on the cluster, this is handled by SLURM
        cfg = from_slurm_env(
            master_port=params.master_port,
            torch_world_size=params.slurm_conf.torch_world_size,
        )
    # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
    elif params.local_rank != -1:
        assert params.exp_id, "exp_id should be set for torch.distributed.launch jobs"
        assert params.slurm_conf.master_port == -1
        cfg = from_torch_launcher_env(local_rank=params.local_rank)
    else:
        assert params.local_rank == -1
        assert params.master_port == -1
        if params.debug.debug and params.debug.rank >= 0:
            assert params.debug.world_size > 0
            local_rank = params.debug.rank
            global_rank = params.debug.rank
            world_size = params.debug.world_size
            cfg = from_cli(local_rank, global_rank, world_size)
            cfg.torch_world_size = params.slurm_conf.torch_world_size
            logger.warning(f"Debug: creating a slurm config {cfg}")
        else:
            cfg = from_cli()

    return cfg


def from_slurm_env(
    master_port: int,
    global_rank: Optional[int] = None,
    world_size: Optional[int] = None,
    master_node_id: Optional[int] = None,
    torch_world_size: int = -1,
):
    world_size = int(os.environ["SLURM_NTASKS"]) if world_size is None else world_size
    global_rank = (
        int(os.environ["SLURM_PROCID"]) if global_rank is None else global_rank
    )
    master_node_id = 0 if master_node_id is None else master_node_id

    assert 10001 <= master_port <= 20000 or world_size == 1 or torch_world_size == 1
    hostnames = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
    )

    return SlurmConf(
        is_slurm_job=True,
        local_rank=int(os.environ["SLURM_LOCALID"]),
        global_rank=global_rank,
        world_size=world_size,
        torch_world_size=torch_world_size,
        master_addr=hostnames.split()[master_node_id].decode("utf-8"),
        master_port=master_port,
        finalized=True,
    )


def from_torch_launcher_env(local_rank: int) -> SlurmConf:
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert global_rank >= 0, global_rank
    assert world_size >= 1, world_size
    return SlurmConf(
        is_slurm_job=False,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
        master_addr="",
        master_port=-1,
        finalized=True,
    )


def from_cli(
    local_rank: Optional[int] = None,
    global_rank: Optional[int] = None,
    world_size: Optional[int] = None,
):
    local_rank = 0 if local_rank is None else local_rank
    global_rank = 0 if global_rank is None else global_rank

    world_size = 1 if world_size is None else world_size

    return SlurmConf(
        is_slurm_job=False,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
        master_addr="",
        master_port=-1,
        finalized=True,
    )
