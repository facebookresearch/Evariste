# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Iterator
import contextlib
import os


@contextlib.contextmanager
def reset_slurm_env() -> Iterator[None]:
    """
        Temporarily unset slurm ids to avoid mistakenly inheriting.
    """
    old_environ = {
        x: os.environ.pop(x, None)
        for x in (
            "SLURM_JOB_ID",
            "SLURM_NTASKS",
            "SLURM_JOB_NUM_NODES",
            "SLURM_NODEID",
            "SLURM_JOB_NODELIST",
            "SLURM_PROCID",
            "SLURM_LOCALID",
            "SLURM_ARRAY_JOB_ID",
            "SLURM_ARRAY_TASK_ID",
        )
    }
    try:
        yield
    finally:
        for k, val in old_environ.items():
            if val is not None:
                os.environ[k] = val
