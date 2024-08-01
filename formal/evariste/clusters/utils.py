# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import re
import subprocess


def clusterify_path(path: str) -> str:
    return path


def clusterify_partitions(partitions: str) -> str:
    return "Theorem_Proving"


def get_max_timeout(partitions: str) -> int:
    """
    Return the maximum time allowed on a set of partitions.
    """
    return 24 * 60


def get_running_partition() -> str:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None:
        return "NO_SLURM_JOB_ID"
    cmd = ["squeue", "--noheader", "-j", job_id, "--Format=Partition"]
    sp = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    return out_str[0].decode("utf-8").strip()
