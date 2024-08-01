# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from evariste import json as json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from logging import getLogger
from typing import TypeVar, Callable, Type, Tuple


from submitit import JobEnvironment

MAX_TRIALS = 100

logger = getLogger()
P = TypeVar("P")


def retry(
    fn: Callable[[], P],
    error_types: Tuple[Type[Exception], ...],
    name: str,
    retry_on_cuda_error: bool = True,
    max_trials: int = MAX_TRIALS,
) -> P:
    trial = 0
    while trial < max_trials:
        try:
            if trial > 0:
                logger.info(f"Retrying {name} [{trial}/{max_trials}]")
            return fn()
        except error_types as err:
            logger.exception(f"Catching this exception in {name}: {err}")
        except RuntimeError as err:
            if retry_on_cuda_error and ("CUDA error" in str(err)):
                logger.exception(f"Catching CUDA error in {name}: {err}")
            else:
                logger.exception(f"Not catching this exception in {name}: {err}")
                raise
        trial += 1
    raise RuntimeError("not reachable")


def reschedule_if_oom(fn: Callable[[], P], name: str) -> Callable[[], P]:
    """
    Sometimes, when relaunching actors because of DeadLean it seems that some
    memory was not correctly freed on gpu (remaining processes?),
    creating a lot of GPU oom errors (even with batch size 1).
    In this case we reschedule the job (if this happen too often we should investigate
    more on this
    """

    def _fn() -> P:
        try:
            return fn()
        except RuntimeError as err:
            if "out of memory" in str(err):
                logger.warning("Requeuing job " + os.environ["SLURM_JOB_ID"])
                os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
                time.sleep(20)
                sys.exit(-1)
            else:
                raise

    return _fn


def dump_safely_state(state, dst_path: Path):
    tmp_dst_path = dst_path.parent / f"{dst_path.name}_tmp_{uuid.uuid4().hex}"
    with tmp_dst_path.open("w") as fp:
        json.dump(state, fp)
    os.rename(tmp_dst_path, dst_path)  # atomic


def _get_state(state_folder, job_id):
    state_path = state_folder / f"{job_id}_state.json"
    if not state_path.exists():
        state = {"job_id": job_id, "n_retry": 0}
    else:
        with state_path.open("r") as fp:
            state = json.load(fp)
    assert state["job_id"] == job_id
    return state


def _save_state(state, state_folder, job_id):
    state_path = state_folder / f"{job_id}_state.json"
    dump_safely_state(state, state_path)


def retry_submitit(func: Callable, max_retries: int, state_folder: Path):
    """
    :param state_folder: Path to the folder where the state (n_retry here) of the
    job will be stored. Needs to be on the NFS to be accessible from all workers
    """

    def _wrapped_func(*args, **kwargs):
        logging.basicConfig(level=logging.INFO)
        state_folder.mkdir(exist_ok=True, parents=True)  # maybe concurrency issues?
        job_env = JobEnvironment()
        state = _get_state(state_folder, job_id=job_env.job_id)

        logger.info(f"Executing {func} - Try {state['n_retry']}")
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.exception("Job failed with error:")
            pass

        state["n_retry"] += 1
        _save_state(state, state_folder, job_env.job_id)

        if state["n_retry"] <= max_retries:
            logger.info(f"Requeuing {job_env.job_id}")
            job_env._requeue(-1)
            sys.exit(-1)
        else:
            logger.info(f"Max retry ({max_retries}) reached for {func} - aborting")

    return _wrapped_func
