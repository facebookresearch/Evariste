# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, asdict
from pprint import pprint
from pathlib import Path
import copy
import pickle
import submitit

from evariste import json as json
from evariste.utils import prepare
from params import Params


def prepare_folder(cfg: Params, verbose: bool = True):
    assert hasattr(cfg, "output_path")
    path = Path(cfg.output_path)  # type: ignore
    path.mkdir()
    config_json = path / "config.json"
    config_pkl = path / "config.pkl"
    if verbose:
        pprint(asdict(cfg))
    with config_json.open("w") as fp:
        json.dump(asdict(cfg), fp, indent=2)
    with config_pkl.open("wb") as fp2:
        pickle.dump(cfg, fp2)


def load_config_dict(path: Path) -> Dict[str, Any]:
    config_json = path / "config.json"
    with config_json.open("r") as fp:
        return json.load(fp)


def launch_with_submitit(
    fn: Callable,
    cfg: Params,
    copy_workdir: bool = True,
    verbose: bool = True,
    exp_name: Optional[str] = None,
):
    assert hasattr(cfg, "output_path")
    assert hasattr(cfg, "n_jobs")
    assert hasattr(cfg, "partition")
    assert hasattr(cfg, "rank")
    assert hasattr(cfg, "world_size")
    # maybe put it somewhere else?
    submitit_folder = Path(cfg.output_path) / "submitit"  # type: ignore
    submitit_folder.mkdir(parents=True)
    assert cfg.n_jobs > 0, cfg.n_jobs  # type: ignore

    mem_gb = getattr(cfg, "slurm_mem_gb", 50)
    if verbose and mem_gb > 50:
        print(f"Using mem_gb: {mem_gb}")

    executor = submitit.AutoExecutor(folder=submitit_folder)
    if exp_name is None:
        exp_name = Path(cfg.output_path).name  # type: ignore
    executor.update_parameters(
        slurm_job_name=exp_name,
        timeout_min=60 * 5,
        gpus_per_node=1,
        cpus_per_task=10,
        slurm_ntasks_per_node=1,
        mem_gb=getattr(cfg, "slurm_mem_gb", 50),
        slurm_partition=cfg.partition,  # type: ignore
    )
    if copy_workdir:
        if verbose:
            print(f"submitit_folder: {submitit_folder}")
        prepare(exp_name)

    jobs = []
    cfgs = []
    with executor.batch():
        for rank in range(cfg.n_jobs):  # type: ignore
            job_cfg = copy.deepcopy(cfg)
            job_cfg.output_path = str(Path(cfg.output_path) / f"job_{rank:03d}")  # type: ignore
            job_cfg.rank = rank  # type: ignore
            job_cfg.world_size = cfg.n_jobs  # type: ignore
            cfgs.append(job_cfg)
            prepare_folder(job_cfg, verbose=False)
            job = executor.submit(fn, job_cfg)
            jobs.append(job)
    for job, job_cfg in zip(jobs, cfgs):
        if verbose:
            print(
                f"job: {job},"
                f"{job_cfg},"
                f"stdout: {job.paths.stdout},"
                f"stderr: {job.paths.stderr}"
            )
