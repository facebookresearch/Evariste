# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from dataclasses import dataclass, asdict
import getpass
import os.path
from typing import Optional
from pathlib import Path

import submitit

from evariste import json
from evariste.adversarial_offline.generator import GeneratorArgs, OfflineGenerator
from evariste.clusters.utils import clusterify_path
from evariste.utils import logged_closing, rstr
from params import Params, ConfStore, cfg_from_cli

from evariste.forward.fwd_eq.gen.proof_search import EqGenProofSearch


@dataclass
class GenRunnerArgs(Params):
    n_jobs: int
    exp_name: str
    gen_args: GeneratorArgs
    exp_id: Optional[str] = None
    root_dir: str = f"YOUR_PATH/generated"
    local: bool = False
    max_parallel_jobs: int = 1000
    slurm_timeout_min: int = 60 * 24
    seed: int = 1764

    def __post_init__(self):
        assert self.n_jobs >= 1
        assert len(self.exp_name.strip()) > 0
        self.root_dir = clusterify_path(self.root_dir)
        assert os.path.isdir(self.root_dir)


def launch_generators(
    cfg: GenRunnerArgs, exp_id: Optional[str] = None, job_folder: Optional[str] = None,
):
    """Adapted from maxi_gen"""
    # random experiment ID if not already set elsewhere
    assert exp_id or (cfg.exp_id is None)
    cfg.exp_id = exp_id if exp_id else rstr(6)

    def start_job(job_id: int):
        assert 0 <= job_id < cfg.n_jobs
        label = f"{cfg.exp_id}_{job_id}"
        dump_path = Path(clusterify_path(f"{cfg.root_dir}/{cfg.exp_name}/{label}/"))
        # update gen_args
        cfg.gen_args.dump_path = dump_path
        cfg.gen_args.job_id = job_id
        cfg.gen_args.n_jobs = cfg.n_jobs
        cfg.gen_args.label = label
        cfg.gen_args.seed = cfg.seed + job_id
        res = None
        gen = OfflineGenerator(cfg.gen_args)
        try:
            with logged_closing(gen, "OfflineGenerator") as gen:
                res = gen.do_filter_generations()
                for _a, _b, c, _d in res:
                    if isinstance(c, EqGenProofSearch):
                        c.clear()
            logging.info(f"Received {len(res)} generations.")
        except Exception as e:
            print(e)
            raise e
        return res

    # workdir folder
    job_folder_str = (
        job_folder if job_folder else f"/gen_workdir/{cfg.exp_name}/{cfg.exp_id}"
    )
    job_folder_path = Path(job_folder_str)

    # export params
    job_folder_path.mkdir(parents=True, exist_ok=True)
    with open(job_folder_path / "params.json", "w") as f:
        json.dump(asdict(cfg), f, sort_keys=True, indent=4)

    logging.info(f"Starting {cfg.n_jobs} jobs in {job_folder_path} ...")

    if cfg.local:
        assert cfg.n_jobs == 1
        start_job(job_id=0)
        # executor = submitit.LocalExecutor(folder=job_folder)    # problem with CUDA
    else:
        executor = submitit.AutoExecutor(
            folder=job_folder_path, slurm_max_num_timeout=-1
        )
        ## if sequential only requires 1 GPU since we first generate then prove on the same GPU
        seq = cfg.gen_args.sequential
        executor.update_parameters(
            slurm_job_name=f"offline_gen__{cfg.exp_name}",
            slurm_array_parallelism=cfg.max_parallel_jobs,
            # slurm_partition="Theorem_Proving,devaccel",
            slurm_partition="Theorem_Proving",
            slurm_cpus_per_task=10 if seq else 2,
            slurm_gpus_per_task=1 if seq else 2,
            slurm_ntasks_per_node=1,
            slurm_mem_gb=60 if seq else 120,
            slurm_srun_args=["-vv"],
            slurm_timeout_min=cfg.slurm_timeout_min,  # in minutes
        )
        jobs = executor.map_array(start_job, range(cfg.n_jobs))

        results = []
        for j in jobs:
            try:
                results.append(j.result())
            except Exception as e:
                logging.info(f"failed: {str(e)}")
        return results


if __name__ == "__main__":
    gen_args: GeneratorArgs = ConfStore["gen_args_512_512_greedy"]
    gen_args.save_chunk_length = 1
    cfg = GenRunnerArgs(n_jobs=1, exp_name="gen_test", gen_args=gen_args)
    cfg = cfg_from_cli(cfg)
    launch_generators(cfg)
