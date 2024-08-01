# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field, asdict
from typing import Optional, Union, List, Tuple, Set
from pathlib import Path
import os
import time
import shutil
import getpass
import submitit
import subprocess
import pandas
import re

from evariste import json
from evariste.envs.lean.utils import EXPECTED_HEADER
from evariste.logger import create_logger
from params import cfg_from_cli, Params
from evariste.utils import clusterify_path, rstr
from evariste.envs.eq.generation_lean import (
    ConfStore,
    RandomGenerationArgs,
    export_random_theorems,
    compute_generation_stats,
)
from evariste.datasets.lean import (
    LEAN_FULL_NAMES_DIR_V2_TACTIC_FULL_NAMES,
    LEAN_SPLITS,
    SYNTHETIC_DATA_DIR,
)


N_SHARDS = 8  # equal to number of GPUS, each local worker loads its id


@dataclass
class RandGenRunnerArgs(Params):
    n_jobs: int
    exp_name: str
    exp_id: Optional[str] = None
    root_dir: str = f"YOUR_PATH/generated"
    random_gen: RandomGenerationArgs = field(
        default_factory=lambda: ConfStore["lean_gen_rwalk_nat"]
    )
    local: bool = False
    max_parallel_jobs: int = 1000
    run_extraction: bool = True

    get_all_exercises: bool = False
    get_all_train: bool = False

    def __post_init__(self):
        assert self.n_jobs >= 1
        assert len(self.exp_name.strip()) > 0
        self.root_dir = clusterify_path(self.root_dir)


def generate_one(cfg: RandGenRunnerArgs, job_id: int) -> float:

    logger = create_logger(None)

    # create dump folder
    assert cfg.exp_id is not None
    assert 0 <= job_id < cfg.n_jobs
    start = time.time()
    dirname = f"{cfg.exp_id}_{job_id}"
    dump_path = Path(clusterify_path(f"{cfg.root_dir}/{cfg.exp_name}/{dirname}/"))
    logger.info(f"Starting generation in {dump_path} ...")
    os.makedirs(dump_path)

    # copy template folders
    shutil.copytree(
        "", dump_path, dirs_exist_ok=True,
    )

    # update generation path
    cfg.random_gen.path = str(dump_path / "src" / "generated.lean")

    # export theorems
    prefix: Optional[str] = None
    if cfg.exp_name is not None:
        prefix = f"{cfg.exp_name}_{job_id}"
    used_theorems = export_random_theorems(prefix, cfg.random_gen)
    logger.info(f"EXPORT DONE -- {time.time() - start:.2f} seconds")

    # print and export stats
    stats = compute_generation_stats(cfg.random_gen.rule_envs, used_theorems)
    logger.info(
        f"Usage: {stats['n_used_rules']}/{stats['n_rules']} ({100 * stats['usage']:.5f}%) -- "
        f"Entropy: {stats['entropy']:.5f}"
    )
    with open(dump_path / "generation_stats.json", "w") as f:
        json.dump(stats, f, sort_keys=True, indent=4)

    # running extraction pipeline
    if cfg.run_extraction:
        logger.info(f"\n========== Running extraction ...")
        with open(dump_path / "src" / "all.lean", "w") as f:
            f.write("import generated\n")
        os.makedirs(dump_path / "extracted")
        os.makedirs(dump_path / "tools")
        for file in os.listdir("lprp/tools"):
            if file.endswith(".py"):
                shutil.copy(f"lprp/tools/{file}", dump_path / "tools" / file)
        with open("lprp/tools/run_all.sh", "r") as f:
            with open(dump_path / "tools" / "run_all.sh", "w") as o:
                o.write(f.read().replace("DATA_DIR", str(dump_path / "extracted")))
        os.chdir(dump_path)
        cmd = f"source tools/run_all.sh f{dump_path / 'extracted'}"
        subprocess.call(["/bin/bash", "-i", "-c", cmd])

    total_time = time.time() - start
    logger.info(f"ALL DONE -- {total_time:.2f} seconds")
    return total_time


def load_one(root: Path) -> Optional[Tuple[int, int, pandas.DataFrame]]:
    """Loads one train csv file to be stitched"""

    csv_path = root / "extracted" / "cleaned_training_data" / "data_and_metadata.csv"
    if not csv_path.exists():
        return None

    df = pandas.read_csv(csv_path)
    with open(root / "extracted" / "lean_errors.json", "r") as f:
        errors = [x for x in json.loads(f.read()) if x["severity"] == "error"]

    # remove rows from dataframe at pos containing errors
    bad_th: Set[str] = set()
    for e in errors:
        try:
            bad_th.add(df[df.line == e["pos_line"]].iloc[0].decl_name)
        except IndexError:
            pass
    df = df[~df["decl_name"].isin(bad_th)]

    # remove rows from dataframe at pos containing sorry
    before_sorry = len(df)
    df = df[~df["human_tactic_code"].str.contains("sorry")]
    after_sorry = len(df)

    return len(bad_th), before_sorry - after_sorry, df


def launch_generators(cfg: RandGenRunnerArgs):
    if cfg.get_all_train:
        assert cfg.run_extraction, "extraction required to get train files"
        assert cfg.random_gen.no_proof is False, "train requires proofs to be generated"
    if cfg.exp_name in LEAN_SPLITS:
        print("split_name already in LEAN_SPLITS")
    data_dir = Path(LEAN_FULL_NAMES_DIR_V2_TACTIC_FULL_NAMES)
    split_file = data_dir / f"split.{cfg.exp_name}"
    lean_file = Path("YOUR_PATH") / f"{cfg.exp_name}.lean"
    if cfg.get_all_exercises:
        for some_f in [split_file, lean_file]:
            if some_f.exists():
                print(f"{some_f} already exists")
        print(f"This will write exercises to:\n{split_file}\n{lean_file}")

    synth_dir = Path(SYNTHETIC_DATA_DIR)
    if cfg.get_all_train:
        p = synth_dir / f"{cfg.exp_name}"
        assert not p.exists(), f"{p} exists !"
        print(
            f"This will write train files to:\n{synth_dir / f'{cfg.exp_name}'/ '{{shard}}.csv'}"
        )

    # random generator batch ID
    assert cfg.exp_id is None
    cfg.exp_id = rstr(6)

    # workdir folder
    job_folder = f"/gen_workdir/{cfg.exp_name}/{cfg.exp_id}"
    executor: Union[submitit.AutoExecutor, submitit.LocalExecutor]
    if cfg.local:
        executor = submitit.LocalExecutor(folder=job_folder)
    else:
        executor = submitit.AutoExecutor(folder=job_folder, slurm_max_num_timeout=-1)

    executor.update_parameters(
        slurm_job_name=f"synth_gen__{cfg.exp_name}",
        slurm_array_parallelism=cfg.max_parallel_jobs,
        slurm_partition="Theorem_Proving",
        slurm_cpus_per_task=1,
        slurm_mem_gb=8,
        timeout_min=60 * 8,  # in minutes
    )
    args = [cfg for _ in range(cfg.n_jobs)]
    jobs = executor.map_array(generate_one, args, range(cfg.n_jobs))

    # export params
    with open(os.path.join(job_folder, "params.json"), "w") as f:
        json.dump(asdict(cfg), f, sort_keys=True, indent=4)

    print(f"Starting {cfg.n_jobs} jobs in {job_folder} ...")

    for j in jobs:
        try:
            res = j.result()
            print(f"Done in {res:.2f}s")
        except Exception as e:
            print(f"failed {str(e)}")

    print("Waiting for all stdouts to be written...")
    time.sleep(5)  # hopefully this is enough for all stdouts to be written ?
    stdouts: List[str] = []
    for j in jobs:
        stdout = j.stdout()
        assert stdout is not None
        stdouts.append(stdout)

    if cfg.get_all_exercises:
        final_str = EXPECTED_HEADER
        all_labels = set()
        for stdout in stdouts:
            m = re.search(r"Exporting theorems to (?P<path>.*) ...\n", stdout)
            assert m is not None, stdout
            generation = m.group("path")
            with open(generation, "r") as f:
                content = f.read()
                final_str += content[len(EXPECTED_HEADER) :]
                all_labels.update(re.findall("theorem (?P<label>.*)\n", content))
        with open(lean_file, "w") as f:
            f.write(final_str)
        with open(split_file, "w",) as f:
            for x in all_labels:
                f.write(x + "\n")
        print(
            f"Generated {len(all_labels)} theorems in files {lean_file} // {split_file}"
        )

    if cfg.get_all_train:
        to_stack = []
        for i, stdout in enumerate(stdouts):
            m = re.search(r"Exporting theorems to (?P<path>.*) ...\n", stdout)
            assert m is not None, stdout
            root = Path(m.group("path")).parents[1]  # root generation folder
            if i % 10 == 0:
                print(f"Stitching worker output {i}")
            to_stitch = load_one(root)
            if to_stitch is None:
                print(f"{root} failed")
                continue
            to_stack.append(to_stitch)
        print(f"Found {len(to_stack)} CSV files")

        if len(to_stack) == 0:
            return

        bads, sorrys, dfs = zip(*to_stack)
        train_folder = Path(synth_dir / f"{cfg.exp_name}")
        os.makedirs(train_folder, exist_ok=False)
        print(f"Stitching into final csv in {train_folder} ...")
        final = pandas.concat(dfs)
        for shard in range(N_SHARDS):
            final[final.index % N_SHARDS == shard].to_csv(
                train_folder / f"{cfg.exp_name}.{shard}.csv"
            )
        print(
            f"Final rows : {len(final)}, removed {sum(sorrys)} sorrys "
            f"and {sum(bads)} bad theorems"
        )


if __name__ == "__main__":

    cfg = RandGenRunnerArgs(n_jobs=1, exp_name="debug")
    cfg = cfg_from_cli(cfg)
    launch_generators(cfg)
