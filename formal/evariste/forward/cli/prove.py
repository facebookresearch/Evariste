# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import itertools
from evariste import json as json
import logging
import time
import getpass
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Optional, Iterator, Tuple, List, cast
import datetime

import numpy as np
import torch

import tqdm

from evariste.comms.zip_store import ZipStore
from evariste.forward.fwd_mm import mm_helpers
from evariste.forward.common import (
    ForwardGoal,
    GenerationHistory,
    GenerationInfos,
    GenerationError,
)
from evariste.forward.forward_prover import ProverConfig
from evariste.forward import forward_model_factory, forward_model_configs
from evariste.forward.proof_search import StandardProofSearch
from evariste.forward.training.helpers import count_unique_theorems
from evariste.forward.utils.retry import retry
from params.params import cfg_from_cli, Params, ConfStore
from evariste.forward.utils.launch_utils import prepare_folder, launch_with_submitit
from evariste.forward.forward_model_configs import ModelConfig


@dataclass
class Config(Params):
    model: ModelConfig
    prover: ProverConfig
    seed: int = 104
    split: str = "valid"
    max_samples: Optional[int] = None
    rank: int = 0
    world_size: int = 1
    debug: bool = False
    tmp: bool = False
    slurm: bool = False
    n_jobs: int = 0
    n_trials_by_trm: int = 1
    dataset_name: Optional[str] = None
    continue_path: Optional[str] = None
    partition: str = "Theorem_Proving"
    output_path: str = "to_replace"
    goal_path: str = ""
    slurm_mem_gb: int = 50


@dataclass
class _Stats:
    done: int = 0
    n_solved: int = 0
    start_time: float = field(default_factory=lambda: time.time())
    time_in_storing: float = 0

    @property
    def duration(self):
        return time.time() - self.start_time


BASE_OUTPUT_FOLDER = Path(f"data/evariste/proving/")


def prove(cfg: Config):
    torch.manual_seed(cfg.seed + cfg.rank)

    output_path = Path(cfg.output_path)
    if cfg.slurm:
        run_uuid = output_path.parent.name
    else:
        run_uuid = output_path.name
    store = ZipStore(output_path)
    # in case of restarting after requeing we restart from scratch
    store.clean_existing(zip_names=["generations"], jsonl_names=["infos"])

    dataset_conf = None
    if cfg.dataset_name:
        dataset_conf = ConfStore[cfg.dataset_name]

    prover, dico, params, env_helper = forward_model_factory.from_checkpoint(
        ckpt_path=cfg.model.ckpt,
        device_str="cuda",
        cfg=cfg.prover,
        overwrite_dataset=dataset_conf,
        prover_dir=output_path,
    )
    if cfg.goal_path == "":
        goal_factory = env_helper.get_goal_factory()
        goals: List[ForwardGoal] = goal_factory.build_forward_goals(
            split=cfg.split, debug=cfg.debug
        )
    else:
        goals = mm_helpers.build_forward_goals_from_samples(
            data_path=cfg.goal_path,
            debug=cfg.debug,
        )
    data = select_goals(cfg, goals)

    def inputs() -> Iterator[Tuple[int, ForwardGoal]]:
        for i, (_, goal, _) in enumerate(data):
            yield i, goal

    def _log(stats: _Stats):
        print(
            f"Solved: {stats.n_solved}, done: {stats.done}, total: {len(data)} "
            f"({100 * stats.n_solved / len(data):.01f}% solved) "
            f"- duration: {stats.duration:.02f}s "
            f"- storing: {stats.time_in_storing:.02f}s"
        )

    stats = _Stats()
    with closing(prover), closing(store):
        proof_iterator = prover.generate_proofs(inputs())
        for i, proof in tqdm.tqdm(proof_iterator, total=len(data)):
            start_storing = time.time()
            if isinstance(proof, GenerationError):
                print(f"Error with proof {i}: {proof}")
                continue

            assert isinstance(proof, StandardProofSearch)

            gen = proof.generation
            gen_info = proof.info
            assert isinstance(gen, GenerationHistory)
            assert isinstance(gen_info, GenerationInfos)

            solved = gen_info.solved
            stats.n_solved += int(solved)
            stats.done += 1
            gen_id, goal, trial = data[i]
            name = goal.label
            if solved:
                assert gen.stack[-1].step is not None, gen
                assert gen.goal.is_solved_by_step(gen.stack[-1].step)

            proof = gen.solving_proof_tree()
            solved_size = count_unique_theorems(proof)

            info = {
                "name": name,
                "gen_id": gen_id,
                "run_uuid": run_uuid,
                "gen_uuid": f"{run_uuid}_{gen_id}",
                "trial": trial,
                "n_valid": len(gen.forward_steps()),
                "n_invalid": len(gen.errors()),
                "stopped": gen_info.stopped,
                "solved": gen_info.solved,
                "solved_size": solved_size,
            }

            store.store_in_jsonl(obj=info, filename="infos")
            if gen.goal.forbidden:
                gen.goal.forbidden = {"<REMOVED/>"}  # we remove forbidden
                # to save space. They can be obtained back with label field
            store.store_in_pickle_zip(obj=gen, zip_name="generations")
            stats.time_in_storing += time.time() - start_storing

            if stats.done % 10 == 0:
                _log(stats)

    _log(stats)
    pprint(prover.stats)
    with store.path("prover_stats.json").open("w") as fp:
        json.dump(prover.stats, fp)

    _analyze(cfg)

    print(f"Output folder: {cfg.output_path}")


def try_prove(cfg: Config):
    outpath = Path(cfg.output_path)
    (outpath / "started").touch()
    try:
        prove_with_retry_if_dead_lean(cfg)
    except Exception:
        (outpath / "failed").touch()
        raise
    else:
        (outpath / "done").touch()


def prove_with_retry_if_dead_lean(cfg: Config):
    from leanml import DeadLean

    retry(
        fn=lambda: prove(cfg),
        error_types=(DeadLean,),
        name="prove",
        max_trials=3,
    )


def _analyze(cfg: Config):
    store = ZipStore(Path(cfg.output_path))
    generations = store.read_pickle_zip("generations")
    infos = store.read_jsonl("infos")

    total = len(infos)
    solved = len([item for item in infos if item["solved"]])
    print(total, solved, solved / total)


GoalId = int
TrialId = int


def select_goals(
    cfg: Config, goals: List[ForwardGoal]
) -> List[Tuple[GoalId, ForwardGoal, TrialId]]:

    if cfg.continue_path:
        solved = load_solved(Path(cfg.continue_path))
        print(f"N previously solved: {len(solved)}")
        goals = [g for g in goals if g.label not in solved]
        print(f"N thrms not solved: {len(goals)}")

    rng = np.random.RandomState(cfg.seed)
    rng.shuffle(goals)
    if cfg.max_samples:
        goals = goals[: cfg.max_samples]
        print(f"Keeping only {len(goals)} theorems")

    data = [
        (gen_id, goal, trial)
        for gen_id, (trial, goal) in enumerate(
            itertools.product(range(cfg.n_trials_by_trm), goals)
        )
    ]
    if cfg.world_size > 1:
        data = [s for s in data if s[0] % cfg.world_size == cfg.rank]
    print(f"N samples: {len(data)}")

    return data


def _make_output_path(cfg: Config) -> str:
    output_folder = BASE_OUTPUT_FOLDER
    output_folder.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{cfg.model.name}_{cfg.prover.name}_{cfg.split}"
    if cfg.n_trials_by_trm > 1:
        name += f"_t{cfg.n_trials_by_trm}"
    if cfg.continue_path:
        assert Path(cfg.continue_path).exists()
        name += "_cont"
    name += f"_{now}"
    if cfg.tmp:
        name = f"tmp_{name}"
    return str(output_folder / name)


def main(cfg: Config):
    prepare_folder(cfg)

    if cfg.slurm:
        launch_with_submitit(prove, cfg=cfg)
    else:
        prove(cfg=cfg)


def load_solved(previous_exp: Path):
    assert previous_exp.exists(), previous_exp

    job_dirs = [p for p in previous_exp.iterdir() if p.name.startswith("job_")]
    if len(job_dirs) == 0:
        job_dirs = [previous_exp]

    infos = []
    for job_path in job_dirs:
        store = ZipStore(job_path)
        infos.extend(store.read_jsonl("infos"))

    return {info["name"] for info in infos if info["solved"]}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    forward_model_configs.register_model_cfgs()
    forward_model_configs.register_prover_cfgs()
    cfg = cfg_from_cli(schema=Config)
    cfg = cast(Config, cfg)
    if cfg.debug:
        cfg.tmp = True
    output_path = _make_output_path(cfg)
    cfg.output_path = output_path
    main(cfg)
