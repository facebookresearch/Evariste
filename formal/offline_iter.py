# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import asdict, dataclass, field
import getpass
import logging
from pathlib import Path
import numpy as np

import submitit
from evariste.adversarial_offline.adv_gen import GenRunnerArgs, launch_generators
from evariste.adversarial_offline.args import GeneratorArgs
from evariste.model.utils import reload_ckpt
from evariste.trainer import launch
from evariste.trainer.args import TrainerArgs
from evariste import json
from evariste.utils import rstr
from evariste.slurm_conf_factory import from_trainer_args
from evariste.model.checkpoints import get_latest_checkpoint
from evariste.model.data.envs.node_sampling_strategy import NodeSamplingStrategy
from params.params import Params, cfg_from_cli
from evariste.slurm import SlurmConf
from evariste.trainer.launch import launch_train

GEN_CFG = GeneratorArgs()
GEN_CFG.prover_params.beam_path = Path("")
GEN_CFG.forward_cfg.search_cfg.max_nodes = 200
GEN_CFG.forward_cfg.search_cfg.max_generations = 200
GEN_CFG.forward_cfg.search_cfg.max_cons_inv_allowed = 200
GEN_CFG.forward_cfg.decoding_params.fixed_temperature = 1.0
GEN_CFG.forward_cfg.search_cfg.n_simultaneous_proofs = 100
GEN_CFG.return_all = True
GEN_CFG.sample_cfg.size_weight = 1.0
GEN_CFG.n_gens = 100
GEN_CFG.save_chunk_length = 10
GEN_CFG.verbose = True
GEN_RUNNER_CFG = GenRunnerArgs(
    n_jobs=1,
    max_parallel_jobs=1000,
    slurm_timeout_min=60 * 24,
    gen_args=GEN_CFG,
    exp_name="MISSING",
)


@dataclass
class RetrainArgs(Params):
    synthetic_weight: str = "1,1"  # offline/synthetic
    epoch_size: int = 10000
    max_epoch: int = 3
    retrain: bool = True
    retrain_on_all_previous: bool = False


@dataclass
class OfflineIterArgs(Params):
    exp_name: str = "debug_offline_iter"
    generator_path: Path = Path("")
    prover_path: Path = Path("")
    gen_runner_args: GenRunnerArgs = field(default_factory=lambda: GEN_RUNNER_CFG)
    iterations: int = 10
    train_nodes: int = 1
    train_timeout_min: int = 120
    offline_dataset_splits: str = "0.85,0.15,0.0"

    train_gen: RetrainArgs = field(default_factory=lambda: RetrainArgs())
    train_prover: RetrainArgs = field(default_factory=lambda: RetrainArgs())

    proved_conditioning: bool = False
    gen_hard_threshold: float = 0
    best_valid: bool = False


def generate_hard(
    args: OfflineIterArgs,
    generator_path: Path,
    prover_path: Path,
    exp_id: str,
    exp_name: str,
    job_folder: Path,
) -> float:
    cfg = args.gen_runner_args
    cfg.gen_args.gen_checkpoint_path = generator_path
    cfg.gen_args.prover_params.beam_path = prover_path
    cfg.gen_args.prefix_with_unproved = args.proved_conditioning
    cfg.exp_name = exp_name
    # start jobs with submitit
    res = launch_generators(cfg, exp_id=exp_id, job_folder=str(job_folder))
    assert res is not None
    total = sum(len(r) for r in res)
    proved = sum(x[3] for r in res for x in r)
    if total != 0:
        logging.info(
            f"Received {total - proved} / {total} = {(total - proved) / total * 100:.1f} % hard statements"
        )
        return (total - proved) / total
    return 0.0


def retrain(
    args: OfflineIterArgs,
    params: TrainerArgs,
    generation_path: str,
    exp_name: str,
    job_folder: Path,
):
    # Update params for this iteration:
    params.eq.dataset.offline_dataset_path = generation_path
    params.master_port = np.random.randint(10001, 20001)
    params.slurm_conf = SlurmConf()
    params.use_checkpoint_dico = True
    logging.info(f"Launching retraining in {job_folder}...")
    logging.info(f"Set master_port to {params.master_port}")

    job_folder.mkdir(exist_ok=True, parents=True)

    # export params
    with open(job_folder / "params.json", "w") as f:
        json.dump(asdict(params), f, sort_keys=True, indent=4)

    return launch_train(
        dump_path=job_folder,
        slurm_job_name=exp_name,
        exp_name=exp_name,
        trainer_args=params,
        partition="Theorem_Proving",
        n_nodes=args.train_nodes,
        timeout_min=args.train_timeout_min,
    )


def make_train_args(
    exp_id: str,
    params: TrainerArgs,
    retrain_args: RetrainArgs,
    is_gen: bool,
    offline_args: OfflineIterArgs,
):
    params.exp_name = offline_args.exp_name
    params.exp_id = exp_id
    params.max_epoch = retrain_args.max_epoch
    params.epoch_size = retrain_args.epoch_size
    params.eq.dataset.offline_dataset_splits = offline_args.offline_dataset_splits
    params.eq.dataset.gen_type = params.eq.dataset.gen_type or "graph"
    params.eq.proved_conditioning = offline_args.proved_conditioning
    params.generation_eval_str = ""

    params.validation_metrics = (
        f"valid-eq_{'newgen' if is_gen else 'bwd_newgen'}_graph_offline_seq2seq-tok-ppl"
    )
    params.stopping_criterion = f"_{params.validation_metrics},3"

    if is_gen:
        if not offline_args.proved_conditioning:
            params.tasks = "eq_newgen_graph_offline_seq2seq,eq_newgen_graph_seq2seq"
            params.tasks_weight = retrain_args.synthetic_weight
        else:
            # with proved conditioning try without synthetic task (maybe enough )
            params.tasks = "eq_newgen_graph_offline_seq2seq"
            params.tasks_weight = ""
    else:
        params.tasks = "eq_bwd_newgen_graph_offline_seq2seq,eq_bwd_newgen_graph_seq2seq"
        params.tasks_weight = retrain_args.synthetic_weight
        params.eq.dataset.fwd_offline_sampling_strategy = (
            NodeSamplingStrategy.AllMinimal
        )

    params.root_dump_path = "PATH/USERNAME/dumped/"
    params.__post_init__()


def maybe_latest(p: Path):
    if p.stem == "LATEST":
        latest = get_latest_checkpoint(p.parent)[0]
        print(f"Using latest: {latest}")
        assert latest is not None
        return Path(latest)


def main(args: OfflineIterArgs):
    exp_name = args.exp_name
    exp_id = rstr(6)
    generator_path = maybe_latest(args.generator_path)
    prover_path = maybe_latest(args.prover_path)
    # take the same Trainer args for generator retraining as in the checkpoint
    params, _, _ = reload_ckpt(ckpt_path=generator_path, only_model_params=False)
    assert isinstance(params, TrainerArgs)
    make_train_args(exp_id, params, args.train_gen, is_gen=True, offline_args=args)

    prover_params, _, _ = reload_ckpt(ckpt_path=prover_path, only_model_params=False)
    assert isinstance(prover_params, TrainerArgs)
    make_train_args(
        exp_id, prover_params, args.train_prover, is_gen=False, offline_args=args
    )

    logging.info(f"Experiment name: {exp_name}")
    logging.info(f"Experiment ID: {exp_id}")
    logging.info(f"All parameters:")
    logging.info(json.dumps(asdict(args), sort_keys=True, indent=4))

    for i in range(args.iterations):
        logging.info(f"============ Starting iteration {i} ============")
        # generate and filter statements
        job_folder_gen = Path(f"/gen_workdir/{exp_name}_gen/{exp_id}/{str(i)}")
        gen_exp_name = f"{args.exp_name}_gen_{i}"
        logging.info(f"Launching generator {gen_exp_name}...")
        logging.info(f"Checkpoint: {generator_path}")
        args.gen_runner_args.seed += 1

        old_proved_conditioning = args.proved_conditioning
        # model is trained without proved conditioning, generate first epoch without
        if i == 0 and args.proved_conditioning:
            args.proved_conditioning = False
        hard_prop = generate_hard(
            args=args,
            generator_path=Path(generator_path),
            prover_path=prover_path,
            exp_id=exp_id,
            exp_name=gen_exp_name,
            job_folder=job_folder_gen,
        )
        args.proved_conditioning = old_proved_conditioning

        retrain_jobs = []
        generation_root = f"YOUR_PATH/generated"

        for is_gen, retrain_args, these_params in [
            (True, args.train_gen, params),
            (False, args.train_prover, prover_params),
        ]:
            if not is_gen and hard_prop < args.gen_hard_threshold:
                logging.info(
                    f"Skipping prover retraining {hard_prop} < {args.gen_hard_threshold} (hard% below threshold)"
                )
                continue
            # retrain generator on hard statements
            if retrain_args.retrain:
                suffix = {True: "train", False: "prover_train"}[is_gen]

                to_reload = {True: generator_path, False: prover_path}[is_gen]

                job_folder_train = Path(
                    f"/gen_workdir/{exp_name}_{suffix}/{exp_id}/{str(i)}"
                )
                these_params.override_dump_path = (
                    f"YOUR_PATH/dumped/{exp_name}_{suffix}/{exp_id}/{str(i)}"
                )
                train_exp_name = f"{exp_name}_{suffix}_{i}"
                these_params.reload_checkpoint = str(to_reload)

                generation_glob = {
                    True: f"{args.exp_name}_gen_*",
                    False: f"{gen_exp_name}",
                }[retrain_args.retrain_on_all_previous]
                generation_glob = (
                    f"{generation_root}/{generation_glob}/{exp_id}_*/generations/"
                )

                logging.info(f"Launching retraining on {generation_glob}...")
                logging.info(f"Reload checkpoint: {to_reload}")

                retrain_jobs.append(
                    retrain(
                        args=args,
                        params=these_params,
                        generation_path=generation_glob,
                        exp_name=train_exp_name,
                        job_folder=job_folder_train,
                    )
                )

        # wait for retraining jobs to finish and check results
        for job in retrain_jobs:
            try:
                logging.info(f"Job results: {job.results()}")
            except Exception as e:
                logging.info(f"failed: {str(e)}")

        # update paths
        if args.train_gen.retrain:
            if args.best_valid:
                generator_path = Path(params.override_dump_path) / ""
            else:
                generator_path_str, _ = get_latest_checkpoint(params.override_dump_path)
                assert isinstance(generator_path_str, str)
                generator_path = Path(generator_path_str)

        if args.train_prover.retrain:
            if args.best_valid:
                prover_path = Path(prover_params.override_dump_path) / ""
            else:
                prover_path_str, _ = get_latest_checkpoint(
                    prover_params.override_dump_path
                )
                assert isinstance(prover_path_str, str)
                prover_path = Path(prover_path_str)


if __name__ == "__main__":
    cfg = OfflineIterArgs()
    cfg = cfg_from_cli(cfg)
    main(cfg)
