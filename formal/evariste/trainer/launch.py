# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time
from typing import Optional, Any, Union, List, Dict
from contextlib import ExitStack
from dataclasses import asdict
from functools import partial
from logging import Logger
from copy import deepcopy
from pathlib import Path
import gc
import os
import re
import sys
import pickle
import random
from evariste.model.checkpoints import get_latest_checkpoint
import submitit
import numpy as np
import torch
import socket

from evariste import json as json
from evariste.slurm import init_signal_handler, init_torch_distributed

# from evariste.trainer.utils import nfs_barrier
from evariste.utils import (
    print_memory,
    logged_closing,
    get_dump_path,
    environment_variables,
    set_TMPDIR,
)
from evariste.logger import create_logger
from evariste.model.modules import build_modules
from evariste.model.trainer import Trainer
from evariste.model.evaluator import Evaluator
from evariste.model.data.envs.builder import build_envs
from evariste.model.data.dictionary import Dictionary
from evariste.slurm_conf_factory import from_trainer_args
from evariste.trainer.args import TrainerArgs
from evariste.backward.prover.utils import GPUMonitor
from evariste.forward.online_generation.fwd_rl_actor import (
    is_fwd_rl_actor,
    run_fwd_rl_actor_from_trainer_args,
)
from evariste.clusters.utils import get_running_partition


logger = create_logger(None)


def initialize_exp(params: TrainerArgs) -> Logger:
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    # dump parameters
    params.dump_path, params.exp_id = get_dump_path(
        root_dump_path=params.root_dump_path,
        exp_name=params.exp_name,
        given_exp_id=params.exp_id,
        overwrite_dump_path=params.override_dump_path,
    )

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith("--"):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match("^[a-zA-Z0-9_]+$", x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command_str = " ".join(command)
    params.command = command_str + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger = create_logger(
        os.path.join(params.dump_path, "train.log"),
        rank=params.slurm_conf.global_rank,
    )

    logger.info(f"============ Initialized logger for PID {os.getpid()} ============")
    if not params.debug.debug:
        logger.info(params.to_json())
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("Running command: %s" % command)
    logger.info("")
    if params.slurm_conf.is_master:
        with open(os.path.join(params.dump_path, "params.pkl"), "wb") as f:
            pickle.dump(params, f)
        json.dump(
            asdict(params),
            open(os.path.join(params.dump_path, "params.json"), "w"),
            sort_keys=True,
            indent=4,
        )
    return logger


def setup_exp_and_update_params(params: TrainerArgs):

    # debug mode
    if params.debug.debug:
        params.exp_name = "debug"
        # exp_id -- dont change it if provided by user. for torch.distributed.launch,
        # should be identical in all workers, so must be set before
        if not params.exp_id:
            params.exp_id = "debug_%08i" % random.randint(0, 100_000_000)
        params.debug.slurm = True
        params.debug.train = True

    if params.debug.train:
        params.batch.collate_queue_size = 500  # faster start up

    params.check_and_mutate_args()

    # initialize the experiment
    logger = initialize_exp(params)  # side effect on params.dump_path and params.exp_id

    # set GPU device (note it is separated from init_torch_distributed since
    # we need to set the GPU even if we don't need torch_distributed (for actors)
    torch.cuda.set_device(params.slurm_conf.local_rank)
    logger.info(
        f"Setting GPU to device {params.slurm_conf.local_rank}. "
        f"Verification: {torch.cuda.current_device()}. "
        f"On hostname: {socket.gethostname()}. "
    )

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()
    return logger


def train(params: TrainerArgs):
    set_TMPDIR()
    logger = setup_exp_and_update_params(params)

    # initialize the multi-GPU / multi-node training
    init_torch_distributed(params.slurm_conf)

    print(f"Training partition: {get_running_partition()}")

    # build environments
    print_memory(logger, "before build envs")
    # TODO make sure we check checkpoint dico is the same
    if params.use_checkpoint_dico:
        if len(params.reload_checkpoint) > 0:
            dico = Dictionary.create_from_checkpoint(params.reload_checkpoint)
        elif len(params.reload_partial_checkpoint) > 0:
            dico = Dictionary.create_from_pretrained(
                params.parsed_reload_partial_checkpoint
            )
        else:
            raise RuntimeError("You should precise a checkpoint to reload.")
    elif params.use_reloaded_dico:
        dico = Dictionary.create_from_pretrained(params.parsed_reload_model)
    else:
        dico = Dictionary.create_empty()
    envs = build_envs(dico, params)  # side effect params.env_base_seed

    # print dico hash (sanity check)
    logger.info(f"Dictionary: {len(dico)} words ({hash(dico)})")

    if params.check_memory:
        new_envs = []
        logger.info("Checking memory")
        for i in range(params.num_workers):
            print_memory(logger, f"Before copy {i}")
            torch.distributed.barrier()
            for x, y in envs.items():
                new_envs.append(deepcopy(y))
        print_memory(logger, "Final expected memory remaining")
        torch.distributed.barrier()
        del new_envs
        gc.collect()
        torch.distributed.barrier()

    # build modules
    print_memory(logger, "before build modules")
    modules = build_modules(dico, params)  # side effect params.

    # build trainer, reload potential checkpoints / build evaluator
    print_memory(logger, "before build trainer")
    trainer = Trainer(modules, envs, dico, params)

    # build evaluator
    print_memory(logger, "before build evaluator")
    evaluator = Evaluator(trainer)

    with ExitStack() as stack:

        # required to correctly close() resources like Tensorboard, Receivers/Senders
        stack.enter_context(logged_closing(trainer, "trainer"))
        stack.enter_context(logged_closing(evaluator, "evaluator"))
        for env_name, env in envs.items():
            if hasattr(env, "close"):
                stack.enter_context(logged_closing(env, env_name))

        # # to prevent nccl timeout errors
        # # (timeout happening currently with minproof rb + send_to_all = False)
        # logger.info("Waiting for all trainers to be ready before beginning training")
        # nfs_barrier(
        #     dump_path=params.dump_path,
        #     rank=params.slurm_conf.global_rank,
        #     world_size=params.slurm_conf.world_size,
        #     timeout_s=3600,
        # )
        # logger.info("All trainers ready!")

        # evaluation
        if params.eval_only:
            _, e = get_latest_checkpoint(params.dump_path)
            trainer.epoch = (
                e  # necessary for bwd_prover_eval to work on checkpoint.-1.pth
            )
            ckpt_path = os.path.join(params.dump_path, f"checkpoint.{e}.pth")
            if not os.path.isfile(ckpt_path):
                assert os.path.isfile(params.reload_model), params.reload_model
                logger.warning(f"Linking {ckpt_path} to {params.reload_model}")
                os.symlink(params.reload_model, ckpt_path)
            print_memory(logger, "before run_all_evals")
            scores = evaluator.run_all_evals()
            print_memory(logger, "after run_all_evals")
            trainer.epoch = 0
            for k, v in scores.items():
                if isinstance(v, (int, str)):
                    logger.info(f"{k} -> {v}")
                else:
                    logger.info(f"{k} -> {v:.6f}")
            logger.info(f"__log__:{json.dumps(scores)}")
            exit()

        tasks = params.parsed_tasks()
        weights = params.parsed_tasks_weight()
        logger.info(f"Training on tasks {tasks}")
        logger.info(f"With weights {weights}")
        # language model training

        # task random iterator
        task_rng = np.random.RandomState(0 if params.sync_worker_tasks else None)

        gpu_mon = GPUMonitor(delay=1.0)  # monitor gpu usage every 1s
        with logged_closing(gpu_mon, "gpu_monitor"):
            for _ in range(params.max_epoch):

                logger.info(
                    f"============ Starting epoch {trainer.epoch} ... ============"
                )
                print_memory(logger, f"loop epoch {trainer.epoch}")
                trainer.n_sentences = 0

                while True:
                    if params.gpu_oom_retry <= 0:
                        choose_and_run_task(trainer, tasks, weights, params, task_rng)
                    else:
                        fn = partial(
                            choose_and_run_task,
                            trainer,
                            tasks,
                            weights,
                            params,
                            task_rng,
                        )
                        retry_if_oom(fn, params.gpu_oom_retry)

                    trainer.iter(gpu_mon)

                    # end of epoch (enough sentences, and not in the middle of gradient accumulation)
                    if (
                        trainer.n_sentences >= trainer.epoch_size
                        and trainer.n_iter % params.accumulate_gradients == 0
                    ):
                        break

                logger.info(f"============ End of epoch {trainer.epoch} ============")

                # moved this here because a checkpoint is required for async eval
                trainer.save_and_roll_checkpoint()
                scores = None
                if trainer.epoch % params.eval_freq == 0:
                    print_memory(logger, f"before EVAL {trainer.epoch}")
                    scores = evaluator.run_all_evals()

                    # print / JSON log
                    for k, v in scores.items():
                        if isinstance(v, (int, str)):
                            logger.info(f"{k} -> {v}")
                        else:
                            logger.info(f"{k} -> {v:.6f}")
                    logger.info(f"__log__:{json.dumps(scores)}")

                    # end of epoch
                    trainer.log_module_weights()
                    trainer.save_best_model(scores)
                trainer.end_epoch(scores)


def choose_and_run_task(
    trainer: Trainer,
    tasks: List[str],
    weights: List[float],
    params: TrainerArgs,
    task_rng: np.random.RandomState,
):
    task = task_rng.choice(tasks, replace=True, p=weights)

    # log current task
    trainer.f_task.write(f"SELECTED {trainer.n_iter} {task} {time.time()}\n")
    trainer.f_task.flush()

    if task in params.no_train_tasks:
        return
    if task.endswith("_clm"):
        trainer.clm_mlm_step(task, causal=True)
    elif task.endswith("_mlm"):
        trainer.clm_mlm_step(task, causal=False)
    elif task.endswith("_mass"):
        trainer.mass_step(task)
    elif task.endswith("_cclm"):
        trainer.cclm_step(task, sample=True)
        for _ in range(params.gan.train_real_freq):
            trainer.cclm_step(task, sample=False)
        disc_steps = params.gan.train_disc_freq
        if disc_steps >= 1:
            n_reps = disc_steps
        else:
            n_reps = 1 if (trainer.n_iter % (-disc_steps) == 0) else 0
        for _ in range(n_reps):
            trainer.cclm_disc_step(task, sample=True)
            trainer.cclm_disc_step(task, sample=False)
    elif (
        task.endswith("_seq2seq")
        or "mcts_tactic" in task
        or "mcts_effect" in task
        or "mcts_minproof" in task
    ):
        # reusing task_rng here means all workers will use hard tgt encoder together
        trainer.seq2seq_step(task, rng=task_rng)
    elif task.endswith("_seq2seqtok"):
        trainer.seq2tok_seq2seqtok_step(task, with_decoder=True)
    elif task.endswith("_seq2tok"):
        trainer.seq2tok_seq2seqtok_step(task, with_decoder=False)
    elif task.endswith("_seq2emb"):
        trainer.seq2emb_step(task)
    elif task.endswith("_bt"):
        trainer.bt_step(task)
    elif task.endswith("_distillation"):
        trainer.distillation_step(task)
    elif task.endswith("_disc"):
        if not params.gan.fixed_generator:
            trainer.seq2seq_step(task, rng=task_rng, use_dicriminator=True)
        trainer.seq2seq_disc_step(task)
    elif "mcts_critic" in task:
        trainer.mcts_critic_step(task)
    elif "rl" in task:
        trainer.rl_step(task)
    else:
        raise Exception(f"Unknown task: {task}")


def retry_if_oom(fn, n_retries: int):
    assert n_retries > 0
    for i in range(n_retries + 1):
        try:
            result = fn()
            if i > 0:
                logger.info(f"OOM caught in trainer: success after {i} retries")
            return result
        except RuntimeError as err:
            if "out of memory" in str(err) and i < n_retries:
                pass
            else:
                raise err
        torch.cuda.empty_cache()
        logger.warning(
            f"OOM caught in trainer: going to do a retry [{i + 1}/{n_retries}]"
        )


class TrainRunner:
    def __init__(self, params: TrainerArgs):
        self.params = params
        self.logger = None

    def __call__(self):
        # TODO: deduplicate this
        self.params.slurm_conf = from_trainer_args(self.params)

        if is_fwd_rl_actor(self.params):
            return run_fwd_rl_actor_from_trainer_args(self.params)
        train(self.params)

    def checkpoint(
        self, *args: Any, **kwargs: Any
    ) -> Optional[submitit.helpers.DelayedSubmission]:
        # For the moment we don't want to be rescheduled if the trainer job is preempted

        # assert self.logger is not None
        # prod_id = int(os.environ["SLURM_PROCID"])
        # if prod_id == 0:
        #     self.logger.warning("Requeuing job " + os.environ["SLURM_JOB_ID"])
        #     # submits to requeuing
        #     return submitit.helpers.DelayedSubmission(self, *args, **kwargs)
        # self.logger.warning("Not the master process, no need to requeue.")
        return None


@environment_variables(SBATCH_NO_REQUEUE="1")
def launch_train(
    dump_path: Path,
    slurm_job_name: str,
    exp_name: str,
    partition: str,
    trainer_args: TrainerArgs,
    gpus_per_node: int = 8,
    cpus_per_task: int = 10,
    mem_per_gpu: int = 60,
    ntasks_per_node: int = 8,
    n_nodes: int = 1,
    timeout_min: Optional[int] = None,
    local: bool = False,
    exclude_nodes: str = "",
) -> submitit.Job:
    assert type(mem_per_gpu) is int and mem_per_gpu > 0
    logger.info(
        f"LAUNCH TRAIN: Starting job on partition {partition} "
        f"for at most {timeout_min} minutes. local={local}"
    )
    if not local:
        trainer_executor: Union[submitit.AutoExecutor, submitit.LocalExecutor]
        kwargs = {}
        if exclude_nodes:
            kwargs["slurm_exclude"] = exclude_nodes
        trainer_executor = submitit.AutoExecutor(
            folder=dump_path, slurm_max_num_timeout=-1
        )
        trainer_executor.update_parameters(
            slurm_job_name=slurm_job_name,
            slurm_timeout_min=timeout_min,
            slurm_gpus_per_node=gpus_per_node,
            slurm_cpus_per_task=cpus_per_task,
            slurm_ntasks_per_node=ntasks_per_node,
            slurm_nodes=n_nodes,
            slurm_partition=partition,
            # these work better than the inherited cluster defaults : mask_cpu // cyclic
            slurm_additional_parameters={"distribution": "block"},
            slurm_srun_args=["-vv", "--cpu-bind", "none"],
            mem_gb=mem_per_gpu * gpus_per_node,
            **kwargs,
        )
    else:
        # GPU 0: ZMQ prover, GPU 1: trainer
        logger.info(f"LAUNCH TRAIN: Launching train job locally on GPU 1")
        trainer_executor = submitit.LocalExecutor(folder=dump_path)
        trainer_executor.update_parameters(
            timeout_min=timeout_min, gpus_per_node=1, visible_gpus=(1,)
        )
    trainer_args.exp_name = exp_name
    logger.info(f"LAUNCH TRAIN: master port = {trainer_args.master_port}")
    runner = TrainRunner(trainer_args)
    job = trainer_executor.submit(runner)
    return job
