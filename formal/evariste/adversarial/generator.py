# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, List, Dict, Iterator
from dataclasses import dataclass, asdict
from functools import cached_property
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
import os
import time
import torch
import pprint
import psutil
import getpass
import logging
import numpy as np

from evariste.backward.prover.utils import GPUMonitor
from params.params import cfg_from_cli, Params

from evariste.comms.zip import ZipSender
from evariste.comms.comms import make_sender
from evariste.comms.store import AnnotatedGeneration
from evariste.comms.rl_distributed_config import RLDistributedConfig, DistributedSetup

from evariste.forward import forward_model_factory, forward_model_configs
from evariste.forward.common import GenerationHistory, ForwardGoal
from evariste.forward.online_generation.worker_type import WorkerType
from evariste.forward.env_specifics.fwd_env_helper import FwdEnvHelper
from evariste.forward.env_specifics.generation_annotator import NodeSelectionConfig
from evariste.forward.forward_prover import ProverConfig

from evariste.adversarial.mixed_stream import mixed_stream
from evariste.metrics import Logger, DataclassStats, ActionCounter, Avg, AvgDict
from evariste.model.checkpoints import get_latest_checkpoint
from evariste.utils import logged_closing


logger = logging.getLogger(__name__)


@dataclass
class AdvGeneratorConfig(Params):
    prover: ProverConfig
    rl_distributed: RLDistributedConfig
    node_select_cfg: NodeSelectionConfig
    seed: int = 42
    rank: int = 0
    split: str = "train"
    fixed_generator: str = ""
    generator_mix_str: str = "1,generator"
    debug: bool = False

    @cached_property
    def dump_path(self) -> str:
        root_path = self.rl_distributed.exp_root_path
        return os.path.join(root_path, WorkerType.GENERATOR_ACTOR)

    @cached_property
    def use_generation_rewards(self) -> bool:
        return self.rl_distributed.distributed_setup in {
            DistributedSetup.ADV_GENERATOR_REWARD,
            DistributedSetup.GENERATOR_ONLY,
        }

    @cached_property
    def generator_mix(self) -> List[Tuple[float, str]]:
        split = [x.split(",") for x in self.generator_mix_str.split(":")]
        return [(float(x[0]), x[1]) for x in split]

    @staticmethod
    def from_adv_cfg(adv_cfg) -> "AdvGeneratorConfig":
        """
        WARNING this method have a side effect, it calls "setup_exp_and_update_params"
        on intermediate TrainerArgs. Should be removed with a little bit of refacto
        """
        from configs.adv_configs import AdversarialConfig
        from evariste.trainer.launch import setup_exp_and_update_params
        from evariste.forward.forward_model_factory import make_prover_cfg

        assert isinstance(adv_cfg, AdversarialConfig)
        assert adv_cfg.get_worker_type() == WorkerType.GENERATOR_ACTOR

        train_cfg = adv_cfg.make_trainer_args()
        assert train_cfg.debug.debug == adv_cfg.debug
        assert train_cfg.rl_distributed.is_adversarial_training
        assert train_cfg.override_dump_path.endswith("generator_actor")

        # to create the logger, setup cuda, create folder,
        # but can probably be removed with a little work
        setup_exp_and_update_params(train_cfg)
        logger.warning("Called setup_exp_and_update_params for generator TrainerArgs")

        # WARNING: global_rank means rank_in_group
        # (vs local_rank which is rank_in_node) and NOT rank_in_slurm_job
        group_slurm_cfg = train_cfg.slurm_conf
        rank_in_group = group_slurm_cfg.global_rank

        generator_cfg = make_prover_cfg(
            train_cfg, is_generator=True, prover_type="sampling", async_model=True
        )

        return AdvGeneratorConfig(
            rl_distributed=train_cfg.rl_distributed,
            prover=generator_cfg,
            rank=rank_in_group,
            debug=train_cfg.debug.debug,
            generator_mix_str=adv_cfg.generator_mix_str,
            fixed_generator=adv_cfg.fixed_generator,
            node_select_cfg=adv_cfg.node_select_cfg,
        )

    def __post_init__(self):
        assert len(self.rl_distributed.exp_root_path) > 0
        if len(self.rl_distributed.generator_actor_send_to) > 0:
            assert self.node_select_cfg.n_send_to_provers > 0


BASE_OUTPUT_FOLDER = Path(f"")


@dataclass
class DiscardedStats:
    reason: str
    src: str


def generate(cfg: AdvGeneratorConfig):
    logging.info(f"Config: {pprint.pformat(asdict(cfg))}")
    log_interval = 60

    assert cfg.prover.async_model, "not using async_model!"
    checkpoint_dir = os.path.join(
        cfg.rl_distributed.exp_root_path, WorkerType.GENERATOR_TRAINER
    )
    logger.info(f"Using checkpoint dir: {checkpoint_dir}")

    metrics = Logger(outdir=cfg.dump_path, tag="generator", quiet=cfg.rank != 0)
    metrics.log_config(cfg)

    generation_stats: Dict[str, AvgDict] = defaultdict(lambda: AvgDict())
    discarded_stats = DataclassStats(DiscardedStats)
    storing_time = ActionCounter("storing_time", is_rate=False, silent=True)
    generation_time = ActionCounter("generation_time", is_rate=False, silent=True)

    ckpt_path: Optional[str] = None
    if cfg.fixed_generator:
        assert cfg.rl_distributed.n_generator_trainers == 0
        ckpt_path = cfg.fixed_generator
        logger.info(f"Using a fixed checkpoint for generator: {ckpt_path}")

    while ckpt_path is None:
        logging.info(f"Waiting for a generator checkpoint in {checkpoint_dir} ...")
        ckpt_path, cur_version = get_latest_checkpoint(checkpoint_dir, False)
        time.sleep(10.0)
        if ckpt_path is not None:
            logger.info(f"Starting from latest generator: {ckpt_path}")

    # sanity check
    assert ckpt_path is not None and os.path.isfile(ckpt_path)

    generator, dico, params, env_helper = forward_model_factory.from_checkpoint(
        ckpt_path=ckpt_path, device_str="cuda", cfg=cfg.prover
    )
    logging.info(f"Prover built!")
    rng = np.random.RandomState((cfg.seed, cfg.rank))
    goal_stream = env_goal_stream(rng, env_helper, split=cfg.split)
    generation_annotator = env_helper.get_annotator()

    avg_gen_rewards = Avg()

    generation_store: Optional[ZipSender] = None
    if cfg.rank == 0:
        gen_log_dir = Path(cfg.dump_path) / f"generations_logs"
        gen_log_dir.mkdir(parents=True, exist_ok=True)
        # using ZipSender as store
        generation_store = ZipSender(store_path=gen_log_dir, zip_size=200)

    sender = make_sender(
        rank=cfg.rank,
        sender_type=WorkerType.GENERATOR_ACTOR,
        cfg=cfg.rl_distributed,
        debug=cfg.debug,
    )
    logging.info(f"Store done")

    # TODO: restore checkpoint?

    last_log = time.time()
    gpu_mon = GPUMonitor(delay=1.0)  # probes gpu activity every seconds

    logging.info("Starting generation!")

    with ExitStack() as stack:
        stack.enter_context(logged_closing(generator, "generator"))
        stack.enter_context(logged_closing(sender, "sender"))
        stack.enter_context(logged_closing(goal_stream, "goal_stream"))
        stack.enter_context(logged_closing(metrics, "metrics_logger"))
        stack.enter_context(logged_closing(gpu_mon, "gpu_mon"))
        if generation_store:
            stack.enter_context(logged_closing(generator, "generation_store"))

        generation_stream = mixed_stream(
            cfg.generator_mix, generator=generator.generate_proofs(goal_stream)
        )
        in_generation = time.time()

        for goal_id, src, history in generation_stream:
            generation_time.act(time.time() - in_generation)
            start_storing = time.time()
            assert isinstance(history, GenerationHistory)

            if len(history.forward_steps()) == 0:
                # only not generated nodes (e_hyps, first node...), skipping
                discarded_stats.act(DiscardedStats(reason="no_generated", src=src))
            else:
                annotated_goals = generation_annotator.annotate_and_select_goals(
                    history, cfg.node_select_cfg
                )
                for annot_goal in annotated_goals:
                    generation_stats[src].act(asdict(annot_goal.stats))
                    avg_gen_rewards.act(annot_goal.generation_reward)
                    annotated_generation = AnnotatedGeneration(
                        generation=history,
                        src=src,
                        prover_goal=annot_goal.selected_goal,
                        gen_ckpt=cur_version,
                        generation_stats=annot_goal.stats,
                        generation_reward=annot_goal.generation_reward,
                    )
                    sender.store(annotated_generation)
                    if generation_store and np.random.rand() < 0.01:
                        generation_store.store(annotated_generation)

            if not cfg.fixed_generator:
                path, version = get_latest_checkpoint(checkpoint_dir)
                if version > cur_version:
                    logging.info(f"Found version {version}, updating model")
                    generator.reload_model_weights()
                    cur_version = version

            # logging / stats
            storing_time.act(time.time() - start_storing)
            if time.time() - last_log > log_interval:
                gpu_stats = gpu_mon.stats[torch.cuda.current_device()].stats
                last_log = time.time()
                storing = storing_time.rate_and_reset()
                generating = generation_time.rate_and_reset()
                to_log = dict(
                    frac_generation_time=generating / (generating + storing),
                    ram_mem_util_gb=psutil.Process().memory_info().rss / 1024 ** 3,
                    ram_mem_avail_gb=psutil.virtual_memory().available / 1024 ** 3,
                    gpu_usage=gpu_stats["gpu"],
                    gpu_memory=gpu_stats["mem"],
                    model_version=cur_version,
                )
                for src_name, gen_stats in generation_stats.items():
                    for x, y in gen_stats.stats_and_reset().items():
                        to_log[f"{src_name}/{x}"] = y
                for x, y in discarded_stats.stats_and_reset().items():
                    to_log[f"discarded_stats/{x}"] = y
                to_log["sent/s"] = sender.rate_and_reset()
                to_log["gen_rewards"] = avg_gen_rewards.stats_and_reset()
                metrics.log_metrics(to_log)
                logging.info(to_log)
            in_generation = time.time()


GoalStream = Iterator[Optional[Tuple[int, ForwardGoal]]]


def env_goal_stream(
    rng: np.random.RandomState, env_helper: FwdEnvHelper, split: str
) -> GoalStream:
    goal_factory = env_helper.get_goal_factory()
    i = 0
    while True:
        goal = goal_factory.build_generation_goal(rng=rng, split=split)
        yield i, goal
        i += 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    forward_model_configs.register_model_cfgs()
    forward_model_configs.register_prover_cfgs()
    cfg = cfg_from_cli(schema=AdvGeneratorConfig)
    generate(cfg)
