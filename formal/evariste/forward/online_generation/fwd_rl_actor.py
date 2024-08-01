# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import os
import pickle
import shutil
import time
from contextlib import closing
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterator, Tuple, List, Any, Dict
import pprint

from evariste.comms.comms import make_sender
from evariste.forward import forward_model_factory
from evariste.forward.forward_model_configs import SAMPLING_PROVER_CFG
from evariste.forward.env_specifics.fwd_env_helper import FwdEnvHelper
from evariste.forward.fwd_mm.mm_env_helper import MMFwdEnvHelper
from evariste.forward.fwd_mm.mm_helpers import (
    build_forward_goals_from_samples,
    build_forward_goals_from_proof_steps,
)
from evariste.forward.common import (
    ForwardGoal,
    GenerationHistory,
    GenerationInfos,
    GenerationError,
)
from evariste.forward.online_generation.worker_type import WorkerType
from evariste.forward.forward_prover import ProverConfig
from evariste.forward.online_generation.online_generation_common import FwdTrajectory
from evariste.forward.online_generation.goal_selectors.goal_selector import (
    UniformGoalSelector,
    GeneratedGoalSelector,
    MixedGoalSelector,
    GoalSelector,
    GoalSelectionConfig,
    MultiGoalSelector,
)
from evariste.forward.proof_search import StandardProofSearch
from evariste.forward.utils.retry import retry, reschedule_if_oom
from evariste.metrics import Logger
from evariste.model.checkpoints import get_latest_checkpoint, keep_latest_checkpoints
from evariste.trainer.args import TrainerArgs
from evariste.comms.rl_distributed_config import RLDistributedConfig
from params.params import Params
from evariste.forward.utils.launch_utils import prepare_folder


logger = logging.getLogger(__name__)

MAX_ITERATIONS = int(1e12)


@dataclass
class FwdRLActorConfig(Params):
    prover: ProverConfig = field(default_factory=lambda: SAMPLING_PROVER_CFG)
    rl_distributed: RLDistributedConfig = field(
        default_factory=lambda: RLDistributedConfig(
            n_provers=1,
            n_prover_trainers=1,
            is_fwd_rl_training=True,
            use_zmq=False,
            n_generators=0,
            n_generator_trainers=0,
        )
    )
    seed: int = 42

    goal_selection: GoalSelectionConfig = field(
        default_factory=lambda: GoalSelectionConfig()
    )

    rank: int = 0
    world_size: int = 1
    rank_on_machine: int = -1
    debug: bool = False
    output_path: str = ""
    train_dir: str = ""
    splay_wait_time: int = -1

    @classmethod
    def from_trainer_args(cls, params: TrainerArgs):
        assert isinstance(params, TrainerArgs)

        offset = params.slurm_conf.torch_world_size
        world_size = params.slurm_conf.world_size - offset
        rank = params.slurm_conf.global_rank - offset
        assert world_size > rank >= 0
        output_path = os.path.join(params.dump_path, f"fwd_rl_actor_{rank}")

        # hardcoded params for the moment
        from evariste.forward.forward_model_factory import make_prover_cfg

        prover_cfg: ProverConfig = make_prover_cfg(
            params,
            prover_type=params.online_gen_cfg.prover_type,
            is_generator=False,
            async_model=True,
        )
        prover_cfg.decoding_params.fixed_temperature = (
            params.online_gen_cfg.prover_temperature
        )

        return FwdRLActorConfig(
            debug=params.debug.debug,
            train_dir=params.dump_path,
            output_path=output_path,
            prover=prover_cfg,
            rank=rank,
            world_size=world_size,
            rank_on_machine=params.slurm_conf.local_rank,
            goal_selection=params.online_gen_cfg.goal_selection,
            splay_wait_time=params.online_gen_cfg.splay_wait_time,
            rl_distributed=RLDistributedConfig.new_fwd_distributed_rl_cfg(params),
        )


def is_fwd_rl_actor(cfg: TrainerArgs):
    if not cfg.online_fwd_generation:
        return False
    assert cfg.slurm_conf.torch_world_size > 0
    assert cfg.slurm_conf.world_size - cfg.slurm_conf.torch_world_size > 0
    return cfg.slurm_conf.global_rank >= cfg.slurm_conf.torch_world_size


def run_fwd_rl_actor_from_trainer_args(cfg: TrainerArgs):
    # local import, only imported if needed
    from evariste.trainer.launch import setup_exp_and_update_params

    assert is_fwd_rl_actor(cfg)

    setup_exp_and_update_params(cfg)
    if cfg.online_fwd_generation:
        from leanml import DeadLean

        logging.warning("Starting online generation worker")

        def run():
            actor_cfg = FwdRLActorConfig.from_trainer_args(cfg)
            return run_fwd_rl_actor(cfg=actor_cfg)

        run = reschedule_if_oom(run, name="run_fwd_rl_actor")
        return retry(
            run,
            error_types=(DeadLean,),
            name="run_fwd_rl_actor",
            retry_on_cuda_error=True,
        )
    else:
        raise NotImplementedError


@dataclass
class FwdActorStats:
    done: int = 0
    n_solved: int = 0
    duration: float = 0
    time_in_storing: float = 0
    n_chunks: int = 0


def run_fwd_rl_actor(cfg: FwdRLActorConfig):
    assert cfg.rl_distributed.is_fwd_rl_training, cfg.rl_distributed
    train_dir = cfg.train_dir
    actor_path = Path(cfg.output_path)
    if not actor_path.exists():
        prepare_folder(cfg, verbose=False)
        restart = False
    else:
        restart = True
        logger.info("Restarting detected!")

    logger.info(f"Config: {pprint.pformat(asdict(cfg))}")

    model_ckpt_path = None
    cur_version = None
    logger.info("Waiting for a checkpoint...")
    while model_ckpt_path is None:
        model_ckpt_path, version = get_latest_checkpoint(train_dir)
        time.sleep(10.0)
        cur_version = version

    metrics = Logger(outdir=actor_path, tag="fwd_actor", quiet=cfg.rank != 0)
    metrics.log_config(cfg)
    step = 0

    store = make_sender(
        rank=cfg.rank,
        sender_type=WorkerType.PROVER_ACTOR,
        cfg=cfg.rl_distributed,
        debug=cfg.debug,
    )

    if cfg.splay_wait_time and not restart:
        wait_time = cfg.rank_on_machine * cfg.splay_wait_time
        logger.info(
            f"Going to wait {wait_time}s to avoid too much memory consumption"
            f"on same machine while launching envs"
        )
        time.sleep(wait_time)
        logger.info("Done with waiting")

    prover, dico, params, env_helper = forward_model_factory.from_checkpoint(
        ckpt_path=model_ckpt_path, device_str="cuda", cfg=cfg.prover
    )
    if cfg.debug:
        prover.verbose = True
    selector = make_goal_selector(cfg, params, env_helper)

    # restore selector from ckpt
    actor_ckpt, actor_epoch = get_latest_checkpoint(actor_path)
    if actor_ckpt:
        logger.info(f"Detected actor ckpt: {actor_ckpt}")
        restored = load_checkpoint(Path(actor_ckpt))
        assert restored.epoch == actor_epoch
        selector.load_state_dict(restored.selector)
    actor_epoch = max(0, actor_epoch + 1)
    logger.info(f"Starting actor epoch {actor_epoch}")

    logging.info(f"Loading version {cur_version} and {model_ckpt_path}")

    goal_id2selector_id = {}

    def inputs() -> Iterator[Tuple[int, ForwardGoal]]:
        goal_id = 0
        while True:
            sid = selector.select_goal()
            goal_id2selector_id[goal_id] = sid
            yield goal_id, sid
            goal_id += 1

    actor_stats = FwdActorStats()
    start_chunk = time.time()
    last_log = time.time()
    last_ckpt = time.time()
    logger.info("Start prover")
    with closing(prover), closing(store), closing(metrics):
        for id_, proof in prover.generate_proofs(inputs()):
            start_storing = time.time()
            if isinstance(proof, GenerationError):
                err = proof
                logger.info(f"Received error {err.type}:{err}")
                continue

            assert isinstance(proof, StandardProofSearch), type(proof)

            gen = proof.generation
            gen_info = proof.info
            assert isinstance(gen, GenerationHistory)
            assert isinstance(gen_info, GenerationInfos)

            solved = gen_info.solved
            actor_stats.n_solved += int(solved)
            actor_stats.done += 1
            if solved:
                # print("solved", name, len(graph.nodes))
                assert gen.stack[-1].step is not None, gen
                assert gen.goal.is_solved_by_step(gen.stack[-1].step)

            metadata = {
                "name": gen.goal.label,
                "gen_id": id_,
                "actor_id": cfg.rank,
                "n_valid": len(gen.forward_steps()),
                "n_invalid": len(gen.errors()),
                "stopped": gen_info.stopped,
                "solved": gen_info.solved,
            }
            if gen.goal.forbidden:
                gen.goal.forbidden = {"<REMOVED/>"}  # we remove forbidden
                # to save space. They can be obtained back with label field
            store.store(FwdTrajectory(history=gen, metadata=metadata))
            selector_id = goal_id2selector_id.pop(id_)
            selector.update_with_generation(
                generation_id=selector_id, generated=gen, solved=solved
            )
            actor_stats.time_in_storing += time.time() - start_storing

            if time.time() - last_log > (10 if cfg.debug else 600):
                last_log = time.time()
                actor_stats.duration = time.time() - start_chunk

                path, version = get_latest_checkpoint(train_dir)
                log(
                    metrics=metrics,
                    step=step,
                    prover_stats=prover.stats,
                    selector_stats=selector.stats,
                    stats=actor_stats,
                    model_version=cur_version,
                )
                step += 1
                if version > cur_version:
                    logging.info(f"Found version {version}, updating model")
                    prover.reload_model_weights()
                    cur_version = version
                else:
                    logging.info(f"Found version {version}, already current one")
                logging.info(f"Output path: {cfg.output_path}")
                start_chunk = time.time()
                actor_stats = FwdActorStats()

            if time.time() - last_ckpt > (10 if cfg.debug else 3000):
                last_ckpt = time.time()
                save_checkpoint(actor_path, actor_epoch=actor_epoch, selector=selector)
                actor_epoch += 1
                logger.info(f"Starting actor epoch: {actor_epoch}")


def make_goal_selector(
    cfg: FwdRLActorConfig, params: TrainerArgs, env_helper: FwdEnvHelper
) -> GoalSelector:
    goal_cfg = cfg.goal_selection
    goals: List[ForwardGoal] = []

    if goal_cfg.use_human_dataset:
        if not goal_cfg.use_forbidden or goal_cfg.use_subgoals:
            assert isinstance(env_helper, MMFwdEnvHelper)
            mm_data_dir = params.mm.dataset.data_dir
            if cfg.debug:
                mm_data_dir += "/100"
            goals += build_forward_goals_from_proof_steps(
                mm_data_dir=mm_data_dir,
                database_path=params.mm.dataset.database_path,
                split="train",
                use_forbidden=goal_cfg.use_forbidden,
                use_subgoals=goal_cfg.use_subgoals,
            )
        else:
            goal_factory = env_helper.get_goal_factory()
            goals += goal_factory.build_forward_goals(split="train", debug=cfg.debug)

    if goal_cfg.add_goal_path != "":
        assert isinstance(env_helper, MMFwdEnvHelper)
        goals += build_forward_goals_from_samples(
            data_path=goal_cfg.add_goal_path, debug=params.debug.train
        )

    goals = [s for i, s in enumerate(goals) if i % cfg.world_size == cfg.rank]

    assert len(goals) > 0

    selector = UniformGoalSelector(goals=goals, seed=cfg.seed)

    if goal_cfg.use_other_split:
        assert goal_cfg.use_human_dataset
        assert goal_cfg.use_forbidden
        assert not goal_cfg.use_subgoals
        assert not goal_cfg.select_generated_goals
        goal_factory = env_helper.get_goal_factory()
        valid_goals = goal_factory.build_forward_goals(
            split=goal_cfg.use_other_split, debug=cfg.debug
        )
        valid_selector = UniformGoalSelector(goals=valid_goals, seed=cfg.seed)
        logging.info(
            f"Adding {len(valid_goals)} goals from {goal_cfg.use_other_split} split"
        )
        selector = MultiGoalSelector(
            selectors=[selector, valid_selector], seed=cfg.seed
        )
    elif goal_cfg.select_generated_goals:
        # we will sample also goal that are generated in an online fashion
        generated_selector = GeneratedGoalSelector(
            seed=cfg.seed,
            max_goals=goal_cfg.max_generated_goals,
            sampling_strategy=goal_cfg.generated_goals_sampling_strategy,
            alpha=goal_cfg.generated_goals_sampling_alpha,
        )
        selector = MixedGoalSelector(
            supervised=selector, generated=generated_selector, seed=cfg.seed
        )

    return selector


@dataclass
class Ckpt:
    epoch: int
    selector: Any


def save_checkpoint(output_path: Path, actor_epoch: int, selector: GoalSelector):
    assert actor_epoch >= 0, actor_epoch
    ckpt_path = output_path / f"checkpoint.{actor_epoch}.pth"
    tmp_ckpt_path = ckpt_path.parent / f"tmp_{ckpt_path.name}"
    state = Ckpt(epoch=actor_epoch, selector=selector.state_dict())
    with tmp_ckpt_path.open("wb") as fp:
        pickle.dump(state, fp)
    shutil.move(str(tmp_ckpt_path), str(ckpt_path))
    logger.info(f"Wrote ckpt in {ckpt_path}")
    keep_latest_checkpoints(output_path, to_keep=5)


def load_checkpoint(ckpt_path: Path) -> Ckpt:
    with ckpt_path.open("rb") as fp:
        state = pickle.load(fp)
    return state


def log(
    metrics: Logger,
    step: int,
    stats: FwdActorStats,
    prover_stats: Dict,
    selector_stats: Dict,
    **kwargs,
):
    logger.info(
        f"Step: {step} "
        f"Solved: {stats.n_solved}, done: {stats.done} "
        f"({100 * stats.n_solved / stats.done:.01f}% solved) "
        f"- duration: {stats.duration:.02f}s "
        f"- storing: {stats.time_in_storing:.02f}s"
    )

    logs_dict = asdict(stats)
    logs_dict.update({f"fwd_prover/{k}": v for k, v in prover_stats.items()})
    logs_dict.update({f"selector/{k}": v for k, v in selector_stats.items()})
    logs_dict.update(kwargs)
    logger.info(f"{step} {logs_dict}")
    metrics.log_metrics(logs_dict)
