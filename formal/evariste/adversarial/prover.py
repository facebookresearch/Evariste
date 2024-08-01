# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, Union, List, Dict, Deque, Iterator, cast
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from functools import cached_property
from contextlib import ExitStack
from enum import Enum, unique
from pathlib import Path
import os
import time
import pickle
import pprint
import psutil
import random
import getpass
import logging
import datetime
import torch
import numpy as np

from params import ConfStore
from params.params import cfg_from_cli, Params
from evariste.adversarial.generator import GoalStream
from evariste.comms.comms import make_sender, make_receiver
from evariste.comms.rl_distributed_config import RLDistributedConfig
from evariste.comms.store import Receiver, Sender, AnnotatedGeneration
from evariste.comms.zmq import ZMQNotReady
from evariste.envs.eq.graph import eq_nodes_are_equal
from evariste.metrics import Logger, ActionCounter
from evariste.model.checkpoints import get_latest_checkpoint
from evariste.model.transformer_args import DecodingParams
from evariste.trainer.args import TrainerArgs
from evariste.utils import logged_closing

from evariste.forward import forward_model_factory
from evariste.forward.forward_prover import ProverConfig, ForwardProver
from evariste.forward.common import GenerationHistory, ForwardGoal
from evariste.forward.online_generation.worker_type import WorkerType
from evariste.forward.proof_search import StandardProofSearch
from evariste.forward.utils.launch_utils import prepare_folder

from evariste.backward.env.core import BackwardGoal
from evariste.backward.env.equations import EQTheorem
from evariste.backward.graph import Proof as BwdProof
from evariste.backward.prover.prover import (
    ProverKind as BackwardProverKind,
    ProofResult,
    init_and_run_prover,
)
from evariste.backward.prover.prover_args import BeamSearchKind
from evariste.backward.prover.utils import GPUMonitor
from evariste.backward.prover.zmq_prover import ZMQProverParams, ProverParams


logger = logging.getLogger(__name__)


@unique
class AdvProverKind(str, Enum):
    ForwardGreedy = "forward_greedy"
    ForwardSampling = "forward_sampling"
    BackwardGreedy = BackwardProverKind.BackwardGreedy.value
    BackwardMCTS = BackwardProverKind.BackwardMCTS.value

    @property
    def is_forward(self):
        return self in {AdvProverKind.ForwardGreedy, AdvProverKind.ForwardSampling}

    @property
    def is_backward(self):
        return self in {AdvProverKind.BackwardGreedy, AdvProverKind.BackwardMCTS}

    @property
    def backward_kind(self) -> BackwardProverKind:
        assert self.is_backward
        if self == AdvProverKind.BackwardGreedy:
            return BackwardProverKind.BackwardGreedy
        else:
            assert self == AdvProverKind.BackwardMCTS
            return BackwardProverKind.BackwardMCTS


@dataclass
class AdvProverConfig(Params):
    env_name: str
    prover_kind: AdvProverKind
    _prover_cfg: Union[ProverConfig, ZMQProverParams]
    rl_distributed: RLDistributedConfig
    target_for_results: List[Tuple[WorkerType, bool]]  # worker type, send_only_failed
    seed: int = 42
    rank: int = 0
    stop_action: bool = False
    debug: bool = False
    run_uuid: str = "to_replace"
    worker_type: WorkerType = WorkerType.PROVER_ACTOR
    max_input_len: int = 4096
    fixed_prover: str = ""

    @cached_property
    def dump_path(self) -> str:
        root_path = self.rl_distributed.exp_root_path
        return os.path.join(root_path, WorkerType.PROVER_ACTOR)

    @cached_property
    def output_path(self) -> str:
        return os.path.join(self.dump_path, f"rl_prover_worker_{self.rank}")

    @property
    def is_fwd_prover(self) -> bool:
        return self.prover_kind.is_forward

    @property
    def fwd_prover_cfg(self) -> ProverConfig:
        assert self.is_fwd_prover
        assert isinstance(self._prover_cfg, ProverConfig)
        return self._prover_cfg

    @property
    def bwd_prover_cfg(self) -> ZMQProverParams:
        assert not self.is_fwd_prover
        assert isinstance(self._prover_cfg, ZMQProverParams)
        return self._prover_cfg

    @staticmethod
    def get_prover_config(
        env_name: str,
        params: TrainerArgs,
        prover_kind: AdvProverKind,
        fixed_prover: str,
    ) -> Union[ProverConfig, ZMQProverParams]:

        dataset_cfg = {"eq": params.eq.dataset, "mm": params.mm.dataset}[env_name]
        if prover_kind.is_forward:
            return forward_model_factory.make_prover_cfg(
                params,
                prover_type="greedy"
                if prover_kind == AdvProverKind.ForwardGreedy
                else "sampling",
                is_generator=False,
                async_model=True,
            )
        else:
            if prover_kind == AdvProverKind.BackwardGreedy:
                params.n_provers = 500
            assert prover_kind.is_backward
            dec_params: DecodingParams = ConfStore["decoding_bwd_eval"]
            if prover_kind == AdvProverKind.BackwardGreedy:
                dec_params.n_samples = 1
            return ZMQProverParams(
                dataset=dataset_cfg,  # type: ignore
                prover=ProverParams(
                    n_simultaneous_proofs=params.n_provers,
                    beam_path=Path(fixed_prover),
                    mcts=ConfStore["mcts_fast"],
                    beam_kind=BeamSearchKind.AutomaticallyReloading,
                    prover_kind=prover_kind.backward_kind,
                    no_proof_check=True,
                    dump_path=Path(params.dump_path),
                ),
                root_dir=Path(params.dump_path),
                decoding=dec_params,
            )

    @staticmethod
    def from_adv_cfg(adv_cfg) -> "AdvProverConfig":
        """
        WARNING: this method have a side effect, it calls "setup_exp_and_update_params"
        on intermediate TrainerArgs. Should be removed with a little bit of refacto
        """
        from configs.adv_configs import AdversarialConfig
        from evariste.trainer.launch import setup_exp_and_update_params

        assert isinstance(adv_cfg, AdversarialConfig)
        assert adv_cfg.get_worker_type() == WorkerType.PROVER_ACTOR

        train_cfg = adv_cfg.make_trainer_args()
        assert train_cfg.debug.debug == adv_cfg.debug
        assert train_cfg.rl_distributed.is_adversarial_training
        assert train_cfg.override_dump_path.endswith("prover_actor")

        # to create the logger, setup cuda, create folder,
        # but can probably be removed with a little work
        setup_exp_and_update_params(train_cfg)
        logger.warning("Called setup_exp_and_update_params for prover TrainerArgs")

        # WARNING: global_rank means rank_in_group
        # (vs local_rank which is rank_in_node) and NOT rank_in_slurm_job
        group_slurm_cfg = train_cfg.slurm_conf
        rank_in_group = group_slurm_cfg.global_rank

        use_zip = not train_cfg.rl_distributed.use_zmq
        distributed_cfg = train_cfg.rl_distributed
        assert all(wt.is_trainer() for wt in distributed_cfg.prover_actor_send_to)
        assert WorkerType.PROVER_ACTOR in distributed_cfg.generator_actor_send_to

        target_for_results = []
        if WorkerType.GENERATOR_TRAINER in distributed_cfg.prover_actor_send_to:
            target_for_results.append(
                (WorkerType.GENERATOR_TRAINER, adv_cfg.generator_send_only_failed)
            )
        if WorkerType.PROVER_TRAINER in distributed_cfg.prover_actor_send_to:
            if len(target_for_results) > 0 and use_zip:
                pass  # zip. no need to create a second folder
            else:
                target_for_results.append(
                    (WorkerType.PROVER_TRAINER, adv_cfg.prover_send_only_failed)
                )

        if use_zip:
            assert len(target_for_results) == 1
            assert not adv_cfg.generator_send_only_failed
            assert not adv_cfg.prover_send_only_failed

        # build prover configuration
        _prover_cfg = AdvProverConfig.get_prover_config(
            env_name=adv_cfg.env_name,
            params=train_cfg,
            prover_kind=adv_cfg.prover_kind,
            fixed_prover=adv_cfg.fixed_prover,
        )

        return AdvProverConfig(
            env_name=adv_cfg.env_name,
            prover_kind=adv_cfg.prover_kind,
            run_uuid=train_cfg.exp_id,
            _prover_cfg=_prover_cfg,
            rank=rank_in_group,
            rl_distributed=train_cfg.rl_distributed,
            target_for_results=target_for_results,
            max_input_len=train_cfg.batch.max_len,
            debug=adv_cfg.debug,
            fixed_prover=adv_cfg.fixed_prover,
        )

    def __post_init__(self):
        if self.is_fwd_prover:
            assert isinstance(self._prover_cfg, ProverConfig)
        else:
            assert isinstance(self._prover_cfg, ZMQProverParams)
            if self.prover_kind == AdvProverKind.BackwardGreedy:
                assert self.bwd_prover_cfg.decoding.n_samples == 1
        assert all(wt.is_trainer() for wt, _ in self.target_for_results)


BASE_OUTPUT_FOLDER = Path(f"")


def sequence_receiver_to_goal_stream(
    receiver: Receiver[AnnotatedGeneration],
    metadata: Dict[int, AnnotatedGeneration],
    received_stats: Dict[str, ActionCounter],
    debug: bool,
) -> GoalStream:
    log_interval = 1000 if not debug else 10
    total = 0
    goal_id = 0
    n_received = 0
    n_starve = 0
    started = False
    waiting_generations: Deque[AnnotatedGeneration] = deque(maxlen=10_000)
    logger.warning("Waiting to receive sequences")
    while True:
        annotated_generations = receiver.receive_batch()
        waiting_generations.extend(annotated_generations)
        n_received += len(annotated_generations)
        for annotated_gen in annotated_generations:
            received_stats[annotated_gen.src].act()

        if len(waiting_generations) == 0 and not started:
            time.sleep(1.0)
            continue

        if total % log_interval == 0:
            logger.info(
                f"In total, received {n_received} sequences, "
                f"created {goal_id} goals, starved: {n_starve}, "
                f"currently {len(waiting_generations)} waiting sequences."
            )

        total += 1
        if len(waiting_generations) == 0 and started:
            n_starve += 1
            time.sleep(0.1)
            yield None
            continue

        if not started:
            logger.info("Received first sequence!")
            started = True

        annotated_gen = waiting_generations.pop()
        metadata[goal_id] = annotated_gen
        yield goal_id, annotated_gen.prover_goal
        goal_id += 1


def proof_stream_from_backward(
    goal_stream: GoalStream, prover_config: ZMQProverParams, env_name: str
) -> Iterator[Tuple[int, Optional[BwdProof], bool]]:

    label_prefix = "proof_stream_from_bwd_"

    def input_it() -> Iterator[Optional[BackwardGoal]]:
        for maybe_goal in goal_stream:
            if maybe_goal is None:
                yield None  # TODO: have special class
                continue
            goal_id, goal = maybe_goal
            label = f"{label_prefix}{goal_id}"
            # TODO: dirty building of BackwardGoal. Should be cleaner when
            # ForwardGoal will have a field theorem
            if env_name == "mm":
                raise NotImplementedError
                # assert goal.statement is not None
                # assert goal.e_hyps is not None
                # all_toks = {t for t in goal.statement.split()} | {
                #     t for h in goal.e_hyps for t in h.split()
                # }
                # yield BackwardGoal(
                #     # name of the goal will be result.label,
                #     # used to send the sequence to the trainer
                #     label=label,
                #     # TODO: change it for other envs
                #     theorem=MMTheorem(
                #         conclusion=goal.statement,
                #         hyps=[(None, hyp) for hyp in goal.e_hyps],
                #         mand_disj=goal.mand_disj,
                #         # TODO: change this once ForwardGoal will
                #         # have a field mand_vars or theorem
                #         mand_vars=all_toks,
                #     ),
                # )
            elif env_name == "eq":
                goal = cast(ForwardGoal[EQTheorem], goal)
                assert isinstance(goal.thm, EQTheorem)
                assert goal.global_hyps is not None
                theorem = goal.thm

                assert eq_nodes_are_equal(
                    goal.thm.eq_hyps, [hyp.eq_node for hyp in goal.global_hyps]
                )
                yield BackwardGoal(label=label, theorem=theorem)
            else:
                raise NotImplementedError

    for x in init_and_run_prover(
        dataset=prover_config.dataset,
        decoding=prover_config.decoding,
        prover_params=prover_config.prover,
        input_it=input_it(),
        decoder_type="decoder",
    ):
        result = x[0]
        assert isinstance(result, ProofResult)
        assert result.goal.label.startswith(label_prefix)
        goal_id = int(result.goal.label[len(label_prefix) :])
        yield goal_id, result.proof, result.proof is not None


def proof_stream_from_forward(
    goal_stream: GoalStream, prover: ForwardProver
) -> Iterator[Tuple[int, GenerationHistory, bool]]:
    for goal_id, proof_search in prover.generate_proofs(goal_stream):
        assert isinstance(proof_search, StandardProofSearch)
        yield goal_id, proof_search.generation, proof_search.info.solved


def prove(cfg: AdvProverConfig):
    log_interval = 60
    assert (
        cfg.worker_type == WorkerType.PROVER_ACTOR
    ), f"{cfg.worker_type} is not a prover actor"
    checkpoint_dir = os.path.join(
        cfg.rl_distributed.exp_root_path, WorkerType.PROVER_TRAINER
    )
    logger.info(f"Using checkpoint dir: {checkpoint_dir}")

    output_path = Path(cfg.output_path)
    if not output_path.exists():
        prepare_folder(cfg, verbose=False)
    with open(output_path / "config.json", "w") as f:
        f.write(cfg.to_json())
    logging.info(f"Config: {pprint.pformat(asdict(cfg))}")

    metadata: Dict[int, AnnotatedGeneration] = {}
    goal_receiver: Receiver[AnnotatedGeneration] = make_receiver(
        rank=cfg.rank, receiver_type=WorkerType.PROVER_ACTOR, cfg=cfg.rl_distributed
    )
    try:
        goal_receiver.receive_batch()  # init zmq thing if needed
    except ZMQNotReady:
        pass

    received_stats: Dict[str, ActionCounter] = defaultdict(
        lambda: ActionCounter(name="receiver_stat", is_rate=True, silent=True)
    )

    # not using the RLDistributedConfig.prover_actor_send_to inside make_sender method
    # directly because we want to be able to send only failed
    def _output_sender(receiver_type: WorkerType):
        store = make_sender(
            rank=cfg.rank,
            sender_type=WorkerType.PROVER_ACTOR,
            cfg=cfg.rl_distributed,
            debug=cfg.debug,
            receiver_type=receiver_type,
        )
        return store

    senders: List[Tuple[Sender[AnnotatedGeneration], bool, WorkerType]] = []
    for target, only_failed in cfg.target_for_results:
        logging.info(f"Adding sender {target}. Only failed? {only_failed}")
        senders.append((_output_sender(receiver_type=target), only_failed, target))

    logging.info(f"Store done")

    ckpt_path: Optional[str] = None
    if cfg.fixed_prover:
        assert cfg.rl_distributed.n_prover_trainers == 0
        ckpt_path = cfg.fixed_prover
        logger.info(f"Using a fixed checkpoint for prover: {ckpt_path}")

    while ckpt_path is None:
        logging.info(f"Waiting for a prover checkpoint in {checkpoint_dir} ...")
        ckpt_path, cur_version = get_latest_checkpoint(checkpoint_dir)
        time.sleep(10.0)
        if ckpt_path is not None:
            logger.info(f"Starting from latest prover: {ckpt_path}")

    # sanity check
    assert ckpt_path is not None and os.path.isfile(ckpt_path)

    # fwd_prover: Optional[ForwardProver] = None
    fwd_prover: Optional[ForwardProver] = None
    if cfg.prover_kind.is_forward:
        assert cfg.fwd_prover_cfg.async_model, "not using async_model!"
        fwd_prover, dico, params, _ = forward_model_factory.from_checkpoint(
            cfg=cfg.fwd_prover_cfg, ckpt_path=ckpt_path, device_str="cuda"
        )
    else:
        # Automatically reloading requires a folder, so .parent
        cfg.bwd_prover_cfg.prover.beam_path = Path(ckpt_path).parent
        cfg.bwd_prover_cfg.prover.dump_path = Path(cfg.dump_path)
        if cfg.prover_kind == AdvProverKind.BackwardGreedy:
            assert cfg.bwd_prover_cfg.decoding.n_samples == 1

    last_gen_rewards = []
    last_proved = []
    start = time.time()
    logging.info("Starting proving!")
    metrics = Logger(outdir=cfg.dump_path, tag="prover", quiet=cfg.rank != 0)
    metrics.log_config(cfg)
    last_log = time.time()

    gpu_mon = GPUMonitor(delay=1.0)  # probes gpu activity every seconds
    proved_counter: Dict[str, ActionCounter] = defaultdict(
        lambda: ActionCounter(name="proved_counter", is_rate=False, silent=True)
    )
    too_long: Dict[str, ActionCounter] = defaultdict(
        lambda: ActionCounter(name="too_long", is_rate=False, silent=True)
    )
    processed = ActionCounter(name="processed", is_rate=True, silent=True)

    # using ZipSender as store
    gen_log_dir = Path(cfg.dump_path) / f"generations"
    gen_log_dir.mkdir(parents=True, exist_ok=True)
    generation_store = open(gen_log_dir / f"gen_{cfg.rank}.pkll", "wb")

    with ExitStack() as stack:
        stack.enter_context(logged_closing(goal_receiver, "goal_receiver"))
        if cfg.prover_kind.is_forward:
            assert fwd_prover is not None
            stack.enter_context(logged_closing(fwd_prover, "prover"))
        stack.enter_context(logged_closing(metrics, "metrics"))
        stack.enter_context(logged_closing(gpu_mon, "gpu_mon"))
        for sender, _of, wt in senders:
            stack.enter_context(logged_closing(sender, f"sender_{wt}"))

        goal_stream = sequence_receiver_to_goal_stream(
            goal_receiver, metadata, received_stats=received_stats, debug=cfg.debug
        )
        # very likely to be unnecessary
        stack.enter_context(logged_closing(goal_stream, "goal_stream"))

        if cfg.prover_kind.is_backward:
            proof_stream = proof_stream_from_backward(
                goal_stream, cfg.bwd_prover_cfg, cfg.env_name
            )
        else:
            # TODO: unify types
            assert fwd_prover is not None
            proof_stream = proof_stream_from_forward(goal_stream, fwd_prover)  # type: ignore

        # very likely to be unnecessary
        stack.enter_context(logged_closing(proof_stream, "proof_stream"))

        # for each generated goal processed by the prove
        for goal_id, maybe_proof, proved in proof_stream:

            # retrieve associated generation
            annotated: AnnotatedGeneration = metadata.pop(goal_id)
            if len(annotated.generation.forward_steps()) == 0:
                raise RuntimeError("Should have been filtered!")

            # update annotation
            # TODO: make more elaborated reward (based on how difficult the proof was)
            annotated._proved = proved
            annotated._reward = float(not proved)

            # update stats
            processed.act()
            slen = annotated.generation_stats.statement_n_tok
            too_long[annotated.src].act(float(slen > cfg.max_input_len))
            proved_counter[annotated.src].act(float(proved))
            last_gen_rewards.append(annotated.generation_reward)
            last_proved.append(float(proved))

            # dump (debugging purposes only)
            prop_gen_saved = 0.05
            if random.random() < prop_gen_saved:
                pickle.dump(
                    {
                        "timestamp": time.time(),
                        "goal_id": goal_id,
                        "annotated": annotated,
                        "prover_ckpt": cur_version,
                        "maybe_proof": maybe_proof,
                    },
                    generation_store,
                )
                generation_store.flush()

            # send to trainers
            failed = not proved
            for sender, only_failed, _wt in senders:  # potentially both trainers
                assert _wt.is_trainer()
                if failed or not only_failed:
                    sender.store(annotated)

            # potentially reload more recent model (forward only,
            # handled by BeamSearchModel for the backward prover)
            if cfg.prover_kind.is_forward and not cfg.fixed_prover:
                _ckpt_path, prover_version = get_latest_checkpoint(checkpoint_dir)
                if prover_version > cur_version:
                    logging.info(f"Found version {prover_version}, updating model")
                    cur_version = prover_version
                    assert fwd_prover is not None
                    fwd_prover.reload_model_weights()

            # log stats
            if len(last_gen_rewards) == 1000:
                logging.info(
                    f"Last avg final rewards: {np.mean(last_gen_rewards)} -- "
                    f"Last proving proportion: {np.mean(last_proved)} -- "
                    f"Total time: {time.time() - start}"
                )
                last_gen_rewards.clear()
                last_proved.clear()

            if time.time() - last_log > log_interval:
                last_log = time.time()
                gpu_stats = gpu_mon.stats[torch.cuda.current_device()].stats
                to_log = dict(
                    ram_mem_util_gb=psutil.Process().memory_info().rss / 1024 ** 3,
                    ram_mem_avail_gb=psutil.virtual_memory().available / 1024 ** 3,
                    gpu_usage=gpu_stats["gpu"],
                    gpu_memory=gpu_stats["mem"],
                    received=goal_receiver.rate_and_reset(),
                    processed_rate=processed.rate_and_reset(),
                    model_version=cur_version,
                )
                for src, counter in proved_counter.items():
                    to_log[f"proved/{src}"] = counter.rate_and_reset()
                for src, counter in too_long.items():
                    to_log[f"too_long/{src}"] = counter.rate_and_reset()
                for src, counter in received_stats.items():
                    to_log[f"received/{src}"] = counter.rate_and_reset()
                for sender, _of, worker_type in senders:
                    to_log[f"sent/{worker_type}"] = sender.rate_and_reset()
                metrics.log_metrics(to_log)
                logging.info(to_log)


def _make_output_path() -> str:
    output_folder = BASE_OUTPUT_FOLDER
    output_folder.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{now}"
    return str(output_folder / name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = cfg_from_cli(schema=AdvProverConfig)
    output_path = _make_output_path()
    cfg.rl_distributed.exp_root_path = output_path
    cfg.run_uuid = os.path.basename(output_path)
    if not cfg.is_fwd_prover:
        cfg.bwd_prover_cfg.eq_dataset = ConfStore["eq_dataset_basic"]
    prove(cfg)
