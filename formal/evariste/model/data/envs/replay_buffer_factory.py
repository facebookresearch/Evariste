# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

from evariste.comms.comms import make_receiver
from evariste.comms.store import AnnotatedGeneration
from evariste.forward.online_generation.worker_type import WorkerType
from evariste.model.data.envs.replay_buffer_loader import ReplayBuffer
from evariste.trainer.args import TrainerArgs


logger = getLogger()


def build_adv_replay_buffer(
    env_name: str, worker_type: WorkerType, args: TrainerArgs
) -> ReplayBuffer[AnnotatedGeneration]:
    """
    Build replay buffer for annotated generations.
    """
    assert worker_type in [WorkerType.GENERATOR_TRAINER, WorkerType.PROVER_TRAINER]
    assert requires_adv_rb(env_name, args)
    assert args.rl_distributed.is_adversarial_training, args.rl_distributed
    logger.info(f"Building adversarial replay buffer for {worker_type} ({env_name})")

    rb_params = args.rl_params.replay_buffer

    if rb_params.filter_if_rewards_zero:
        if (
            env_name == "mm"
            and rb_params.filter_if_rewards_zero
            and args.mm.graph.proved_conditioning
            and args.parsed_tasks("mm_gen")
        ):
            raise ValueError(
                "You can not filter seqs with zero rewards when using reward "
                "conditioned generation"
            )
        logger.warning("Replay buffer: filtering sequences when all rewards are zero")

    receiver = make_receiver(
        rank=args.slurm_conf.global_rank,
        receiver_type=worker_type,
        cfg=args.rl_distributed,
    )

    logger.info(f"Creating adversarial replay buffer with args: {rb_params}")
    return ReplayBuffer(receiver=receiver, rb_params=rb_params)


def requires_adv_rb(env_name: str, args: TrainerArgs) -> bool:
    use_rb = any(task.endswith("_rl") for task in args.parsed_tasks(env_name))
    if env_name == "mm":
        _use_rb_for_seq2seq_task = (
            args.rl_distributed.is_adversarial_training
            and (
                len(args.parsed_tasks("mm_fwd_seq2seq")) > 0
                or len(args.parsed_tasks("mm_gen_seq2seq")) > 0
                or len(args.parsed_tasks("mm_gen_disc")) > 0
            )
            and args.mm.graph.generated_prob > 0.0
        )
        use_rb |= _use_rb_for_seq2seq_task
    logger.info(f"=== requires_adv_rb? {use_rb}")
    return use_rb
