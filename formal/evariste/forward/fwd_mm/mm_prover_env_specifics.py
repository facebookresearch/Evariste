# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

from evariste.envs.mm.env import MetamathEnv
from evariste.forward.env_specifics.prover_env_specifics import (
    FwdTrainParams,
    ProverEnvSpecifics,
)
from evariste.forward.fwd_mm.mm_fwd_env import MMForwardEnv
from evariste.forward.fwd_mm.mm_fwd_tasks import MMFwdFormat, use_critic
from evariste.forward.fwd_mm.mm_fwd_tokenizer import MMFwdTokenizer
from evariste.trainer.args import TrainerArgs

logger = getLogger()


def mm_env_specifics(
    params: TrainerArgs, mm_env: MetamathEnv = None
) -> ProverEnvSpecifics:
    """
    Create all the objects needed by the forward prover to work on MM
    """
    fmt = MMFwdFormat.from_trainer_args(params)
    logger.info(f"Using fmt for forward: {fmt}")

    discr_conditioning = False
    if fmt.hum_vs_gen_disc:
        discr_conditioning = True
        if params.gan.fixed_generator:
            logger.warning(
                "Disactivating discr_conditioning at proving time"
                "HACK to be able to reload a used a fixed generator"
                "trained without discr embedding"
            )
            discr_conditioning = False

    fwd_train_params = FwdTrainParams(
        max_inp_len=params.batch.max_len,
        stop_symbol=fmt.stop_symbol,
        is_generator=fmt.is_generation,
        use_stop_action=fmt.use_stop_action,
        use_critic=any(use_critic(t) for t in params.parsed_tasks("mm")),
        train_dir=params.dump_path,
        command_prefix=fmt.cmd_prefix,
        discr_conditioning=discr_conditioning,
    )

    logger.info(f"Using fwd_train_params for forward: {fwd_train_params}")

    fwd_tokenizer = MMFwdTokenizer.from_trainer_args(params)
    fwd_env = MMForwardEnv.from_trainer_args(mm_env=mm_env, params=params)

    # print("WARNING\n" * 10)
    # print("Using fake async env to test")
    # fwd_env = DebugMMAsyncFwdEnv(fwd_env)

    return ProverEnvSpecifics(fwd_train_params, fwd_tokenizer, fwd_env)


def build_mm_env(params: TrainerArgs) -> MetamathEnv:
    from evariste.model.data.envs.metamath import MetamathDataEnvironment

    return MetamathDataEnvironment.build_mm_env(params.mm.dataset.database_path)
