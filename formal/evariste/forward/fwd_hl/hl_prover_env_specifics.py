# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from evariste.envs.hl.api import HOLLightAPI
from evariste.forward.env_specifics.prover_env_specifics import (
    FwdTrainParams,
    ProverEnvSpecifics,
)
from evariste.forward.fwd_hl.hl_fwd_env import HLForwardEnv
from evariste.forward.fwd_hl.hl_fwd_tokenizer import HLFwdTokenizer
from evariste.model.data.dictionary import EOS_WORD
from evariste.trainer.args import TrainerArgs


def hl_env_specifics(params: TrainerArgs) -> ProverEnvSpecifics:
    """
    Create all the objects needed by the forward prover to work on HL
    """
    is_generation = are_hl_generation_tasks(params)

    fwd_train_params = FwdTrainParams(
        max_inp_len=params.batch.max_len,
        stop_symbol=EOS_WORD,
        is_generator=is_generation,
        use_stop_action=False,
        use_critic=False,
        train_dir=params.dump_path,
        command_prefix=None,
    )

    fwd_tokenizer = HLFwdTokenizer(is_generation=is_generation)
    fwd_env = HLForwardEnv(
        hl_bwd_env=HOLLightAPI(
            checkpoint_path=params.hl.dataset.checkpoint_path,
            timeout=params.hl.dataset.timeout,
        )
    )

    return ProverEnvSpecifics(fwd_train_params, fwd_tokenizer, fwd_env)


def are_hl_generation_tasks(params: TrainerArgs):
    tasks = params.parsed_tasks("hl")
    are_generation = []
    for task in tasks:
        if task.startswith("hl_fwd_"):
            is_generation = False
        elif task.startswith("hl_gen_"):
            is_generation = True
        else:
            raise NotImplementedError(task)
        are_generation.append(is_generation)
    if len(set(are_generation)) != 1:
        raise NotImplementedError(f"Mismatch between tasks {tasks}: {are_generation}")
    is_generation = are_generation[0]
    return is_generation
