# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from evariste.envs.eq.env import EquationEnv
from evariste.forward.env_specifics.prover_env_specifics import (
    FwdTrainParams,
    ProverEnvSpecifics,
)
from evariste.forward.fwd_eq.eq_fwd_env import EqForwardEnv
from evariste.forward.fwd_eq.eq_fwd_tokenizer import EqFwdTokenizer
from evariste.model.data.dictionary import EOS_WORD
from evariste.trainer.args import TrainerArgs


def eq_env_specifics(params: TrainerArgs, eq_env: EquationEnv) -> ProverEnvSpecifics:
    """
    Create all the objects needed by the forward prover to work on EQ
    """
    is_generation = are_eq_generation_tasks(params)

    fwd_train_params = FwdTrainParams(
        max_inp_len=params.batch.max_len,
        stop_symbol=EOS_WORD,
        is_generator=is_generation,
        use_stop_action=params.eq.stop_action,
        use_critic=False,
        train_dir=params.dump_path,
        command_prefix=None,
    )

    fwd_tokenizer = EqFwdTokenizer(is_generation=is_generation)
    fwd_env = EqForwardEnv(eq_env=eq_env, rule_env=params.eq.dataset.rule_env)

    return ProverEnvSpecifics(fwd_train_params, fwd_tokenizer, fwd_env)


def are_eq_generation_tasks(params: TrainerArgs):
    tasks = params.parsed_tasks("eq")
    are_generation = []
    for task in tasks:
        if task == "eq_notask":
            return False
        if task == "eq_gen_notask":
            return True

        if task.startswith("eq_fwd_"):
            is_generation = False
        elif task.startswith("eq_gen_"):
            is_generation = True
        else:
            raise NotImplementedError(task)
        are_generation.append(is_generation)
    if len(set(are_generation)) != 1:
        raise NotImplementedError(f"Mismatch between tasks {tasks}: {are_generation}")
    is_generation = are_generation[0]
    return is_generation
