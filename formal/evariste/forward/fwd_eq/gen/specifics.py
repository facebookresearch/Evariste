# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Any
from evariste.datasets.equations import EquationsDatasetConf
from evariste.envs.eq.env import EquationEnv
from evariste.forward.env_specifics.prover_env_specifics import (
    FwdTrainParams,
    ProverEnvSpecifics,
)
from evariste.forward.fwd_eq.gen.env import EqGenForwardEnv
from evariste.forward.fwd_eq.gen.tokenizer import EqGenFwdTokenizer
from evariste.model.data.dictionary import EOS_WORD
from evariste.trainer.args import TrainerArgs


def eq_gen_env_specifics_from_trainer_args(
    params: TrainerArgs, eq_env: EquationEnv
) -> ProverEnvSpecifics:
    fwd_train_params = FwdTrainParams(
        max_inp_len=params.batch.max_len,
        stop_symbol=EOS_WORD,
        is_generator=True,
        use_stop_action=False,
        use_critic=False,
        train_dir=params.dump_path,
        command_prefix=None,
    )
    fwd_tokenizer = EqGenFwdTokenizer()
    fwd_env = EqGenForwardEnv(eq_env=eq_env, params=params.eq.dataset,)

    return ProverEnvSpecifics(fwd_train_params, fwd_tokenizer, fwd_env)


def eq_gen_env_specifics_for_gen(
    params: EquationsDatasetConf,
    eq_env: Optional[EquationEnv] = None,
    seed: Optional[int] = None,
) -> ProverEnvSpecifics:
    fwd_train_params = FwdTrainParams(
        max_inp_len=0,
        stop_symbol="unused",
        is_generator=True,
        use_stop_action=False,
        use_critic=False,
        train_dir="unused",
        command_prefix=None,
    )
    fwd_tokenizer = EqGenFwdTokenizer()
    eq_env = eq_env or EquationEnv.build(params.env, seed=seed)
    fwd_env = EqGenForwardEnv(eq_env=eq_env, params=params,)
    return ProverEnvSpecifics(fwd_train_params, fwd_tokenizer, fwd_env)
