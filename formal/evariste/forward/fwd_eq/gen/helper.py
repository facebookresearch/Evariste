# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from evariste.forward.fwd_eq.eq_env_helper import EQFwdEnvHelper

from evariste.forward.env_specifics.prover_env_specifics import ProverEnvSpecifics
from evariste.forward.fwd_eq.gen.specifics import eq_gen_env_specifics_from_trainer_args


class EQGenFwdEnvHelper(EQFwdEnvHelper):
    def get_env_name(self) -> str:
        return "eq_gen"

    def get_prover_env_specifics(self) -> ProverEnvSpecifics:
        return eq_gen_env_specifics_from_trainer_args(
            self.params, self.eq_data_env.eq_env
        )
