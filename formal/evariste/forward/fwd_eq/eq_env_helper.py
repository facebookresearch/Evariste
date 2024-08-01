# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from typing import Optional

from evariste.forward.env_specifics.fwd_goal_factory import ForwardGoalFactory
from evariste.forward.fwd_eq.eq_fwd_goal_factory import EQForwardGoalFactory
from evariste.forward.fwd_eq.eq_prover_env_specifics import eq_env_specifics
from evariste.forward.env_specifics.prover_env_specifics import ProverEnvSpecifics
from evariste.trainer.args import TrainerArgs
from evariste.forward.fwd_eq.eq_generation_annotator import EQGenerationAnnotator
from evariste.forward.env_specifics.fwd_env_helper import FwdEnvHelper
from evariste.model.data.envs.equations import EquationsEnvironment
from evariste.model.data.dictionary import Dictionary


logger = getLogger()


class EQFwdEnvHelper(FwdEnvHelper):
    def __init__(
        self,
        params: TrainerArgs,
        dico: Dictionary,
        eq_data_env: Optional[EquationsEnvironment] = None,
    ):
        self.params = params
        self.dico = dico
        self._eq_data_env = eq_data_env

    def get_env_name(self) -> str:
        return "eq"

    def get_prover_env_specifics(self) -> ProverEnvSpecifics:
        eq_env = self.eq_data_env.eq_env
        return eq_env_specifics(self.params, eq_env)

    def get_annotator(self) -> EQGenerationAnnotator:
        return EQGenerationAnnotator(self.params)

    @property
    def eq_data_env(self) -> EquationsEnvironment:
        if self._eq_data_env is None:
            eq_data_env = EquationsEnvironment(
                Dictionary.create_empty(), self.params, fast=True
            )
            self._eq_data_env = eq_data_env
        return self._eq_data_env

    def get_goal_factory(self) -> ForwardGoalFactory:
        return EQForwardGoalFactory(self.eq_data_env, self.params)

    def close(self):
        if self._eq_data_env:
            self._eq_data_env.close()
