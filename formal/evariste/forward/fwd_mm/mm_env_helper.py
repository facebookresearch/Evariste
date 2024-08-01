# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from typing import Optional, List, Dict


from evariste.envs.mm.env import MetamathEnv
from evariste.forward.env_specifics.fwd_env_helper import FwdEnvHelper
from evariste.forward.env_specifics.fwd_goal_factory import ForwardGoalFactory
from evariste.forward.fwd_mm.mm_fwd_goal_factory import MMForwardGoalFactory
from evariste.forward.fwd_mm.mm_prover_env_specifics import (
    build_mm_env,
    mm_env_specifics,
)
from evariste.forward.fwd_mm.mm_fwd_tasks import MMFwdFormat
from evariste.forward.fwd_mm.mm_generation_annotator import MMGenerationAnnotator
from evariste.forward.env_specifics.prover_env_specifics import ProverEnvSpecifics
from evariste.model.data.dictionary import Dictionary
from evariste.trainer.args import TrainerArgs

logger = getLogger()


class MMFwdEnvHelper(FwdEnvHelper):
    def __init__(
        self,
        params: TrainerArgs,
        dico: Dictionary,
        mm_env: Optional[MetamathEnv] = None,
    ):
        self.params = params
        self.dico = dico
        self._mm_env = mm_env
        self._splits: Optional[Dict[str, List[str]]] = None

        self._fmt: Optional[MMFwdFormat] = None

    def get_env_name(self) -> str:
        return "mm"

    def get_prover_env_specifics(self) -> ProverEnvSpecifics:
        return mm_env_specifics(self.params, mm_env=self.mm_env())

    def get_annotator(self) -> MMGenerationAnnotator:
        return MMGenerationAnnotator(
            params=self.params, mm_env=self.mm_env(), dico=self.dico
        )

    @property
    def fmt(self) -> MMFwdFormat:
        if self._fmt is None:
            self._fmt = MMFwdFormat.from_trainer_args(self.params)
            logger.info(f"Forward prover fmt: {self._fmt}")
        return self._fmt

    def mm_env(self) -> MetamathEnv:
        if self._mm_env is None:
            self._mm_env = build_mm_env(self.params)
        return self._mm_env

    def get_goal_factory(self) -> ForwardGoalFactory:
        return MMForwardGoalFactory(
            params=self.params, mm_env=self.mm_env(), fmt=self.fmt
        )
