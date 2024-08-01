# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from pathlib import Path
from typing import Optional

from evariste.backward.env.lean.env import LeanExpanderEnv
from evariste.datasets import LeanDatasetConf
from evariste.forward.env_specifics.fwd_env_helper import FwdEnvHelper
from evariste.forward.fwd_lean.lean_fwd_env import LeanAsyncForwardEnv
from evariste.forward.fwd_lean.lean_fwd_goal_factory import LeanForwardGoalFactory
from evariste.forward.fwd_lean.lean_fwd_thm_tokenizer import LeanFwdThmTokenizer
from evariste.forward.fwd_lean.lean_fwd_tokenizer import LeanFwdTokenizer
from evariste.forward.env_specifics.generation_annotator import GenerationAnnotator
from evariste.forward.env_specifics.prover_env_specifics import (
    ProverEnvSpecifics,
    FwdTrainParams,
)
from evariste.model.data.dictionary import EOS_WORD, EOU_WORD, Dictionary
from evariste.model.data.envs.lean import LeanDataEnvironment
from evariste.trainer.args import TrainerArgs

logger = getLogger()


class LeanFwdEnvHelper(FwdEnvHelper):
    def __init__(
        self,
        params: TrainerArgs,
        dico: Dictionary,
        prover_dir: Optional[str],
        overwrite_dataset: Optional[LeanDatasetConf] = None,
    ):
        self.params = params
        self.dico = dico
        self.overwrite_dataset = overwrite_dataset
        if self.overwrite_dataset:
            assert isinstance(overwrite_dataset, LeanDatasetConf)
        self._expander_env: Optional[LeanExpanderEnv] = None
        self._data_env: Optional[LeanDataEnvironment] = None
        self._dataset_conf: Optional[LeanDatasetConf] = None
        self._prover_dir = prover_dir

    def get_dataset_conf(self) -> LeanDatasetConf:
        if self._dataset_conf is None:
            conf = (
                self.overwrite_dataset
                if self.overwrite_dataset
                else self.params.lean.dataset
            )
            conf = conf.get_materialized()
            self._dataset_conf = conf
        if self._prover_dir is None:
            assert not self._dataset_conf.dump_proof_logs
        return self._dataset_conf

    def _get_expander_env(self) -> LeanExpanderEnv:
        if self._expander_env is None:
            logger.info(
                f"Creating LeanExpanderEnv with config: {self.get_dataset_conf()}"
            )
            if self._prover_dir is None:
                assert not self._dataset_conf.dump_proof_logs
                prover_dir = Path("not_used")
            else:
                prover_dir = Path(self._prover_dir)

            dataset_conf = self.get_dataset_conf()
            self._expander_env = LeanExpanderEnv(
                dataset=dataset_conf, dump_path=prover_dir
            )
            logger.info("Done with creating LeanExpanderEnv")
        return self._expander_env

    def get_env_name(self) -> str:
        return "lean"

    def get_prover_env_specifics(self) -> ProverEnvSpecifics:
        is_generation = False

        fwd_tokenizer = LeanFwdTokenizer.from_trainer_args(self.params)
        stop_symbol = fwd_tokenizer.last_token

        fwd_train_params = FwdTrainParams(
            max_inp_len=self.params.batch.max_len,
            stop_symbol=stop_symbol,
            is_generator=is_generation,
            use_stop_action=False,
            use_critic=False,
            train_dir=self.params.dump_path,
            command_prefix=None,
        )

        fwd_env = LeanAsyncForwardEnv(expander_env=self._get_expander_env())

        return ProverEnvSpecifics(fwd_train_params, fwd_tokenizer, fwd_env)

    def get_goal_factory(self) -> LeanForwardGoalFactory:
        return LeanForwardGoalFactory(dataset=self.get_dataset_conf(), dico=self.dico)

    def get_annotator(self) -> GenerationAnnotator:
        raise NotImplementedError
