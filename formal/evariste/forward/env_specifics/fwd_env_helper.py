# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Any
import abc

from evariste.datasets import DatasetConf, LeanDatasetConf
from evariste.forward.env_specifics.fwd_goal_factory import ForwardGoalFactory
from evariste.forward.env_specifics.generation_annotator import GenerationAnnotator
from evariste.forward.env_specifics.prover_env_specifics import ProverEnvSpecifics
from evariste.model.data.dictionary import Dictionary
from evariste.trainer.args import TrainerArgs


class FwdEnvHelper(abc.ABC):
    @abc.abstractmethod
    def get_env_name(self) -> str:
        pass

    @abc.abstractmethod
    def get_prover_env_specifics(self) -> ProverEnvSpecifics:
        pass

    # helper to make the annotation of generation env agnostic
    @abc.abstractmethod
    def get_annotator(self) -> GenerationAnnotator:
        pass

    # goal factory methods
    @abc.abstractmethod
    def get_goal_factory(self) -> ForwardGoalFactory:
        pass

    # factory method
    @classmethod
    def from_trainer_args(
        cls,
        dico: Dictionary,
        params: Any,
        env: Optional[Any] = None,
        env_name: Optional[str] = None,
        overwrite_dataset: Optional[DatasetConf] = None,
        prover_dir: Optional[str] = None,
    ) -> "FwdEnvHelper":

        if not env_name:
            from evariste.forward.forward_model_factory import get_env_name

            env_name = get_env_name(params)

        env_helper: FwdEnvHelper
        if env_name == "mm":
            from evariste.forward.fwd_mm.mm_env_helper import MMFwdEnvHelper
            from evariste.model.data.envs.metamath import MetamathDataEnvironment

            assert overwrite_dataset is None

            mm_env = None
            if env:
                assert isinstance(env, MetamathDataEnvironment)
                mm_env = env.mm_env
            env_helper = MMFwdEnvHelper(params=params, dico=dico, mm_env=mm_env)
        elif env_name == "eq":
            from evariste.forward.fwd_eq.eq_env_helper import EQFwdEnvHelper

            assert overwrite_dataset is None

            if env:
                from evariste.model.data.envs.equations import EquationsEnvironment

                assert isinstance(env, EquationsEnvironment)
            env_helper = EQFwdEnvHelper(params=params, dico=dico, eq_data_env=env)
        elif env_name == "eq_gen":
            from evariste.forward.fwd_eq.gen.helper import EQGenFwdEnvHelper

            assert overwrite_dataset is None
            if env:
                from evariste.model.data.envs.equations import EquationsEnvironment

                assert isinstance(env, EquationsEnvironment)
            env_helper = EQGenFwdEnvHelper(params=params, dico=dico, eq_data_env=env)

        elif env_name == "hl":
            from evariste.forward.fwd_hl.hl_env_helper import HLFwdEnvHelper

            assert overwrite_dataset is None

            env_helper = HLFwdEnvHelper(dico=dico, params=params)
        elif env_name == "lean":
            from evariste.forward.fwd_lean.lean_env_helper import LeanFwdEnvHelper

            assert overwrite_dataset is None or isinstance(
                overwrite_dataset, LeanDatasetConf
            )

            env_helper = LeanFwdEnvHelper(
                params=params,
                overwrite_dataset=overwrite_dataset,
                prover_dir=prover_dir,
                dico=dico,
            )
        else:
            raise RuntimeError(f"Unknown environment: {env_name}")

        return env_helper


def get_env_name(params: TrainerArgs):
    env_names = [task.split("_")[0] for task in params.parsed_tasks()]
    if len(set(env_names)) != 1:
        raise NotImplementedError(f"Not implemented for mix of envs: {env_names}")
    if env_names[0] == "eq" and any(["newgen" in t for t in params.parsed_tasks("eq")]):
        return "eq_gen"
    return env_names[0]
