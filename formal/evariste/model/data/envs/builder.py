# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from typing import Dict
import numpy as np

from evariste.model.data.envs.latex import LatexDataEnvironment
from evariste.model.data.envs.hol_light import HOLLightDataEnvironment
from evariste.model.data.envs.metamath import MetamathDataEnvironment
from evariste.model.data.envs.equations import EquationsEnvironment
from evariste.model.data.envs.lean import LeanDataEnvironment
from evariste.model.data.envs.isabelle import IsabelleDataEnvironment
from evariste.model.data.envs.multi import MultiEnvironment
from evariste.model.data.envs.sr import SRDataEnvironment

from evariste.model.data.dictionary import Dictionary
from evariste.trainer.args import TrainerArgs


logger = getLogger()


def build_envs(dico: Dictionary, params: TrainerArgs) -> Dict:
    """
    Build environments.
    """
    # environment random seed
    if params.env_base_seed < 0:
        params.env_base_seed = np.random.randint(1_000_000_000)
        logger.info(f"env_base_seed -- setting to {params.env_base_seed}")
    else:
        logger.info(f"env_base_seed -- set to {params.env_base_seed}")

    # build environments
    latex_env = LatexDataEnvironment(dico, params)
    hl_env = HOLLightDataEnvironment(dico, params)
    mm_env = MetamathDataEnvironment(dico, params)
    eq_env = EquationsEnvironment(dico, params)
    lean_env = LeanDataEnvironment(dico, params)
    isabelle_env = IsabelleDataEnvironment(dico, params)
    sr_env = SRDataEnvironment(dico, params)

    # available environments
    envs = {
        "latex": latex_env,
        "hl": hl_env,
        "mm": mm_env,
        "eq": eq_env,
        "lean": lean_env,
        "isabelle": isabelle_env,
        "sr": sr_env,
    }

    # build multi environment
    multi_env = MultiEnvironment(dico, params, envs)
    envs["multi"] = multi_env

    # update dictionary parameters
    if params.dico.n_words < 0:
        params.dico = dico.conf
    else:
        assert params.dico == dico.conf, (params.dico.to_json(), dico.conf.to_json())

    return envs
