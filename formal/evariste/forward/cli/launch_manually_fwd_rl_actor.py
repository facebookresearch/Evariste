# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import datetime
import getpass
import logging
from pathlib import Path
from typing import cast

from evariste.forward import forward_model_configs
from evariste.forward.online_generation.fwd_rl_actor import (
    FwdRLActorConfig,
    run_fwd_rl_actor,
)
from params.params import cfg_from_cli

BASE_OUTPUT_FOLDER = Path(f"evariste/online_generation_tests/")


def _make_output_path(cfg: FwdRLActorConfig) -> str:
    output_folder = BASE_OUTPUT_FOLDER
    output_folder.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"generations_{cfg.prover.name}_{now}"
    return str(output_folder / name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    forward_model_configs.register_model_cfgs()
    forward_model_configs.register_prover_cfgs()
    cfg = cfg_from_cli(base_config=FwdRLActorConfig())
    cfg = cast(FwdRLActorConfig, cfg)
    output_path = _make_output_path(cfg)
    cfg.output_path = output_path
    cfg.rl_distributed.exp_root_path = output_path
    run_fwd_rl_actor(cfg)
