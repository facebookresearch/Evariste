# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from params.params import cfg_from_cli, ConfStore
from configs.fwd_configs import register_fwd_cfgs
from configs.bwd_configs import register_bwd_cfgs

from evariste import slurm_conf_factory
from evariste.forward.online_generation.fwd_rl_actor import (
    is_fwd_rl_actor,
    run_fwd_rl_actor_from_trainer_args,
)
from evariste.trainer.args import TrainerArgs
from evariste.trainer.launch import train


def main(args: TrainerArgs) -> None:
    if is_fwd_rl_actor(args):
        run_fwd_rl_actor_from_trainer_args(args)
    else:
        # run experiment
        train(args)


if __name__ == "__main__":
    register_fwd_cfgs()
    register_bwd_cfgs()

    cfg: TrainerArgs = cfg_from_cli(ConfStore["default_cfg"])
    cfg.slurm_conf = slurm_conf_factory.from_trainer_args(cfg)
    main(cfg)
