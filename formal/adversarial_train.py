# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from dataclasses import asdict
from logging import getLogger
import os
import pickle

from evariste import json as json
from params.params import cfg_from_cli
from configs.adv_configs import AdversarialConfig
from evariste.adversarial.generator import AdvGeneratorConfig
from evariste.adversarial.prover import AdvProverConfig
from evariste.adversarial import generator, prover
from evariste.forward.online_generation.worker_type import WorkerType
from evariste.trainer.launch import train


logger = getLogger()


def main(cfg: AdversarialConfig) -> None:
    cfg.check_adversarial_config()
    if cfg.is_master():
        with open(os.path.join(cfg.exp_path, "params.pkl"), "wb") as f:
            pickle.dump(cfg, f)
        json.dump(
            asdict(cfg),
            open(os.path.join(cfg.exp_path, "params.json"), "w"),
            sort_keys=True,
            indent=4,
        )
    worker_type = cfg.get_worker_type()
    logger.info(f"========== Running {worker_type} ==========")
    if worker_type == WorkerType.GENERATOR_ACTOR:
        generator_cfg = AdvGeneratorConfig.from_adv_cfg(cfg)
        generator.generate(generator_cfg)
    elif worker_type == WorkerType.PROVER_ACTOR:
        prover_cfg = AdvProverConfig.from_adv_cfg(cfg)
        prover.prove(prover_cfg)
    elif worker_type in {WorkerType.PROVER_TRAINER, WorkerType.GENERATOR_TRAINER}:
        train_cfg = cfg.make_trainer_args()
        train(train_cfg)
    else:
        raise NotImplementedError(worker_type)


if __name__ == "__main__":
    from configs.adv_configs import register_adversarial_configs

    register_adversarial_configs()
    cfg = cfg_from_cli(schema=AdversarialConfig)
    main(cfg)
