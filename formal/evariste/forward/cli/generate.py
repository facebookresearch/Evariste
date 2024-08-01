# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Script to generate from a checkpoint.

Usage:
python -m evariste.forward.cli.generate \
   --model.ckpt /YOUR_MODEL \
   --prover sampling --n_generations 1000
"""
from collections import Counter
from typing import Optional, Tuple, List, Iterator
from dataclasses import dataclass
from contextlib import closing
from pprint import pprint
from pathlib import Path
import tqdm
import torch
import getpass
import logging
import datetime
import numpy as np

from evariste import json as json
from params.params import cfg_from_cli, Params
from evariste.comms.zip_store import ZipStore
from evariste.forward import forward_model_factory, forward_model_configs
from evariste.forward.cli.cli_common import Generation
from evariste.forward.common import ForwardGoal, GenerationHistory, GenerationInfos
from evariste.forward.env_specifics.fwd_env_helper import FwdEnvHelper
from evariste.forward.forward_model_configs import ModelConfig
from evariste.forward.forward_prover import ProverConfig
from evariste.forward.proof_search import StandardProofSearch
from evariste.forward.utils.launch_utils import prepare_folder, launch_with_submitit
from evariste.model.data.dictionary import Dictionary


BASE_OUTPUT_FOLDER = Path(f"")


@dataclass
class Config(Params):
    model: ModelConfig
    prover: ProverConfig
    seed: int = 42
    split: str = "train"
    n_generations: int = 100
    rank: int = 0
    world_size: int = 1
    debug: bool = False
    slurm: bool = False
    n_jobs: int = 1
    partition: str = "Theorem_Proving"
    output_path: Optional[str] = None

    def make_output_path(self):
        # post init?
        assert self.output_path is None
        output_folder = BASE_OUTPUT_FOLDER
        output_folder.mkdir(parents=True, exist_ok=True)
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = f"{self.model.name}_{self.prover.name}_{self.split}_{self.n_generations}"
        name += f"_{now}"
        if self.debug:
            name = f"debug_{name}"
        self.output_path = str(output_folder / name)


def generate(cfg: Config):

    print(f"===== Starting generation =====")
    print(f"Output folder: {cfg.output_path}")

    torch.manual_seed(cfg.seed)
    rng = np.random.RandomState((cfg.seed, cfg.rank))

    assert cfg.output_path is not None
    store = ZipStore(Path(cfg.output_path))

    generator, dico, params, env_helper = forward_model_factory.from_checkpoint(
        ckpt_path=cfg.model.ckpt, device_str="cuda", cfg=cfg.prover
    )

    def inputs() -> Iterator[Tuple[int, ForwardGoal]]:
        i = 0
        goal_factory = env_helper.get_goal_factory()
        while i < cfg.n_generations:
            yield i, goal_factory.build_generation_goal(rng, cfg.split)
            i += 1

    with closing(generator), closing(store):
        generations = generator.generate_proofs(inputs())
        for out_id, output in tqdm.tqdm(generations, total=cfg.n_generations):
            assert isinstance(output, StandardProofSearch)
            assert isinstance(output.generation, GenerationHistory)
            assert isinstance(output.info, GenerationInfos)
            assert not output.info.solved, output.info

            # print(f"{out_id} {len(output.generation.forward_steps())}")

            generation = Generation(history=output.generation, infos=output.info)
            store.store_in_pickle_zip(obj=generation, zip_name="generations")

    metrics = _analyze(cfg, dico, env_helper)
    assert cfg.output_path is not None
    with (Path(cfg.output_path) / "generations_metrics.json").open("w") as fp:
        json.dump(metrics, fp)
    pprint(metrics)


def _analyze(cfg: Config, dico: Dictionary, env_helper: FwdEnvHelper):
    from evariste.forward.cli.analyze_generations import evaluate_generations

    assert cfg.output_path is not None
    store = ZipStore(Path(cfg.output_path))
    generations: List[Generation] = store.read_pickle_zip("generations")

    histories = [g.history for g in generations]

    return evaluate_generations(histories, dico, env_helper)


def main(cfg: Config):
    prepare_folder(cfg)

    if cfg.slurm:
        launch_with_submitit(generate, cfg=cfg)
    else:
        generate(cfg=cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    forward_model_configs.register_model_cfgs()
    forward_model_configs.register_prover_cfgs()
    cfg = cfg_from_cli(schema=Config)
    cfg.make_output_path()
    main(cfg)
