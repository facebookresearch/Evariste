# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from evariste import json as json
from pathlib import Path
from pprint import pprint
from typing import cast, List, Dict

from evariste.forward import forward_model_configs
from evariste.forward.cli.cli_common import Generation
from evariste.forward.env_specifics.fwd_env_helper import FwdEnvHelper
from evariste.forward.fwd_mm.mm_env_helper import MMFwdEnvHelper
from evariste.forward.fwd_eq.eq_env_helper import EQFwdEnvHelper
from evariste.forward.fwd_mm.mm_helpers import evaluate_mm_generations
from evariste.forward.fwd_eq.eq_helpers import evaluate_eq_generations
from evariste.comms.zip_store import ZipStore
from evariste.model.data.dictionary import Dictionary
from evariste.model.utils import reload_ckpt
from params import Params
from params.params import cfg_from_cli


@dataclass
class Config(Params):
    path: str


def analyze(cfg: Config):
    generation_path = Path(cfg.path)
    assert generation_path.exists(), generation_path

    job_dirs = [p for p in generation_path.iterdir() if p.name.startswith("job_")]
    if len(job_dirs) == 0:
        job_dirs = [generation_path]

    generations = []
    for job_path in job_dirs:
        print(f"Loading {job_path}")
        store = ZipStore(job_path)
        generations.extend(store.read_pickle_zip("generations"))

    generations = cast(List[Generation], generations)
    histories = [g.history for g in generations]

    master_path = job_dirs[0]
    with (master_path / "config.json").open("r") as fp:
        previous_cfg = json.load(fp)
    print("previous_cfg", previous_cfg)
    ckpt_path = previous_cfg["model"]["ckpt"]
    params, dico, _ = reload_ckpt(ckpt_path, only_model_params=False)
    env_helper = FwdEnvHelper.from_trainer_args(dico, params)
    metrics = evaluate_generations(histories, dico, env_helper)
    pprint(metrics)
    print(
        f"{metrics['avg_proof_size']:.02f}",
        f"{metrics['avg_statement_len']:.02f}",
        metrics["n_statements_distinct"],
        metrics["n_tok_used"],
        metrics["n_label_in_proof"],
    )


def evaluate_generations(histories, dico: Dictionary, env_helper: FwdEnvHelper) -> Dict:
    """
    Not putting evaluate_generations in FwdEnvBuilder for the moment
    for simplicity
    """
    if isinstance(env_helper, MMFwdEnvHelper):
        n_labels = len(env_helper.mm_env().labels)
        dico_size = len(dico) - n_labels
        train_set = set()
        for x in env_helper.mm_env().labels.values():
            train_set.add(x[1]["tokens_str"])
        return evaluate_mm_generations(histories, dico_size, n_labels, train_set)
    elif isinstance(env_helper, EQFwdEnvHelper):
        return evaluate_eq_generations(histories, env_helper)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    forward_model_configs.register_model_cfgs()
    forward_model_configs.register_prover_cfgs()
    cfg = cfg_from_cli(schema=Config)
    cfg = cast(Config, cfg)
    analyze(cfg)
