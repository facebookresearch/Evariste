# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
from dataclasses import dataclass
from typing import Dict

from evariste.forward.forward_prover import ProverConfig, SearchConfig
from evariste.forward.core.forward_policy import MAX_BATCH_SIZE, BatchConfig
from evariste.forward.proof_search import SearchType
from evariste.model.transformer_args import DecodingParams
from params import ConfStore, Params


@dataclass
class ModelConfig(Params):
    ckpt: str
    name: str = "unnamed_model"


DEFAULT_MAX_BATCH_MEMORY = 100_000_000

GREEDY_PROVER_CFG = ProverConfig(
    SearchConfig(max_nodes=500, max_generations=500, max_cons_inv_allowed=1),
    DecodingParams(
        max_gen_len=1024,
        fixed_temperature=None,
        use_beam=False,
        n_samples=1,
        stop_symbol="set_at_reloading_time",
    ),
    BatchConfig(max_batch_mem=DEFAULT_MAX_BATCH_MEMORY),
    name="greedy",
)

SAMPLING_PROVER_CFG = ProverConfig(
    SearchConfig(max_nodes=500, max_generations=500, max_cons_inv_allowed=5),
    DecodingParams(
        max_gen_len=1024,
        fixed_temperature=1.0,
        use_beam=False,
        use_sampling=True,
        n_samples=1,
        stop_symbol="set_at_reloading_time",
    ),
    BatchConfig(max_batch_mem=DEFAULT_MAX_BATCH_MEMORY),
    name="sampling",
)


STOCHASTIC_BEAM_PROVER_CFG = ProverConfig(
    SearchConfig(
        max_nodes=500,
        max_generations=500,
        max_cons_inv_allowed=1,
        n_simultaneous_proofs=128,
    ),
    DecodingParams(
        max_gen_len=1024,
        fixed_temperature=1.0,
        use_beam=True,
        use_sampling=True,
        n_samples=32,
        stop_symbol="set_at_reloading_time",
    ),
    BatchConfig(
        max_batch_mem=DEFAULT_MAX_BATCH_MEMORY // 32,
        max_batch_size=MAX_BATCH_SIZE // 32,
    ),
    name="stochastic_beam",
)

lean_greedy_config = copy.deepcopy(GREEDY_PROVER_CFG)
lean_greedy_config.search_cfg.n_simultaneous_proofs = 128

PROVER_CFGS = dict(
    greedy=GREEDY_PROVER_CFG,
    sampling=SAMPLING_PROVER_CFG,
    stochastic_beam=STOCHASTIC_BEAM_PROVER_CFG,
    sampling2=ProverConfig(
        SearchConfig(max_nodes=512, max_generations=512, max_cons_inv_allowed=20),
        DecodingParams(
            max_gen_len=300,
            fixed_temperature=1.0,
            use_beam=False,
            use_sampling=True,
            n_samples=1,
            stop_symbol="set_at_reloading_time",
        ),
        BatchConfig(max_batch_mem=DEFAULT_MAX_BATCH_MEMORY),
    ),
    lean_greedy=lean_greedy_config,
)

MODELS: Dict[str, str] = {}  # register models here with name and path


def register_model_cfgs():
    for name, path in MODELS.items():
        model_cfg = ModelConfig(name=name, ckpt=path)
        ConfStore[name] = model_cfg


def register_prover_cfgs():
    for name, cfg in PROVER_CFGS.items():
        cfg.name = name
        ConfStore[name] = cfg
