# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import getpass
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field

from evariste.backward.model.beam_search_kind import BeamSearchKind
from evariste.backward.prover.args import MCTSParams
from evariste.backward.prover.prover_args import ProverParams, ProverKind
from evariste.envs.eq.env import EqGraphSamplerParams
from evariste.forward.core.forward_policy import BatchConfig
from evariste.forward.forward_model_configs import DEFAULT_MAX_BATCH_MEMORY
from evariste.forward.proof_search import SearchType
from evariste.model.transformer_args import DecodingParams
from evariste.forward.forward_prover import ProverConfig, SearchConfig
from evariste.model_zoo import ZOO
from params.params import Params, ConfStore, MISSING
from evariste.datasets import EquationsDatasetConf


ConfStore["fast_forward_gen"] = ProverConfig(
    SearchConfig(
        max_nodes=20,
        max_generations=20,
        max_cons_inv_allowed=5,
        proof_search_type=SearchType.STD,  # or SearchType.PROVER_LOSS
        # n_simultaneous_proofs=64,
    ),
    DecodingParams(
        max_gen_len=1024,
        fixed_temperature=1.0,
        use_beam=False,
        use_sampling=True,
        n_samples=1,
        stop_symbol="set_at_reloading_time",
    ),
    BatchConfig(max_batch_mem=DEFAULT_MAX_BATCH_MEMORY),
    name="fast_forward_gen",
)


@dataclass
class GeneratorArgs(Params):
    # params of the prover for filtering
    # If forward_cfg.searcg_cfg.proof_search_type == SearchType.PROVER_LOSS,
    # also used for prover-loss search.
    prover_params: ProverParams = field(
        default_factory=lambda: ProverParams(
            prover_kind=ProverKind.BackwardGreedy,
            mcts=MCTSParams(),
            beam_path=Path("."),
            dump_path=Path("."),
            n_simultaneous_proofs=100,
            beam_kind=BeamSearchKind.Fixed,
        )
    )
    # decoding params for the prover for filtering
    # If forward_cfg.searcg_cfg.proof_search_type == SearchType.PROVER_LOSS,
    # also used for prover-loss search.
    decoding_params: DecodingParams = field(
        default_factory=lambda: ConfStore["decoding_greedy"]
    )
    dump_path: Path = Path(".")
    forward_cfg: ProverConfig = field(
        default_factory=lambda: ConfStore["fast_forward_gen"]
    )
    sample_cfg: EqGraphSamplerParams = field(
        default_factory=lambda: EqGraphSamplerParams(size_weight=1.0)
    )
    gen_checkpoint_path: Path = Path(".")
    dataset_conf: EquationsDatasetConf = field(
        default_factory=lambda: ConfStore["eq_dataset_lean"]
    )
    max_tokens: int = 5000  # for batching in a prover-loss search bwd prover
    use_bwd_critic: bool = (
        False  # instead of bwd prover loss, use bwd prover critic scores
    )
    max_inp_len: int = 1024
    include_errors: bool = False
    log_interval: int = 60  # in seconds
    save_chunk_length: int = 100
    verbose: bool = False
    label: Optional[str] = None
    job_id: int = 0
    n_jobs: int = 1
    load_path_globs: Optional[str] = None
    seed: int = 0
    all_time_averages: bool = (
        True  # if True, log global averages of counters, otherwise interval averages
    )
    sequential: bool = False
    n_gens: Optional[int] = None
    return_all: bool = False

    prefix_with_unproved: bool = False
    check_backward_proof: bool = False


ConfStore["gen_args_512_512_greedy"] = GeneratorArgs(
    gen_checkpoint_path=Path("."),
    dump_path=Path("."),
    prover_params=ProverParams(
        prover_kind=ProverKind.BackwardGreedy,
        mcts=MCTSParams(),
        beam_path=Path("."),
        dump_path=Path("."),
        n_simultaneous_proofs=100,
        beam_kind=BeamSearchKind.Fixed,
    ),
    decoding_params=ConfStore["decoding_greedy"],
    dataset_conf=ConfStore["eq_dataset_lean"],
)

ConfStore["gen_args_512_512_mcts"] = GeneratorArgs(
    gen_checkpoint_path=Path("."),
    dump_path=Path("."),
    prover_params=ProverParams(
        prover_kind=ProverKind.BackwardMCTS,
        mcts=ConfStore["mcts_fast"],
        beam_path=Path("."),
        dump_path=Path("."),
        n_simultaneous_proofs=20,
        beam_kind=BeamSearchKind.Fixed,
    ),
    decoding_params=ConfStore["decoding_bwd_eval"],
    dataset_conf=ConfStore["eq_dataset_lean"],
)
