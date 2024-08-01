# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, Iterator, Dict, Callable

from pathlib import Path
import functools
from evariste.backward.prover.prover_args import CleaningLevel

import torch
import logging

from evariste.logger import create_logger
from evariste.backward.env.sr.env import SREnvGenerator
from evariste.model.data.dictionary import EOS_WORD, EOU_WORD
from evariste.datasets import (
    DatasetConf,
    MetamathDatasetConf,
    EquationsDatasetConf,
    HolLightDatasetConf,
    LeanDatasetConf,
    lang_from_dataset,
    SRDatasetConf,
    IsabelleDatasetConf,
)
from evariste.datasets.lean import LEAN_CONDITIONING
from evariste.backward.env.equations.env import EQEnvGenerator, EnvGen
from evariste.backward.env.metamath.env import MMEnvGenerator
from evariste.backward.env.hol_light.env import HLEnvGenerator
from evariste.backward.env.lean.env import LeanExpanderEnv
from evariste.backward.prover.core import HandlerType
from evariste.backward.prover.mcts_prover import MCTSProofHandlerGenerator
from evariste.backward.prover.greedy_prover import GreedyProofTreeHandler
from evariste.backward.prover.bfs_prover import BFSProofHandler
from evariste.backward.prover.igreedy_prover import ImprovedGreedy
from evariste.backward.prover.expander import MPExpander
from evariste.backward.model.beam_search_kind import BeamSearchKind
from evariste.backward.model.beam_search import (
    AutomaticallyReloadingBeamSearch,
    ManuallyReloadingBeamSearch,
    IncreasingQualityBeamSearch,
    FixedBeamSearch,
    BeamSearchModel,
)
from evariste.model.transformer_args import DecodingParams
from evariste.backward.prover.prover_args import (
    ProverParams,
    ProverKind,
    ConditioningKind,
)
from evariste.backward.env.core import BackwardEnv

logger = create_logger(None)


def backward_init(
    dataset: DatasetConf,
    decoding: DecodingParams,
    prover: ProverParams,
    decoder_type: str,
):
    """
    Init the async expander and the envs needed by a backward prover.
    @return: env_gen, env_gen(), expander,
    """
    env_gen: Optional[EnvGen] = None
    conditioning_vectors: Optional[Path] = None
    if (
        not isinstance(dataset, LeanDatasetConf)
        or prover.prover_kind != ProverKind.BackwardMCTS
    ):
        assert (
            prover.try_stitch is False
            and prover.proof_cleaning_params.level == CleaningLevel.No
        )

    if isinstance(dataset, MetamathDatasetConf):
        logging.info("Loading Metamath env")
        env_gen = MMEnvGenerator(dataset)
        env = env_gen()
        decoding.stop_symbol = EOU_WORD
    elif isinstance(dataset, EquationsDatasetConf):
        logging.info("Loading Equation env")
        env_gen = EQEnvGenerator(dataset, n_async_envs=dataset.n_async_envs)
        env = env_gen()
        decoding.stop_symbol = EOS_WORD
    elif isinstance(dataset, SRDatasetConf):
        logging.info("Loading SR env")
        env_gen = SREnvGenerator(dataset, n_async_envs=dataset.n_async_envs)
        env = env_gen()
        decoding.stop_symbol = EOS_WORD
    elif isinstance(dataset, HolLightDatasetConf):
        logging.info("Loading HOL-Light env")
        env_gen = HLEnvGenerator(
            dataset, debug=prover.debug, dump_path=prover.dump_path
        )
        env = env_gen()
        decoding.stop_symbol = EOU_WORD
    elif isinstance(dataset, LeanDatasetConf):
        logging.info(f"Loading Lean env {dataset}")
        env = BackwardEnv(
            LeanExpanderEnv(dataset, debug=prover.debug, dump_path=prover.dump_path,)
        )
        decoding.stop_symbol = EOU_WORD
        if prover.conditioning_kind != ConditioningKind.No:
            assert dataset.conditioning != ""
            conditioning_vectors = Path(LEAN_CONDITIONING[dataset.conditioning])
    elif isinstance(dataset, IsabelleDatasetConf):
        logging.info(f"Loading Isabelle env {dataset}")
        raise RuntimeError("not implemented")
        # TODO: Create backward environment for Isabelle
        # env = BackwardEnv(IsabelleExpanderEnv(dataset, debug=prover.debug, dump_path=prover.dump_path))
        decoding.stop_symbol = EOS_WORD
    else:
        raise RuntimeError(f"Unhandled dataset type {type(dataset)}")

    # CUDA here!
    device = "cpu" if decoding.cpu else f"cuda:{torch.cuda.current_device()}"
    logging.info(f"Setting GPU to device {device} for backward!")

    if prover.beam_kind == BeamSearchKind.AutomaticallyReloading:
        logging.info(
            f"Automatically reloading beam search loading in path {prover.beam_path}"
        )
        beam_search: BeamSearchModel = AutomaticallyReloadingBeamSearch(
            prover.beam_path,
            decoding,
            device,
            decoder_type,
            conditioning_vectors,
            min_reload_time=prover.min_reload_time,
        )
    elif prover.beam_kind == BeamSearchKind.IncreasingQuality:
        logging.info(
            f"Increasing quality beam search loading in path {prover.beam_path}"
        )
        beam_search = IncreasingQualityBeamSearch(
            path=prover.beam_path,
            split=prover.eval_split,
            decoding_params=decoding,
            device=device,
            lang=lang_from_dataset(dataset),
            decoder_type=decoder_type,
            conditioning_vectors=conditioning_vectors,
        )

    elif prover.beam_kind == BeamSearchKind.Manual:
        beam_search = ManuallyReloadingBeamSearch(
            prover.beam_path, decoding, device, decoder_type, conditioning_vectors
        )
    elif prover.beam_kind == BeamSearchKind.Fixed:
        beam_search = FixedBeamSearch(
            prover.beam_path, decoding, device, decoder_type, conditioning_vectors
        )
    else:
        raise RuntimeError(f"Unknown beam kind {prover.beam_kind}")
    beam_search.load_dico()  # blocking until a model exists

    logging.info("Beam search loaded")

    expander = MPExpander(
        beam_search=beam_search,
        expander_params=prover.mcts.expander,
        dataset=dataset,
        profile=False,
        n_gpus=prover.n_gpus,
        prover_dump_path=prover.dump_path,
        quiet=prover.quiet,
        only_jsonl=prover.only_jsonl,
    )
    if (
        prover.beam_kind == BeamSearchKind.Manual
        or prover.beam_kind == BeamSearchKind.Fixed
    ):
        expander.reload_model_weights()  # load model on new process only
    return env_gen, env, expander


def prior_search(prover_params: ProverParams):
    from evariste.backward.prover.prior_search_prover import PriorSearchProofHandler

    return (
        functools.partial(
            PriorSearchProofHandler,
            n_expansions_max=prover_params.mcts.n_expansions,
            n_concurrent_expansions_max=prover_params.mcts.succ_expansions,
        ),
    )


def get_prover_handler(prover_params: ProverParams) -> HandlerType:
    # Chose a single proof handler
    kind_to_handler: Dict[ProverKind, Callable[[], HandlerType]] = {
        ProverKind.BackwardGreedy: lambda: GreedyProofTreeHandler,
        ProverKind.BackwardIGreedy: lambda: ImprovedGreedy,
        ProverKind.BackwardMCTS: lambda: MCTSProofHandlerGenerator(
            prover_params=prover_params
        ),
        ProverKind.BackwardBreadthFS: lambda: functools.partial(
            BFSProofHandler,
            n_expansions_max=prover_params.mcts.n_expansions,
            n_concurrent_expansions_max=None,
        ),
        ProverKind.PriorSearch: lambda: prior_search(prover_params),
    }
    proof_handler: HandlerType = kind_to_handler[prover_params.prover_kind]()
    return proof_handler
