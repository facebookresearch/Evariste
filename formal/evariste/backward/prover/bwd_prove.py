# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Iterator, Iterable, Sequence, List, Set, Dict
from logging import getLogger
from evariste.backward.prover.mcts import MCTSResult
from params import MISSING
import os
import pickle

from evariste import json as json
from evariste.datasets import DatasetConf
from evariste.model.transformer_args import DecodingParams
from evariste.backward.prover.prover import (
    ProverParams,
    ProofResult,
    ProverKind,
    init_and_run_prover,
)
from evariste.backward.prover.utils import WeightedAvgStats
from evariste.backward.env.core import BackwardGoal
from evariste.backward.env.lean.env import DeclNotFound


logger = getLogger(__name__)


def bwd_prove(
    dataset: DatasetConf,
    decoding: DecodingParams,
    prover_params: ProverParams,
    to_prove: Iterable[BackwardGoal],
    n_attempts: int = 1,
    decoder_type: str = "decoder",
    dump_proofs: bool = False,
    dump_stats: bool = False,
) -> Sequence[ProofResult]:

    # dump proofs
    if dump_proofs:
        assert prover_params.dump_path != MISSING
        assert prover_params.dump_path.exists(), prover_params.dump_path
        logger.info(f"Dumping proofs in {prover_params.dump_path}")
        os.makedirs(prover_params.dump_path / "proofs", exist_ok=True)

    # sanity check
    assert n_attempts == 1 or n_attempts > 1 and decoding.use_sampling
    is_greedy = prover_params.prover_kind == ProverKind.BackwardGreedy
    if is_greedy and decoding.n_samples > 1:
        raise RuntimeError(
            f"Greedy decoding requires to decode with 1 sample! "
            f"Provided n_samples={decoding.n_samples}."
        )
    decoding.__post_init__()

    to_prove = list(to_prove)
    goals: Dict[str, BackwardGoal] = {goal.label: goal for goal in to_prove}
    assert len(goals) == len(to_prove), "Some names are not unique"

    attempts_remaining = {goal.label: n_attempts for goal in to_prove}
    finished: Set[str] = set()

    gpu_stats = WeightedAvgStats()

    def input_it() -> Iterator[BackwardGoal]:
        while len(attempts_remaining) > 0:
            for name, n_remaining in list(attempts_remaining.items()):
                if n_remaining == 0:
                    del attempts_remaining[name]
                else:
                    attempts_remaining[name] -= 1
                    try:
                        yield goals[name]
                    except DeclNotFound:
                        pass
        logger.info("Everything sent, waiting for prover to finish")

    results_received: Set[str] = set()
    results: List[ProofResult] = []

    for maybe_result in init_and_run_prover(
        dataset=dataset,
        decoding=decoding,
        prover_params=prover_params,
        decoder_type=decoder_type,
        input_it=input_it(),
    ):
        proof_result, gpu_stats_update = maybe_result
        goal = proof_result.goal
        label = goal.label
        assert label not in results_received or n_attempts > 1, label
        results_received.add(label)
        gpu_stats.update(gpu_stats_update)
        if proof_result.exception is not None:
            logger.warning(f"{type(proof_result.exception)} - {proof_result.exception}")
            attempts_remaining[label] = 0
        elif proof_result.proof is not None:
            attempts_remaining[label] = 0
            results.append(proof_result)
            if dump_proofs:
                fp = prover_params.dump_path / "proofs" / f"{label}.pkl"
                with open(fp, "wb") as f:
                    pickle.dump(proof_result.proof, f)

        if dump_stats:
            from evariste.backward.prover.prior_search_prover import (
                ProofResultWithStats,
            )

            assert isinstance(proof_result, ProofResultWithStats)
            path = prover_params.dump_path / "proof_stats.json"
            print(f"Dumping stats in {path}")
            stats = proof_result.proof_stats
            stats["label"] = label
            with path.open("a") as fp:  # type: ignore[assignment]
                fp.write(f"{json.dumps(stats)}\n")  # type: ignore[attr-defined]

        if label not in attempts_remaining or attempts_remaining[label] == 0:
            finished.add(label)

        if len(finished) > 0:
            acc = 100 * len(results) / len(finished)
            extra_info = ""
            if isinstance(proof_result, MCTSResult):
                extra_info = f"nodes: {proof_result.stats['n_nodes']} "
            if prover_params.print_status:
                logger.info(
                    f"BWD_PROVE: received {goal.label[:40]:<40} "
                    f"{extra_info}"
                    f"has_proof: {int(proof_result.proof is not None)} "
                    f"Proved: {len(results)}/{len(finished)} ({acc:.2f}%) "
                    f"{gpu_stats_update}"
                )

    logger.info(f"GPU STATS: {json.dumps(gpu_stats.stats)}")
    return results
