# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import asdict
from pathlib import Path

from numpy.random import RandomState

from evariste import json as json
from evariste.backward.model.beam_search_kind import BeamSearchKind
from evariste.backward.prover.args import MCTSParams
from evariste.backward.prover.prover_args import ProverParams
from evariste.backward.prover.zmq_prover import ZMQProverParams
from evariste.backward.remote.search_params_sampler import (
    maybe_make_search_params_samplers,
    HyperOptKind,
    WorkerParamsSampler,
    MCTSSolvingStats,
)
from evariste.model.transformer_args import DecodingParams

NO_BEAM = "{'n_expansions':[1000,10000],'n_samples':[8,16,32,48],'temperature':[0.8,2.0],'exploration':[0.01,100],'depth_penalty':[0.8,0.9,0.95,1]}"
BEAM = "{'n_expansions':[1000,10000],'n_samples':[8,16,32,48],'beam':[true,false]}"


def test_factory():
    goal_sampler, worker_sampler = maybe_make_search_params_samplers(
        hyperopt=HyperOptKind.Random,
        hyperopt_param_str=NO_BEAM,
        n_machines=1,
        n_simultaneous_proofs=1,
    )
    assert goal_sampler is not None
    assert worker_sampler is not None  # always created when goal_sampler


def test_goal_sampler():
    rng = RandomState(0)
    for hyperopt_str in [NO_BEAM]:
        goal_sampler, _ = maybe_make_search_params_samplers(
            hyperopt=HyperOptKind.Random,
            hyperopt_param_str=hyperopt_str,
            n_machines=1,
            n_simultaneous_proofs=1,
        )

        for idx in range(10):
            name = f"{idx}"
            result = goal_sampler.sample_goal_params(name)

            for key, value in asdict(result).items():
                if key in hyperopt_str:
                    assert value is not None, (key, value)
            success = bool(rng.randint(2))
            goal_sampler.update(name, success=success)

        reco = goal_sampler.get_recommendation()
        _ = json.dumps(reco)  # check we can jsonify it


def _mock_zmq_prover_params():
    return ZMQProverParams(
        prover=ProverParams(
            mcts=MCTSParams(n_expansions=50, early_stop=True),
            beam_path=Path("blabl"),
            dump_path=Path("blabla"),
            n_simultaneous_proofs=1,
            beam_kind=BeamSearchKind.Manual,
        ),
        decoding=DecodingParams(),
        root_dir=Path("blabla"),
    )


def test_worker_sampler_no_beam():
    _, worker_sampler = maybe_make_search_params_samplers(
        hyperopt=HyperOptKind.Random,
        hyperopt_param_str=NO_BEAM,
        n_machines=1,
        n_simultaneous_proofs=1,
    )
    params = _mock_zmq_prover_params()
    sampled = [
        worker_sampler.sample_prover_params(params, verbose=False) for _ in range(50)
    ]
    # n_samples should have change
    assert len({s.decoding.n_samples for s in sampled}) > 1
    assert len({s.decoding.use_beam for s in sampled}) == 1


def test_worker_sampler_with_beam():
    _, worker_sampler = maybe_make_search_params_samplers(
        hyperopt=HyperOptKind.Random,
        hyperopt_param_str=BEAM,
        n_machines=1,
        n_simultaneous_proofs=1,
    )
    params = _mock_zmq_prover_params()
    sampled = [
        worker_sampler.sample_prover_params(params, verbose=False) for _ in range(50)
    ]
    # n_samples should have change
    assert len({s.decoding.n_samples for s in sampled}) > 1
    assert len({s.decoding.use_beam for s in sampled}) == 2


def test_mcts_solving_stats():
    goal_sampler, worker_sampler = maybe_make_search_params_samplers(
        hyperopt=HyperOptKind.Random,
        hyperopt_param_str=BEAM,
        n_machines=1,
        n_simultaneous_proofs=1,
    )
    solving_stats = MCTSSolvingStats(hyperopt_param_str=BEAM, n_buckets=2)
    rng = RandomState(0)

    for idx in range(1000):
        goal_params = goal_sampler.sample_goal_params(f"{idx}")
        worker_params = worker_sampler.sample_prover_params(
            params=_mock_zmq_prover_params(), verbose=False
        )

        dict1, dict2 = WorkerParamsSampler.rebuild_goal_and_worker_params(
            goal_params=goal_params,
            use_beam=worker_params.decoding.use_beam,
            n_samples=worker_params.decoding.n_samples,
        )
        # checking we can dump
        _ = json.dumps(dict1)
        _ = json.dumps(dict2)
        for key in dict1:
            assert key in BEAM

        if dict2["beam"]:
            assert "n_samples" in dict2
            assert "n_samples" not in dict1
        else:
            assert "n_samples" not in dict2
            assert "n_samples" in dict1

        solving_stats.update(
            solved=bool(rng.randint(2)), g_params=dict1, w_params=dict2
        )

    stats = solving_stats.get_stats()
    for key, value in stats.items():
        assert 40 < value < 60, (key, value)
