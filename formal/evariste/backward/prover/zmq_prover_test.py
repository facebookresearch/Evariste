# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evariste.backward.goal_factory import get_goals_to_prove

from evariste.backward.prover.prover import BackwardGoal
from datetime import datetime

from evariste.backward.prover.prover_args import (
    BeamSearchKind,
    ProverParams,
    ProverKind,
)
from evariste.datasets import DatasetConf
import pytest

# from evariste.backward.prover.async_prover import AsyncProver

from evariste.model.transformer_args import DecodingParams
from evariste.model.data.dictionary import EOS_WORD
from evariste.async_workers.async_worker import (
    AsyncWorker,
    RequestId,
    AsyncWorkerDeadError,
)
from evariste.backward.prover.zmq_prover import (
    ZMQProverParams,
    ProverHandler,
    ZMQProver,
    run_async,
)

from params import ConfStore


beam_path, dataset = (
    "",
    ConfStore["eq_dataset_exp_trigo_hyper"],
)

# ##########################

# user = os.environ["USER"]
now = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
dump_path = f"/tmp/prover_async_worker_test/{now}/"


decoding_params = DecodingParams(
    max_gen_len=256,
    n_samples=1,
    use_beam=False,
    use_sampling=False,
    use_cache=True,
    stop_symbol=EOS_WORD,
)

mcts_params = ConfStore["mcts_very_fast"]
mcts_params.tokens_per_batch = 1000

zmq_test_params = ZMQProverParams(
    eq_dataset=dataset,
    prover=ProverParams(
        n_simultaneous_proofs=2,
        mcts=mcts_params,
        beam_path=Path(beam_path),
        beam_kind=BeamSearchKind.Fixed,
        prover_kind=ProverKind.BackwardMCTS,
        dump_mcts=True,
        dump_path=Path(dump_path),
        heartbeats_freq=60,
    ),
    decoding=decoding_params,
    n_machines=1,
    max_attempts=1,
    partition="",
    root_dir=Path(dump_path),
    dump_proofs=True,
    n_th_to_prove=3,
    shuffle_seed=43,
    max_restarts=0,
    copy_model=False,
    local=True,
)


class AsyncProverTest(AsyncWorker[BackwardGoal, Dict]):
    def __init__(self, params: ZMQProverParams):
        self.input_queue: List[Tuple[RequestId, BackwardGoal]] = []
        self._is_alive = False
        self.sent = 0

    def is_alive(self):
        return self._is_alive

    def start(self):
        self._is_alive = True

    def stop(self):
        self._is_alive = False

    def submit(self, input: BackwardGoal) -> RequestId:
        assert self._is_alive
        rid = self.sent
        self.sent += 1
        self.input_queue.append((rid, input))
        return rid

    def ready(self) -> List[Tuple[RequestId, Dict]]:
        print("call ready")
        if not self.is_alive():
            return []
        if len(self.input_queue) == 0:
            return []
        request_id, goal = self.input_queue.pop(0)
        result = {
            "type": "result",
            "label": goal.label,
            "name": goal.name,
            "model": "MODEL_NAME",
            "success": True,
            "src": "prover",  # not used
            "n_samples_sent": 0,
            "mcts_stats": {},
            "mcts_hist_stats": {},
            "proof_stats": {},
            "gpu_stats": {},
            "exception": None,
            "goal_params": {},
            "worker_params": {},
        }
        return [(request_id, result)]

    def close(self):
        pass


class AsyncProverTestRaisingStart(AsyncProverTest):
    def start(self):
        raise RuntimeError("Error in starting")


class AsyncProverTestRaisingSubmit(AsyncProverTest):
    def submit(self):
        raise RuntimeError("Error in submit")


class AsyncProverTestRaisingReady(AsyncProverTest):
    def ready(self):
        raise RuntimeError("Error in ready")


@pytest.mark.slow
def test_prover_handler_logic():
    """
    Test code of prover handler with stupid test prover that always return success=True and do nothing more.
    """
    handler = ProverHandler(
        params=zmq_test_params,
        name="test_prover_handler",
        split="identities",
        prover_fn=AsyncProverTest,
        folder_exists_ok=True,
    )
    handler.run()
    print(handler.proved)
    print(handler.failed)
    assert len(handler.proved) == zmq_test_params.n_th_to_prove
    assert len(handler.failed) == 0


@pytest.mark.slow
@pytest.mark.parametrize(
    "prover_fn",
    [
        AsyncProverTestRaisingStart,
        AsyncProverTestRaisingReady,
        AsyncProverTestRaisingSubmit,
    ],
)
def test_prover_handler_with_raise_logic(prover_fn):
    """
    Test code of prover handler with stupid test prover that crash when starting, calling ready or calling submit.
    It should raise AsyncWorkerDeadError.
    """

    handler = ProverHandler(
        params=zmq_test_params,
        name="test_prover_handler",
        split="identities",
        prover_fn=prover_fn,
        folder_exists_ok=True,
    )
    with pytest.raises(AsyncWorkerDeadError):
        handler.run()
    assert len(handler.proved) == 0
    assert len(handler.failed) == 0


@pytest.mark.slow
def test_prover_handler_zmq_prover():
    """
    Test code of prover handler with real backward prover, but with number of expansion too low to have success.
    This is more to check if code runs.
    """
    handler = ProverHandler(
        params=zmq_test_params,
        name="test_prover_handler",
        split="identities",
        prover_fn=ZMQProver,
        folder_exists_ok=True,
    )
    handler.run()
    print(handler.proved)
    print(handler.failed)
    assert len(handler.proved) == 0
    assert len(handler.failed) == zmq_test_params.n_th_to_prove


@pytest.mark.slow
def test_prover_handler_zmq_prover_run_async():
    """
    Just here to test that run_async code is not broken.
    """
    run_async(
        params=zmq_test_params,
        name="test_prover_handler",
        split="identities",
    )
