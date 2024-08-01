# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import dataclasses
import tempfile
from typing import Tuple, List
import time
from contextlib import closing, ExitStack
import pytest

from evariste.async_workers.async_worker import AsyncWorker, RequestId
from evariste.async_workers.zmq_submitit_worker import (
    ZMQSubmititParams,
    SubmititConfig,
    ZMQSubmititWorker,
    AliveError,
    ZmqSubmititWorkerLauncher,
    _RouterZMQSocket,
)


class SimpleWorker(AsyncWorker[int, Tuple[int, int]]):
    def __init__(self, worker_id: int = -1, full_logging: bool = False):
        self.input_queue: List[Tuple[RequestId, int]] = []
        self.total_request = 0
        self.total_sent = 0
        self._is_alive = False
        self.worker_id = worker_id

    def is_alive(self):
        return self._is_alive

    def start(self):
        self._is_alive = True

    def stop(self):
        self._is_alive = False

    def submit(self, inp: int) -> RequestId:
        rid = self.total_request
        self.input_queue.append((rid, inp))
        self.total_request += 1
        return rid

    def ready(self) -> List[Tuple[RequestId, Tuple[int, int]]]:
        if not self.is_alive():
            return []
        if len(self.input_queue) == 0:
            return []
        request_id, input = self.input_queue.pop(0)
        self.total_sent += 1
        return [(request_id, (self.worker_id, input ** 2))]

    def close(self):
        pass


class RaisingSimpleWorker(SimpleWorker):
    def is_alive(self):
        if self.total_sent > 1:  # die after 2 outputs
            raise RuntimeError("Square worker crashed after two outputs")
        return self._is_alive


class SleepingSimpleWorker(SimpleWorker):
    def is_alive(self):
        if self.total_sent > 1:  # sleep after 2 outputs
            time.sleep(100)
        return self._is_alive


@dataclasses.dataclass
class Timeout:
    max: float
    start: float = dataclasses.field(default_factory=time.time)

    def ok(self):
        if time.time() - self.start > self.max:
            raise RuntimeError("Too long")
        return True


@pytest.mark.slow
def test_with_simple_inner_worker():
    with ExitStack() as stack:
        tempdir = stack.enter_context(tempfile.TemporaryDirectory())
        socket = _RouterZMQSocket()
        stack.enter_context(closing(socket))
        worker = ZMQSubmititWorker(
            SimpleWorker,
            ZMQSubmititParams(
                SubmititConfig(folder=tempdir, slurm_job_name="test_requests_async",),
                check_status_freq=0.0,
            ),
            _RouterZMQSocket(),
        )
        stack.enter_context(closing(worker))

        with pytest.raises(AliveError):
            worker.submit(1)

        worker.start()

        assert worker.is_alive()

        rid_0 = worker.submit(2)
        rid_1 = worker.submit(3)
        results = []
        timeout = Timeout(max=5.0)
        while timeout.ok():
            results.extend(worker.ready())
            if len(results) >= 2:
                break
        assert results == [(rid_0, (0, 4)), (rid_1, (0, 9))]

        results = worker.ready()
        assert results == []
        assert worker.is_alive()

        rid_2 = worker.submit(4)
        worker.stop()
        timeout = Timeout(max=1.0)
        while worker.is_alive() and timeout.ok():
            assert len(worker.ready()) == 0
        assert not worker.is_alive()


@pytest.mark.slow
def test_is_dead_when_inner_worker_raise():
    with ExitStack() as stack:
        tempdir = stack.enter_context(tempfile.TemporaryDirectory())
        launcher = ZmqSubmititWorkerLauncher(
            RaisingSimpleWorker,
            ZMQSubmititParams(
                SubmititConfig(folder=tempdir, slurm_job_name="test_requests_async",),
                check_status_freq=0.0,
            ),
        )
        stack.enter_context(closing(launcher))
        worker = launcher.launch_workers(1)[0]
        stack.enter_context(closing(worker))
        assert worker.is_alive()

        worker.submit(0)
        worker.submit(1)
        worker.submit(2)

        # receive 2 outputs then the worker die
        results = []
        timeout = Timeout(max=5.0)
        while timeout.ok():
            result = worker.ready()
            results.extend(result)
            if len(results) == 2:
                break

        # Now the worker is dead
        assert len(worker.ready()) == 0
        worker.submit(
            3
        )  # can submit as I didnt call is_alive or stop so _status == READY
        assert len(worker.ready()) == 0

        timeout = Timeout(max=1.0)
        while worker.is_alive() and timeout.ok():
            assert len(worker.ready()) == 0

        # now I check is_alive
        assert not worker.is_alive()


@pytest.mark.slow
def test_is_dead_if_no_heartbeat():
    with ExitStack() as stack:
        tempdir = stack.enter_context(tempfile.TemporaryDirectory())
        launcher = ZmqSubmititWorkerLauncher(
            RaisingSimpleWorker,
            ZMQSubmititParams(
                SubmititConfig(folder=tempdir, slurm_job_name="test_requests_async",),
                heartbeat_freq=0.1,
                check_status_freq=0.0,
            ),
        )
        stack.enter_context(closing(launcher))
        worker = launcher.launch_workers(1)[0]
        stack.enter_context(closing(worker))
        assert worker.is_alive()

        worker.submit(0)
        worker.submit(1)
        worker.submit(2)

        timeout = Timeout(5.0)
        results = []
        while timeout.ok() and len(results) < 2:
            results.extend(worker.ready())

        assert len(results) == 2

        timeout = Timeout(2.0)
        while worker.is_alive() and timeout.ok():
            assert len(worker.ready()) == 0
        assert not worker.is_alive()


@pytest.mark.slow
def test_zmq_submitit_launcher():
    with ExitStack() as stack:
        tempdir = stack.enter_context(tempfile.TemporaryDirectory())

        params = ZMQSubmititParams(
            SubmititConfig(folder=tempdir, slurm_job_name="test_requests_async"),
        )
        launcher = ZmqSubmititWorkerLauncher(SimpleWorker, params)
        stack.enter_context(closing(launcher))
        workers = launcher.launch_workers(n_workers=5)
        assert len(workers) == 5
        for worker in workers:
            assert worker.is_alive()

        for worker_id, worker in enumerate(workers):
            worker.submit(worker_id)

        timeout = Timeout(5.0)
        res = []
        while timeout.ok() and len(res) < 5:
            for wid, worker in enumerate(workers):
                outs = worker.ready()
                if len(outs) > 0:
                    print(outs)
                    res.extend([(wid, out) for out in outs])

        assert sorted(res) == [(wid, (0, (wid, wid ** 2))) for wid in range(5)]
