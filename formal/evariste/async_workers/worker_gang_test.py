# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import os
import tempfile
from typing import Tuple, List, Iterator, Union, Generic
from contextlib import closing
from typing import Callable
import pytest
import copy
from contextlib import ExitStack

from evariste.async_workers.async_worker_helpers import make_iterator
from evariste.async_workers.async_worker import (
    AsyncWorkerDeadError,
    AsyncWorkerLauncher,
)
from evariste.async_workers.worker_gang import AsyncWorkerGang
from evariste.async_workers.zmq_submitit_worker import (
    ZmqSubmititWorkerLauncher,
    ZMQSubmititParams,
    SubmititConfig,
)
from evariste.async_workers.zmq_submitit_worker_test import (
    SimpleWorker,
    RaisingSimpleWorker,
    Timeout,
)
from evariste.utils import logged_closing


class DeadSimpleWorker(SimpleWorker):
    def is_alive(self):
        if self.total_sent > 1:  # die after 2 outputs
            self._is_alive = False
        return self._is_alive


def simple_worker_with_death_factory(id: int = 0) -> SimpleWorker:
    return DeadSimpleWorker(id)


class SimpleAsyncWorkerLauncher(AsyncWorkerLauncher[int, Tuple[int, int]]):
    def __init__(self, worker_factory: Callable[[int], SimpleWorker]):
        self.n_worker_lanched = 0
        self.worker_factory = worker_factory

    def launch_workers(self, n_workers, first_with_full_logging: bool = False):
        new_workers = []
        for i in range(n_workers):
            worker = self.worker_factory(
                self.n_worker_lanched + i
            )  # hack to get the "global" worker id in test
            worker.start()
            new_workers.append(worker)
        self.n_worker_lanched += n_workers

        return new_workers

    def close(self):
        pass


def run_gang(
    worker_launcher,
    input_it,
    n_workers,
    max_queued_inputs,
    expected,
    total_worker_launched,
    max_restarts,
    local=True,
    check_alive_freq=0,
):
    n_inputs = len(list(copy.copy(input_it)))
    print(f"Local gang with {n_inputs} inputs.")

    assert isinstance(input_it, List) or isinstance(input_it, Iterator)
    if isinstance(input_it, List):
        input_it = iter(input_it)

    gang: AsyncWorkerGang = AsyncWorkerGang(
        worker_launcher=worker_launcher,
        n_workers=n_workers,
        max_restarts=max_restarts,
        max_queued_inputs=max_queued_inputs,
        check_alive_freq=check_alive_freq,
    )
    should_raise = len(expected) != n_inputs

    outputs = []

    def _run():
        for output in make_iterator(
            gang,
            max_in_worker=gang.max_queued_inputs * gang.n_workers,
            input_it=input_it,
        ):
            outputs.append(output)

    with closing(gang):
        if should_raise:
            with pytest.raises(AsyncWorkerDeadError):
                _run()
        else:
            _run()

    print(outputs)
    print(expected)
    assert len(gang.workers) == 0, gang.workers.keys()
    assert len(gang.dead_workers) == total_worker_launched, gang.dead_workers

    if local:
        assert outputs == expected
    else:
        assert len(outputs) == len(expected)
        # we check worker id / rid / output
        if len(expected):
            if len(expected[0]) == 2:
                assert all(output in outputs for output in expected)
            # we cannot predict worker_id so we only predict the output
            elif len(expected[0]) == 1:
                expected = [el[0] for el in expected]
                assert all(output[1] in expected for output in outputs)
            else:
                raise NotImplementedError

    print(gang.jobs_stats)


@pytest.mark.parametrize(
    "input_it,n_workers,expected",
    [
        (iter([]), 1, [],),  # empty input, single worker
        (iter([]), 4, [],),  # empty input, several workers
        (
            iter(range(100)),
            1,
            [(0, i ** 2) for i in range(100)],  # (worker_id, output)
        ),  # single worker
        (
            iter(range(5)),
            2,
            [(0, 0), (1, 1), (0, 4), (1, 9), (0, 16)],
        ),  # workers < inputs
        (
            iter(range(5)),
            5,
            [(0, 0), (1, 1), (2, 4), (3, 9), (4, 16)],
        ),  # workers == inputs
        (
            iter(range(5)),
            8,
            [(0, 0), (1, 1), (2, 4), (3, 9), (4, 16)],
        ),  # workers > inputs
    ],
)
def test_gang_local_no_death(input_it, n_workers, expected):
    """
    No death in workers.
    Check we have expected outputs (worker_id, request_id, output),
    that the workers have been stopped.
    """
    run_gang(
        worker_launcher=SimpleAsyncWorkerLauncher(SimpleWorker),
        input_it=input_it,
        n_workers=n_workers,
        max_queued_inputs=1,
        expected=expected,
        total_worker_launched=n_workers,
        max_restarts=0,
    )


@pytest.mark.parametrize(
    "input_it,n_workers,expected,max_restarts",
    [
        (iter([]), 1, [], 10),
        (iter([]), 4, [], 10),
        (iter(range(5)), 1, [(0, 0), (0, 1), (0, 4), (0, 9), (0, 16)], 10,),
        (iter(range(5)), 2, [(0, 0), (1, 1), (0, 4), (1, 9), (0, 16)], 10,),
    ],
)
def test_gang_local_no_death_with_restarts(input_it, n_workers, expected, max_restarts):
    """
    No death in workers tested with restarts - as there are no death, there should not be restarted.
    Check we have expected outputs (worker_id, request_id, output),
    that the workers have been stopped.
    """
    run_gang(
        worker_launcher=SimpleAsyncWorkerLauncher(SimpleWorker),
        input_it=input_it,
        n_workers=n_workers,
        max_queued_inputs=1,
        expected=expected,
        total_worker_launched=n_workers,
        max_restarts=max_restarts,
    )


@pytest.mark.parametrize(
    "n_inputs,n_workers,expected,total_worker_launched",
    [
        (5, 1, [(0, 0), (0, 1)], 1),  # single worker
        (3, 2, [(0, 0), (1, 1), (0, 4)], 2),  # workers < inputs
        (5, 2, [(0, 0), (1, 1), (0, 4), (1, 9)], 2),  # workers < inputs
        (5, 5, [(0, 0), (1, 1), (2, 4), (3, 9), (4, 16)], 5,),  # workers == inputs
        (5, 8, [(0, 0), (1, 1), (2, 4), (3, 9), (4, 16)], 8,),  # workers > inputs
    ],
)
def test_gang_local_deaths_no_restart_no_dead_inputs(
    n_inputs, n_workers, expected, total_worker_launched
):
    """
    Workers die after it sent 2 inputs.
    We check that we have correct outputs and correct worker status at the end:
    max_queued_inputs=1,
    so we can submit one input at a time and get the output before die, so no use of dead input.
    """
    run_gang(
        worker_launcher=SimpleAsyncWorkerLauncher(simple_worker_with_death_factory),
        input_it=iter(range(n_inputs)),
        n_workers=n_workers,
        max_queued_inputs=1,
        expected=expected,
        total_worker_launched=total_worker_launched,
        max_restarts=0,
    )


@pytest.mark.parametrize(
    "n_inputs,n_workers,expected,total_worker_launched",
    [
        (
            5,
            1,
            [(0, 0), (0, 1)],
            1,
        ),  # single worker, submit all inputs to it (as there max_queued -1), it dies after 2 outputs, no restart
        (
            5,
            2,
            [(0, 0), (0, 1), (1, 4), (1, 9)],
            2,
        ),  # 2 workers, we submit all input to worker 0, then it dies after 2 outputs, dead_input should contain the rest, and resubmit to worker 1, which die after 2, so we have 4 outputs
        (
            5,
            3,
            [(0, 0), (0, 1), (1, 4), (1, 9), (2, 16)],
            3,
        ),  # the third worker only processed 1 input, so still alive
    ],
)
def test_gang_local_deaths_no_restart_with_dead_inputs(
    n_inputs, n_workers, expected, total_worker_launched
):
    """
    Workers die after it sent 2 inputs.
    We check that we have correct outputs and correct worker status at the end:
    max_queued_inputs=math.inf,
    so we can submit all inputs and then the worker dies, and we will have to use dead_input i.e requeue inputs.
    """
    run_gang(
        worker_launcher=SimpleAsyncWorkerLauncher(simple_worker_with_death_factory),
        input_it=iter(range(n_inputs)),
        n_workers=n_workers,
        max_queued_inputs=math.inf,
        expected=expected,
        total_worker_launched=total_worker_launched,
        max_restarts=0,
    )


@pytest.mark.parametrize(
    "n_inputs,n_workers,expected,total_worker_launched,max_restarts",
    [
        (5, 1, [(0, 0), (0, 1)], 1, 0),  # single worker - no restart
        (5, 1, [(0, 0), (0, 1), (1, 4), (1, 9)], 2, 1,),  # single worker - one restart
        (
            5,
            1,
            [(0, 0), (0, 1), (1, 4), (1, 9), (2, 16)],
            3,
            2,
        ),  # single worker - two restarts
        (
            5,
            1,
            [(0, 0), (0, 1), (1, 4), (1, 9), (2, 16)],
            3,
            10,
        ),  # single worker - ten restarts
        (
            4,
            2,
            [(0, 0), (1, 1), (0, 4), (1, 9)],
            2,  # we restart one worker and then next_job() gives stop iteration -> the least worker didnt do anything
            1,
        ),  # two workers - use of restart for no job as directly we got stopiteration
        (
            5,
            2,
            [(0, 0), (1, 1), (0, 4), (1, 9), (2, 16)],
            3,
            1,
        ),  # two workers die, one restart available, used for one input
        (
            5,
            2,
            [(0, 0), (1, 1), (0, 4), (1, 9), (2, 16)],
            4,
            10,
        ),  # two workers die, two restarts the first new worker will do one input, the second zero
    ],
)
def test_gang_local_deaths_with_restarts_no_dead_inputs(
    n_inputs, n_workers, expected, total_worker_launched, max_restarts
):
    """
    Workers die after it sent 2 inputs.
    We check that we have correct outputs and correct worker status at the end:
    max_queued_inputs=1,
    so we can submit one input at a time and get the output before die, so no use of dead input.
    """
    run_gang(
        worker_launcher=SimpleAsyncWorkerLauncher(simple_worker_with_death_factory),
        input_it=iter(range(n_inputs)),
        n_workers=n_workers,
        max_queued_inputs=1,
        expected=expected,
        total_worker_launched=total_worker_launched,
        max_restarts=max_restarts,
    )


@pytest.mark.parametrize(
    "n_inputs,n_workers,expected,total_worker_launched,max_restarts",
    [
        (
            5,
            1,
            [(0, 0), (0, 1)],
            1,
            0,
        ),  # single worker, submit all inputs to it (as there max_queued -1), it dies after 2 outputs, zero restart
        (
            5,
            1,
            [(0, 0), (0, 1), (1, 4), (1, 9)],
            2,
            1,
        ),  # single worker, submit all inputs to it (as there max_queued -1), it dies after 2 outputs, one restart
        (
            5,
            1,
            [(0, 0), (0, 1), (1, 4), (1, 9), (2, 16)],
            3,
            2,
        ),  # single worker, submit all inputs to it (as there max_queued -1), it dies after 2 outputs, no restart
        (
            5,
            2,
            [(0, 0), (0, 1), (1, 4), (1, 9), (2, 16)],
            3,
            1,
        ),  # 2 workers, we submit all input to worker 0, then it dies after 2 outputs, dead_input should contain the rest, and resubmit to worker 1, which die after 2, so we have 4 outputs
        (
            5,
            2,
            [(0, 0), (0, 1), (1, 4), (1, 9), (2, 16)],
            4,
            2,
        ),  # 2 workers, we submit all input to worker 0, then it dies after 2 outputs, dead_input should contain the rest, and resubmit to worker 1, which die after 2, so we have 4 outputs
    ],
)
def test_gang_local_deaths_with_restarts_with_dead_inputs(
    n_inputs, n_workers, expected, total_worker_launched, max_restarts
):
    """
    Workers die after it sent 2 inputs. But here we restart workers.
    We check that we have correct outputs and correct worker status at the end:
    max_queued_inputs=math.inf,
    so we can submit all inputs and then the worker dies, and we will have to use dead_input i.e requeue inputs.
    """
    run_gang(
        worker_launcher=SimpleAsyncWorkerLauncher(simple_worker_with_death_factory),
        input_it=iter(range(n_inputs)),
        n_workers=n_workers,
        max_queued_inputs=math.inf,
        expected=expected,
        total_worker_launched=total_worker_launched,
        max_restarts=max_restarts,
    )


def test_requests_remote_gang_square():
    with ExitStack() as stack:
        tempdir = stack.enter_context(tempfile.TemporaryDirectory())
        zmq_cfg = ZMQSubmititParams(
            SubmititConfig(folder=tempdir, slurm_job_name=f"test_requests_async_{id}",),
            check_status_freq=0.0,
        )
        gang_handler = AsyncWorkerGang(
            worker_launcher=ZmqSubmititWorkerLauncher(SimpleWorker, zmq_cfg),
            n_workers=4,
            max_restarts=0,
            max_queued_inputs=1,
        )
        # wait until it is connected before sending
        with closing(gang_handler):
            outputs = []
            for output in make_iterator(gang_handler, 1, iter(range(11))):
                outputs.append(output)
        print(outputs)

        outputs = [output[1] for output in outputs]
        assert all(i ** 2 in outputs for i in range(11))


@pytest.mark.slow
@pytest.mark.parametrize(
    "input_it,n_workers,expected",
    [
        (iter([]), 1, [],),  # empty input, single worker
        (iter([]), 4, [],),  # empty input, several workers
        (
            iter(range(5)),
            1,
            [
                (0, 0),
                (0, 1),
                (0, 4),
                (0, 9),
                (0, 16),
            ],  # as default worker id in simpleworker is zero
        ),  # single worker
        (
            iter(range(5)),
            2,
            [(0,), (1,), (4,), (9,), (16,)],
        ),  # here we cannot predict wich worker will have wich input, so only check that we hve all outputs
        (
            iter(range(5)),
            5,
            [(0, 0), (1, 1), (2, 4), (3, 9), (4, 16)],
        ),  # workers == inputs
        (
            iter(range(5)),
            8,
            [(0, 0), (1, 1), (2, 4), (3, 9), (4, 16)],
        ),  # workers > inputs
    ],
)
def test_gang_zmq_no_death(input_it, n_workers, expected):
    """
    No death in workers, test zmq gang.
    Check we have expected outputs (request_id, output),
    that the workers have been stopped.
    """
    with ExitStack() as stack:
        tempdir = stack.enter_context(tempfile.TemporaryDirectory())
        zmq_cfg = ZMQSubmititParams(
            SubmititConfig(folder=tempdir, slurm_job_name=f"test_requests_async_{id}",),
            check_status_freq=0.0,
        )
        run_gang(
            worker_launcher=ZmqSubmititWorkerLauncher(SimpleWorker, zmq_cfg),
            input_it=input_it,
            n_workers=n_workers,
            max_queued_inputs=1,
            expected=expected,
            total_worker_launched=n_workers,
            max_restarts=0,
            local=False,
            check_alive_freq=1,
        )


@pytest.mark.slow
def test_gang_zmq_death():
    """
    Test zmq gang, worker dies after 2 outputs. Test that thanks to  restarts we have all outputs.
    """
    with ExitStack() as stack:
        tempdir = stack.enter_context(tempfile.TemporaryDirectory())
        zmq_cfg = ZMQSubmititParams(
            SubmititConfig(folder=tempdir, slurm_job_name=f"test_requests_async_{id}",),
            check_status_freq=0.0,
        )
        run_gang(
            worker_launcher=ZmqSubmititWorkerLauncher(RaisingSimpleWorker, zmq_cfg),
            input_it=iter(range(8)),
            n_workers=2,
            max_queued_inputs=1,
            expected=[(i ** 2,) for i in range(8)],
            total_worker_launched=4,
            max_restarts=2,
            local=False,
            check_alive_freq=10,
        )


class AsyncWorkerGangRaising(AsyncWorkerGang):
    def manage_workers(self):
        super().manage_workers()
        raise RuntimeError


@pytest.mark.slow
def test_remote_gang_closing_if_make_it_raises():
    """
    Check that workers are properly closed if make_iterator raises.
    Here input_it will raise runtime error after 2s.
    """
    with ExitStack() as stack:
        tempdir = stack.enter_context(tempfile.TemporaryDirectory())
        zmq_cfg = ZMQSubmititParams(
            SubmititConfig(folder=tempdir, slurm_job_name=f"test_requests_async_{id}",),
            check_status_freq=1,
        )
        gang: AsyncWorkerGang = AsyncWorkerGang(
            worker_launcher=ZmqSubmititWorkerLauncher(SimpleWorker, zmq_cfg),
            n_workers=3,
            max_restarts=0,
            max_queued_inputs=1,
            check_alive_freq=1,
        )

        def input_it():
            timeout = Timeout(2)
            while timeout.ok():
                yield 1

        with pytest.raises(RuntimeError):
            with logged_closing(gang, "gang"):
                for _ in make_iterator(
                    gang,
                    max_in_worker=gang.max_queued_inputs * gang.n_workers,
                    input_it=input_it(),
                ):
                    pass

        assert all(worker.closed for worker in gang._all_workers())


@pytest.mark.slow
def test_remote_gang_closing_if_manage_workers_raises():
    """
    Check that workers are properly closed if manage_workers raises.
    Use of AsyncWorkerGangRaising that raises in manage_worker.
    """
    with ExitStack() as stack:
        tempdir = stack.enter_context(tempfile.TemporaryDirectory())
        zmq_cfg = ZMQSubmititParams(
            SubmititConfig(folder=tempdir, slurm_job_name=f"test_requests_async_{id}",),
            check_status_freq=1,
        )
        gang: AsyncWorkerGang = AsyncWorkerGangRaising(
            worker_launcher=ZmqSubmititWorkerLauncher(SimpleWorker, zmq_cfg),
            n_workers=3,
            max_restarts=0,
            max_queued_inputs=1,
            check_alive_freq=1,
        )

        def input_it():
            while True:
                yield 1

        with pytest.raises(RuntimeError):
            with logged_closing(gang, "gang"):
                for _ in make_iterator(
                    gang,
                    max_in_worker=gang.max_queued_inputs * gang.n_workers,
                    input_it=input_it(),
                ):
                    pass

        assert all(worker.closed for worker in gang._all_workers())
