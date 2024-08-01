# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
from typing import Tuple, List, Dict, Any, Generic

from evariste.async_workers.async_worker import (
    AsyncWorker,
    AsyncWorkerLauncher,
    Input,
    Output,
    RequestId,
)
from evariste.logger import create_logger

logger = create_logger(None)


class AsyncWorkerGang(Generic[Input, Output], AsyncWorker[Input, Output]):
    """
   Create a pool of restartable AsyncWorker. It is useful when individual async worker can die and should be
   restarted, which can happen if async worker are distributed (like :class:ZmqSubmititWorker).

   When an async worker die it is restarted, and the inputs it was processing are rescheduled on other workers

   Gang worker is not alive anymore there is no more workers

   :param worker_launcher: launcher for the inner async workers
   :param n_workers: number of workers needed
   :param max_restarts: number of maximum restarts for workers. When maximum is reached, workers are not restarted.
   :param max_queued_inputs: number max of concurrent inputs for each worker
   :param check_alive_freq: time interval in seconds at which we check which worker is alive or not.
    """

    def __init__(
        self,
        worker_launcher: AsyncWorkerLauncher[Input, Output],
        n_workers: int,
        max_restarts: int,
        max_queued_inputs: int,
        check_alive_freq: int = 60,
    ):

        self.worker_launcher = worker_launcher
        assert n_workers > 0
        self.n_workers = n_workers
        self.started = False
        self.max_queued_inputs = max_queued_inputs
        assert self.max_queued_inputs > 0

        self.workers: Dict[int, AsyncWorker[Input, Output]] = {}
        self.dead_workers: Dict[int, AsyncWorker[Input, Output]] = {}
        self.stopping_workers: Dict[
            int, AsyncWorker[Input, Output]
        ] = {}  # waiting for close()

        self.curr_inputs: Dict[int, Dict[RequestId, Tuple[RequestId, Input]]] = {}
        self._cur_worker_id: int = 0
        self.max_restarts: int = max_restarts

        self._stop_workers_if_no_inputs: bool = False
        self.n_stopped_workers: int = 0

        self.input_buffer: List[Tuple[RequestId, Input]] = []
        self.n_dead_inputs = 0

        self.cur_req_id = 0

        self.check_alive_freq = check_alive_freq
        self.jobs_stats: Dict[str, Any] = {}
        self.worker_with_full_logging = None

        self.last_check = time.time()

    def stop_workers_if_no_inputs(self):
        """
        Once it is called, workers without any inputs are going to be stopped.
        """
        self._stop_workers_if_no_inputs = True

    def start(self):
        assert not self.started, "Gang should be started only once!"
        self._launch_workers_if_needed()

    def _launch_workers_if_needed(self):
        if len(self.input_buffer) == 0 and self.started:
            return
        n_start = self.n_workers - (len(self.workers) + self.n_stopped_workers)
        n_start = min(n_start, self.max_restarts + self.n_workers - self._cur_worker_id)
        if n_start == 0:
            return
        logger.info(
            f"WorkerGang: Start {n_start} new workers to process {len(self.input_buffer)} waiting inputs"
        )
        first_with_full_logging = False
        if self.worker_with_full_logging is None:
            first_with_full_logging = True
        new_workers = self.worker_launcher.launch_workers(
            n_workers=n_start, first_with_full_logging=first_with_full_logging,
        )
        if first_with_full_logging:
            self.worker_with_full_logging = self._cur_worker_id

        assert len(new_workers) == n_start
        for worker in new_workers:
            self.workers[self._cur_worker_id] = worker
            self.curr_inputs[self._cur_worker_id] = {}
            self._cur_worker_id += 1

        self.started = True

    def submit(self, inp: Input) -> RequestId:
        rid = self.cur_req_id
        self.input_buffer.append((rid, inp))
        self.cur_req_id += 1
        return rid

    def handle_result(self, result):
        return result

    def ready(self) -> List[Tuple[RequestId, Output]]:

        # submit inputs
        self._submit_input_buffers()

        # check status
        if time.time() - self.last_check > self.check_alive_freq:
            self.manage_workers()
            self.last_check = time.time()

        all_outputs = []
        # receive outputs
        for worker_id, worker in self.workers.items():
            outputs = worker.ready()
            for worker_rid, output in outputs:
                rid, _ = self.curr_inputs[worker_id].pop(worker_rid)
                all_outputs.append((rid, self.handle_result(output)))

        return all_outputs

    def close(self):
        logger.info(f"Closing {self.__class__.__name__} ...")
        for worker in self._all_workers():
            worker.close()
        self.worker_launcher.close()
        logger.info(f"{self.__class__.__name__} closed.")

    def is_alive(self):
        return len(self.workers) > 0

    def stop(self):
        logger.info("Stopping gang.")
        self.manage_workers()
        for worker_id in list(self.workers.keys()):
            self._stop_worker(worker_id)

    def _submit_input_buffers(self):
        for worker_id, worker in self.workers.items():
            while self._can_receive(worker_id=worker_id) and len(self.input_buffer) > 0:
                rid, inp = self.input_buffer.pop(0)
                worker_rid = worker.submit(inp)
                self.curr_inputs[worker_id][worker_rid] = (rid, inp)

    def _all_workers(self) -> List[AsyncWorker]:
        # should be iterator ?
        all_workers: List[AsyncWorker] = []
        all_workers.extend(self.workers.values())
        all_workers.extend(self.dead_workers.values())
        return all_workers

    def manage_workers(self):
        logger.info(
            f"{len(self.workers)} alive workers, {len(self.dead_workers)} dead workers."
        )
        assert self._cur_worker_id == len(self.workers) + len(self.dead_workers)

        # close stopped worker not closed (do it before stopping some workers)
        for worker_id in list(self.stopping_workers.keys()):
            worker = self.stopping_workers.pop(worker_id)
            worker.close()

        # handle dead workers
        for worker_id in list(self.workers.keys()):
            worker = self.workers[worker_id]
            if worker.is_alive():
                continue
            logger.info(f"worker not alive {worker_id}")
            worker.close()
            new_dead_inputs = [
                (rid, inp) for _, (rid, inp) in self.curr_inputs[worker_id].items()
            ]
            logger.info(
                f"Putting {new_dead_inputs} from worker {worker_id} in dead inputs"
            )
            self.n_dead_inputs += len(new_dead_inputs)
            self.input_buffer.extend(new_dead_inputs)
            self.dead_workers[worker_id] = worker
            del self.curr_inputs[worker_id]
            del self.workers[worker_id]
            if worker_id == self.worker_with_full_logging:
                self.worker_with_full_logging = None

        # check if we need to stop some workers
        if self._stop_workers_if_no_inputs and len(self.input_buffer) == 0:
            for worker_id in list(self.workers.keys()):
                # If the worker has computed all its inputs, stop worker
                if len(self.curr_inputs[worker_id]) == 0:
                    self._stop_worker(worker_id)

        # relaunch dead workers, but not the stopped one
        self._launch_workers_if_needed()

        if len(self.workers) > self.n_workers:
            logger.error(
                f"Unexpected number of workers: {len(self.workers)}, "
                f"with n_machines={self.n_workers}"
            )

        all_workers = self._all_workers()
        if all_workers:
            self.jobs_stats = all_workers[0].batch_stats(all_workers)

    def _can_receive(self, worker_id: int) -> bool:
        return len(self.curr_inputs[worker_id]) < self.max_queued_inputs

    def _stop_worker(self, worker_id: int):
        assert worker_id in self.workers
        worker = self.workers[worker_id]
        logger.info(f"Stopping worker {worker_id}.")
        worker.stop()
        assert len(self.curr_inputs[worker_id]) == 0
        self.dead_workers[worker_id] = worker
        # we want to close it later, after the worker stopped
        self.stopping_workers[worker_id] = worker
        self.n_stopped_workers += 1
        del self.curr_inputs[worker_id]
        del self.workers[worker_id]
