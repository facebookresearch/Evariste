# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field, asdict
from typing import (
    Callable,
    Tuple,
    Generic,
    List,
    Dict,
    Optional,
    Union,
    cast,
)

from evariste.async_workers.async_worker import (
    AsyncWorker,
    AsyncWorkerLauncher,
    Input,
    Output,
    RequestId,
)
from evariste.backward.prover.utils import set_MKL_env_vars
from evariste.comms.zmq import ManagedZmqContext
from evariste.utils import (
    this_job_id,
    logged_closing,
    environment_variables,
    set_TMPDIR,
    OurSignalHandler,
)
import zmq
from zmq import ZMQError, Socket
import socket
import submitit
from enum import Enum
import pickle
import time
from contextlib import ExitStack
from evariste.logger import create_logger

logger = create_logger(None)


WorkerId = int
IsFullLogging = bool


@dataclass
class HeartBeat:
    pass


@dataclass
class Connect:
    worker_id: int


class WorkerStatus(str, Enum):
    NOT_STARTED = "not_started"
    STARTING = "starting"
    READY = "ready"
    DONE = "done"
    DEAD = "dead"

    def is_alive(self):
        return self in {WorkerStatus.STARTING, WorkerStatus.READY}


@dataclass
class SubmititConfig:
    """
    Config for submitit executor
    """

    folder: str
    slurm_job_name: str
    local: bool = True
    slurm_timeout_min: int = 240
    slurm_gpus_per_task: int = 0
    slurm_cpus_per_task: int = 1
    slurm_ntasks_per_node: int = 1
    slurm_partition: str = "Theorem_Proving"
    slurm_mem_gb: int = 10
    slurm_array_parallelism: int = 256  # number max of tasks executed in parallel
    slurm_constraint: str = ""
    slurm_exclude: str = ""
    slurm_max_num_timeout: int = -1
    slurm_srun_args: List[str] = field(default_factory=list)

    def make_executor(
        self, visible_gpus: Optional[List[int]] = None
    ) -> submitit.Executor:
        if visible_gpus is None:
            visible_gpus = [1]
        # Submitit
        if self.local:
            executor: submitit.Executor = submitit.LocalExecutor(folder=self.folder)
            executor.update_parameters(
                timeout_min=self.slurm_timeout_min,
                gpus_per_node=1,
                visible_gpus=visible_gpus,
            )
        else:
            executor = submitit.AutoExecutor(
                folder=self.folder, slurm_max_num_timeout=self.slurm_max_num_timeout,
            )
            parameters = asdict(self)
            del parameters["folder"]
            del parameters["local"]
            del parameters["slurm_max_num_timeout"]
            executor.update_parameters(**parameters)
        return executor


@dataclass
class ZMQSubmititParams:
    submitit_cfg: SubmititConfig
    heartbeat_freq: float = 60.0
    max_heartbeat_missed: int = 5
    check_status_freq: float = 60.0


class AliveError(Exception):
    pass


class ZMQSubmititWorker(Generic[Input, Output], AsyncWorker[Input, Output]):
    def __init__(
        self,
        worker_factory: Callable[[WorkerId, IsFullLogging], AsyncWorker[Input, Output]],
        cfg: ZMQSubmititParams,
        router_socket: "_RouterZMQSocket",
        full_logging: bool = True,
    ):
        """
        Wrapper around an async worker to make it distributed. The inner async worker will be created on a new slurm job
        and this job will communicate with this object via ZMQ.

         Create it using ZmqSubmititWorkerLauncher (and not this __init__).

        The interface of this an instance will be the same than the inner worker for the :meth:`ZMQSubmititWorker.submit`
        and :meth:`ZMQSubmititWorker.ready`.

        In addition :meth:`ZMQSubmititWorker.is_alive` says if the remote job is dead (which can happen because of
        host failure or preemption for instance).



        :param worker_factory: factory for the inner worker. Take as input the worker_id and if the worker needs
         to have full logging.
        :param cfg:
        :param router_socket: shared zmq socket across clients
        :param full_logging: If the remote inner worker needs to have full logging
        """
        self.cfg = cfg
        self.worker_factory = worker_factory
        self._status = WorkerStatus.NOT_STARTED
        self.death_reason: Optional[str] = None
        self.last_heartbeat = 0.0
        self.last_check = 0.0
        self.job: Optional[submitit.Job] = None
        self.job_id: Optional[str] = None
        self.total_request_id = 0
        self._input_buffer: List[Tuple[RequestId, Input]] = []
        self._has_submitted_buffer = False

        self.heartbeat_freq = cfg.heartbeat_freq
        self.max_heartbeat_missed = cfg.max_heartbeat_missed
        self.socket: _RouterZMQSocket[Input, Output]
        self.socket = router_socket
        self.socket_worker_id = self.socket.get_new_worker_id()
        self.started = False
        self.stopped = False
        self.closed = False

        self.check_status_freq = cfg.check_status_freq
        self.last_check = time.time()
        self.full_logging = full_logging

    def start(self):
        if self.started:
            raise RuntimeError("Worker is already started. Can be started only once")
        self.launch_with_executor(self.cfg.submitit_cfg.make_executor())

    def launch_with_executor(self, executor: submitit.Executor):
        try:
            with environment_variables(SBATCH_NO_REQUEUE="1"):
                self.job = executor.submit(
                    _run_async_worker,
                    self.worker_factory,
                    self.socket.controller_addr,
                    self.heartbeat_freq,
                    self.socket_worker_id,
                    self.full_logging,
                )
            self._status = WorkerStatus.STARTING
        except submitit.core.utils.FailedJobError as e:
            self._status = WorkerStatus.DEAD
            self.death_reason = "submission"
        self.started = True

    def is_alive(self):
        """Check if worker was alive on last_check, happening in `ready`"""
        # no update in is_alive(), only in ready()
        assert self._status in WorkerStatus
        return self._status.is_alive()

    def _update_status(self):
        if not self._status.is_alive():
            return
        assert self.job is not None
        if self.job.done():
            try:
                _ = self.job.result()
                self._status = WorkerStatus.DONE
                logger.warning(
                    f"ZMQSubmititWorker {self.job_id} finished. \n {self.job.result()}."
                )
            except Exception as e:
                self._status = WorkerStatus.DEAD
                if "timed-out" in str(e):
                    self.death_reason = "timed-out"
                elif "Sigterm" in str(e):
                    self.death_reason = "sigterm"
                else:
                    self.death_reason = "other"
                logger.warning(f"ZMQSubmititWorker {self.job_id} died. \n {str(e)}.")
        # if not done or dead, can still be stopping
        # or I check that I am still starting or now connected i.e ready
        elif self._status == WorkerStatus.STARTING:
            if self.socket.is_connected(self.socket_worker_id, refresh=True):
                logger.info(f"ZMQSubmititWorker {self.job_id} is connected.")
                self.last_heartbeat = time.time()
                self._status = WorkerStatus.READY
                self.job_id = self.job.job_id
        elif self._status == WorkerStatus.READY:
            if self._is_lagging(self.max_heartbeat_missed):
                assert self.job is not None
                logger.warning(
                    f"ZMQSubmititWorker {self.job_id} is lagging. Killing it. Dead"
                )
                self.job.cancel(check=False)
                self._status = WorkerStatus.DEAD
                self.death_reason = "lagging"

        else:
            raise RuntimeError(f"Unknow status {self._status}")

        return self._status.is_alive()

    def _is_lagging(self, heartbeat_missed):
        return (
            time.time() - self.last_heartbeat > heartbeat_missed * self.heartbeat_freq
        )

    def _send_input_buffer(self):
        while len(self._input_buffer):
            rid, inp = self._input_buffer.pop(0)
            self.socket.send_for_worker(self.socket_worker_id, rid, inp)

    def submit(self, inp: Input) -> RequestId:
        """
        Submit input to remote worker.

        :raise: exception `AliveError` if not started

        :param inp: input
        :return:
        """
        if self._status == "not_started":
            raise AliveError(
                "Worker has to be started before submit, call start() first."
            )
        request_id = self.total_request_id
        self.total_request_id += 1
        self._input_buffer.append((request_id, inp))
        return request_id

    def ready(self) -> List[Tuple[RequestId, Output]]:

        # must be done before connection check below to catch workers that died before connecting
        if time.time() - self.last_check > self.check_status_freq:
            self._update_status()
            self.last_check = time.time()

        if not self.socket.is_connected(worker_id=self.socket_worker_id, refresh=True):
            return []

        # the buffer has to be submitted here if we have not called submit once ready
        # otherwise we will never get the outputs
        self._send_input_buffer()

        all_outputs: List[Tuple[RequestId, Output]] = []
        results = self.socket.receive_for_worker(worker_id=self.socket_worker_id)
        for result in results:
            if isinstance(result, HeartBeat):
                self.last_heartbeat = time.time()
            else:
                all_outputs.append(result)

        return all_outputs

    @classmethod
    def batch_stats(cls, workers: List["ZMQSubmititWorker"]) -> Dict:
        stats = {
            "n_jobs_submitted": 0,
            "n_jobs_finished": 0,
            "n_jobs_died": 0,
            "n_jobs_died__timed-out": 0,
            "n_jobs_died__sigterm": 0,
            "n_jobs_died__lagging": 0,
            "n_jobs_died__other": 0,
            "n_jobs_died__submission": 0,
            "n_jobs_lagging": 0,
            "n_jobs_starting": 0,
            "n_jobs_ready": 0,
        }
        n_not_started = 0
        for worker in workers:
            assert isinstance(worker, cls)
            if worker._status == WorkerStatus.STARTING:
                stats["n_jobs_starting"] += 1
            elif worker._status == WorkerStatus.READY:
                stats["n_jobs_ready"] += 1
                if worker._is_lagging(heartbeat_missed=2):
                    stats["n_jobs_lagging"] += 1
            elif worker._status == WorkerStatus.DONE:
                stats["n_jobs_finished"] += 1
            elif worker._status == WorkerStatus.DEAD:
                stats["n_jobs_died"] += 1
                assert worker.death_reason is not None
                stats[f"n_jobs_died__{worker.death_reason}"] += 1
            elif worker._status == WorkerStatus.NOT_STARTED:
                n_not_started += 1
            else:
                raise RuntimeError(f"Unkown status {worker._status}")
        stats["n_jobs_submitted"] = len(workers) - n_not_started
        return stats

    def stop(self):
        """
        Send a "stop" signal to remote host for this one to close properly.

        :return:
        """
        if (
            not self._status.is_alive()
            or not self.socket.is_connected(self.socket_worker_id, refresh=False)
            or self.stopped
        ):
            return
        self.socket.send_for_worker(self.socket_worker_id, -1, "stop_worker")
        self.stopped = True

    def close(self):

        if self.closed:
            return

        logger.info(f"Closing ZMQSubmititWorker {self.job_id}...")

        if self.job is not None and self._status.is_alive():
            logger.info(f"Cancelling ZMQSubmititWorker {self.job_id}...")
            try:
                self.job.cancel(check=False)
            except Exception as e:
                logger.warning(
                    f"Could not cancelled ZMQSubmititWorker {self.job_id}. Error: {str(e)}..."
                )

        self.closed = True
        logger.info(f"ZMQSubmititWorker {self.job_id} closed.")


class ZmqSubmititWorkerLauncher(
    Generic[Input, Output], AsyncWorkerLauncher[Input, Output]
):
    def __init__(
        self,
        worker_factory: Callable[[WorkerId, IsFullLogging], AsyncWorker[Input, Output]],
        cfg: ZMQSubmititParams,
    ):
        """
        AsyncWorkerLauncher implementation for ZMQSubmitWorker.

        :param worker_factory: inner factory of async_worker (that will be executed remotely)
        :param cfg: cfg for Submitit and ZMQSubmititWorker
        """
        self.cfg = cfg
        self.local = self.cfg.submitit_cfg.local
        self.executor: Optional[submitit.Executor] = None
        if not self.local:
            self.executor = self.cfg.submitit_cfg.make_executor()
        self.worker_factory = worker_factory
        self.router_socket: _RouterZMQSocket = _RouterZMQSocket()

    @environment_variables(SBATCH_NO_REQUEUE="1")
    def launch_workers(
        self, n_workers: int, first_with_full_logging: bool = False
    ) -> List[AsyncWorker[Input, Output]]:
        workers: List[ZMQSubmititWorker[Input, Output]] = []
        for worker_id in range(n_workers):
            full_logging = False
            if first_with_full_logging and worker_id == 0:
                full_logging = True
            workers.append(
                ZMQSubmititWorker(
                    self.worker_factory,
                    self.cfg,
                    self.router_socket,
                    full_logging=full_logging,
                )
            )
        if self.local:
            for i, worker in enumerate(workers):
                executor = self.cfg.submitit_cfg.make_executor(visible_gpus=[i])
                worker.launch_with_executor(executor)
        else:
            assert self.executor is not None  # mypy
            with self.executor.batch():
                for worker in workers:
                    worker.launch_with_executor(self.executor)
        for worker in workers:
            assert worker.job is not None
            worker.job_id = worker.job.job_id
            logger.info(
                f"ZMQSubmititWorker {worker.job_id} submited. Logs: \n {worker.job.paths.stdout} \n {worker.job.paths.stderr}"
            )
            logger.info(
                f"ZMQSubmititWorker {worker.job_id} controller_addr: {worker.socket.controller_addr}"
            )
        return cast(List[AsyncWorker[Input, Output]], workers)

    def close(self):
        logger.info("Closing ZmqSubmititWorkerLauncher...")
        self.router_socket.close()
        logger.info("ZmqSubmititWorkerLauncher closed.")


###########
# Internals
###########


def _run_async_worker(
    worker_factory: Callable[[WorkerId, IsFullLogging], AsyncWorker],
    controller_addr: str,
    heartbeat_freq: int,
    zmq_worker_id: int,
    full_logging: bool,
):
    """
    Internal function used to execute the inner async worker on remote host
    """
    set_TMPDIR()
    OurSignalHandler.start()
    set_MKL_env_vars()

    job_id = this_job_id()
    with ExitStack() as stack:
        context = ManagedZmqContext()
        stack.enter_context(logged_closing(context, "controller_zmq_context"))
        controller_socket: Socket = context.socket(zmq.DEALER)
        controller_socket.identity = job_id.encode("ascii")
        controller_socket.connect(controller_addr)
        logger.info(f"RUN_ASYNC_WORKER: Controller_addr: {controller_addr}")
        worker = worker_factory(zmq_worker_id, full_logging)
        stack.enter_context(logged_closing(worker, "inner_async_worker"))
        worker.start()

        # only send for connection after the start that can be long
        controller_socket.send_pyobj(Connect(worker_id=zmq_worker_id))
        logger.info(f"RUN_ASYNC_WORKER: Send {Connect(worker_id=zmq_worker_id)}")

        last_heartbeat = time.time()

        inner_to_outer_rid = {}
        while True:
            now = time.time()
            if now - last_heartbeat > heartbeat_freq:
                logger.info("RUN_ASYNC_WORKER: send heartbeat")
                controller_socket.send_pyobj([HeartBeat()])
                last_heartbeat = now
            # check if new input has been sent and submit to worker to compute res, send res
            try:
                msg = controller_socket.recv(zmq.NOBLOCK)
            except ZMQError:  # no new input
                pass
            else:
                outer_rid, inp = pickle.loads(msg)  # type: ignore
                if inp == "stop_worker":
                    logger.warning("RUN_ASYNC_WORKER: receive stop, leaving.")
                    return
                inner_rid = worker.submit(inp)
                inner_to_outer_rid[inner_rid] = outer_rid
            outputs = [
                (inner_to_outer_rid.pop(inner_rid), out)
                for inner_rid, out in worker.ready()
            ]
            if len(outputs) > 0:
                controller_socket.send_pyobj(outputs)


class _RouterZMQSocket(Generic[Input, Output]):
    """
        Internal wrapper on a router socket that allows to handle multiple clients with one socket.

        Clients are identified by their worker_id: int.

        We can first get a new worker_id (`get_new_worker_id`) to send it as id to our client
        (sending and launching of client not handled by this object).

        For a given worker_id we can check if it is connected. Once it is connected we can check what
        was received from the given client, or send something to this given client
        """

    def __init__(self):
        context = ManagedZmqContext()
        sock = context.socket(zmq.ROUTER)
        port = sock.bind_to_random_port(f"tcp://*")
        hostname = socket.gethostname()
        controller_addr = f"tcp://{hostname}:{port}"

        self.context = context
        self.sock = sock
        self.port = port
        self.controller_addr = controller_addr

        self.worker_id_to_waiting_results: Dict[
            WorkerId, List[Union[HeartBeat, Tuple[RequestId, Output]]]
        ] = {}
        self.worker_id_to_worker_uuid: Dict[WorkerId, bytes] = {}
        self.worker_uuid_to_worker_id: Dict[bytes, WorkerId] = {}
        self.n_workers = 0

    def is_connected(self, worker_id: WorkerId, refresh: bool = True) -> bool:
        assert worker_id < self.n_workers
        connected = worker_id in self.worker_id_to_worker_uuid

        if refresh and not connected:
            self._receive_until_empty()
            connected = worker_id in self.worker_id_to_worker_uuid

        if connected:
            assert worker_id in self.worker_id_to_waiting_results, (
                worker_id,
                self.worker_id_to_waiting_results.keys(),
            )
        return connected

    def _receive_until_empty(self):
        while True:
            try:
                worker_uuid, result = self.sock.recv_multipart(zmq.NOBLOCK)
            except ZMQError:
                return
            assert worker_uuid is not None
            result = pickle.loads(result)
            if isinstance(result, Connect):
                logger.info(f"received {worker_uuid} {result}")
                assert worker_uuid not in self.worker_uuid_to_worker_id
                assert (
                    result.worker_id not in self.worker_id_to_waiting_results
                ), self.worker_id_to_waiting_results
                self.worker_id_to_worker_uuid[result.worker_id] = worker_uuid
                self.worker_uuid_to_worker_id[worker_uuid] = result.worker_id
                self.worker_id_to_waiting_results[result.worker_id] = []
            else:
                # print(f"received {worker_uuid} {result}")
                worker_id = self.worker_uuid_to_worker_id[worker_uuid]
                # should exist since connected
                self.worker_id_to_waiting_results[worker_id].extend(result)

    def receive_for_worker(
        self, worker_id: WorkerId
    ) -> List[Union[HeartBeat, Tuple[RequestId, Output]]]:
        assert self.is_connected(worker_id, refresh=False)
        self._receive_until_empty()
        results = self.worker_id_to_waiting_results[worker_id]
        self.worker_id_to_waiting_results[worker_id] = []
        return results

    def send_for_worker(
        self, worker_id: WorkerId, request_id: RequestId, inp: Union[str, Input]
    ):
        assert self.is_connected(worker_id, refresh=False)
        worker_uuid = self.worker_id_to_worker_uuid[worker_id]
        self.sock.send(worker_uuid, zmq.SNDMORE)
        self.sock.send_pyobj((request_id, inp))

    def close(self):
        logger.info("Closing _RouterZMQSocket ...")
        self.context.close()
        logger.info("_RouterZMQSocket closed.")

    def get_new_worker_id(self) -> WorkerId:
        worker_id = self.n_workers
        self.n_workers += 1
        return worker_id
