# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Set, Dict, Any, List, Tuple
import zmq
import pickle
from zmq import ZMQError
import socket
from pathlib import Path
import os
import logging
import submitit
from dataclasses import dataclass, field
import time
import psutil
from evariste import json as json
from evariste.comms.zmq import ManagedZmqContext
from evariste.logger import create_logger
from evariste.utils import (
    this_job_id,
    environment_variables,
    logged_closing,
    OurSignalHandler,
)
from evariste.clusters.utils import clusterify_partitions
from evariste.datasets.lean import LeanDatasetConf
from evariste.metrics import Rate, StatsCollection, Timer, ActivityTracker
from lean_cluster.utils import reset_slurm_env

from leanml import get_api, WrongModule, DeadLean


logger = create_logger(None)


LeanInstanceDied = "LEAN_INSTANCE_DIED"


@dataclass
class ReqID:
    worker_id: str
    req_id: str


class HeartBeat:
    def __init__(self, cpu_usage: float, mem_usage: float):
        self.cpu_usage = cpu_usage
        self.mem_usage = mem_usage


MAX_ALIVE = 3600 * 5


@dataclass
class InstanceStats(StatsCollection):
    # Ideally, process rate should be >= recv_rate, otherwise the queue will grow.
    recv_rate: Rate = field(default_factory=lambda: Rate())
    recv_rate_mb: Rate = field(default_factory=lambda: Rate())
    process_rate: Rate = field(default_factory=lambda: Rate())
    sent_rate_mb: Rate = field(default_factory=lambda: Rate())
    lean_time: Timer = field(default_factory=lambda: Timer())
    full_time: Timer = field(default_factory=lambda: Timer())

    new_session_rate: Rate = field(default_factory=lambda: Rate())
    del_session_rate: Rate = field(default_factory=lambda: Rate())

    # starving detection
    no_tactic_in_queue: Timer = field(
        default_factory=lambda: Timer(cum_ratio=True, overall_cum_time=True)
    )
    less_than_100_tactic_in_queue: Timer = field(
        default_factory=lambda: Timer(cum_ratio=True, overall_cum_time=True)
    )
    no_session_with_tacs: Timer = field(
        default_factory=lambda: Timer(cum_ratio=True, overall_cum_time=True)
    )
    less_than_1_session_with_tacs: Timer = field(
        default_factory=lambda: Timer(cum_ratio=True, overall_cum_time=True)
    )

    # timers
    cycle: Timer = field(default_factory=lambda: Timer())
    heartbeat: Timer = field(default_factory=lambda: Timer())
    incoming: Timer = field(default_factory=lambda: Timer())
    outputting: Timer = field(default_factory=lambda: Timer())

    # tracks answers from lean threads
    active_threads: ActivityTracker = field(default_factory=lambda: ActivityTracker())


class LeanInstance:
    """
    Boots a lean instance and communicates with it.
    If lean raises DeadLean, kill all sessions, send back InstanceDied to currently open sessions
    """

    def __init__(
        self,
        comms_folder: Path,
        dataset: LeanDatasetConf,
        heartbeat: int,
        debug: bool = False,
    ):
        self.comms_folder = comms_folder
        self.dataset = dataset.get_materialized()
        self.debug = debug
        self.dealers: Set[str] = set()
        self.open_sessions: Set[str] = set()
        self.req_to_dealer: Dict[str, ReqID] = {}

        self.req_to_timestamp: Dict[str, float] = {}

        self.stats = InstanceStats()
        self.last_cpu_util = 0
        self.last_mem = 0.0

        logger.info(f"Using lean checkpoint : {Path(self.dataset.checkpoint_path)}")
        logger.info(f"Lean dataset: {self.dataset}")
        self.api = get_api(
            Path(self.dataset.checkpoint_path),
            fast=self.dataset.fast,
            preload="ml_server" in self.dataset.checkpoint_path,  # preload if not ckpt
            quiet=not debug,
            path_to_fifos=self.dataset.path_to_fifos,
            num_threads=self.dataset.num_threads,
            dump_comms=debug,
            old=self.dataset.is_old,
            additional_roots=None
            if dataset.is_old
            else [Path(self.dataset.statement_splits_path)],
        )
        self.lean_process = psutil.Process(self.api._proc.pid)  # type: ignore
        logger.info("API BOOTED")

        self.zmq_context = ManagedZmqContext()
        self.sock = self.zmq_context.socket(zmq.ROUTER)
        self.port = self.sock.bind_to_random_port(f"tcp://*")
        self.sockaddr = f"{socket.gethostname()}:{self.port}"
        filename = f"{self.sockaddr}__{this_job_id()}"
        os.makedirs(self.comms_folder, exist_ok=True)
        (self.comms_folder / filename).touch()  # now accept connexions
        logger.info(f"TOUCHED {self.comms_folder / filename}")
        self.heartbeat = heartbeat
        self.last_heartbeat = time.time()
        # tactic_str, start_time, session
        self.outstanding_tactics: Dict[int, Tuple[str, float, str]] = {}

    def run(self):
        try:
            self._run()
        except DeadLean:
            logger.error("LEAN INSTANCE DIED")

    def close(self):
        # if we've been stopped, we don't care about lost messages
        self.zmq_context.close()

    def do_send(self, request) -> str:
        assert self.api is not None
        req_type = request.pop("req_type")
        send_func = {
            "new_session": self.api.new_session,
            "del_session": self.api.del_session,
            "parse_goal": self.api.parse_goal,
            "parse_goal_and_apply_tactic": self.api.parse_goal_and_apply_tactic,
            "parse_children": self.api.parse_children,
            "parse_command": self.api.parse_command,
            "send_tactic": self.api.send_tactic,
            "eval_cmd": self.api.eval_cmd,
        }[req_type]

        # annoyingly, del_session uses name and not session_name...
        if req_type == "del_session":
            self.stats.del_session_rate.act(1)
            request["name"] = request.pop("session_name")
            try:
                self.open_sessions.remove(request["name"])
            except KeyError:
                logger.warning(
                    f"Tried to remove session {request['name']} from open "
                    f"sessions it is not open"
                )
        elif req_type == "new_session":
            self.stats.new_session_rate.act(1)

        to_ret = send_func(**request)  # type: ignore  # all functions are of different types...
        if req_type == "send_tactic":
            self.outstanding_tactics[to_ret] = (
                request["tactic_str"],
                time.time(),
                request["session_name"],
            )
        return to_ret

    def _run(self):
        last_log = time.time()
        self.stats.cycle.start()
        while True:

            self.stats.cycle.stop()
            self.stats.cycle.start()

            if time.time() - last_log > 60:
                last_log = time.time()
                mem_av_gb = psutil.virtual_memory().available / (1024 ** 3)
                self.last_cpu_util = self.lean_process.cpu_percent(interval=None)
                self.last_mem = self.lean_process.memory_info().rss / (1024 ** 3)
                logger.info(f"[Lean CPU] {self.last_cpu_util}")
                logger.info(f"[Lean Mem] {int(self.last_mem)}G")
                logger.info(f"[Mem] Available memory on host: {mem_av_gb}GB")

                # to prevent being at the middle of a cycle
                self.stats.no_tactic_in_queue.stop_if_not_stopped()
                self.stats.less_than_100_tactic_in_queue.stop_if_not_stopped()
                self.stats.no_session_with_tacs.stop_if_not_stopped()
                self.stats.less_than_1_session_with_tacs.stop_if_not_stopped()

                logs = self.stats.rate_and_reset()
                logs["in_queue"] = len(self.req_to_dealer)
                logs["n_open_sessions"] = len(self.open_sessions)
                logs["n_sessions_with_outstanding_tactics"] = len(
                    {name for _, _, name in self.outstanding_tactics.values()}
                )

                logger.info(json.dumps(logs) + "\n")
                cur = time.time()
                for tactic_str, process_start, _ in self.outstanding_tactics.values():
                    delta = cur - process_start
                    if delta > 300:
                        logger.error(f"SLOW TACTIC: {tactic_str}")
                        raise RuntimeError("Slow tac -> killed")

            self._inspect_outstanding_tactics()

            self.stats.heartbeat.start()
            if time.time() - self.last_heartbeat > self.heartbeat:
                sent_heartbeat = False
                for worker_id in self.dealers:
                    self.sock.send_multipart(
                        [
                            worker_id,
                            pickle.dumps(HeartBeat(self.last_cpu_util, self.last_mem)),
                        ]
                    )
                    sent_heartbeat = True
                if sent_heartbeat:
                    self.last_heartbeat = time.time()
                    logger.info("sent heartbeat")
                else:
                    logger.info("No heartbeat sent since no dealer")
            self.stats.heartbeat.stop()

            self.stats.incoming.start()
            # receive from zmq and send to api
            try:
                worker_id, data = self.sock.recv_multipart(zmq.NOBLOCK)
                self.dealers.add(worker_id)
                if len(self.dealers) > 1:
                    logger.error(f"Received more than one dealer: {self.dealers}")
                    raise RuntimeError(f"Too many dealers: {self.dealers}")
            except ZMQError:
                pass
            else:
                received: Dict[str, Any] = pickle.loads(data)
                assert isinstance(received, dict), (type(received), received)
                if received.get("req_type") == "dealer_connected":
                    client_job_id = received["client_job_id"]
                    logger.info(
                        f"Connected to client with job_id: {client_job_id} - "
                        f"worker_id: {worker_id}"
                    )
                else:
                    original_req_id = received.pop("req_id")
                    rid = ReqID(worker_id=worker_id, req_id=original_req_id)
                    self.stats.recv_rate.act()
                    self.stats.recv_rate_mb.act(len(data) / 1024 ** 2)
                    try:
                        internal_req_id = self.do_send(received)
                    except WrongModule as e:
                        # == DeclNotFound
                        self.sock.send_multipart(
                            [
                                worker_id,
                                pickle.dumps(
                                    {"req_id": original_req_id, "error": str(e)}
                                ),
                            ]
                        )
                    else:
                        self.req_to_dealer[internal_req_id] = rid
                        self.req_to_timestamp[internal_req_id] = time.time()
            self.stats.incoming.stop()

            # receive from api, send to zmq
            self.stats.outputting.start()
            try:
                received: Dict[str, Any] = self.api.recv(timeout=0.01)
            except TimeoutError:
                pass
            else:
                req_id = self.req_to_dealer.pop(received["req_id"])
                if "thread_id" in received:
                    self.stats.active_threads.act(received["thread_id"])
                    received["thread_id"] = f"{this_job_id()}_{received['thread_id']}"

                self.outstanding_tactics.pop(
                    received["req_id"], None
                )  # not outstanding anymore
                if "name" in received:
                    self.open_sessions.add(received["name"])
                else:
                    self.stats.process_rate.act()
                    if "eval_time" in received:
                        self.stats.lean_time.add_interval(float(received["eval_time"]))

                    process_time = time.time() - self.req_to_timestamp.pop(
                        received["req_id"]
                    )
                    self.stats.full_time.add_interval(float(process_time))

                received["req_id"] = req_id.req_id
                data = pickle.dumps(received)
                self.stats.sent_rate_mb.act(len(data) / 1024 ** 2)
                self.sock.send_multipart([req_id.worker_id, data])
            self.stats.outputting.stop()

    def _inspect_outstanding_tactics(self):

        n_outstanding_tactics = len(self.outstanding_tactics)
        n_sessions_with_outstanding_tactics = len(
            {name for _, _, name in self.outstanding_tactics.values()}
        )

        def _toogle(timer: Timer, cond: bool):
            timer.start_if_not_started() if cond else timer.stop_if_not_stopped()

        _toogle(self.stats.no_tactic_in_queue, n_outstanding_tactics == 0)
        _toogle(self.stats.less_than_100_tactic_in_queue, n_outstanding_tactics <= 100)
        _toogle(
            self.stats.no_session_with_tacs, n_sessions_with_outstanding_tactics == 0
        )
        _toogle(
            self.stats.less_than_1_session_with_tacs,
            n_sessions_with_outstanding_tactics <= 1,
        )


def run_lean_instance(
    comms_folder: Path, dataset: LeanDatasetConf, heartbeat: int, debug: bool = False
):
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Initializing lean instance on {dataset}")
    OurSignalHandler.start()
    # raise Exception on sigterm to close "properly" rather than being killed
    # otherwise submitit bypasses this signal
    instance = LeanInstance(
        comms_folder=comms_folder, dataset=dataset, heartbeat=heartbeat, debug=debug
    )
    with logged_closing(instance, "lean_instance"):
        instance.run()

    logger.info("Exiting lean instance")


class LeanCluster:
    """ Lean Cluster is used for slurm accounting, launching and checking instance status.
    All standard out/err and submission.sh are stored in *prover_dump_path / lean_instances / expander_id*

    :param name: a name to identify the subsequently created slurm jobs
    :type name: str
    :param comm_folder: Where the zmq sockets are stored. See lean_cluster.client.LeanClusterClient
    :type comm_folder: Path
    :param dataset: Contains all relevant options for calling leanml.comms.get_api
    :type dataset: evariste.datasets.LeanDatasetConf 
    :param expander_id: Identify the specific lean cluster. In an online mcts context with 200 provers sharing the same prover dump path, each prover will have its own lean cluster identified by its *expander_id*.
    :type expander_id: str
    :param heartbeat: Frequency at which we expect to receive instance heartbeats. No received heartbeats for 5**heartbeats* and we cycle the instance.
    :type heartbeat: str

    """

    def __init__(
        self,
        name: str,
        comm_folder: Path,
        dataset: LeanDatasetConf,
        expander_id: str,
        heartbeat: int = 60,
    ):
        self.jobs: List[submitit.Job] = []
        self.name = name
        self.heartbeat = heartbeat
        self.comm_folder = comm_folder
        self.expander_id = expander_id
        self.job_folder = (
            self.comm_folder.parent.parent / "lean_instances" / self.expander_id
        )
        self.dataset = dataset
        self.launch_instance(dataset.num_instances)

    @environment_variables(SBATCH_NO_REQUEUE="1")
    def launch_instance(self, n_to_launch: int):
        if self.dataset.partition != "local":
            executor: submitit.Executor = submitit.AutoExecutor(
                folder=self.job_folder, slurm_max_num_timeout=-1
            )
            executor.update_parameters(
                slurm_job_name=self.name,
                slurm_timeout_min=24 * 60,
                slurm_gpus_per_node=0,
                slurm_gpus_per_task=0,
                slurm_cpus_per_task=self.dataset.num_threads,
                slurm_ntasks_per_node=1,
                slurm_partition=clusterify_partitions("Theorem_Proving"),
                # these work better than the inherited cluster defaults : mask_cpu // cyclic
                slurm_additional_parameters={"distribution": "block"},
                slurm_srun_args=["-vv", "--cpu-bind", "none"],
                slurm_mem_gb=100,
            )
        else:
            executor = submitit.LocalExecutor(folder=self.job_folder)
            executor.update_parameters(
                timeout_min=24 * 60, gpus_per_node=0,
            )
        with reset_slurm_env():
            with executor.batch():
                logger.info(f"Launching {n_to_launch} instances with submitit")
                for _ in range(n_to_launch):
                    job = executor.submit(
                        run_lean_instance,
                        self.comm_folder,
                        self.dataset,
                        debug=False,
                        heartbeat=self.heartbeat,
                    )
                    self.jobs.append(job)

    def check(self):
        """ update job statuses and relaunch if needed """
        logger.info("lean_cluster.check")
        to_relaunch = 0
        next_jobs = []
        for job in self.jobs:
            if job.done():
                logger.info(f"Detected dead instance ({job})")
                logger.info(
                    f"Client will detect this when it will not receive "
                    f"heartbeat for some time"
                )
                to_relaunch += 1
            else:
                next_jobs.append(job)

        self.jobs = next_jobs
        if to_relaunch > 0:
            logger.info(f"lean_cluster.check relaunching n_jobs: {to_relaunch}")
            self.launch_instance(to_relaunch)
            logger.info(f"lean_cluster.check after relaunch: self.jobs: {self.jobs}")
        logger.info("Done for lean_cluster.check")

    def current_job_ids(self) -> Set[str]:
        return {j.job_id for j in self.jobs}

    def kill(self, job_id: str):
        logger.info(f"lean_cluster.kill request for job_id: {job_id}")
        logger.info(f"lean_cluster.kill self.jobs: {self.jobs}")
        next_jobs = []
        to_relaunch = 0
        for j in self.jobs:
            if j.job_id == job_id:
                logger.info(f"lean_cluster.kill canceling {job_id}")
                j.cancel()
                to_relaunch += 1
            else:
                next_jobs.append(j)
        self.jobs = next_jobs
        if to_relaunch > 0:
            logger.info(f"lean_cluster.kill relaunching n_jobs: {to_relaunch}")
            self.launch_instance(to_relaunch)
            logger.info(f"lean_cluster.kill after relaunch: self.jobs: {self.jobs}")
        else:
            # can happen if job has been detected done and already killed
            logger.warning(f"lean_cluster.kill request {job_id} not found in jobs")

    def close(self):
        logger.info("Closing LeanCluster ...")
        for j in self.jobs:
            logger.info(f"Canceling: {j.job_id}")
            j.cancel(check=False)
            logger.info(f"Canceled: {j.job_id}")
        logger.info("Closed LeanCluster")
