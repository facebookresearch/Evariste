# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field
from logging import getLogger
from typing import Dict, Optional, List, Any, Set, Union
from zmq import Socket, Context, ZMQError
from collections import defaultdict
from pathlib import Path
import os
import zmq
import time
import pickle
import random
import string

from evariste.comms.zmq import ManagedZmqContext
from evariste.metrics import ActionCounter, Max, StatsCollection, Timer, Avg
from evariste.utils import this_job_id
from lean_cluster.instance import (
    LeanCluster,
    LeanDatasetConf,
    LeanInstanceDied,
    HeartBeat,
)


logger = getLogger()


@dataclass
class ClientStats(StatsCollection):
    send: Timer = field(default_factory=lambda: Timer())
    recv: Timer = field(default_factory=lambda: Timer())
    refresh: Timer = field(default_factory=lambda: Timer())
    refresh_instance_every_max_time: Max = field(default_factory=lambda: Max())
    refresh_instance_every_mean_time: Avg = field(default_factory=lambda: Avg())
    died_no_heartbeat_rate: ActionCounter = field(
        default_factory=lambda: ActionCounter("", silent=True, is_rate=True)
    )
    died_job_failed_rate: ActionCounter = field(
        default_factory=lambda: ActionCounter("", silent=True, is_rate=True)
    )


class LeanClusterClient:
    """
    The goal of this class is to replicate the LeanAPI api, but with
    many potentially remote lean instances.
    We distribute create_sessions evenly, and save where each sessions are.
    If an instance dies:

    * it returns "error": LeanInstanceDied for all its outstanding requests

    * any subsequent requests that should have been sent to it are cancelled

    * the instance is relaunched

    Communications are handled via ZMQ.

    *__init__* is blocking, as it waits for all lean instances to be up and running

    :param name: a name to identify the subsequently created slurm jobs
    :type name: str
    :param comm_folder: ZMQ socket addresses will be stored in *comm_folder / expander_id*. In general, this is *prover_dump_path / lean_cluster_comms*
    :type comm_folder: Path
    :param dataset: Contains all relevant options for calling leanml.comms.get_api
    :type dataset: evariste.datasets.LeanDatasetConf
    :param expander_id: Identify the specific lean cluster. In an online mcts context with 200 provers sharing the same prover dump path, each prover will have its own lean cluster identified by its *expander_id*.
    :type expander_id: str
    """

    def __init__(
        self,
        name: str,
        comm_folder: Path,
        dataset: LeanDatasetConf,
        expander_id: str,
    ):
        """Waits for the lean cluster server to be running and ready to accept work"""
        self.comm_folder = comm_folder / expander_id
        os.makedirs(self.comm_folder, exist_ok=True)
        logger.info(f"lean cluster comm_folder {self.comm_folder}")
        self.heartbeat = 60
        self.lean_cluster = LeanCluster(
            name, self.comm_folder, dataset, expander_id, self.heartbeat
        )
        self.session_to_node: Dict[str, str] = {}
        self.zmq_context = ManagedZmqContext()
        self.lean_instances: Dict[str, Socket] = {}
        self.sock_addr_to_job_id: Dict[str, str] = {}
        self.last_heartbeats: Dict[str, float] = {}
        self.active_sessions_on_worker: Dict[str, int] = {}
        self.requests_on_worker: Dict[str, Dict[str, Dict]] = defaultdict(dict)
        self.next_req_id = 0
        self.closed = False
        self.dead_instances: Set[str] = set()
        self.to_receive: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        self.timers = ClientStats()

        self.cpu_usage = ActionCounter("cpu_usage", is_rate=False)
        self.mem_usage = ActionCounter("mem_usage", is_rate=False)
        self.last_refresh = time.time()

        try:
            self.refresh_instances(wait_for=dataset.num_instances)
        except Exception:
            # close lean cluster if crashing
            self.close()
            raise

    def alive(self):
        self.lean_cluster.check()

    def instance_is_dead(self, sockaddr):
        if sockaddr in self.dead_instances:
            return  # missing heartbeat after death
        self.dead_instances.add(sockaddr)
        self.lean_instances.pop(sockaddr).close(linger=0)
        self.active_sessions_on_worker.pop(sockaddr)
        self.last_heartbeats.pop(sockaddr)
        to_fail = self.requests_on_worker.pop(sockaddr, {})
        job_id = self.sock_addr_to_job_id.pop(sockaddr)
        logger.info(
            f"instance_is_dead: job_id: {job_id} sock: {sockaddr} "
            f"n dead requests: {len(to_fail)}"
        )
        for req_id in to_fail:
            self.to_receive[sockaddr].append(
                {"req_id": req_id, "error": LeanInstanceDied}
            )
        os.unlink(self.comm_folder / f"{sockaddr[len('tcp://'):]}__{job_id}")
        self.lean_cluster.kill(job_id)

    def refresh_instances(self, wait_for: int = 0):
        """
        * Mark any instance that missed its heartbeat as dead
        * wait for at least "wait_for" instances to be online (ie, API loaded)
        * check job status of all instances every 60s:

        :param wait_for: Block until *wait_for*  instances are ready to accept work
        :type wait_for: int
        """
        self.timers.refresh.start()

        now = time.time()
        since_last_refresh = now - self.last_refresh

        if (len(self.lean_instances) >= wait_for) and (since_last_refresh < 10):
            self.timers.refresh.stop()
            return

        self.last_refresh = now
        self.timers.refresh_instance_every_mean_time.act(since_last_refresh)
        self.timers.refresh_instance_every_max_time.act(since_last_refresh)

        for sockaddr, last_heartbeat in list(self.last_heartbeats.items()):

            if sockaddr in self.dead_instances:
                logger.warning(f"{sockaddr} is in dead instance, should not happen!")
                continue

            job_id = self.sock_addr_to_job_id[sockaddr]
            if time.time() - last_heartbeat > 5 * self.heartbeat:
                # instance is as good as dead
                logger.info(
                    f"No heartbeat for {sockaddr} for {time.time() - last_heartbeat}s "
                    f"job_id: {job_id} "
                    f"(time since previous refresh: {since_last_refresh:.02f})"
                )
                logger.warning(f"{sockaddr} job_id: {job_id} is DEAD (no heartbeat)")
                self.instance_is_dead(sockaddr)
                self.timers.died_no_heartbeat_rate.act(1)
            elif job_id not in self.lean_cluster.current_job_ids():
                logger.warning(
                    f"{sockaddr} job_id: {job_id} is DEAD "
                    f"(not anymore in lean_cluster jobs)"
                )
                self.instance_is_dead(sockaddr)
                self.timers.died_job_failed_rate.act(1)

        first = True
        last_check = time.time()

        while len(self.lean_instances) < wait_for or first:
            # update job status every 60 seconds
            if time.time() - last_check > 60:
                self.lean_cluster.check()
                last_check = time.time()

            first = False

            for f in os.listdir(self.comm_folder):
                addr, job_id = f.split("__")
                sock_addr = f"tcp://{addr}"
                if (
                    sock_addr not in self.lean_instances
                    and sock_addr not in self.dead_instances
                ):
                    sock = self.zmq_context.socket(zmq.DEALER)
                    letters = string.ascii_lowercase
                    identity = "".join(random.choice(letters) for _ in range(5))
                    sock.identity = identity.encode("utf-8")
                    logger.info(f"Adding client {sock_addr}, job_id: {job_id}")

                    sock.connect(sock_addr)
                    self.lean_instances[sock_addr] = sock
                    self.last_heartbeats[sock_addr] = time.time()
                    self.active_sessions_on_worker[sock_addr] = 0
                    self.sock_addr_to_job_id[sock_addr] = job_id

                    # WARNING we can have more than 4 sockets if
                    # lean_cluster detected dead instance (via submitit/slurm)
                    # but we didn't detected it (since we wait for hearbeat to consider
                    # as dead)
                    logger.info(
                        f"Currently n instances: "
                        f"{len(self.sock_addr_to_job_id)} with job_ids"
                        f" {self.sock_addr_to_job_id}"
                    )

                    to_send = {
                        "req_type": "dealer_connected",
                        "client_job_id": this_job_id(),
                    }
                    self._send(to_send, sock_addr, sock)

            if len(self.lean_instances) < wait_for:
                time.sleep(10)
                logger.info(
                    f"Client waiting for {wait_for - len(self.lean_instances)} lean instances"
                )

        self.timers.refresh.stop()

    def new_session(
        self,
        module_path: str,
        decl_name: str,
        merge_alpha_equiv: bool,
        pp_opts: Optional[Dict] = None,
    ) -> str:

        return self.send(
            {
                "req_type": "new_session",
                "module_path": module_path,
                "decl_name": decl_name,
                "merge_alpha_equiv": merge_alpha_equiv,
                "pp_opts": pp_opts,
            }
        )

    def eval_cmd(self, to_run: str, module_path: Optional[str], timeout: int) -> str:

        return self.send(
            {
                "req_type": "eval_cmd",
                "to_run": to_run,
                "module_path": module_path,
                "timeout": timeout,
            }
        )

    def del_session(self, session_name: str):
        to_ret = self.send(
            {
                "req_type": "del_session",
                "session_name": session_name,
            }
        )
        sockaddr = self.session_to_node[session_name]
        if sockaddr in self.active_sessions_on_worker:
            self.active_sessions_on_worker[sockaddr] -= 1
            assert self.active_sessions_on_worker[sockaddr] >= 0

        return to_ret

    def parse_goal(self, goal_pp: str, session_name: str, timeout: int = 2000):
        return self.send(
            {
                "req_type": "parse_goal",
                "session_name": session_name,
                "goal_pp": goal_pp,
                "timeout": timeout,
            }
        )

    def parse_goal_and_apply_tactic(
        self,
        goal_pp: str,
        session_name: str,
        tactic_str: str,
        timeout: int = 2000,
        max_size: int = 1000**3,
        max_subgoals: int = 10_000,
        max_metavars: int = 10_000,
        max_repeated_hyps: int = 10_000,
        strip_tags: bool = True,
    ):
        return self.send(
            {
                "req_type": "parse_goal_and_apply_tactic",
                "session_name": session_name,
                "goal_pp": goal_pp,
                "tactic_str": tactic_str,
                "timeout": timeout,
                "max_size": max_size,
                "max_subgoals": max_subgoals,
                "max_metavars": max_metavars,
                "max_repeated_hyps": max_repeated_hyps,
                "strip_tags": strip_tags,
            }
        )

    def parse_children(
        self,
        children: List[str],
        session_name: str,
        timeout: int = 2000,
        max_metavars: int = 10_000,
        max_repeated_hyps: int = 10_000,
    ):
        return self.send(
            {
                "req_type": "parse_children",
                "session_name": session_name,
                "children": children,
                "timeout": timeout,
                "max_metavars": max_metavars,
                "max_repeated_hyps": max_repeated_hyps,
            }
        )

    def parse_command(self, goal_pp: str, session_name: str, timeout: int = 2000):
        return self.send(
            {
                "req_type": "parse_command",
                "session_name": session_name,
                "goal_pp": goal_pp,
                "timeout": timeout,
            }
        )

    def send_tactic(
        self,
        session_name: str,
        state_id: int,
        tactic_str: str,
        timeout: int = 2000,
        max_size: int = 1000**3,
        max_subgoals: int = 10_000,
        max_metavars: int = 10_000,
        max_repeated_hyps: int = 10_000,
        nosplit: bool = False,
    ) -> str:
        return self.send(
            {
                "req_type": "send_tactic",
                "session_name": session_name,
                "state_id": state_id,
                "tactic_str": tactic_str,
                "timeout": timeout,
                "max_size": max_size,
                "max_subgoals": max_subgoals,
                "max_metavars": max_metavars,
                "max_repeated_hyps": max_repeated_hyps,
                "nosplit": nosplit,
            }
        )

    def send(self, to_send: Dict[str, Any]) -> str:
        assert not self.closed, "Lean cluster closed"
        self.refresh_instances(wait_for=1)
        self.timers.send.start()
        to_send["req_id"] = str(self.next_req_id)
        self.next_req_id += 1
        if to_send["req_type"] in {"new_session", "eval_cmd"}:
            assert (
                len(self.active_sessions_on_worker) > 0
            ), "No worker available for new session!"
            worker = sorted(
                [
                    (v, sockaddr)
                    for sockaddr, v in self.active_sessions_on_worker.items()
                ]
            )
            sockaddr = worker[0][1]
            sock: Optional[Socket] = self.lean_instances[sockaddr]
            if to_send["req_type"] == "new_session":
                self.active_sessions_on_worker[worker[0][1]] += 1
        else:
            assert "session_name" in to_send
            sockaddr = self.session_to_node[to_send["session_name"]]
            sock = self.lean_instances.get(
                sockaddr, None
            )  # None -> sockaddr in self.dead_instances

        if sockaddr in self.dead_instances:
            assert (
                to_send["req_type"] != "new_session"
            ), "active_sessions_on_worker not properly maintained"
            self.to_receive[sockaddr].append(
                {"req_id": to_send["req_id"], "error": LeanInstanceDied}
            )
        else:
            self.requests_on_worker[sockaddr][to_send["req_id"]] = to_send
            assert sock is not None
            self._send(to_send, sockaddr, sock)
        self.timers.send.stop()
        return to_send["req_id"]

    def _send(self, to_send: Dict, sockaddr: str, sock: Socket):
        attempt = 0
        while True:
            try:
                assert sock is not None
                sock.send(pickle.dumps(to_send), flags=zmq.NOBLOCK)
                break
            except ZMQError:
                # Either the instance died or it's stuck somehow. Count it as dead.
                time.sleep(0.1)
                attempt += 1
                if attempt > 100:
                    logger.info(
                        f"Didn't manage to send to job_id: "
                        f"{self.sock_addr_to_job_id[sockaddr]} "
                        f"socket: {sockaddr} for 100 attemps, marking as dead"
                    )
                    self.instance_is_dead(sockaddr)
                    break

    def recv(self, timeout: float = -1) -> Optional[Dict[str, Any]]:
        self.refresh_instances(wait_for=1)
        self.timers.recv.start()
        for sockaddr, results in self.to_receive.items():
            to_ret = results.pop()
            if len(results) == 0:
                self.to_receive.pop(sockaddr)
            self.timers.recv.stop()
            return to_ret
        for sockaddr, sock in list(self.lean_instances.items()):
            try:
                received: Union[Dict, HeartBeat] = sock.recv_pyobj(zmq.NOBLOCK)
                if isinstance(received, HeartBeat):
                    self.last_heartbeats[sockaddr] = time.time()
                    self.cpu_usage.act(received.cpu_usage)
                    self.mem_usage.act(received.mem_usage)
                    continue
            except ZMQError:
                continue

            # If answer to session creation, store mapping
            if "name" in received:
                # if this raises, juse use sockaddr__session for session names.
                assert (
                    received["name"] not in self.session_to_node
                ), "Two lean instances used the same session name!"
                self.session_to_node[received["name"]] = sockaddr
            self.requests_on_worker[sockaddr].pop(received["req_id"])
            self.timers.recv.stop()
            return received

        self.timers.recv.stop()
        raise TimeoutError

    def close(self):
        """Call lean_cluster.instance.LeanCluster.close() and closes the zmq context."""
        if not self.closed:
            logger.info("Closing LeanClusterClient ...")
            self.lean_cluster.close()  # calls cancel on all jobs
            logger.info("Closing zmq_context ...")
            self.zmq_context.close()
            logger.info("Closed zmq_context")
            logger.info("Closed LeanClusterClient")
            self.closed = True
        else:
            logger.info("Already closed LeanClusterClient")
