# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, List, Set, Generic, TypeVar
from threading import Thread, Event
from queue import Empty, Queue
from logging import getLogger
from pathlib import Path
from zmq import Socket
import os
import re
import zmq
import time
import torch
import socket
import pickle

from evariste.metrics import ActionCounter
from evariste.comms.store import Sender, Receiver
from evariste.utils import logged_closing, rstr


class ServerDisconnected(Exception):
    pass


class ZMQNotReady(Exception):
    pass


class ZMQNotReadySample:
    pass


S = TypeVar("S")


logger = getLogger()


class ManagedZmqContext:
    def __init__(self):
        self._context = zmq.Context()
        self._sockets: List[Socket] = []
        self.closed = False

    def socket(self, socket_type: int, **kwargs) -> Socket:
        assert not self.closed
        socket_ = self._context.socket(socket_type, **kwargs)
        self._sockets.append(socket_)
        return socket_

    def close(self):
        if self.closed:
            logger.warning("Context already closed!")
            return
        logger.info(f"Closing {len(self._sockets)} sockets...")
        for socket_ in self._sockets:
            socket_.close(linger=0)

        logger.info(f"Closed {len(self._sockets)} sockets")

        # maybe term() hang since not all sockets are closed yet
        # despite the close(linger=0) ?
        if not all(s.closed for s in self._sockets):
            logger.warning("Some sockets are still not closed!")

        self._sockets = []  # no more pointers on sockets

        # maybe we should not term here and let the context being destroyed by __del__?
        logger.info(f"Terminating zmq context...")
        self._context.term()
        logger.info(f"Terminated zmq context")
        self.closed = True

    def __del__(self):
        if not self.closed:
            raise RuntimeError("ManagedZmqContext was not closed")


class ZMQSender(Generic[S], Sender[S]):
    def __init__(
        self, client_id: str, socket_file_path: str, server_timeout: float = 600.0
    ):
        self.client_id = client_id
        self.socket_file_path = socket_file_path
        self.store_socket: Optional[Socket] = None
        self.old_addr = ""
        self.context = ManagedZmqContext()

        # for logging
        self.counter = ActionCounter(f"ZMQ_{client_id}", is_rate=True)
        self.output_flow = ActionCounter(f"ZMQ_OUTPUT_RATE_{client_id}", is_rate=True)

        # exponential backoff for send
        self.backoff = 1
        self.last_alive = 0
        self.server_timeout = server_timeout  # raise ServerDisconnected if no "alive" received before this timeout

    def rate_and_reset(self) -> float:
        return self.counter.rate_and_reset()

    def recreate_sock(self, store_addr: str):
        self.context.close()
        self.context = ManagedZmqContext()
        self.store_socket = self.context.socket(zmq.DEALER)
        assert self.store_socket is not None
        self.store_socket.setsockopt(zmq.RCVTIMEO, 1000)
        self.store_socket.identity = self.client_id.encode("utf-8")
        self.store_socket.connect(store_addr)
        self.old_addr = store_addr

    def maybe_update_sock(self):
        once = True
        while self.store_socket is None or once:
            if os.path.exists(self.socket_file_path):
                with open(self.socket_file_path, "r") as f:
                    new_addr = f.readline().strip()
                    if new_addr != self.old_addr:
                        self.recreate_sock(new_addr)
            else:
                logger.info(f"ZMQSender: Waiting for sock {self.socket_file_path}")
                time.sleep(5.0)
            once = False

    def wait_for_store(self):
        while True:
            try:
                self.maybe_update_sock()
                self.store_socket.send_pyobj(None)
                assert self.store_socket.recv_pyobj() == "alive"
                self.last_alive = time.time()
                return
            except zmq.error.Again:
                logger.info("ZMQSender: Waiting for server to be alive")
                time.sleep(5.0)

    def check_alive_or_die(self):
        """
        Consume all the input queue for 'alive' messages.
        Error if the last one is too old.
        """
        try:
            while True:
                assert self.store_socket.recv_pyobj(zmq.NOBLOCK) == "alive"
                self.last_alive = time.time()
        except zmq.error.Again:
            if time.time() - self.last_alive > self.server_timeout:
                # didn't receive any alive token
                raise ServerDisconnected(
                    f"ZMQSender: No alive token for "
                    f"{time.time() - self.last_alive:.3f}s"
                )

    def store(self, obj: S):
        assert self.store_socket is not None
        try:
            self.store_socket.send_pyobj(obj, zmq.NOBLOCK)
            self.check_alive_or_die()  # make sure the server isn't dead for real.
        except zmq.error.Again:
            logger.warning(
                f"ZMQSender: Send failed, generator too fast! "
                f"Waiting for {self.backoff:.3f}s"
            )
            time.sleep(self.backoff)
            # at most 1 min back-off seems plenty
            self.backoff = min(self.backoff * 2, 60)
            self.maybe_update_sock()
            return

        # If we can send, reset backoff to 1 sec
        self.backoff = 1
        self.counter.act()

        # output in kb/s
        # self.output_flow.act(len(obj) / 1000)

    def close(self):
        self.context.close()


class MultiWorkerZMQSender(Sender[S]):
    def __init__(
        self,
        client_id: str,
        socket_file_root: Path,
        socket_file_pattern: str,
        server_timeout: float = 600.0,
    ):
        self.client_id = client_id
        self.socket_file_root = socket_file_root
        self.socket_file_pattern = socket_file_pattern
        self.server_timeout = server_timeout
        self.known_senders: Set[str] = set()
        self.senders: List[ZMQSender] = []
        self.next_sender = 0
        self.last_refresh = 0

    def refresh_senders(self, force_refresh=False):
        if time.time() - self.last_refresh < 60 and not force_refresh:
            return
        try:
            for f in os.listdir(self.socket_file_root):
                if f not in self.known_senders and re.match(
                    self.socket_file_pattern, f
                ):
                    sender = ZMQSender(
                        self.client_id,
                        str(self.socket_file_root / f),
                        self.server_timeout,
                    )
                    sender.wait_for_store()
                    self.known_senders.add(f)
                    self.senders.append(sender)
                    logger.info(
                        f"MultiWorkerZMQSender: {self.client_id} has "
                        f"{len(self.senders)} senders."
                    )
            self.last_refresh = time.time()
        except FileNotFoundError:
            logger.info(
                f"MultiWorkerZMQSender: Waiting for folder "
                f"{self.socket_file_root} to exist"
            )

    def wait_for_store(self):
        while True:
            self.refresh_senders(force_refresh=True)
            if len(self.senders) > 0:
                break
            time.sleep(10)

    def store(self, obj: S) -> None:
        self.refresh_senders()
        self.senders[self.next_sender % len(self.senders)].store(obj)
        self.next_sender += 1

    def rate_and_reset(self) -> float:
        return sum(
            [sender.rate_and_reset() for sender in self.senders], 0
        )  # assumes all _rate_avg_time are equal...

    def close(self) -> None:
        for sender in self.senders:
            sender.close()


class ZMQReceiver(Generic[S], Receiver[S]):
    def __init__(
        self,
        dump_path: Path,
        global_rank: int,
        name: str = "",
        heartbeat_freq: float = 60.0,
    ):
        # TODO: should this be a deque to prioritize recent elements ?
        self.queue: Queue[Tuple[S, float]] = Queue()  # not a multiproc queue.

        # for zmq comms
        self.started: bool = False
        self.comm_thread: Optional[Thread] = None
        self.stop = Event()
        self.has_started = Event()
        self.zmq_addr = None
        self.heartbeat_freq = heartbeat_freq

        self.name = name
        self.dump_path = dump_path
        self.bad_data_path = dump_path / "bad_data"
        # store bad data for inspection
        os.makedirs(self.bad_data_path, exist_ok=True)
        self.global_rank = global_rank
        self.bad_recv = 0
        self.recv_counter = ActionCounter(name, is_rate=True, silent=True)
        self.recv_mb = ActionCounter(f"{name}_mb", is_rate=True, silent=True)

    def comms(self):
        # for logging / avoid cyclic dependencies when importing TrainerArgs
        assert self.zmq_addr is None
        context = ManagedZmqContext()
        with logged_closing(context, "comms_context"):
            sock = context.socket(zmq.ROUTER)
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            port = sock.bind_to_random_port(f"tcp://*")
            self.zmq_addr = f"tcp://{socket.gethostname()}:{port}"
            self.has_started.set()

            clients = set()
            last_heartbeat = time.time()
            while not self.stop.is_set():
                # send heartbeat if needed
                if time.time() - last_heartbeat > self.heartbeat_freq:
                    last_heartbeat = time.time()
                    for client_id in clients:
                        try:
                            sock.send(client_id, zmq.SNDMORE)
                            sock.send_pyobj("alive", zmq.NOBLOCK)
                        except zmq.error.Again:
                            # if the client died for some reason, we're ok with that.
                            clients.remove(client_id)

                try:
                    client_id, data = sock.recv_multipart()
                    size = float(len(data)) / 1024 ** 2
                    try:
                        data = pickle.loads(data)
                    except pickle.UnpicklingError:
                        fname = rstr() + ".bad"
                        logger.warning(
                            f"Bad data received in comms by {client_id} dumped in {self.bad_data_path / fname}"
                        )
                        with open(self.bad_data_path / fname, "rb") as f:
                            f.write(data)
                        self.bad_recv += 1
                        continue
                    if data is None:
                        clients.add(client_id)
                        sock.send(client_id, zmq.SNDMORE)
                        sock.send_pyobj("alive", zmq.NOBLOCK)
                        logger.info(f"ADDING CLIENT -- Client ID: {client_id}")
                        continue
                    self.queue.put((data, size))
                except zmq.error.Again:
                    continue

    @property
    def sockets_path(self) -> Path:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        return (
            self.dump_path
            / f"{self.name}_sockets"
            / f"{self.global_rank}_{worker_id}.addr"
        )

    def start_zmq_if_needed(self) -> bool:
        if self.comm_thread is not None:
            return False
        self.comm_thread = Thread(target=self.comms)
        self.comm_thread.start()
        # block until zmq_addr is defined
        while not self.has_started.is_set():
            time.sleep(1.0)

        logger.info(f"Bound rb socket to {self.zmq_addr}")
        os.makedirs(self.dump_path / f"{self.name}_sockets", exist_ok=True)
        # would be cleaner to file lock and
        dst_path = self.sockets_path
        # Here we don't append. Overwrite happens after pre-emption and an append wouldn't work.
        # Make sure that each zmq receiver is writing to a different file...
        with open(dst_path, "w+",) as port_file:
            port_file.write(f"{self.zmq_addr}\n")
        logger.info(f"Wrote socket infos in {dst_path}")
        return True

    def rate_and_reset(self) -> float:
        return self.recv_counter.rate_and_reset()

    def receive_batch(self) -> List[S]:
        if self.start_zmq_if_needed():
            raise ZMQNotReady()
        if self.comm_thread is None or not self.comm_thread.is_alive():
            raise RuntimeError("ZMQ store thread seems to be dead")
        seqs: List[S] = []
        while True:
            try:
                data, size = self.queue.get_nowait()
                seqs.append(data)
                self.recv_counter.act(1)
                self.recv_mb.act(size)
            except Empty:
                break
        return seqs

    def close(self) -> None:
        logger.info("Closing ZMQReceiver ...")
        if self.comm_thread:
            logger.info(f"Killing ZMQ store {self.name}")
            self.stop.set()
            self.comm_thread.join()
            logger.info("ZMQ store killed")
            self.comm_thread = None
        logger.info("Closed ZMQReceiver")
