# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from typing import Generic, TypeVar
import time
from logging import getLogger
from pathlib import Path
from typing import List, Set, Optional

from evariste.metrics import ActionCounter
from evariste.comms.zip_store import ChunkedZipStore, ZipStore
from evariste.comms.store import (
    Sender,
    Receiver,
)

logger = getLogger(__name__)

S = TypeVar("S")


class ZipSender(Generic[S], Sender[S]):
    def __init__(self, store_path: Path, zip_size: int):
        self.zip_store = ChunkedZipStore(store_path)
        self.zip_size = zip_size
        self.cur_chunk_id = 0
        self.cur_seq_id = 0
        self.reload()
        self.zip_store.start_chunk(self.cur_chunk_id)

        self.counter = ActionCounter(self.sender_uuid, is_rate=True)

    @property
    def sender_uuid(self) -> str:
        return self.zip_store.root_path.name

    def rate_and_reset(self) -> float:
        return self.counter.rate_and_reset()

    def reload(self):
        ready_chunks = self.zip_store.ready_chunks()
        if ready_chunks:
            logger.info(
                f"[Sender {self.sender_uuid}] Detected {len(ready_chunks)} chunks"
            )
            self.cur_chunk_id = max(ready_chunks) + 1

    def store(self, seq: S) -> None:
        assert self.cur_seq_id < self.zip_size
        self.zip_store.store_in_pickle_zip(seq, zip_name="sequences")
        self.cur_seq_id += 1
        if self.cur_seq_id >= self.zip_size:
            self.zip_store.finish_chunk()
            self.cur_chunk_id += 1
            self.zip_store.start_chunk(self.cur_chunk_id)
            self.cur_seq_id = 0
        self.counter.act()

    def close(self):
        self.zip_store.close()

    def __del__(self):
        self.close()

    @classmethod
    def from_dump_path(
        cls,
        zip_size: int,
        dump_path: Path,
        store_name: str,
        rank: int,
        worker_id: int = 0,
    ) -> "ZipSender":
        root_path = dump_path / f"{store_name}_{rank}_{worker_id}"
        logger.info(f"Starting a sender for {store_name} at {root_path}")
        root_path.mkdir(parents=True, exist_ok=True)
        return cls(store_path=root_path, zip_size=zip_size)


class ZipReceiver(Generic[S], Receiver[S]):
    def __init__(
        self, store_path: Path, receiver_uuid: str = "sequence_receiver",
    ):
        self.store_path = store_path
        self.receiver_uuid = receiver_uuid
        self.zip_store: Optional[ChunkedZipStore] = None
        self.loaded_chunks: Set = set()
        self.recv_counter = ActionCounter(receiver_uuid, is_rate=True)

    def rate_and_reset(self) -> float:
        return self.recv_counter.rate_and_reset()

    def init_store(self):
        # TODO bind here with worker_id ?
        assert self.zip_store is None
        duration = 1

        while not self.store_path.exists():
            logger.info(
                f"(PID: {os.getpid()}) Waiting for {self.store_path} to exist. Wait {duration}s"
            )
            time.sleep(duration)
            duration *= 2

        self.zip_store = ChunkedZipStore(self.store_path)
        # ignoring present zips
        self.loaded_chunks = set(self.zip_store.ready_chunks())
        logger.info(
            f"(PID: {os.getpid()}) - {self.__class__.__name__}: "
            f"Ignoring {len(self.loaded_chunks)} already present chunks"
        )

    def reload_last_chunks(self, n_sequences: int) -> List[S]:
        if self.zip_store is None:
            self.init_store()
        assert self.zip_store is not None
        reloaded_sequences = []
        reloaded = []
        for chunk_id in reversed(sorted(self.loaded_chunks)):
            reloaded.append(chunk_id)
            chunk: ZipStore = self.zip_store.get_chunk(chunk_id)
            sequences: List[S] = chunk.read_pickle_zip("sequences")
            reloaded_sequences.extend(sequences)
            if len(reloaded_sequences) > n_sequences:
                reloaded_sequences = reloaded_sequences[:n_sequences]
                break
        logger.info(
            f"(PID: {os.getpid()}) - {self.__class__.__name__}: "
            f"reloaded {len(reloaded_sequences)} sequences "
            f"with chunks {reloaded}"
        )
        return list(reversed(reloaded_sequences))

    def receive_batch(self) -> List[S]:
        if self.zip_store is None:
            self.init_store()
        assert self.zip_store is not None
        loaded = []
        loaded_sequences = []
        for chunk_id in sorted(self.zip_store.ready_chunks()):
            if chunk_id not in self.loaded_chunks:
                loaded.append(chunk_id)
                self.loaded_chunks.add(chunk_id)
                chunk: ZipStore = self.zip_store.get_chunk(chunk_id)
                sequences: List[S] = chunk.read_pickle_zip("sequences")
                loaded_sequences.extend(sequences)
                for i in range(len(sequences)):
                    self.recv_counter.act()
        if loaded:
            logger.info(
                f"(PID: {os.getpid()}) - {self.__class__.__name__}: received data"
                f" loaded {len(loaded)} new chunks, "
                f" with {len(loaded_sequences)} sequences"
            )
        return loaded_sequences

    def close(self):
        if self.zip_store:
            self.zip_store.close()

    def __del__(self):
        self.close()

    @classmethod
    def from_dump_path(
        cls,
        dump_path: Path,
        rank: int,
        name: str,
        worker_id: int,
        src_rank: int,
        src_name: str,
        src_worker_id: int,
        wait_for_store: bool = True,
    ) -> "ZipReceiver":
        root_path = dump_path / f"{src_name}_{src_rank}_{src_worker_id}"
        logger.info(f"Starting a receiver for {name} at {root_path}")

        store = cls(store_path=root_path, receiver_uuid=f"{name}_{rank}_{worker_id}",)

        if wait_for_store:
            store.init_store()

        return cls(store_path=root_path, receiver_uuid=f"{name}_{rank}_{worker_id}",)


def get_other_ranks(rank: int, world_size: int, other_world_size: int) -> List[int]:
    # TODO: find a better name
    smaller_world = min(world_size, other_world_size)
    return [
        r for r in range(other_world_size) if r % smaller_world == rank % smaller_world
    ]
