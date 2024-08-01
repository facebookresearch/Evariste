# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import tempfile
from contextlib import closing
from pathlib import Path

from evariste.comms.zip import ZipSender, ZipReceiver


def test_send_and_read():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        sender = ZipSender[int](store_path=tmp_dir, zip_size=10)
        receiver = ZipReceiver[int](store_path=tmp_dir)
        with closing(sender), closing(receiver):
            received = receiver.receive_batch()
            assert len(received) == 0

            for i in range(15):
                sender.store(i)

            received = receiver.receive_batch()
            assert len(received) == 10
            for j in range(10):
                assert received[j] == j
            for i in range(15, 30):
                sender.store(i)
            received = receiver.receive_batch()
            assert len(received) == 20
            for j in range(20):
                assert received[j] == j + 10


def test_reload():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        sender = ZipSender(store_path=tmp_dir, zip_size=10)
        with closing(sender):
            for i in range(21):
                sender.store(i)
        assert sender.cur_chunk_id == 2
        del sender
        sender = ZipSender(store_path=tmp_dir, zip_size=10)
        assert sender.cur_chunk_id == 2
        del sender


def test_receive_reload_last_chunks():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        sender = ZipSender(store_path=tmp_dir, zip_size=5)
        with closing(sender):
            for i in range(21):
                sender.store(i)
        assert sender.cur_chunk_id == 4
        del sender

        receiver = ZipReceiver(store_path=tmp_dir)

        seqs = receiver.receive_batch()
        assert len(seqs) == 0  # chunk already exist when receiver is created

        receiver = ZipReceiver(store_path=tmp_dir)
        seqs = receiver.reload_last_chunks(n_sequences=10)
        assert len(seqs) == 10
        assert set(seqs) == set(range(10, 20))

        receiver = ZipReceiver(store_path=tmp_dir)
        seqs = receiver.reload_last_chunks(n_sequences=0)
        assert len(seqs) == 0
