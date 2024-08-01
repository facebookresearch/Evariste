# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path
import os
import time
import pytest
import tempfile

from evariste.comms.zmq import ZMQReceiver, ZMQSender, ServerDisconnected


def receive_until_timeout(s, timeout: float = 1.0):
    received = set()
    start = time.time()
    while time.time() - start < timeout:
        r = s.receive_batch()
        if not r:
            time.sleep(0.1)
        received.update(r)
    return received


def do_test_store(s, clients):
    s.start_zmq_if_needed()
    for i, c in enumerate(clients):
        c.wait_for_store()
        for seq_num in range(3):
            c.store((i, seq_num))
    received = receive_until_timeout(s, timeout=0.2)
    expected = set(((i, seq_num) for i in range(len(clients)) for seq_num in range(3)))
    assert received == expected


class TestZMQ:
    def test_zmq(self):
        with tempfile.TemporaryDirectory() as temp_dirname:
            dump_path = Path(temp_dirname)
            s = ZMQReceiver(
                dump_path=dump_path, global_rank=0, name="test", heartbeat_freq=0.01,
            )

            clients = [
                ZMQSender(f"{i}", s.sockets_path, server_timeout=0.1) for i in range(5)
            ]
            do_test_store(s, clients)
            s.close()
            with pytest.raises(ServerDisconnected):
                clients[0].store("ok")
                time.sleep(0.1)
                clients[0].store(
                    "lol"
                )  # woops, haven't received heartbeats in a while !

            os.unlink(s.sockets_path)
            s2 = ZMQReceiver(
                dump_path=dump_path, global_rank=0, name="test", heartbeat_freq=0.01
            )
            do_test_store(s2, clients)
            s2.close()
            for c in clients:
                c.close()
