# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from concurrent.futures.thread import ThreadPoolExecutor
from logging import getLogger
from tempfile import TemporaryDirectory

import pytest

from evariste.trainer.utils import nfs_barrier

logger = getLogger()


def test_nfs_barrier_raise():
    with TemporaryDirectory() as tmp:
        with pytest.raises(TimeoutError):
            nfs_barrier(dump_path=tmp, rank=0, world_size=2, timeout_s=0.001)


def test_nfs_barrier():
    timeout = 1.0  # hoping that's enough

    def _nfs_barrier(a_tuple):
        return nfs_barrier(*a_tuple)

    world_size = 4
    with TemporaryDirectory() as tmp:
        with ThreadPoolExecutor(max_workers=world_size) as executor:
            arguments = [(tmp, rank, world_size, timeout) for rank in range(world_size)]
            results = executor.map(_nfs_barrier, arguments)
        for _ in results:
            pass
