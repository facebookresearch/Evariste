# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import tempfile
from pathlib import Path

from evariste.backward.remote.state_dump_utils import StateDumper, _get_id


def test_state_dumper():
    with tempfile.TemporaryDirectory() as temp_dirname:
        folder = Path(temp_dirname)
        dumper = StateDumper(folder=folder, n_states_max=10, keep_first=True)

        inside = 0
        for _ in range(10):
            inside += 1
            dumper.maybe_dump_state(f"state {inside}")
            print("inside", inside)
            assert len(dumper.get_present()) == inside
            assert {_get_id(p) for p in dumper.get_present()} == set(range(inside))

        for _ in range(120):
            inside += 1
            dumper.maybe_dump_state(f"state {inside}")
            assert len(dumper.get_present()) <= 10
            prs_ids = [_get_id(p) for p in dumper.get_present()]
            assert prs_ids[0] == 0
            assert prs_ids[-1] == inside - 1

        print(prs_ids)
        assert {_get_id(p) for p in dumper.get_present()} == {
            0,
            16,
            32,
            48,
            64,
            80,
            96,
            112,
            128,
            129,  # last dumped
        }


def test_state_dumper_2():
    with tempfile.TemporaryDirectory() as temp_dirname:
        folder = Path(temp_dirname)
        dumper = StateDumper(folder=folder, n_states_max=10, keep_first=False)

        inside = 0
        for _ in range(1):
            inside += 1
            dumper.maybe_dump_state(f"state {inside}")
            assert {_get_id(p) for p in dumper.get_present()} == {0}

        for _ in range(10):
            inside += 1
            dumper.maybe_dump_state(f"state {inside}")
            assert {_get_id(p) for p in dumper.get_present()} == set(range(1, inside))

        for _ in range(100):
            inside += 1
            dumper.maybe_dump_state(f"state {inside}")
            assert len(dumper.get_present()) <= 10
            prs_ids = [_get_id(p) for p in dumper.get_present()]
            assert prs_ids[-1] == inside - 1
