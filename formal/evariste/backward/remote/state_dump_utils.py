# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pickle
from logging import getLogger
from pathlib import Path
from typing import Any, List

logger = getLogger()


class StateDumper:
    """
  Make sure that there are at most n_states_max in the folder.
  If n_states_max is reached, we double 'keep_every' and keep only state every
   'keep_every'.
   keep_first we keep the state 0 if not we remove it.
  """

    def __init__(
        self,
        folder: Path,
        n_states_max: int,
        keep_first: bool = False,
        is_master: bool = True,
    ):
        self.folder = folder
        self.is_master = is_master
        self.n_states_max = n_states_max
        self.keep_every = 1
        self.keep_first = keep_first

        if self.is_master:
            self.folder.mkdir(exist_ok=True)
            present = self.get_present()

            self.n = 0 if not present else _get_id(present[-1])
        else:
            self.n = -1

    def maybe_dump_state(self, state: Any):
        if not self.is_master:
            return

        # removing old states
        present = self.get_present()
        to_keep = [p for p in present if self._keep(_get_id(p))]
        if len(to_keep) >= self.n_states_max:
            self.keep_every *= 2
        to_remove = [p for p in present if not self._keep(_get_id(p))]
        logger.info(
            f"State dumper for folder: {self.folder} - "
            f"Going to remove {len(to_remove)} states"
        )
        for p in to_remove:
            p.unlink()

        # dumping latest
        _safe_dump(state, self.folder / f"state.{self.n}")

        self.n += 1

    def _keep(self, idx: int) -> bool:
        if idx == 0:
            return self.keep_first
        return idx % self.keep_every == 0

    def get_present(self) -> List[Path]:
        return sorted(
            (p for p in self.folder.iterdir() if p.name.startswith("state.")),
            key=lambda p: _get_id(p),
        )


def _get_id(state: Path) -> int:
    assert state.name.startswith("state.")
    return int(state.name[len("state.") :])


def _safe_dump(state: Any, path: Path):
    assert not path.exists()
    tmp_path = path.parent / f"tmp.{path.name}"
    with tmp_path.open("wb") as fp:
        pickle.dump(state, fp)
    tmp_path.rename(path)
