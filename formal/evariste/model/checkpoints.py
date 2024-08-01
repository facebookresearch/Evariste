# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple, Optional
from logging import getLogger
import os
import re


logger = getLogger()


def get_latest_checkpoint(
    dump_dir, should_exist: bool = True
) -> Tuple[Optional[str], int]:
    latest_path, latest_epoch = None, -2
    if not os.path.exists(dump_dir) and not should_exist:
        logger.info(f"Checkpoint dir {dump_dir} doesn't exist")
        return latest_path, latest_epoch
    files = os.listdir(dump_dir)

    for f in files:
        if re.match(r"^checkpoint\.[0-9-]+\.pth$", f):
            epoch = int(f.split(".")[1])  # checkpoint.{epoch}.pth
            if epoch > latest_epoch:
                latest_path, latest_epoch = os.path.join(dump_dir, f), epoch
    return latest_path, latest_epoch


def keep_latest_checkpoints(dump_dir, to_keep: int = 5):
    files = os.listdir(dump_dir)
    epoch_to_path = {}
    for f in files:
        if re.match(r"^checkpoint\.[0-9-]+\.pth$", f):
            epoch = int(f.split(".")[1])  # checkpoint.{epoch}.pth
            epoch_to_path[epoch] = os.path.join(dump_dir, f)
    if len(epoch_to_path) > to_keep:
        to_delete = sorted(epoch_to_path.keys())[:-to_keep]
        for e in to_delete:
            os.unlink(epoch_to_path[e])
