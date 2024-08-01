# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from enum import Enum, unique


@unique
class BeamSearchKind(str, Enum):
    Manual = "manual"
    AutomaticallyReloading = "automatically_reloading"
    IncreasingQuality = "increasing_quality"
    Fixed = "fixed"
