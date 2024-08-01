# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from enum import Enum


class SamplingMethod(str, Enum):
    FULL = "full"


class FilterKind(str, Enum):
    All = "all"
