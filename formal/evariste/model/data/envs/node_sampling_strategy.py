# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from enum import Enum, unique


@unique
class NodeSamplingStrategy(str, Enum):
    AllFull = "all_full"  # all proof steps, including all ancestor nodes in each step
    AllMinimal = "all_minimal"  # all proof steps, including only the necessary ancestor nodes in each step
    SamplingMinimal = "sampling_minimal"  # sampling according to sampler_params, including only the necessary ancestor nodes in each step
    # SamplingFull = "sampling_full"    # not supported
