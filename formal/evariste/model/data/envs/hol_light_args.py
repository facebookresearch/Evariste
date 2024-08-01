# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from params import Params, ConfStore
from dataclasses import dataclass

from evariste.datasets import HolLightDatasetConf


@dataclass
class HOLLightArgs(Params):
    dataset: HolLightDatasetConf
    # list_split_path: str
    # list_subset_path: str
    # remove_cmds_path: str


ConfStore["hl_plus_default_args"] = HOLLightArgs(
    dataset=ConfStore["hl_plus_default_dataset"]
)
