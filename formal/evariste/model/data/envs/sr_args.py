# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass

from params import Params, ConfStore
from evariste.datasets import SRDatasetConf


@dataclass
class SRArgs(Params):
    dataset: SRDatasetConf


ConfStore["default_sr"] = SRArgs(dataset=ConfStore["sr_dataset_default"])
