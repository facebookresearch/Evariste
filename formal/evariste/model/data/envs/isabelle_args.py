# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass

from params import Params, ConfStore
from evariste.datasets import IsabelleDatasetConf


@dataclass
class IsabelleArgs(Params):
    dataset: IsabelleDatasetConf


ConfStore["default_isabelle"] = IsabelleArgs(
    dataset=ConfStore["isabelle_dataset_default"]
)
