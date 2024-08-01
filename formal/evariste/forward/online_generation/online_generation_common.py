# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from typing import Dict

from evariste.forward.common import GenerationHistory


@dataclass
class FwdTrajectory:
    history: GenerationHistory
    metadata: Dict