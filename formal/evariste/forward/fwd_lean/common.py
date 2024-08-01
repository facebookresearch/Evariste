# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass

from evariste.backward.env.lean.graph import LeanTheorem, LeanTactic
from evariste.forward.common import EnvInfo, GenericForwardTactic


# todo: delete this class and create a simple type instead
@dataclass
class LeanForwardTactic(GenericForwardTactic[LeanTheorem, LeanTactic]):
    pass


class LeanEnvInfo(EnvInfo):
    pass
