# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass

from evariste.forward.common import GenerationHistory, GenerationInfos


@dataclass
class Generation:
    """
    Defined here because of weird error with submitit pickling if defined in same
    script than __main__
    """

    history: GenerationHistory
    infos: GenerationInfos
