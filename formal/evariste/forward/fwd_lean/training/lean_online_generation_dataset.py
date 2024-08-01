# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from evariste.forward.fwd_lean.training.common import LeanProofNode
from evariste.forward.online_generation.online_generation_dataset import (
    OnlineGenerationDataset,
)


class LeanOnlineGenerationDataset(OnlineGenerationDataset[LeanProofNode]):
    """Using env agnostic OnlineGenerationDataset implementation"""

    pass
