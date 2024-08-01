# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List

import pytest

from evariste.forward.common import GenerationHistory
from evariste.forward.env_specifics.generation_annotator import NodeSelectionConfig
from evariste.forward.fwd_eq.conftest import trainer_args
from evariste.forward.fwd_eq.eq_generation_annotator import EQGenerationAnnotator


@pytest.mark.parametrize("select_cfg", [NodeSelectionConfig()])
def test_eq_generation_annotator(
    eq_histories: List[GenerationHistory], select_cfg: NodeSelectionConfig
):
    annotator = EQGenerationAnnotator(params=trainer_args())
    for history in eq_histories:
        goals = annotator.annotate_and_select_goals(
            history=history, select_cfg=select_cfg
        )
        assert len(goals) == select_cfg.n_send_to_provers
