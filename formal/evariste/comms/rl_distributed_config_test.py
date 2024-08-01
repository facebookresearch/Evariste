# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

def test_distributed_setup_cfgs():
    from evariste.comms.rl_distributed_config import (
        DistributedSetup,
        _SETUP_TO_CFG,
    )

    for name in list(DistributedSetup):
        # check that it doesn't crash
        assert name in _SETUP_TO_CFG, name
