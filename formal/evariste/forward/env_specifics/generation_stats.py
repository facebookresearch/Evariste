# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass


@dataclass
class GenerationStats:
    """
    Put here stats that are common between envs
    NOTE: `None` to new attributes to avoid pickling reloading issues
    """

    statement_n_tok: int
    last_node_proof_size: int
    last_node_depth: int
    n_forward_steps: int
    n_forward_errors: int
    last_node_size_by_depth: float
