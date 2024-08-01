# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field

from params import Params, ConfStore
from evariste.envs.eq.env import EquationsEnvArgs
from evariste.envs.sr.env import SREnvArgs


@dataclass
class SRDatasetConf(Params):

    sr_env: SREnvArgs
    # rule_types_str: str  # TODO: implement

    min_init_ops: int = 1
    max_init_ops: int = 10
    nan_rejection_proportion: float = 0.5
    simplify_equations: bool = True

    n_async_envs: int = field(
        default=8, metadata={"help": "Number of async SR envs (0 for sync)"}
    )

    def __post_init__(self):
        pass


def register_sr_datasets():
    ConfStore["sr_eq_env_default"] = EquationsEnvArgs(
        unary_ops_str="neg,inv,ln,pow2,sqrt,abs,cos,sin,tan",
        binary_ops_str="add,sub,mul,div",
        leaf_probs_str="0.2,0.8,0.0",
        n_vars=1,
    )
    ConfStore["sr_env_default"] = SREnvArgs(
        eq_env=ConfStore["sr_eq_env_default"],
        until_depth=1,
        min_n_points=20,
        max_n_points=20,
        # max_backward_steps
        # rule_criterion
        # sampling_strategy
    )

    ConfStore["sr_polynomial_eq_env_default"] = EquationsEnvArgs(
        unary_ops_str="neg,pow2", binary_ops_str="add,sub,mul", n_vars=1
    )
    ConfStore["sr_polynomial_env_default"] = SREnvArgs(
        eq_env=ConfStore["sr_polynomial_eq_env_default"],
        until_depth=1,
        min_n_points=20,
        max_n_points=20,
    )

    ConfStore["sr_int_eq_env"] = EquationsEnvArgs(
        vtype="int", unary_ops_str="neg,abs", binary_ops_str="add,sub,mul,%", n_vars=1
    )

    ConfStore["sr_int_env_default"] = SREnvArgs(
        eq_env=ConfStore["sr_int_eq_env"],
        until_depth=1,
        min_n_points=20,
        max_n_points=20,
    )

    ConfStore["sr_dataset_default"] = SRDatasetConf(sr_env=ConfStore["sr_env_default"])
    ConfStore["sr_int_dataset_default"] = SRDatasetConf(
        sr_env=ConfStore["sr_int_env_default"]
    )
    ConfStore["sr_polynomial_dataset_default"] = SRDatasetConf(
        sr_env=ConfStore["sr_polynomial_env_default"], min_init_ops=8, max_init_ops=10
    )
