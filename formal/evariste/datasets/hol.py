# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field

from params import Params, ConfStore


@dataclass
class HolLightDatasetConf(Params):
    checkpoint_path: str = field(
        default="", metadata={"help": "Path to checkpoint"},
    )
    timeout: float = field(
        default=5.0, metadata={"help": "OCaml timeout in s"},
    )
    n_envs: int = field(
        default=5, metadata={"help": "Number of OCaml processes"},
    )
    data_dir: str = field(
        default="",
        metadata={
            "help": "Path to dataset files (data.tok and split.{train|test|valid})"
        },
    )
    custom_dag: str = field(
        default="",
        metadata={
            "help": 'If non-empty, the DAG is directly loaded from f"{custom_dag}.json". '
            'Typically, custom_dag="order_from_repo"'
            "the DAG is made such that for a given OCaml object (e.g., a `thm`), "
            "its parents are all OCaml objects defined beforehand in the same file "
            "as well as all OCaml objects defined in files that need to be loaded"
        },
    )
    goals_with_num_steps: bool = field(
        default=False,
        metadata={
            "help": "Each goal includes the number of steps taken to prove it by a human"
        },
    )


def register_hol_datasets():

    ConfStore["hl_new1"] = HolLightDatasetConf(checkpoint_path="", data_dir="",)

    ConfStore["hl_new1_temp"] = HolLightDatasetConf(checkpoint_path="", data_dir="")

    ConfStore["hl_default_dataset"] = HolLightDatasetConf(
        checkpoint_path="", data_dir="", n_envs=10, timeout=5.0,
    )
    ConfStore["hl_plus_default_dataset"] = HolLightDatasetConf(
        checkpoint_path="", data_dir="", n_envs=10, timeout=5.0,
    )
