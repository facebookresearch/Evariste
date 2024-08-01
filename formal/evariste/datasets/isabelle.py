# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field

from matplotlib.pyplot import step

from params import Params, ConfStore


@dataclass
class IsabelleDatasetConf(Params):
    data_dir: str = field(
        default="/datasets/isabelle_data",
        metadata={
            "help": "Path to dataset files test_theorem_names.json and {train|valid|test}.jsonl"
        },
    )
    step_timeout_in_seconds: float = field(
        default=0.2, metadata={"help": "Timeout for each step in seconds"},
    )
    sledgehammer_timeout_in_seconds: float = field(
        default=35.0, metadata={"help": "Timeout for sledgehammer in seconds"},
    )

    def __post_init__(self):
        pass


def register_isabelle_datasets():
    ConfStore["isabelle_dataset_default"] = IsabelleDatasetConf()
