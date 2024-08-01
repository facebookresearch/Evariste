# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field
import os

from params import Params, ConfStore
from evariste.clusters.utils import clusterify_path


@dataclass
class MetamathDatasetConf(Params):
    database_path: str = field(metadata={"help": "Path to set.mm"},)
    data_dir: str = field(
        metadata={
            "help": "Path to dataset files (proof_trees.pkl and split.{train|test|valid})"
        },
    )

    parser: str = field(metadata={"help": "Name or path to grammar.in"},)

    def __post_init__(self):
        self.database_path = clusterify_path(self.database_path)
        self.data_dir = clusterify_path(self.data_dir)

    def _check_and_mutate_args(self):
        self.__post_init__()  # redo here to avoid  not clusterifying on reload before checking if path is ok
        assert os.path.isfile(self.database_path), self.database_path
        assert os.path.isdir(self.data_dir), self.data_dir


NEW3_MINIF2F = MetamathDatasetConf(database_path="", data_dir="", parser="new3",)


def register_metamath_datasets():
    root = ""
    data_dirs = {
        "holophrasm": "DATASET_HOLOPHRASM/",
        "new3": "DATASET_3",
        "new2": "DATASET_2",
    }

    for dataset in data_dirs.keys():
        ConfStore[f"{dataset}"] = MetamathDatasetConf(
            database_path=os.path.join(root, data_dirs[dataset], "set.mm"),
            data_dir=os.path.join(root, data_dirs[dataset]),
            parser=dataset,
        )
        ConfStore[f"{dataset}_100"] = MetamathDatasetConf(
            database_path=os.path.join(root, data_dirs[dataset], "set.mm"),
            data_dir=os.path.join(root, data_dirs[dataset], "100"),
            parser=dataset,
        )

    ConfStore["syntax_gen_100k"] = MetamathDatasetConf(
        database_path="", data_dir="", parser=None,
    )

    # About 660k theorems
    ConfStore["inequal1"] = MetamathDatasetConf(
        database_path="", data_dir="", parser="inequalities",
    )
    ConfStore["new3_minif2f"] = NEW3_MINIF2F
