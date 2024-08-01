# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from params import Params, ConfStore
from dataclasses import dataclass, field
from evariste.clusters.utils import clusterify_path


@dataclass
class LatexArgs(Params):
    data_dir: str = field(
        default="", metadata={"help": "Informal data directory"},
    )

    def __post_init__(self):
        self.data_dir = clusterify_path(self.data_dir)


ConfStore["latex_arxiv"] = LatexArgs(data_dir="YOUR_PATH/bpe")
ConfStore["latex_python"] = LatexArgs(data_dir="YOUR_PATH/bpe")
ConfStore["default_latex"] = LatexArgs(data_dir="")
