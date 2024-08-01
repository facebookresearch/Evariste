# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from typing import Set, Iterator
import os

from evariste.clusters.utils import clusterify_path


@dataclass
class ZOOModel:
    path: str
    dataset: str
    lang: str

    def __post_init__(self):
        self.path = clusterify_path(self.path)


@dataclass
class ZOO:
    @staticmethod
    def get_model(name: str) -> ZOOModel:
        model = getattr(ZOO, name, None)
        assert isinstance(model, ZOOModel)
        return model

    @staticmethod
    def all_zoo_models() -> Iterator[ZOOModel]:
        for x in sorted(list(ZOO.__dict__.keys())):
            if x.startswith("_") or x in {"get_model", "is_in_zoo", "all_zoo_models"}:
                continue
            zoo_model = getattr(ZOO, x)
            assert isinstance(zoo_model, ZOOModel), x
            yield zoo_model

    @staticmethod
    def is_in_zoo(path: str) -> bool:
        zoo_paths = {clusterify_path(model.path) for model in ZOO.all_zoo_models()}
        return clusterify_path(path) in zoo_paths


ZOO_MODELS: Set[str] = set()
for k, v in ZOO.__dict__.items():
    if isinstance(v, ZOOModel):
        if v.path in ZOO_MODELS:
            raise RuntimeError(f"Model with path {v.path} is defined multiple times.")
        ZOO_MODELS.add(v.path)


if __name__ == "__main__":
    not_found = []
    for model_path in ZOO_MODELS:
        if not os.path.isfile(model_path):
            not_found.append(model_path)
    if len(not_found) > 0:
        print(f"{len(not_found)} models not found:")
        for model_path in sorted(not_found):
            print(f"\t{model_path}")
