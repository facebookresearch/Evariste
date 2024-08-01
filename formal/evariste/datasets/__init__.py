# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Union

from evariste.datasets.metamath import MetamathDatasetConf, register_metamath_datasets
from evariste.datasets.hol import HolLightDatasetConf, register_hol_datasets
from evariste.datasets.lean import LeanDatasetConf, register_lean_datasets
from evariste.datasets.equations import (
    EquationsDatasetConf,
    register_equations_datasets,
)
from evariste.datasets.isabelle import IsabelleDatasetConf, register_isabelle_datasets
from evariste.datasets.sr import SRDatasetConf, register_sr_datasets


register_metamath_datasets()
register_hol_datasets()
register_lean_datasets()
register_equations_datasets()
register_isabelle_datasets()
register_sr_datasets()

DatasetConf = Union[
    MetamathDatasetConf,
    HolLightDatasetConf,
    LeanDatasetConf,
    EquationsDatasetConf,
    IsabelleDatasetConf,
    SRDatasetConf,
]


def lang_from_dataset(dataset: DatasetConf) -> str:
    assert dataset is not None
    if isinstance(dataset, MetamathDatasetConf):
        return "mm"
    elif isinstance(dataset, HolLightDatasetConf):
        return "hl"
    elif isinstance(dataset, LeanDatasetConf):
        return "lean"
    elif isinstance(dataset, EquationsDatasetConf):
        return "eq"
    elif isinstance(dataset, IsabelleDatasetConf):
        return "isabelle"
    elif isinstance(dataset, SRDatasetConf):
        return "sr"
    else:
        raise RuntimeError(f"Unknown dataset configuration: {dataset}")
