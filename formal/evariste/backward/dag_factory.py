# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from evariste.datasets import (
    DatasetConf,
    MetamathDatasetConf,
    EquationsDatasetConf,
    HolLightDatasetConf,
    LeanDatasetConf,
    SRDatasetConf,
)
from evariste.envs.hl.utils import get_dag_and_th
from evariste.envs.mm.env import MetamathEnv


def get_dag(dataset: DatasetConf):
    if isinstance(dataset, MetamathDatasetConf):
        if "minif2f" in dataset.database_path:
            return None
        mm_env = MetamathEnv(
            filepath=dataset.database_path,
            rename_e_hyps=True,
            decompress_proofs=False,
            verify_proofs=False,
            log_level="info",
        )
        mm_env.process()
        return mm_env.parse_label_dag()
    elif isinstance(dataset, EquationsDatasetConf):
        return None
    elif isinstance(dataset, SRDatasetConf):
        return None
    elif isinstance(dataset, HolLightDatasetConf):
        return get_dag_and_th(dataset.data_dir, custom_dag=dataset.custom_dag)[0]
    elif isinstance(dataset, LeanDatasetConf):
        return None
    else:
        raise RuntimeError("Not implemented")
