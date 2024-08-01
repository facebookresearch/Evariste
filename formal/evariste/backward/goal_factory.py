# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, List
import os
import re
import numpy as np
from params import ConfStore

from evariste.model.data.envs.equations import EquationsEnvironment
from evariste.datasets.isabelle import IsabelleDatasetConf
from evariste.backward.env.core import BackwardGoal
from evariste.datasets import (
    DatasetConf,
    MetamathDatasetConf,
    EquationsDatasetConf,
    HolLightDatasetConf,
    LeanDatasetConf,
    SRDatasetConf,
)


def get_labels(
    dataset: DatasetConf,
    split: str,
    n_to_prove: Optional[int] = None,
    sharding: Optional[Tuple[int, int]] = None,
    shuffle_seed: int = 43,
) -> List[str]:
    """
    Load labels.
    """

    # Equations
    if isinstance(dataset, EquationsDatasetConf):
        if split in {"eq_bwd_rwalk_seq2seq", "eq_bwd_graph_seq2seq"}:
            assert n_to_prove is not None
            labels = [split] * n_to_prove  # TODO: fix?
        else:
            assert split == "identities", split
            labels = list(EquationsEnvironment.get_identities(dataset).keys())
    # Symbolic Regression
    elif isinstance(dataset, SRDatasetConf):
        assert n_to_prove is not None
        labels = [f"{split}_SAMPLE_{i}" for i in range(n_to_prove)]
    # HOL-Light
    elif isinstance(dataset, HolLightDatasetConf) and split.startswith("miniF2F"):
        from evariste.benchmark.miniF2F.hl_miniF2F import get_miniF2F_goals_from_repo

        if split == "miniF2F":
            goals = get_miniF2F_goals_from_repo(proving_mode="bwd", splitted=False)
        else:
            assert split in {"miniF2F-valid", "miniF2F-test"}
            goals_all_split = get_miniF2F_goals_from_repo(
                proving_mode="bwd", splitted=True
            )
            goals = goals_all_split[split[len("miniF2F-") :]]  # type: ignore  # type is a (false) union
        labels = [goal.name for goal in goals]  # type: ignore  # same thing. requires fixing get_minif2f_goals
    elif isinstance(dataset, LeanDatasetConf) and not dataset.is_old:
        labels = [x[1] for x in dataset.get_labels_and_splits()[split]]
    # Lean / Metamath
    else:
        assert isinstance(
            dataset, (LeanDatasetConf, MetamathDatasetConf, IsabelleDatasetConf)
        ), dataset
        # filter IMO problems (Lean only)
        if split == "imo":
            assert isinstance(dataset, LeanDatasetConf)
            labels = []
            for split_ in ["minif2f_valid", "minif2f_test"]:
                new_labels = get_labels(
                    dataset=dataset,
                    split=split_,
                    n_to_prove=None,
                    sharding=sharding,
                    shuffle_seed=0,
                )
                assert set(new_labels).isdisjoint(set(labels))
                labels.extend(new_labels)
            init_labels = len(labels)
            labels = [x for x in labels if "imo" in x]
            labels = sorted(labels)
            print(f"Found {len(labels)}/{init_labels} IMO problems.")

        # filter IMO problems in annotations (Lean only)
        else:
            m = re.match(r"annotations_v(?P<version>\d+)-imo", split)
            if m:
                assert isinstance(dataset, LeanDatasetConf)
                labels = get_labels(
                    dataset=dataset,
                    split=f"annotations_v{m.group('version')}",
                    n_to_prove=None,
                    sharding=sharding,
                    shuffle_seed=0,
                )
                init_labels = len(labels)
                labels = [x for x in labels if "IMO" in x]
                labels = sorted(labels)
                print(f"Found {len(labels)}/{init_labels} IMO problems.")

            else:
                with open(os.path.join(dataset.data_dir, f"split.{split}"), "r") as f:
                    labels = [line.strip() for line in f]

    # shuffle labels (always have a seed to ensure workers process different labels)
    assert shuffle_seed is not None
    rng = np.random.RandomState(shuffle_seed)
    rng.shuffle(labels)

    # optionally create shards
    all_labels = labels[:n_to_prove]
    if sharding is None:
        labels = all_labels
    else:
        shard_id, n_shards = sharding
        labels = all_labels[shard_id::n_shards]

    return labels


def get_goals_to_prove(
    dataset: DatasetConf,
    split: str,
    n_to_prove: Optional[int],
    sharding: Optional[Tuple[int, int]] = None,
    shuffle_seed: int = 43,
    labels: Optional[List[str]] = None,
) -> List[BackwardGoal]:
    print(f"===== Getting goals to prove for {split} set =====")

    if labels is None:
        labels = get_labels(
            dataset=dataset,
            split=split,
            n_to_prove=n_to_prove,
            sharding=sharding,
            shuffle_seed=shuffle_seed,
        )

    assert len(labels) > 0, "found no label"
    return [
        BackwardGoal.create_unmat(label, split=split) for label in labels[:n_to_prove]
    ]
