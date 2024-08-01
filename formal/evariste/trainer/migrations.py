# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Dict, Any
from functools import partial
from evariste import json as json

from params.params import flatten_dict
from params.migrate import FlatDict, migrations, migrate, warn_migration
from evariste.trainer.args import TrainerArgs, ModelArgs
from evariste.backward.prover.args import MCTSParams


def rename_prefix(flat_dict: FlatDict, old: str, new: str) -> FlatDict:
    """
    old and new are prefixes (without the dot)
    """
    warn_migration(f"Changing {old} to {new} prefixes in flatdict")
    new_dict = dict(flat_dict)  # copy
    old_prefix = f"{old}."
    new_prefix = f"{new}."
    for key, value in flat_dict.items():
        if key.startswith(old_prefix):
            new_dict.pop(key)
            new_key = new_prefix + key[len(old_prefix) :]
            new_dict[new_key] = value
    return new_dict


def add_hol_dataset(flat_dict: FlatDict) -> FlatDict:
    new_dict = dict(flat_dict)  # copy
    if (
        "hl.dataset.checkpoint_path" not in new_dict
        or new_dict["hl.dataset.checkpoint_path"] == ""
    ):
        warn_migration("Adding default hol dataset to config")
        to_copy = {
            "hl.dataset.checkpoint_path": "resources/hol-light/checkpoint-hl-multivariate-complex",
            "hl.dataset.data_dir": "resources/hol-light/DATASET_last",
            "hl.dataset.cwd": "resources/hol-light/",
        }
        for x, y in to_copy.items():
            new_dict[x] = y
    return new_dict


migrations[TrainerArgs].append(add_hol_dataset)


def add_lean_dataset(flat_dict: FlatDict) -> FlatDict:
    new_dict = dict(flat_dict)  # copy
    if (
        "lean.dataset.checkpoint_path" not in new_dict
        or not new_dict["lean.dataset.checkpoint_path"]
    ):
        warn_migration("Adding default lean dataset to config")
        to_copy = {
            "lean.dataset.checkpoint_path": "",
            "lean.dataset.tokenizer": "bpe_arxiv_utf8",
            "lean.dataset.data_dir": "",
        }
        for x, y in to_copy.items():
            new_dict[x] = y
    return new_dict


migrations[TrainerArgs].append(add_lean_dataset)
migrations[TrainerArgs].append(
    partial(rename_prefix, old="decoding", new="decoding_params")
)
migrations[TrainerArgs].append(
    partial(rename_prefix, old="lean.past_config", new="lean.dataset")
)


migrations[MCTSParams].append(
    partial(rename_prefix, old="batch_size", new="succ_expansions")
)


def to_int(flat_dict: FlatDict) -> FlatDict:
    new_dict = dict(flat_dict)  # copy
    for key in ["min_layers", "enc_min_layers", "dec_min_layers"]:
        new_dict[key] = int(new_dict[key])
    return new_dict


migrations[ModelArgs].append(to_int)


def migrate_train_args(nested_dict: Dict[str, Any]) -> TrainerArgs:
    flat_dict = flatten_dict(nested_dict)
    flat_dict = migrate(TrainerArgs, flat_dict)
    return TrainerArgs.from_flat(flat_dict)


if __name__ == "__main__":
    paths = ["YOUR PATH"]
    for path in paths:
        dict_ = json.load(open(path))
        cfg = migrate_train_args(dict_)
