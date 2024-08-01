# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Any, Dict, Callable, List, Type
from dataclasses import fields, is_dataclass
from collections import defaultdict

from params import Params


FlatDict = Dict[str, Any]
Migration = Callable[[FlatDict], FlatDict]

migrations: Dict[Type[Params], List[Migration]] = defaultdict(list)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def migrate(schema: Type[Params], flat_dict: FlatDict) -> FlatDict:
    for migration in migrations.get(schema, []):
        flat_dict = migration(flat_dict)
    for field in fields(schema):
        if is_dataclass(field.type):
            prefix = f"{field.name}."
            selected = {
                k[len(prefix) :]: v
                for k, v in flat_dict.items()
                if k.startswith(prefix)
            }
            assert isinstance(field.type, Params)
            new_entries = migrate(field.type, selected)
            new_entries = {f"{prefix}{k}": v for k, v in new_entries.items()}
            for k, v in selected.items():
                key = f"{prefix}{k}"
                if key not in new_entries:
                    flat_dict.pop(key)
            flat_dict.update(new_entries)
    return flat_dict


def warn_migration(txt):
    print(f"{bcolors.WARNING}[MIGRATION] - {txt}{bcolors.ENDC}")
