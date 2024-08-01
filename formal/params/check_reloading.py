# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from evariste import json as json

from evariste.trainer.migrations import migrate_train_args


TRAINER_ARGS_PATHS = [""]


def reload_trainer_args(path: str):
    dict_ = json.load(open(path))
    # at least we don't crash ;)
    _ = migrate_train_args(dict_)


def check_reloading():
    for p in TRAINER_ARGS_PATHS:
        reload_trainer_args(p)


if __name__ == "__main__":
    check_reloading()
