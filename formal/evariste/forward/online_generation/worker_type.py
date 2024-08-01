# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from enum import unique, Enum


@unique
class WorkerType(str, Enum):
    GENERATOR_TRAINER = "generator_trainer"
    GENERATOR_ACTOR = "generator_actor"
    PROVER_TRAINER = "prover_trainer"
    PROVER_ACTOR = "prover_actor"

    def is_trainer(self):
        return self in {self.GENERATOR_TRAINER, self.PROVER_TRAINER}

    def is_actor(self):
        return self in {self.GENERATOR_ACTOR, self.PROVER_ACTOR}

    def is_generator(self):
        return self in {self.GENERATOR_TRAINER, self.GENERATOR_ACTOR}

    def is_prover(self):
        return self in {self.PROVER_TRAINER, self.PROVER_ACTOR}
