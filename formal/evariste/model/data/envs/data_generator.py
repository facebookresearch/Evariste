# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod


class DataGenerator(ABC):
    @property
    @abstractmethod
    def data(self):
        ...

    @abstractmethod
    def set_rng(self, rng):
        pass

    @abstractmethod
    def get_sample(self, task, split, index):
        pass

    @abstractmethod
    def get_stats(self):
        pass

    def maybe_load(self, task: str, split: str):
        """
        load only the required data.
        By default does nothing because all envs except lean can afford to load everything across workers
        """
        pass
