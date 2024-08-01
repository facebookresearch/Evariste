# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from evariste.model.data.envs.batch_iterator import BatchIterator
from evariste.model.data.envs.env import DataGenerator
from evariste.model.data.dictionary import Dictionary
from evariste.comms.zmq import ZMQNotReady, ZMQNotReadySample
from params import ConfStore
import torch
from torch.utils.data import DataLoader
import numpy as np
from evariste.trainer.args import TrainerArgs
import pytest

RNG = np.random.RandomState(1)


class MyEnv(DataGenerator):
    def __init__(self, task, split):
        self._data = {}
        self._data[task] = {}
        self.dico = Dictionary.create_empty()
        if split in ["valid", "test"]:
            self._data[task][split] = [
                {
                    "x": list(RNG.randint(0, 100, (RNG.randint(0, 5)))),
                    "y": list(RNG.randint(0, 100, (RNG.randint(0, 5)))),
                }
                for i in range(100)
            ]

    @property
    def data(self):
        return self._data

    def set_rng(self, rng):
        self.rng = rng

    def get_sample(self, task, split, index):
        if split == "train":
            return {
                "x": list(self.rng.randint(0, 100, (self.rng.randint(0, 20)))),
                "y": list(self.rng.randint(0, 100, (self.rng.randint(0, 20)))),
            }
        else:
            return self.data[task][split][index]

    def get_stats(self):
        return {}


class MyEnvZMQ(DataGenerator):
    def __init__(self, task, split):
        self._data = {}
        self.dico = Dictionary.create_empty()
        self._data[task] = {}
        self.zmq_ready = False

    @property
    def data(self):
        return self._data

    def set_rng(self, rng):
        self.rng = rng

    def get_sample(self, task, split, index):
        if split == "train":
            if not self.zmq_ready:
                self.zmq_ready = True
                raise ZMQNotReady()
            return {
                "x": list(self.rng.randint(0, 100, (np.random.randint(0, 20)))),
                "y": list(self.rng.randint(0, 100, (np.random.randint(0, 20)))),
            }
        else:
            return self.data[task][split][index]

    def get_stats(self):
        return {}


class MyEnvFixedRng(MyEnv):
    def set_rng(self, rng):
        self.rng = np.random.RandomState(0)


def test_basic_valid():
    cfg = ConfStore["default_cfg"]
    env = MyEnv("my_task", "valid")
    batch_iterator = BatchIterator(env, "valid", "my_task", cfg, env.dico.pad_index)
    assert batch_iterator.batch_size == 32
    data_loader = DataLoader(
        batch_iterator, batch_size=None, num_workers=cfg.num_workers
    )
    i = 0
    for batch in data_loader:
        i += 1
        if i < 4:
            assert batch["x"].shape[0] == 32
        else:
            assert batch["x"].shape[0] == 4
    assert i == 4


@pytest.mark.parametrize(
    "queue_strategy,num_workers",
    [
        ("uniform_sampling", 0),
        ("uniform_sampling", 10),
        ("uniform_sampling_replacement", 0),
        ("uniform_sampling_replacement", 10),
    ],
)
def test_basic_train(queue_strategy, num_workers):
    cfg = ConfStore["default_cfg"]
    cfg.env_base_seed = 10
    cfg.num_workers = num_workers
    cfg.slurm_conf.global_rank = 0
    cfg.batch.queue_strategy = queue_strategy
    env = MyEnv("my_task", "train")
    batch_iterator = BatchIterator(env, "train", "my_task", cfg, env.dico.pad_index)
    data_loader = DataLoader(
        batch_iterator, batch_size=None, num_workers=cfg.num_workers
    )
    i = 0
    for batch in data_loader:
        if i > 50:
            break
        assert batch["x"].numel() + batch["y"].numel() <= cfg.batch.tokens
        i += 1


def test_wo_worker_seed():
    cfg = ConfStore["default_cfg"]
    cfg.env_base_seed = 10
    cfg.slurm_conf.global_rank = 0
    cfg.batch.queue_strategy = "uniform_sampling"

    # If we do not set a different seed perworker, then the 10 workers will see the data in same order,
    # and we will have the same 10 batches
    # Test first that we do have this behaviour
    # First check that with one worker there is no pb i.e all batches are different
    cfg.num_workers = 0
    env = MyEnvFixedRng("my_task", "train",)
    batch_iterator = BatchIterator(env, "train", "my_task", cfg, env.dico.pad_index)
    data_loader = DataLoader(
        batch_iterator, batch_size=None, num_workers=cfg.num_workers
    )
    i = 0
    batches = []
    for batch in data_loader:
        if i > 30:
            break
        batches.append(batch["x"])
        i += 1
    assert all(not torch.equal(batches[0], batch) for batch in batches[1:])

    # Then check that with 10 workers we effectivly have same batches
    cfg.num_workers = 10
    env = MyEnvFixedRng("my_task", "train")
    batch_iterator = BatchIterator(env, "train", "my_task", cfg, env.dico.pad_index)
    data_loader = DataLoader(
        batch_iterator, batch_size=None, num_workers=cfg.num_workers
    )
    i = 0
    batches = []
    for batch in data_loader:
        if i > 30:
            break
        batches.append(batch["x"])
        i += 1
    assert all(torch.equal(batches[0], batches[i]) for i in range(10))
    assert all(torch.equal(batches[10], batches[i]) for i in range(10, 20))
    assert all(torch.equal(batches[20], batches[i]) for i in range(20, 30))
    assert not torch.equal(batches[0], batches[10])
    assert not torch.equal(batches[0], batches[20])
    assert not torch.equal(batches[10], batches[20])


def test_w_worker_seed():
    cfg = ConfStore["default_cfg"]
    cfg.env_base_seed = 10
    cfg.num_workers = 10
    cfg.slurm_conf.global_rank = 0
    cfg.batch.queue_strategy = "uniform_sampling"

    # Here we set a different seed perworker, then the 10 workers will NOT see the data in same order,
    # So batches should be different
    env = MyEnv("my_task", "test")
    batch_iterator = BatchIterator(env, "train", "my_task", cfg, env.dico.pad_index)
    data_loader = DataLoader(
        batch_iterator, batch_size=None, num_workers=cfg.num_workers
    )
    i = 0
    batches = []
    for batch in data_loader:
        if i > 30:
            break
        batches.append(batch["x"])
        i += 1
    assert not all(torch.equal(batches[0], batches[i]) for i in range(10))
    assert not all(torch.equal(batches[10], batches[i]) for i in range(10, 20))
    assert not all(torch.equal(batches[20], batches[i]) for i in range(20, 30))


@pytest.mark.parametrize("num_workers", [(0), (10)])
def test_trainer_seed(num_workers):
    cfg = ConfStore["default_cfg"]
    cfg.env_base_seed = 10
    cfg.num_workers = num_workers
    cfg.batch.queue_strategy = "uniform_sampling"
    env = MyEnv("my_task", "train")

    # Sanity check: If I sample twice with same trainer seed, I have the same data
    cfg.slurm_conf.global_rank = 0
    batch_iterator = BatchIterator(env, "train", "my_task", cfg, env.dico.pad_index)
    data_loader = DataLoader(
        batch_iterator, batch_size=None, num_workers=cfg.num_workers
    )

    i = 0
    batches_0 = []
    for batch in data_loader:
        if i > 30:
            break
        batches_0.append(batch["x"])
        i += 1

    env = MyEnv("my_task", "train")
    batch_iterator = BatchIterator(env, "train", "my_task", cfg, env.dico.pad_index)
    data_loader = DataLoader(
        batch_iterator, batch_size=None, num_workers=cfg.num_workers
    )
    i = 0
    batches_1 = []
    for batch in data_loader:
        if i > 30:
            break
        batches_1.append(batch["x"])
        i += 1

    assert all(torch.equal(batches_0[i], batches_1[i]) for i in range(30))

    # then I we change the global rank, i.e different trainer, we have different data
    env = MyEnv("my_task", "train")
    cfg.slurm_conf.global_rank = 1
    batch_iterator = BatchIterator(env, "train", "my_task", cfg, env.dico.pad_index)
    data_loader = DataLoader(
        batch_iterator, batch_size=None, num_workers=cfg.num_workers
    )
    i = 0
    batches_1 = []
    for batch in data_loader:
        if i > 30:
            break
        batches_1.append(batch["x"])
        i += 1
    assert not all(torch.equal(batches_0[i], batches_1[i]) for i in range(30))


def test_zmq():
    cfg = ConfStore["default_cfg"]
    cfg.env_base_seed = 10
    cfg.num_workers = 1
    cfg.slurm_conf.global_rank = 0
    env = MyEnvZMQ("my_task", "train")
    batch_iterator = BatchIterator(env, "train", "my_task", cfg, env.dico.pad_index)
    data_loader = DataLoader(
        batch_iterator, batch_size=None, num_workers=cfg.num_workers
    )
    batches = []
    i = 0
    for batch in data_loader:
        if i > 20:
            break
        i += 1
        batches.append(batch)
    assert isinstance(batches[0], ZMQNotReadySample)
    assert all(not isinstance(batches[i], ZMQNotReadySample) for i in range(1, 20))
