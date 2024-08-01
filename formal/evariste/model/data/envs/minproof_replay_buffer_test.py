# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, List, Dict, Any

import numpy
import pytest
from numpy.random import RandomState

from evariste import json
from evariste.backward.prover.mcts_samples import MCTSSampleTactics
from evariste.backward.remote.state_dump_utils import StateDumper
from evariste.comms.store import Receiver
from evariste.model.data.envs.minproof_replay_buffer import (
    MinProofPriorityStore,
    MinProofReplayBufferCfg,
    MCTSSampleProof,
    MCTSReplayBuffer,
    PriorityStore,
    LabelSampling,
    _normalized_softmax_probs,
    MultiSplitStore,
)
from evariste.testing.common import MyTheorem, MyTactic


def _thm(i):
    return MyTheorem(conclusion=f"conc{i}")


def _tactic():
    return MyTactic("sa")


def _proof(rng: RandomState, label: str, size: Optional[int] = None):
    size = rng.randint(1, 10) if size is None else size
    samples = [_tactics(i) for i in range(size)]
    return MCTSSampleProof(label=label, size=size, samples=samples, stype="size")


def _tactics(i: int):
    return MCTSSampleTactics(
        goal=_thm(i), tactics=[_tactic()], target_pi=[1], inproof=True, visit_count=10,
    )


N = 100


def test_minproof_priority_store():
    for sampling in LabelSampling:
        for beta in [0, 0.5, 1, 5]:
            for dedup in [True, False]:
                _test_minproof_priority_store(sampling, beta=beta, deduplicate=dedup)


def _test_minproof_priority_store(sampling: str, beta: float, deduplicate: bool):
    store = MinProofPriorityStore(
        cfg=MinProofReplayBufferCfg(
            label_sampling=sampling,
            n_proofs_by_label=1,
            min_recv_proofs=1,
            should_dump=False,
            proof_sampling_beta=beta,
            deduplicate_proofs=deduplicate,
        )
    )
    rng = RandomState(42)
    _ = store.get_stats()
    assert not store.ready()
    for i in range(N):
        proof = _proof(label=f"{i}", rng=rng)
        store.add(proof)

    assert len(store.labels) == N
    assert store.ready()
    assert isinstance(store.sample(rng), MCTSSampleTactics)
    _ = store.get_stats()

    if sampling == LabelSampling.Uniform:
        assert numpy.allclose(store.cum_weights, 1 + numpy.arange(len(store.labels)))

    for proofs in store.data:
        assert len(proofs) == 1


def test_minproof_priority_store_2():
    for sampling in LabelSampling:
        assert isinstance(sampling, LabelSampling)
        for beta in [0, 0.5, 1, 5]:
            for dedup in [True, False]:
                _test_store_2(sampling, beta=beta, deduplicate=dedup)


def _test_store_2(sampling: LabelSampling, beta: float, deduplicate: bool):
    store = MinProofPriorityStore(
        cfg=MinProofReplayBufferCfg(
            n_proofs_by_label=3,
            min_recv_proofs=1,
            label_sampling=sampling,
            should_dump=False,
            proof_sampling_beta=beta,
            deduplicate_proofs=deduplicate,
        )
    )
    rng = RandomState(42)
    stats = store.get_stats()
    dump_stats(stats)
    assert not store.ready()
    for i in range(N // 10):
        for j in range(2, 2 + 10):
            # one on two is duplicated
            proof = _proof(label=f"{i}", rng=rng, size=i + j // 2)
            store.add(proof)

    stats = store.get_stats()
    dump_stats(stats)

    assert len(store.labels) == N // 10

    for idx in range(len(store.labels)):
        assert len(store.data[idx]) == 3
        if deduplicate:
            assert [mp.size for mp in store.data[idx]] == [idx + j for j in [1, 2, 3]]
        else:
            assert [mp.size for mp in store.data[idx]] == [idx + j for j in [1, 1, 2]]

    for _ in range(10):
        _ = store.sample(rng)

    assert store.ready()
    assert isinstance(store.sample(rng), MCTSSampleTactics)
    stats = store.get_stats()
    dump_stats(stats)


def test_with_state_dumper():
    for sampling in LabelSampling:
        with TemporaryDirectory() as tmp:
            store = fill_minproof_rb(sampling, tmp)

            # restore
            new_state_dumper = StateDumper(folder=Path(tmp) / "dump", n_states_max=10)
            cfg = MinProofReplayBufferCfg(
                min_recv_proofs=1, dump_interval=1e-5, label_sampling=sampling
            )
            new_store = MinProofPriorityStore(cfg=cfg, state_dumper=new_state_dumper)
            assert len(new_store.labels) == 0
            assert new_store.recv_proofs == 0
            new_store.restore_from_state_dump()
            assert new_store.labels == store.labels
            assert new_store.data == store.data
            assert new_store.recv_proofs == store.recv_proofs
            assert numpy.allclose(new_store.cum_weights, store.cum_weights)
            assert new_store.labels_stats == store.labels_stats

            stats = store.get_stats()
            dump_stats(stats)


def fill_minproof_rb(sampling: str, folder: str) -> MinProofPriorityStore:
    state_dumper = StateDumper(folder=Path(folder) / "dump", n_states_max=10)
    cfg = MinProofReplayBufferCfg(
        min_recv_proofs=1, dump_interval=1e-5, label_sampling=sampling
    )
    with pytest.raises(AssertionError):
        _ = MinProofPriorityStore(cfg=cfg, state_dumper=None)
    rng = numpy.random.RandomState(0)
    store = MinProofPriorityStore(cfg=cfg, state_dumper=state_dumper)
    stats = store.get_stats()
    dump_stats(stats)
    time.sleep(1e-4)  # time to dump!
    proof = _proof(label=f"{0}", rng=rng, size=1)
    store.add(proof)
    _ = store.sample(rng)
    assert len(state_dumper.get_present()) == 1
    time.sleep(1e-4)  # time to dump!
    proof = _proof(label=f"{1}", rng=rng, size=2)
    store.add(proof)
    _ = store.sample(rng)
    assert len(state_dumper.get_present()) == 2
    return store


def test_multi_split_store():
    label_to_split = {}
    splits = []
    stores = []
    for split_idx in [1, 2]:
        split = f"s{split_idx}"
        splits.append(split)
        stores.append(
            MinProofPriorityStore(
                cfg=MinProofReplayBufferCfg(min_recv_proofs=2, should_dump=False),
                name=f"split{split}",
            )
        )
        for label_idx in range(10):
            label = f"s{split_idx}_l{label_idx}"
            label_to_split[label] = split

    labels = list(label_to_split.keys())

    store = MultiSplitStore(
        label_to_split=label_to_split, splits=splits, stores=stores, probs=[0.9, 0.1]
    )

    dump_stats(store.get_stats())
    assert not store.ready()

    rng = numpy.random.RandomState(0)
    for _ in range(100):
        label = rng.choice(labels)
        proof = _proof(rng, label=label)
        store.add(proof)

    assert store.ready()
    dump_stats(store.get_stats())

    _ = store.sample(rng=rng)

    dump_stats(store.get_stats())


class DummyReceiver(Receiver[int]):
    def receive_batch(self) -> List[int]:
        return [0]

    def rate_and_reset(self) -> float:
        return 1.0

    def close(self):
        pass


class DummyPriorityStore(PriorityStore[int, int]):
    def get_stats(self) -> Dict[str, float]:
        return {}

    def __init__(self):
        self.data = []

    def ready(self) -> bool:
        return len(self.data) >= 10

    def sample(self, rng: RandomState) -> int:
        return rng.choice(self.data)

    def add(self, inp: int):
        self.data.append(inp)


def test_minproof_mcts_data_loader():
    store = DummyPriorityStore()
    rb = MCTSReplayBuffer(receiver=DummyReceiver(), priority_store=store)
    rng = RandomState(42)
    _ = rb.get_stats()
    _ = rb.get_mcts_sample("train", index=None, rng=rng)
    assert len(store.data) == 10
    _ = rb.get_mcts_sample("train", index=None, rng=rng)
    assert len(store.data) == 11
    stats = rb.get_stats()
    dump_stats(stats)


def dump_stats(stats: Dict[str, Any]):
    json.dumps(stats)


def test_normalized_softmax_probs():
    for sizes in [
        [3, 2, 20, 10],
        [200, 400, 200, 100],
        [2],
        [10000, 10000, 10000, 9999],
        [2.0, 2.0],
        [10000, 100, 5000],
    ]:
        for beta in [0, 1, -1]:
            sizes = numpy.array(sizes, dtype=numpy.float_)
            probs = _normalized_softmax_probs(sizes, beta=beta)
            assert len(probs) == len(sizes)
            assert not numpy.isnan(probs).any()
            assert numpy.isclose(probs.sum(), 1)

            print(probs)


def test_reload_minproof_rb():
    with TemporaryDirectory() as tmp:
        store = fill_minproof_rb(sampling=LabelSampling.Uniform, folder=tmp)
        present = StateDumper(
            folder=Path(tmp) / "dump", is_master=False, n_states_max=-1
        ).get_present()
        if len(present) == 0:
            raise RuntimeError(f"No dump in {Path(tmp) / 'dump'}")

        path = present[-1]

        cfg = MinProofReplayBufferCfg(
            min_recv_proofs=1, label_sampling=LabelSampling.Uniform, should_dump=False
        )
        new_store = MinProofPriorityStore(cfg=cfg, state_dumper=None)
        new_store.reload_dump(path)
        assert new_store.data == store.data
        rng = RandomState(0)

        # checking that new store work correctly
        _ = new_store.sample(rng=rng)
        stats = store.get_stats()
        dump_stats(stats)
        new_store.add(_proof(label=f"{0}", rng=rng, size=3))
        new_store.add(_proof(label=f"{2}", rng=rng, size=1))
        _ = new_store.sample(rng=rng)
        stats = store.get_stats()
        dump_stats(stats)

        assert len(new_store.data[0]) == 2
        assert len(new_store.data[1]) == 1
        assert len(new_store.data[2]) == 1
