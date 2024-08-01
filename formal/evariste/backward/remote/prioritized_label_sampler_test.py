# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, List

from numpy.random import RandomState
import numpy as np

from evariste.backward.prover.mcts_samples import MCTSSampleCritic
from evariste.backward.remote.prioritized_label_sampler import (
    SingleSplitLabelSampler,
    MultiSplitLabelSampler,
    WARM_UP_STATUS,
    PRIORITY_STATUS,
    PriorityKind,
    PLSStats,
    LabelSamplerCfg,
)
from evariste.testing.common import MyTheorem

N = 1000


def test_sampler():
    for kind in PriorityKind:
        assert isinstance(kind, PriorityKind)
        print(f"kind, {kind}")
        _test_sampler(kind)


def test_new_staleness():
    for kind in PriorityKind:
        assert isinstance(kind, PriorityKind)
        print(f"kind, {kind}")
        _test_sampler(kind, use_uniform_as_staleness=True)


def test_uniform_probs_for_non_solved():
    for kind in PriorityKind:
        assert isinstance(kind, PriorityKind)
        print(f"kind, {kind}")
        _test_sampler(kind, uniform_probs_for_non_solved=True)


def _test_sampler(
    priority_kind: PriorityKind,
    use_uniform_as_staleness: bool = False,
    uniform_probs_for_non_solved: bool = False,
):
    cfg = LabelSamplerCfg(
        beta=1.0,
        rho=0.1,
        priority_kind=priority_kind,
        use_uniform_as_staleness=use_uniform_as_staleness,
        uniform_probs_for_non_solved=uniform_probs_for_non_solved,
    )
    rng = RandomState(42)
    labels = [f"{i}" for i in range(100)]
    sampler = SingleSplitLabelSampler(labels=labels, cfg=cfg)
    first_labels = []
    stats = sampler.stats()
    assert stats["status"] == WARM_UP_STATUS
    for i in range(len(labels)):
        label = sampler.sample_label(rng)
        p_solved = _p_solved(label, len(labels))
        solved = rng.random() < p_solved

        pls_stats = _pls_stats(
            solved,
            samples=[_random_critic_sample(rng, label) for _ in range(rng.randint(10))],
        )
        sampler.update_score(label, pls_stats=pls_stats)
        first_labels.append(label)
    assert set(first_labels) == set(labels)

    stats = sampler.stats()
    assert stats["status"] == PRIORITY_STATUS

    for i in range(N):
        label = sampler.sample_label(rng)
        p_solved = _p_solved(label, len(labels))
        solved = rng.random() < p_solved
        sampler.update_score(label, pls_stats=_pls_stats(solved))
    if uniform_probs_for_non_solved:
        probs = sampler._priority_probs()
        non_solved = [
            i
            for i, label in enumerate(sampler.labels)
            if sampler.label2stats[label].n_proved == 0
        ]
        assert np.allclose(probs[non_solved], 1.0 / len(sampler.labels))

    stats = sampler.stats()
    assert stats["status"] == PRIORITY_STATUS
    assert stats["total_sampled"] == N + len(labels)

    assert np.sum(
        [sampler.label2stats[label].n_sampled for label in labels]
    ) == N + len(labels)
    _ = sampler.state()


def test_sampler_no_update():
    for kind in PriorityKind:
        assert isinstance(kind, PriorityKind)
        print(f"kind, {kind}")
        _test_sampler_no_update(kind)


def _test_sampler_no_update(priority_kind: PriorityKind):
    rng = RandomState(42)
    labels = [f"{i}" for i in range(100)]
    cfg = LabelSamplerCfg(beta=0.1, rho=0.1, priority_kind=priority_kind)
    sampler = SingleSplitLabelSampler(labels=labels, cfg=cfg)
    stats = sampler.stats()
    _ = sampler.state()
    assert stats["status"] == WARM_UP_STATUS
    for i in range(2 * len(labels)):
        _ = sampler.sample_label(rng)
        if i == N // 2:
            _ = sampler.state()
            _ = sampler.stats()
        # no update of sampler
    _ = sampler.state()

    stats = sampler.stats()
    assert stats["status"] == PRIORITY_STATUS
    assert stats["total_sampled"] == 2 * len(labels)
    assert np.isclose(stats["not_solved_prob"], 1.0)
    _ = sampler.state()


MARGIN = 0.1


def _pls_stats(
    solved: bool, samples: Optional[List[MCTSSampleCritic]] = None
) -> PLSStats:
    return PLSStats.from_mcts(
        samples=samples, proved=solved, minproof_size=1 if solved else None
    )


def test_multi_split():
    rng = RandomState(42)
    splits = ["s1", "s2"]
    split_probs = np.array([0.9, 0.1])
    label_to_split = {}
    split_to_sampler = {}
    for split in splits:
        labels = []
        for i in range(100):
            label = f"{split}_l{i}"
            label_to_split[label] = split
            labels.append(label)
        sampler = SingleSplitLabelSampler(labels=labels, cfg=LabelSamplerCfg())
        split_to_sampler[split] = sampler
    sampler = MultiSplitLabelSampler(
        splits=splits,
        split_probs=split_probs,
        split_to_sampler=split_to_sampler,
        label_to_split=label_to_split,
    )
    _ = sampler.stats()
    for _ in range(1000):
        label = sampler.sample_label(rng)
        solved = rng.random() < 0.5
        sampler.update_score(label, pls_stats=_pls_stats(solved))
    _ = sampler.stats()
    counts = [
        sum([s.n_sampled for s in split_to_sampler[split].label2stats.values()])
        for split in sampler.splits
    ]
    assert sum(counts) == 1000
    p0, p1 = counts[0] / sum(counts), counts[1] / sum(counts)
    assert 0.9 - MARGIN < p0 < 0.9 + MARGIN
    assert 0.1 - MARGIN < p1 < 0.1 + MARGIN
    _ = sampler.state()


def _p_solved(label: str, n_labels: int) -> float:
    label_ = int(label)
    if label_ < 0.5 * n_labels:
        return 0.0
    elif label_ < 0.75 * n_labels:
        return (label_ - (0.5 * n_labels)) / (0.25 * n_labels)
    else:
        return 1.0


def _random_critic_sample(rng: RandomState, label: str) -> MCTSSampleCritic:
    return MCTSSampleCritic(
        goal=MyTheorem("a"),
        solved=bool(rng.randint(1)),
        q_estimate=rng.random(),
        critic=rng.random(),
        bad=False,
        label=label,
        visit_count=500,
    )


def _test_speed():
    import time

    start = time.time()
    rng = RandomState(42)
    labels = [f"{i}" for i in range(30_000)]
    sampler = SingleSplitLabelSampler(
        labels=labels,
        cfg=LabelSamplerCfg(
            priority_kind=PriorityKind.Uniform, use_uniform_as_staleness=False
        ),
    )
    first_labels = []
    stats = sampler.stats()
    assert stats["status"] == WARM_UP_STATUS
    for i in range(len(labels)):
        label = sampler.sample_label(rng)
        p_solved = _p_solved(label, len(labels))
        solved = rng.random() < p_solved
        sampler.update_score(label, _pls_stats(solved))
        first_labels.append(label)
    assert set(first_labels) == set(labels)
    end_warmup_up = time.time()
    print(f"warmup done in {end_warmup_up - start}")

    stats = sampler.stats()
    assert stats["status"] == PRIORITY_STATUS

    for i in range(N):
        label = sampler.sample_label(rng)
        p_solved = _p_solved(label, len(labels))
        solved = rng.random() < p_solved
        sampler.update_score(label, pls_stats=_pls_stats(solved))
    print(f"{N} trials done in {time.time() - end_warmup_up}")

    stats = sampler.stats()
    assert stats["status"] == PRIORITY_STATUS
    assert stats["total_sampled"] == N + len(labels)

    assert np.sum(
        [sampler.label2stats[label].n_sampled for label in labels]
    ) == N + len(labels)

    print(stats)

    assert False


def test_pls_stats():
    for solved in [True, False]:
        stats = _pls_stats(solved=solved)
        assert PLSStats.from_mcts_stats(stats.to_stats()) == stats

    samples = [
        MCTSSampleCritic(
            solved=True,
            goal=MyTheorem("a"),
            q_estimate=0.5,
            critic=0.0,
            visit_count=500,
            bad=False,
        )
        for _ in range(10)
    ]
    stats = PLSStats.from_mcts(proved=True, minproof_size=1, samples=samples)

    assert np.isclose(stats.avg_abs_error_to_solved, 1.0)
    assert np.isclose(stats.avg_abs_error_to_v_mcts, 0.5)
