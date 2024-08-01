# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, List, Dict, Any, Tuple

from numpy.random import RandomState
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import unique, Enum
import time
import numpy as np

from evariste.backward.prover.mcts_samples import MCTSSampleCritic
from params import Params


WARM_UP_STATUS = 0
PRIORITY_STATUS = 1


@unique
class PriorityKind(str, Enum):
    PropNotProved = "prop_not_proved"
    FirstTimeProved = "first_time_proved"
    MinProofSize = "minproof_size"
    Uniform = "uniform"
    PosValueLoss = "pos_value_loss"


@dataclass
class LabelSamplerCfg(Params):
    # paper: https://openreview.net/forum?id=NfZ6g2OmXEk
    use_prioritized_label_sampler: bool = False
    # prob smoothing (see paper)
    beta: float = 0.5
    # to control staleness (see paper)
    rho: float = 0.1
    priority_kind: PriorityKind = PriorityKind.PosValueLoss
    # alpha coef for exponential moving average (for PriorityKind using critic error)
    ema_alpha: float = 0.1
    # if True we give a fixed probability for each theorem non solved, corresponding
    # they would have under the 'uniform' distribution.
    uniform_probs_for_non_solved: bool = False
    use_uniform_as_staleness: bool = False


def prioritized_label_sampler_factory(
    cfg: LabelSamplerCfg,
    splits: List[str],
    split_probs: np.ndarray,
    split_to_labels: Dict[str, List[str]],
    label_to_split: Dict[str, str],
) -> Optional["PrioritizedLabelSampler"]:
    if not cfg.use_prioritized_label_sampler:
        return None
    assert len(splits) > 0
    split_to_sampler = {}
    for split in splits:
        labels = split_to_labels[split]
        sampler = SingleSplitLabelSampler(labels=labels, cfg=cfg)
        split_to_sampler[split] = sampler
    return MultiSplitLabelSampler(
        splits=splits,
        split_probs=split_probs,
        split_to_sampler=split_to_sampler,
        label_to_split=label_to_split,
    )


class PrioritizedLabelSampler(ABC):
    @abstractmethod
    def sample_label(self, rng: RandomState) -> str:
        pass

    @abstractmethod
    def update_score(self, label: str, pls_stats: "PLSStats"):
        pass

    @abstractmethod
    def stats(self) -> Dict:
        pass

    @abstractmethod
    def state(self) -> Dict:
        pass


@dataclass
class _PrioStats:
    # can be sampled many time before receiving results
    n_sampled: int
    last_sampled: int
    first_proved: Optional[int] = None
    n_proved: int = 0
    n_finished: int = 0
    minproof_size: Optional[int] = None
    # pos value loss (mcts adaptation of positive value loss of
    # https://arxiv.org/pdf/2110.02439.pdf)
    # ema = exponential moving average
    # using 1 / N  SUM ( int(solved) - critic ) for N the solved node in the MCTS
    ema_pos_value_loss: Optional[float] = None

    def prop_solved(self) -> float:
        return self.n_proved / self.n_finished

    def waiting_for_result(self) -> bool:
        return self.n_sampled > self.n_finished


class SingleSplitLabelSampler(PrioritizedLabelSampler):
    def __init__(self, labels: List[str], cfg: LabelSamplerCfg):
        self.cfg = cfg
        self.labels = labels
        assert len(set(labels)) == len(labels), "duplicated"
        self.use_uniform_as_staleness = self.cfg.use_uniform_as_staleness
        self.uniform_probs_for_non_solved = self.cfg.uniform_probs_for_non_solved

        # state
        self.first = True
        self.warm_up: List[str] = []
        self.total_sampled: int = 0
        self.total_finished: int = 0
        self.label2stats: Dict[str, _PrioStats] = {}
        self.rho = self.cfg.rho
        self.beta = self.cfg.beta
        self.priority_kind = self.cfg.priority_kind
        self.ema_alpha = self.cfg.ema_alpha

        self.cum_time_in_sampling: float = 0

    def _warming(self) -> bool:
        return self.first or len(self.warm_up) > 0

    def sample_label(self, rng: RandomState) -> str:
        start = time.time()
        if self.first:
            assert not self.warm_up
            labels = sorted(self.labels)
            rng.shuffle(labels)
            self.warm_up = labels
            self.first = False

        # sample all labels first
        if self.warm_up:
            label = self.warm_up.pop()
            self.label2stats[label] = _PrioStats(
                n_sampled=1, last_sampled=self.total_sampled
            )
        else:
            label = self.labels[
                rng.choice(np.arange(len(self.labels)), p=self._probs())
            ]
            self.label2stats[label].n_sampled += 1
            self.label2stats[label].last_sampled = self.total_sampled
        self.total_sampled += 1
        self.cum_time_in_sampling += time.time() - start
        return label

    def update_score(self, label: str, pls_stats: "PLSStats"):
        assert label in self.label2stats
        stats: _PrioStats = self.label2stats[label]
        assert stats.waiting_for_result()
        if pls_stats.proved:
            assert pls_stats.minproof_size > 0
            if stats.n_proved == 0:
                assert stats.first_proved is None, type(stats.first_proved)
                assert stats.minproof_size is None, type(stats.minproof_size)
                stats.first_proved = self.total_finished
                stats.minproof_size = int(pls_stats.minproof_size)
            else:
                assert stats.minproof_size is not None
                stats.minproof_size = min(
                    stats.minproof_size, int(pls_stats.minproof_size)
                )
        else:
            assert pls_stats.minproof_size == -1

        stats.ema_pos_value_loss = self._update_moving_average(
            stats.ema_pos_value_loss, pls_stats.avg_pos_error_to_solved
        )

        stats.n_proved += int(pls_stats.proved)
        stats.n_finished += 1
        self.total_finished += 1

    def _staleness_probs(self):
        assert not self._warming()
        if self.use_uniform_as_staleness:
            scores = np.ones(len(self.labels))
        else:
            # naive re-computation of all scores each sample
            scores = np.array(
                [
                    float(self.total_sampled - self.label2stats[label].last_sampled)
                    for label in self.labels
                ]
            )
        return _normalize(scores, beta=1.0)

    def _priority_probs(self):
        assert not self._warming()
        non_solved_mask = np.array(
            [self.label2stats[label].n_proved == 0 for label in self.labels], dtype=bool
        )
        # naive re-computation of all scores each sample
        stats = (self.label2stats[label] for label in self.labels)
        if self.priority_kind == PriorityKind.PropNotProved:
            scores = np.array(
                [(s.n_finished - s.n_proved) / max(s.n_finished, 1) for s in stats]
            )
        elif self.priority_kind == PriorityKind.FirstTimeProved:
            scores = np.array(
                [
                    float(s.first_proved)
                    if s.first_proved is not None
                    else float(self.total_finished)
                    for s in stats
                ]
            )
        elif self.priority_kind == PriorityKind.MinProofSize:
            scores = np.array(
                [float(s.minproof_size) if s.n_proved > 0 else -1.0 for s in stats]
            )
            scores = np.where(scores == -1, max(scores.max(), 1), scores)
            assert scores.min() >= 1, scores.min()
        elif self.priority_kind == PriorityKind.Uniform:
            scores = np.ones(len(self.labels))
        elif self.priority_kind == PriorityKind.PosValueLoss:
            scores = np.array(
                [
                    float(s.ema_pos_value_loss)
                    if s.ema_pos_value_loss is not None
                    else 0.0
                    for s in stats
                ]
            )
        else:
            raise NotImplementedError(self.priority_kind)
        if self.uniform_probs_for_non_solved:
            scores[non_solved_mask] = 0
            if scores.sum() == 0:
                scores[~non_solved_mask] = 1

        probs = _normalize(scores, beta=self.beta)
        if self.uniform_probs_for_non_solved:
            probs[non_solved_mask] = 1.0 / len(probs)
            probs[~non_solved_mask] *= (~non_solved_mask).sum() / len(probs)
        assert np.isclose(probs.sum(), 1.0), probs.sum()
        return probs

    def _probs(self):
        assert not self._warming()
        return (
            1 - self.rho
        ) * self._priority_probs() + self.rho * self._staleness_probs()

    def _update_moving_average(
        self, avg: Optional[float], update: float
    ) -> Optional[float]:
        update = max(update, 0)  # setting to 0 if -1 (no critic nodes)
        if avg is None:
            return update
        else:
            return avg * (1 - self.ema_alpha) + self.ema_alpha * update

    def stats(self) -> Dict:
        status = (
            WARM_UP_STATUS if self.first or len(self.warm_up) > 0 else PRIORITY_STATUS
        )
        stats = {
            "status": status,
            "cum_time_in_sampling": self.cum_time_in_sampling,
            "total_sampled": self.total_sampled,
            "n_labels": len(self.labels),
            "n_solved": len([s for s in self.label2stats.values() if s.n_proved > 0]),
            "n_always_solved": len(
                [s for s in self.label2stats.values() if s.n_proved == s.n_finished > 0]
            ),
        }
        if not self._warming():
            probs = self._probs()
            not_solved_prob = sum(
                [
                    probs[i]
                    for i, label in enumerate(self.labels)
                    if self.label2stats[label].n_proved == 0
                ]
            )
            always_solved_prob = sum(
                [
                    probs[i]
                    for i, label in enumerate(self.labels)
                    if self.label2stats[label].n_proved
                    == self.label2stats[label].n_finished
                    > 0
                ]
            )
            last_10_solved = [
                i
                for _, i in sorted(
                    [
                        (self.label2stats[l].first_proved, i)
                        for i, l in enumerate(self.labels)
                        if self.label2stats[l].n_proved > 0
                    ],
                    reverse=True,
                )[:10]
            ]
            last_10_solved_prob = sum(self._probs()[i] for i in last_10_solved)
            stats.update(
                {
                    "not_solved_prob": not_solved_prob,
                    "always_solved_prob": always_solved_prob,
                    "last_10_solved_prob": last_10_solved_prob,
                }
            )

        return stats

    def state(self) -> Dict:
        state = {
            "labels": self.labels,
            "stats": [self.label2stats.get(l, None) for l in self.labels],
        }
        if not self._warming():
            state["priority_probs"] = self._priority_probs()
            state["staleness_probs"] = self._staleness_probs()
        return state


class MultiSplitLabelSampler(PrioritizedLabelSampler):
    def __init__(
        self,
        splits: List[str],
        split_probs: np.ndarray,
        split_to_sampler: Dict[str, SingleSplitLabelSampler],
        label_to_split: Dict[str, str],
    ):
        self.splits = splits
        self.split_probs = split_probs
        self.split_to_sampler = split_to_sampler
        self.label_to_split = label_to_split
        assert len(splits) == len(self.split_probs)
        assert set(splits) == set(split_to_sampler.keys())
        assert set(splits) == set(self.label_to_split.values())

    def sample_label(self, rng: RandomState) -> str:
        next_split = rng.choice(self.splits, size=1, p=self.split_probs)[0]
        return self.split_to_sampler[next_split].sample_label(rng)

    def update_score(self, label: str, pls_stats: "PLSStats"):
        split = self.label_to_split[label]
        self.split_to_sampler[split].update_score(label, pls_stats)

    def stats(self) -> Dict:
        stats = {}
        for split in self.splits:
            for k, v in self.split_to_sampler[split].stats().items():
                stats[f"{split}/{k}"] = v
        return stats

    def state(self) -> Any:
        return {
            f"{split}/{k}": v
            for split in self.splits
            for k, v in self.split_to_sampler[split].state().items()
        }


def _normalize(scores: np.ndarray, beta: float) -> np.ndarray:
    if np.sum(scores) == 0:
        scores += 1
    assert np.min(scores) >= 0
    scores = scores ** (1 / beta)
    return scores / np.sum(scores)


PREFIX = "pls/"


@dataclass
class PLSStats:
    # float because of serialization / deserialization
    proved: float
    minproof_size: float
    critic_n_samples: float
    critic_prop_solved: float
    critic_prop_bad: float
    avg_abs_error_to_v_mcts: float
    avg_abs_error_to_solved: float
    avg_pos_error_to_v_mcts: float
    avg_pos_error_to_solved: float
    avg_critic_for_solved: float
    avg_critic_for_non_solved: float

    def __post_init__(self):
        assert self.proved == (self.minproof_size > 0)

    def to_stats(self) -> Dict[str, Tuple[float, float]]:
        dict_ = asdict(self)
        return {f"{PREFIX}{k}": (float(v), 1.0) for k, v in dict_.items()}

    @staticmethod
    def from_mcts_stats(mcts_stats: Dict[str, Tuple[float, float]]) -> "PLSStats":
        mcts_stats = {k: v for k, v in mcts_stats.items() if k.startswith(PREFIX)}
        assert all(c == 1 for _, c in mcts_stats.values()), mcts_stats
        dict_ = {
            k[len(PREFIX) :]: v[0]
            for k, v in mcts_stats.items()
            if k.startswith(PREFIX)
        }
        return PLSStats(**dict_)

    @staticmethod
    def from_mcts(
        proved: bool,
        minproof_size: Optional[float],
        samples: Optional[List[MCTSSampleCritic]],
    ) -> "PLSStats":
        if samples is None or len(samples) == 0:
            return PLSStats(
                proved=proved,
                minproof_size=minproof_size if minproof_size is not None else -1,
                critic_n_samples=0,
                critic_prop_solved=-1,
                critic_prop_bad=-1,
                avg_abs_error_to_v_mcts=-1,
                avg_abs_error_to_solved=-1,
                avg_pos_error_to_v_mcts=-1,
                avg_pos_error_to_solved=-1,
                avg_critic_for_solved=-1,
                avg_critic_for_non_solved=-1,
            )
        solved_samples = [s for s in samples if s.solved]
        non_solved_samples = [s for s in samples if not s.solved]
        pos_error_samples = [s for s in samples if (s.q_estimate - s.critic) >= 0]

        return PLSStats(
            proved=proved,
            minproof_size=minproof_size if minproof_size is not None else -1,
            critic_n_samples=len(samples),
            critic_prop_solved=np.mean([s.solved for s in samples]).item(),
            critic_prop_bad=np.mean([s.bad for s in samples]).item(),
            avg_abs_error_to_v_mcts=np.mean(
                [abs(s.q_estimate - s.critic) for s in samples]
            ).item(),
            avg_abs_error_to_solved=np.mean(
                [abs(float(s.solved) - s.critic) for s in samples]
            ).item(),
            avg_pos_error_to_v_mcts=np.mean(
                [(s.q_estimate - s.critic) for s in pos_error_samples]
            ).item()
            if pos_error_samples
            else -1,
            avg_pos_error_to_solved=np.mean(
                [float(s.solved) - s.critic for s in solved_samples]
            ).item()
            if solved_samples
            else -1,
            avg_critic_for_solved=np.mean([s.critic for s in solved_samples]).item()
            if solved_samples
            else -1,
            avg_critic_for_non_solved=np.mean(
                [s.critic for s in non_solved_samples]
            ).item()
            if non_solved_samples
            else -1,
        )
