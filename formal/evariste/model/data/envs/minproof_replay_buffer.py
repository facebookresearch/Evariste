# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, unique
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Generic, TypeVar, Optional, Any, Tuple

import numpy as np
import torch
from numpy.random import RandomState

from evariste.backward.prover.mcts_samples import MCTSSampleTactics
from evariste.backward.remote.state_dump_utils import StateDumper
from evariste.clusters.utils import clusterify_path
from evariste.comms.store import Receiver
from evariste.datasets import DatasetConf
from evariste.forward.training.helpers import sample_from_cumulative
from evariste.model.data.envs.mcts_loader_common import MCTSDataLoader
from params import Params

logger = getLogger()


@dataclass
class MCTSSampleProof:
    label: str
    size: float
    stype: str
    samples: List[MCTSSampleTactics]


@unique
class LabelSampling(str, Enum):
    Uniform = "uniform"
    MinSize = "min_size"  # proportional to minproof size


@dataclass
class MinProofReplayBufferCfg(Params):
    n_proofs_by_label: int = 4
    min_recv_proofs: int = 10  # low for send_to_all = False
    label_sampling: str = LabelSampling.Uniform
    # given a label we sample a proof with prob prop to
    # exp( - beta * (size - avg_size_label) / std_size_label))
    # beta = 0 is uniform distribution
    proof_sampling_beta: float = 0.0
    # deduplication of proofs by using all the samples in MMSampleProof (even if
    # MMSampleProof.samples represent an ensemble of proofs and not a single proof)
    deduplicate_proofs: bool = False
    split_props_str: str = ""  # "train:1,valid:1"

    # dump
    should_dump: bool = True
    dump_interval: float = 3600.0
    max_dumps: int = 20

    # "path/to/state.xxx"
    # or "train:path/to/train/state.xxx,valid:path/to/valid/state.xxx" if using split
    # probs
    reload_str: str = ""

    def _check_and_mutate_args(self):
        if self.reload_paths is not None:
            if self.split_props is not None:
                splits_a, _ = self.reload_paths
                splits_b, _ = self.split_props
                assert set(splits_a) == set(splits_b), (splits_a, splits_b)
            else:
                splits, _ = self.reload_paths
                assert len(splits) == 1, splits
                assert splits[0] == ""

    @property
    def split_props(self) -> Optional[Tuple[List[str], List[float]]]:
        if self.split_props_str == "":
            return None
        splits, props = zip(*[x.split(":") for x in self.split_props_str.split(",")])
        assert len(set(splits)) == len(splits)
        return splits, [float(p) for p in props]

    @property
    def reload_paths(self) -> Optional[Tuple[List[str], List[Path]]]:
        if self.reload_str == "":
            return None
        if "," not in self.reload_str:
            splits = [""]
            paths = [self.reload_str]
        else:
            splits, paths = zip(*[x.split(":") for x in self.reload_str.split(",")])
        assert len(set(splits)) == len(splits)
        paths = [Path(clusterify_path(p)) for p in paths]
        assert all(p.name.startswith("state.") for p in paths)
        return splits, paths


Inp = TypeVar("Inp")
Out = TypeVar("Out")


class PriorityStore(Generic[Inp, Out], ABC):
    @abstractmethod
    def add(self, inp: Inp):
        pass

    @abstractmethod
    def sample(self, rng: RandomState) -> Out:
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def ready(self) -> bool:
        pass


@dataclass
class _LabelStat:
    # stats to compute get_stats or to be reloaded in notebook

    # last timestamp we updated the best proof
    last_best: int
    # last timestamp we update the k proofs with a new one
    last_new: int
    # last timestamp we received a proof
    last_recv: int
    n_proofs_recv: int = 1

    @staticmethod
    def create(cur_timestamp: int) -> "_LabelStat":
        return _LabelStat(
            last_recv=cur_timestamp, last_new=cur_timestamp, last_best=cur_timestamp
        )


class MinProofPriorityStore(PriorityStore[MCTSSampleProof, MCTSSampleTactics]):
    def __init__(
        self,
        cfg: MinProofReplayBufferCfg,
        name: str = "minproof_store",
        state_dumper: Optional[StateDumper] = None,
    ):
        self.cfg = cfg

        self.labels: List[str] = []
        self.labels_to_idx: Dict[str, int] = {}
        self.data: List[List[MCTSSampleProof]] = []
        self.name = name
        self.recv_proofsteps = 0
        self.recv_proofs = 0

        self.last_log_ready = time.time()

        assert cfg.should_dump == (state_dumper is not None)
        self.last_dump = time.time()
        self.state_dumper = state_dumper

        self.time_in_dumping: float = 0

        self.cum_weights: np.ndarray = np.array([])

        # stats on when received last best minproof by label
        self.labels_stats: Dict[str, _LabelStat] = {}

    def add(self, inp: MCTSSampleProof):
        cur_timestamp = self._cur_timestamp()

        if inp.label.startswith("eq_bwd_"):
            raise RuntimeError(
                f"Not implemented for generated data! (label: {inp.label})"
            )
        if inp.label not in self.labels_to_idx:
            idx = len(self.labels)
            self.labels.append(inp.label)
            self.labels_to_idx[inp.label] = idx
            self.labels_stats[inp.label] = _LabelStat.create(cur_timestamp)
            assert len(self.data) == idx
            self.data.append([])
        else:
            idx = self.labels_to_idx[inp.label]

        self.recv_proofs += 1
        self.recv_proofsteps += len(inp.samples)

        proofs = self.data[idx]

        # update stats for label
        label_stats = self.labels_stats[inp.label]
        label_stats.n_proofs_recv += 1
        label_stats.last_recv = cur_timestamp
        if len(proofs) == 0 or inp.size < proofs[0].size:
            label_stats.last_best = cur_timestamp
        if len(proofs) == 0 or inp.size <= proofs[-1].size:
            label_stats.last_new = self.recv_proofs

        # TODO: handle duplicate
        proofs.insert(0, inp)  # adding at the beginning to remove old one in priority

        if self.cfg.deduplicate_proofs:
            # deduplicate by keeping order (received proof at the beginning)
            present = set()
            new_proofs = []
            for proof in proofs:
                key = tuple(sorted((s.goal, tuple(s.tactics)) for s in proof.samples))
                if key in present:
                    continue
                present.add(key)
                new_proofs.append(proof)
            proofs = new_proofs

        proofs = sorted(proofs, key=lambda p: p.size)[: self.cfg.n_proofs_by_label]
        self.data[idx] = proofs

        self._update_cum_weights()

    def sample(self, rng: RandomState) -> MCTSSampleTactics:
        proofs = self._sample_proofs(rng)
        assert len(proofs) > 0

        if self.cfg.proof_sampling_beta == 0.0:
            # uniform sampling
            pid = rng.randint(len(proofs))
        else:
            sizes = np.array([p.size for p in proofs])
            p = _normalized_softmax_probs(sizes, beta=self.cfg.proof_sampling_beta)
            pid = rng.choice(np.arange(len(proofs)), p=p)
        proof = proofs[pid]
        sample = proof.samples[rng.randint(len(proof.samples))]

        self._maybe_dump()

        return sample

    def ready(self) -> bool:
        ready = self._ready()
        if not ready and time.time() - self.last_log_ready > 60:
            self.last_log_ready = time.time()
            self._log(
                f"not ready, received "
                f"{self.recv_proofs}/{self.cfg.min_recv_proofs}"
                f"- stats: {self.get_stats()}"
            )
        return ready

    def _update_cum_weights(self):
        # naive recomputation for now
        if self.cfg.label_sampling == LabelSampling.Uniform:
            if len(self.data) > len(self.cum_weights):
                self.cum_weights = np.cumsum(np.ones(len(self.data)))
        elif self.cfg.label_sampling == LabelSampling.MinSize:
            minsizes = np.array([proofs[0].size for proofs in self.data])
            assert minsizes.min() > 0
            self.cum_weights = np.cumsum(minsizes)
        else:
            raise NotImplementedError(self.cfg.label_sampling)

    def _sample_proofs(self, rng: RandomState) -> List[MCTSSampleProof]:
        assert self.ready()
        assert (
            len(self.labels)
            == len(self.labels_to_idx)
            == len(self.data)
            == len(self.cum_weights)
        )
        return sample_from_cumulative(
            cumulative=self.cum_weights, data=self.data, rng=rng
        )

    def _ready(self) -> bool:
        """Use this function for get_stats() to avoid logging on other
        dataloader processes"""
        return self.recv_proofs >= self.cfg.min_recv_proofs

    def _cur_timestamp(self) -> int:
        return self.recv_proofs

    def get_stats(self) -> Dict[str, float]:
        if self.recv_proofs == 0:
            return {}

        minsizes = [proofs[0].size for proofs in self.data]

        all_last_best = np.array([s.last_best for s in self.labels_stats.values()])

        # delta with now
        avg_last_best_delta = np.mean(self._cur_timestamp() - all_last_best).item()
        min_last_best_delta = np.min(self._cur_timestamp() - all_last_best).item()

        stats = {
            "n_labels": len(self.labels),
            "n_recv_proofs": self.recv_proofs,
            "n_recv_proofsteps": self.recv_proofsteps,
            "n_proof_steps_in_store": sum(
                [len(p.samples) for proofs in self.data for p in proofs]
            ),
            "time_in_dumping": self.time_in_dumping,
            "avg_minsize": np.mean(minsizes).item(),
            "max_minsize": np.max(minsizes).item(),
            "median_minsize": np.median(minsizes).item(),
            "avg_found_best_delta": avg_last_best_delta,
            "min_found_best_delta": min_last_best_delta,
            "ready": self._ready(),
        }
        stats = {k: float(v) for k, v in stats.items()}
        return stats

    def _log(self, msg: str):
        return logger.info(f"[{self.name} (PID: {os.getpid()})] {msg}")

    def _maybe_dump(self):
        if self.state_dumper is None:
            return
        if time.time() - self.last_dump > self.cfg.dump_interval:
            start = time.time()
            self._log("dumping state")
            self.state_dumper.maybe_dump_state(
                {
                    "labels": self.labels,
                    "data": self.data,
                    "labels_to_idx": self.labels_to_idx,
                    "recv_proofs": self.recv_proofs,
                    "recv_proofsteps": self.recv_proofsteps,
                    "labels_stats": self.labels_stats,
                }
            )
            self.last_dump = time.time()
            self.time_in_dumping += self.last_dump - start

    def restore_from_state_dump(self):
        assert self.state_dumper is not None
        present = self.state_dumper.get_present()
        if not present:
            return
        path = present[-1]
        with path.open("rb") as fp:
            state = pickle.load(fp)

        self.labels = state["labels"]
        self.data = state["data"]
        self.labels_to_idx = state["labels_to_idx"]
        self.recv_proofs = state["recv_proofs"]
        self.recv_proofsteps = state["recv_proofsteps"]
        self.labels_stats = state["labels_stats"]
        self._update_cum_weights()

    def reload_dump(self, dump_path: Path):
        assert dump_path.exists()
        with dump_path.open("rb") as fp:
            state = pickle.load(fp)
        data: List[List[MCTSSampleProof]] = state["data"]
        self._log(
            f"Going to reload proofs for {len(data)} labels"
            f" ({sum(len(p) for p in data)} proofs in total)"
        )
        for label_proofs in data:
            for proof in reversed(label_proofs):
                self.add(inp=proof)
        self._reset_label_stats()

    def _reset_label_stats(self):
        self.labels_stats = {
            label: _LabelStat.create(cur_timestamp=0) for label in self.labels
        }
        for stats in self.labels_stats.values():
            stats.n_proofs_recv = 0


class MultiSplitStore(PriorityStore[MCTSSampleProof, MCTSSampleTactics]):
    def __init__(
        self,
        label_to_split: Dict[str, str],
        splits: List[str],
        probs: List[float],
        stores: List[MinProofPriorityStore],
    ):
        self.splits = splits
        self.probs = np.array(probs)
        self.stores = stores

        assert len(set(self.splits)) == len(splits)
        assert np.isclose(self.probs.sum(), 1.0)
        assert len(self.probs) == len(self.splits) == len(self.stores)
        assert {s for s in label_to_split.values()} == set(self.splits), (
            {s for s in label_to_split.values()},
            set(self.splits),
        )

        split_to_idx = {s: idx for idx, s in enumerate(self.splits)}
        self.label_to_idx = {l: split_to_idx[s] for l, s in label_to_split.items()}

    def sample(self, rng: RandomState) -> Out:
        idx = rng.choice(np.arange(len(self.splits)), p=self.probs)
        return self.stores[idx].sample(rng)

    def add(self, inp: MCTSSampleProof):
        idx = self.label_to_idx[inp.label]
        return self.stores[idx].add(inp)

    def ready(self) -> bool:
        return all(s.ready() for s in self.stores)

    def get_stats(self) -> Dict[str, float]:
        stats = {}
        for split, store in zip(self.splits, self.stores):
            stats.update({f"{split}/{k}": v for k, v in store.get_stats().items()})
        return stats

    def reload_dump(self, splits: List[str], dump_paths: List[Path]):
        assert set(splits) == set(self.splits)
        split_to_path = {s: p for s, p in zip(splits, dump_paths)}
        for i, split in enumerate(self.splits):
            path = split_to_path[split]
            logger.info(f"Reloading split: {split} with dump_path: {path}")
            self.stores[i].reload_dump(path)


def _normalized_softmax_probs(sizes: np.ndarray, beta: float) -> np.ndarray:
    if len(sizes) == 1:
        return np.ones_like(sizes, dtype=np.float_)
    assert len(sizes) > 1
    normalized_sizes = (sizes - sizes.mean()) / (sizes.std() + 1e-5)
    return torch.softmax(torch.from_numpy(-beta * normalized_sizes), dim=-1).numpy()


class MCTSReplayBuffer(Generic[Inp, Out], MCTSDataLoader):
    """
    TODO: I would like that other ReplayBuffer and different ZMQDataLoader,
      BaseMCTSDataLoader, SubProofDataLoader use this class to simplify the taxonomy
      (specific mcts reloading  / adversarial logic could be contained in
      the PriorityStore abstraction) - in this case I should change the interface
      for the main method to 'get_sample' and rename class to something like
      ReplayBuffer
    """

    def __init__(
        self,
        receiver: Receiver[Inp],
        priority_store: PriorityStore[Inp, Out],
        name: Optional[str] = None,
        mcts_fmt: Optional[str] = None,
    ):
        self.receiver = receiver
        self.priority_store = priority_store
        self.name = name if name is not None else self.__class__.__name__

        self.last_log_sample = time.time()

        # stats
        self.received_samples = 0
        self.wait_time_before_start = 0
        self.time_in_receiver: float = 0
        self.time_in_adding: float = 0
        self.time_in_sampling: float = 0

        self._mcts_fmt = mcts_fmt

    def get_mcts_sample(
        self, split: str, index: Optional[int], rng: np.random.RandomState,
    ) -> Out:
        assert split == "train"
        assert index is None
        return self._get_sample(rng)

    def _log(self, msg: str):
        return logger.info(f"[{self.name} (PID: {os.getpid()})] {msg}")

    def _get_sample(self, rng: np.random.RandomState) -> Out:
        first = True
        start = time.time()
        was_ready = self.priority_store.ready()
        while not self.priority_store.ready() or first:
            start_revc = time.time()
            received = self.receiver.receive_batch()
            self.time_in_receiver += time.time() - start_revc
            start_add = time.time()
            for rec in received:
                self.received_samples += 1
                self.priority_store.add(rec)
            self.time_in_adding += time.time() - start_add
            first = False
            if not self.priority_store.ready():
                self.wait_time_before_start = time.time() - start
                time.sleep(0.001)
            elif not was_ready:
                self.wait_time_before_start = time.time() - start
                self._log("rb ready!")

        assert self.priority_store.ready()
        start_sampling = time.time()
        sample = self.priority_store.sample(rng)
        self.time_in_sampling += time.time() - start_sampling

        if time.time() - self.last_log_sample > 60:  # change this when working
            self.last_log_sample = time.time()
            self._log(f"stats: {self.get_stats()}")

        return sample

    def get_stats(self) -> Dict[str, float]:
        stats = {
            "received_samples": self.received_samples,
            "wait_time_before_start": self.wait_time_before_start,
            "time_in_sampling": self.time_in_sampling,
            "time_in_adding": self.time_in_adding,
            "time_in_receiver": self.time_in_receiver,
        }
        store_stats = self.priority_store.get_stats()
        assert set(store_stats.keys()).isdisjoint(set(stats.keys()))
        stats.update(store_stats)
        return stats

    def close(self):
        self.receiver.close()

    @property
    def mcts_fmt(self) -> Optional[str]:
        return self._mcts_fmt


class MinProofMCTSReplayBuffer(MCTSReplayBuffer[MCTSSampleProof, MCTSSampleTactics]):
    @staticmethod
    def from_trainer_args(
        params: Any, mcts_task: str, dataset: DatasetConf
    ) -> "MinProofMCTSReplayBuffer":
        from evariste.trainer.args import TrainerArgs  # circular import TrainerArgs
        from evariste.model.data.envs.mcts_loader import (
            mcts_subtask_receiver_factory,
        )  # circular import TrainerArgs

        assert isinstance(params, TrainerArgs)

        receiver = mcts_subtask_receiver_factory(params, mcts_subtask="minproof")
        cfg = params.mcts_train.minproof_rb

        if cfg.split_props is None:
            store = _make_store(params, cfg)
            if cfg.reload_paths:
                (split,), (path,) = cfg.reload_paths
                assert split == ""
                store.reload_dump(dump_path=path)
        else:
            splits, props = cfg.split_props
            label_to_split: Dict[str, str] = {}
            for split in splits:
                # circular imports TrainerArgs
                from evariste.backward.goal_factory import get_labels

                labels = get_labels(dataset=dataset, split=split)
                print(f"debug: labels {split}: {len(labels)}")
                for label in labels:
                    assert label not in label_to_split, (
                        label,
                        split,
                        label_to_split[label],
                    )
                    label_to_split[label] = split
            probs = list(np.array(props) / sum(props))
            logger.info(
                f"Creating MultiSplitStore with splits: {splits} and probs: {probs}"
            )
            store = MultiSplitStore(
                splits=splits,
                probs=probs,
                stores=[_make_store(params, cfg, split=split) for split in splits],
                label_to_split=label_to_split,
            )
            if cfg.reload_paths is not None:
                store.reload_dump(*cfg.reload_paths)

        assert "mcts_minproof" in mcts_task
        s = mcts_task.split("_")  # lang, mcts, tactic, str_fmt
        assert len(s) == 4, mcts_task
        mcts_fmt = s[3]

        return MinProofMCTSReplayBuffer(
            receiver=receiver,
            priority_store=store,
            name="minproof_rb",
            mcts_fmt=mcts_fmt,
        )


def _make_store(
    params, cfg: MinProofReplayBufferCfg, split: Optional[str] = None,
) -> MinProofPriorityStore:
    from evariste.trainer.args import TrainerArgs

    assert isinstance(params, TrainerArgs)
    postfix = f"_{split}" if split is not None else ""
    if cfg.should_dump:
        state_dumper = StateDumper(
            folder=Path(params.dump_path) / f"minproof_rb_dumps{postfix}",
            n_states_max=cfg.max_dumps,
            is_master=params.slurm_conf.is_master,
        )
    else:
        state_dumper = None

    return MinProofPriorityStore(
        cfg=params.mcts_train.minproof_rb,
        state_dumper=state_dumper,
        name=f"minproof_store{postfix}",
    )
