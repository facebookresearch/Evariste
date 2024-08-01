# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Union, Tuple, List, Dict
from collections import defaultdict
from logging import getLogger
from threading import Thread
from pathlib import Path
import os
import math
import time
import pynvml
import shutil
import numpy as np
import torch


logger = getLogger()


Number = Union[float, int]


def logmeanexp(values: List[float]) -> float:
    """
    Returns log (sum_{i=1}^N exp(v_i)) / N for v_i in values.
    Used for averaging probabilities given in log-form.
    """
    return torch.logsumexp(torch.tensor(values), 0).item() - math.log(len(values))


class WeightedAvgStats:
    """provides an average over a bunch of stats"""

    def __init__(self):
        self.raw_stats: Dict[str, float] = defaultdict(float)
        self.total_weights: Dict[str, float] = defaultdict(float)

    def update(self, vals: Dict[str, Tuple[Number, Number]]) -> None:
        for key, (value, weight) in vals.items():
            self.raw_stats[key] += value * weight
            self.total_weights[key] += weight

    @property
    def non_zero_stats(self) -> Dict[str, float]:
        return {x: y for x, y in self.raw_stats.items() if self.total_weights[x] > 0}

    @property
    def stats(self) -> Dict[str, float]:
        return {x: y / self.total_weights[x] for x, y in self.non_zero_stats.items()}

    @property
    def tuple_stats(self) -> Dict[str, Tuple[float, float]]:
        return {
            x: (y / self.total_weights[x], self.total_weights[x])
            for x, y in self.non_zero_stats.items()
        }

    def reset(self) -> None:
        self.raw_stats = defaultdict(float)
        self.total_weights = defaultdict(float)


class HistStats:
    """Histogram stats, which cumulates stats into a global histogram and output usage and entropy."""

    def __init__(self):
        self.hist = {}

    def update(self, vals):
        for x, y in vals.items():
            if len(y) == 0:
                continue
            if isinstance(y, list):
                y = np.array(y)
            assert type(y) == np.ndarray, y
            if x not in self.hist:
                self.hist[x] = y
            else:
                assert len(y) == len(self.hist[x]), print(y, self.hist[x])
                self.hist[x] += y

    @property
    def stats(self):
        stats = {}
        for x, y in self.hist.items():
            usage, entropy = compute_usage_entropy(y)
            stats.update({f"{x}_usage": usage, f"{x}_entropy": entropy})
        return stats

    def reset(self):
        self.hist = {}


class GPUMonitor(Thread):
    def __init__(self, delay: float):
        super().__init__()
        self.delay = delay
        self.stopped = False
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        self.handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)
        ]
        self.stats = [WeightedAvgStats() for _ in range(device_count)]
        self.last_time = time.time()
        self.start()

    def update_stats(self) -> None:
        time_delta = time.time() - self.last_time
        for i, h in enumerate(self.handles):
            utilization = pynvml.nvmlDeviceGetUtilizationRates(h)
            memory = pynvml.nvmlDeviceGetMemoryInfo(h)
            mem = 100.0 * memory.used / memory.total
            self.stats[i].update(
                {"mem": (mem, time_delta), "gpu": (utilization.gpu, time_delta)}
            )
        self.last_time = time.time()

    @property
    def main_stats(self) -> Dict[str, float]:
        return self.stats[torch.cuda.current_device()].stats

    def run(self) -> None:
        self.last_time = time.time()
        while not self.stopped:
            self.update_stats()
            time.sleep(self.delay)

    def close(self):
        self.stopped = True
        self.join()
        return [s.tuple_stats for s in self.stats]


def set_MKL_env_vars():
    for k in ["MKL_THREADING_LAYER", "MKL_SERVICE_FORCE_INTEL"]:
        print(f"{k}: {os.environ.get(k)}")
        if os.environ.get(k) is None:
            print(f"Setting {k} to GNU")
            os.environ[k] = "GNU"


def copy_model(src_path: Path, tgt_dir: Path, hard_copy: bool = True) -> Path:
    """
    Copy the checkpoint to be evaluated immediately to prevent it from
    being deleted. Optionally just create a symlink to save disk usage.
    """
    if not src_path.is_file():
        raise FileNotFoundError(f"No model found in {src_path}")
    tmp_path = tgt_dir / "tmp_checkpoint.-1.pth"
    tgt_path = tgt_dir / "checkpoint.-1.pth"

    # copy the model, or create a simple symlink
    if not tgt_path.exists():
        os.makedirs(tgt_dir, exist_ok=True)
        if hard_copy:
            logger.info(f"Copying model from {src_path} to {tgt_path} ...")
            shutil.copy(src_path, tmp_path)
            os.rename(tmp_path, tgt_path)
        else:
            logger.info(f"Linking model from {src_path} to {tgt_path} ...")
            os.symlink(src_path, tgt_path)
    return tgt_path


def compute_usage_entropy(counts: np.ndarray) -> Tuple[float, float]:
    assert isinstance(counts, np.ndarray)
    # compute usage
    usage = (counts != 0).sum() / len(counts)
    # compute entropy
    if counts.sum() > 0:
        p = counts / counts.sum()
        p[p == 0] = 1
        entropy = -(np.log(p) * p).sum()
        return usage, entropy
    else:
        return 0.0, 0.0
