# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import abc
import math
from abc import ABC
from contextlib import AbstractContextManager, contextmanager
from typing import (
    Type,
    TypeVar,
    Optional,
    Dict,
    Union,
    Generic,
    Callable,
    Any,
    TextIO,
    List,
    Set,
)
from torch.utils.tensorboard import SummaryWriter
from dataclasses import fields, asdict, dataclass, field
from collections import defaultdict
from json import JSONDecodeError
from collections import Counter
from functools import wraps
from pathlib import Path
import sys
import time
import logging
import datetime
import numpy as np
import torch
import torch.utils.tensorboard

from evariste import json as json
from evariste.utils import tail

SimpleStatValue = Union[float, str, int]
StatValue = Union[SimpleStatValue, Dict[str, SimpleStatValue], Dict[str, float]]


def _sanitize(value):
    if isinstance(value, torch.Tensor):
        return value.detach().item()
    return value


def _sanitize_before_dump(value):
    assert type(value) == np.ndarray
    return value.tolist()


def _restore_hist_to_values(hist: np.ndarray) -> np.ndarray:
    assert type(hist) == np.ndarray
    s = hist.sum()
    if s == 0:
        return np.array([])
    normalized = hist / s
    to_plot = normalized / normalized[normalized > 0].min()
    values = []
    for i in range(len(to_plot)):
        values += [i for _ in range(int(to_plot[i]))]
    return np.array([values])


def _retrieve_last_global_step(path: Path) -> int:
    """
    Retrieve last global step from file.

    """
    assert path.is_file()
    last_lines = tail(path, n=10)
    if len(last_lines) == 0:
        return 0
    for line in last_lines[::-1]:
        try:
            global_step = json.loads(line)["global_step"]
            return global_step
        except JSONDecodeError:
            print(
                f'Couldn\'t reload line "{line}" from {path}',
                file=sys.stderr,
                flush=True,
            )
        except KeyError:
            print(
                f'Couldn\'t reload global step from "{line}" in {path}',
                file=sys.stderr,
                flush=True,
            )
    return 0


class Logger:
    def __init__(
        self,
        outdir,
        tag: Optional[str] = None,
        quiet: bool = False,
        only_jsonl: bool = False,
    ):
        self.quiet = quiet
        self.only_jsonl = only_jsonl
        self.global_step = 0
        self.closed = False
        self.outdir = outdir
        self.tag = tag
        self.writer: Optional[SummaryWriter] = None
        self.jsonl_writer: Optional[TextIO] = None
        if not self.quiet:
            assert outdir is not None
            outdir = Path(outdir)
            if not only_jsonl:
                self.writer = SummaryWriter(log_dir=outdir / "tb")
            fname = f"metrics.{tag}.jsonl" if tag else "metrics.jsonl"
            fpath = outdir / fname

            # retrieve global step from existing file (if any)
            if fpath.exists():
                self.global_step = _retrieve_last_global_step(fpath)

            self.jsonl_writer = open(fpath, "a")

    def log_config(self, cfg):
        if self.writer is None:
            return
        key = "cfg" if self.tag is None else f"{self.tag}/cfg"
        self.writer.add_text(key, str(cfg))

    def log_metrics(self, metrics: Dict[str, Any], sanitize: bool = False):
        if self.quiet:
            return
        if sanitize:
            metrics = {k: _sanitize(v) for k, v in metrics.items()}
        for key, value in metrics.items():
            if type(value) is str:
                print(f'WARNING: not logging "{key}" of type {type(value)}: {value}')
            elif self.writer is not None:
                assert not isinstance(value, dict), f"{key} {value}"
                key = key if self.tag is None else f"{self.tag}/{key}"
                self.writer.add_scalar(key, value, global_step=self.global_step)
        created_at = datetime.datetime.utcnow().isoformat()
        json_metrics = dict(global_step=self.global_step, created_at=created_at)
        json_metrics.update(metrics)
        print(json.dumps(json_metrics), file=self.jsonl_writer, flush=True)
        self.global_step += 1

    def log_histograms(
        self, hist: Dict[str, Any], restore_hist_to_values: bool = False
    ):
        if self.quiet:
            return
        if restore_hist_to_values:
            hist = {k: _restore_hist_to_values(v) for k, v in hist.items()}
        if self.writer is not None:
            for key, value in hist.items():
                assert type(value) == np.ndarray
                key = key if self.tag is None else f"{self.tag}/{key}"
                try:
                    self.writer.add_histogram(key, value, global_step=self.global_step)
                except ValueError as e:
                    logging.warning(f"Ignoring histogram {key} = {value} --> {e}")
        created_at = datetime.datetime.utcnow().isoformat()
        json_metrics = dict(global_step=self.global_step, created_at=created_at)
        json_metrics.update(hist)
        json_metrics = {k: _sanitize_before_dump(v) for k, v in hist.items()}
        print(json.dumps(json_metrics), file=self.jsonl_writer, flush=True)
        self.global_step += 1

    def close(self):
        if self.quiet:
            return
        self.closed = True
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.jsonl_writer is not None:
            self.jsonl_writer.close()
            self.jsonl_writer = None


class _Stats(ABC):
    @abc.abstractmethod
    def rate_and_reset(self) -> StatValue:
        pass


class ActionCounter(_Stats):
    def __init__(self, name: str, is_rate: bool, silent: bool = False):
        self.name = name
        self.last_log = time.time()
        self.cum_count = 0.0
        self.cum_actions = 0
        self.total_actions = 0
        self.is_rate = is_rate
        self.silent = silent

    def reset(self) -> None:
        self.last_log = time.time()
        self.cum_count = 0
        self.cum_actions = 0

    def act(self, v: Union[int, float] = 1) -> None:
        self.cum_count += v
        self.cum_actions += 1
        self.total_actions += 1

    def rate(self) -> float:
        delta_t = time.time() - self.last_log
        if self.is_rate:
            rate = self.cum_count / delta_t
            stat_type = "rate/s"
        else:
            rate = 0.0 if self.cum_actions == 0 else self.cum_count / self.cum_actions
            stat_type = "avg"
        if not self.silent:
            logging.info(
                f"[Counter {self.name}] - {rate} ({stat_type}) | {self.total_actions} total"
            )
        return rate

    def rate_and_reset(self) -> float:
        rate = self.rate()
        self.reset()
        return rate


T = TypeVar("T")


class DataclassStats(Generic[T]):
    def __init__(self, schema: Type[T]):
        self.schema = schema
        self.counters: Dict[str, Union[ActionCounter, Counter]] = {}
        self.rate = ActionCounter(name="rate", is_rate=True, silent=True)
        self.reset_counters()

    def reset_counters(self):
        for field_ in fields(self.schema):
            if isinstance(field_.type(), (int, float)):
                self.counters[field_.name] = ActionCounter(
                    name=field_.name, is_rate=False, silent=True
                )
            elif field_.type == str:
                self.counters[field_.name] = Counter()

    def act(self, elem: T) -> None:
        for name, value in asdict(elem).items():  # type: ignore
            this_counter = self.counters.get(name, None)
            if this_counter is None:
                continue
            if isinstance(this_counter, ActionCounter):
                this_counter.act(value)
            elif isinstance(this_counter, Counter):
                this_counter[value] += 1
            else:
                raise RuntimeError(f"How could this happen ? {this_counter} // {name}")
        self.rate.act()

    def stats_and_reset(self) -> Dict[str, float]:
        res = {"rate": self.rate.rate_and_reset()}
        for name, counter in self.counters.items():
            if isinstance(counter, ActionCounter):
                res[name] = counter.rate_and_reset()
            elif isinstance(counter, Counter):
                for subname, count in dict(counter.most_common(5)).items():
                    res[f"{name}/{subname}"] = count
        return res


class Avg(ActionCounter):
    """
    Allow to take average of a value.
    """

    def __init__(self):
        super(Avg, self).__init__(name="unk", is_rate=False, silent=True)

    def stats_and_reset(self) -> float:
        return self.rate_and_reset()


class Rate(ActionCounter):
    """
    Allow to take average of a value.
    """

    def __init__(self):
        super(Rate, self).__init__(name="unk", is_rate=True, silent=True)

    def stats_and_reset(self) -> float:
        return self.rate_and_reset()


class AvgDict:
    """
    Allow to take average of dict of values.

    Init lazily on first value
    """

    def __init__(self):
        self._action_counters: Dict[str, ActionCounter] = {}
        self._initialized = False

    def init_with_value(self, value: Dict[str, Union[int, float]]):
        assert isinstance(value, dict)
        for key in value.keys():
            self._action_counters[key] = ActionCounter(
                name="unused", is_rate=False, silent=True
            )
        self._initialized = True

    def act(self, value: Dict[str, Union[int, float]]):
        if not self._initialized:
            self.init_with_value(value)

        assert set(value.keys()) == set(
            self._action_counters.keys()
        ), f"{set(value.keys())} != {set(self._action_counters.keys())}"
        for key, value_ in value.items():
            assert isinstance(value_, (int, float))
            self._action_counters[key].act(value_)

    def stats_and_reset(self) -> Dict[str, float]:
        if not self._initialized:
            return {}
        stats = {}
        for key, counter in self._action_counters.items():
            stats[key] = counter.rate_and_reset()
        return stats


class Timings(defaultdict):
    def __init__(self, log_method: Callable[[str], None]):
        defaultdict.__init__(self, list)
        self.log_method = log_method

    def log(self):
        self.log_method(f"Timings: {json.dumps(dict(self))}")


def timeit(method):
    @wraps(method)
    def timed_method(self, *args, **kwargs):
        if not hasattr(self, "timings"):
            self.timings = Timings(self.log)
        start = time.time()
        result = method(self, *args, **kwargs)
        self.timings[method.__name__].append((start, time.time() - start))
        return result

    return timed_method


def log_timeit(close_method):
    @wraps(close_method)
    def wrapped(self, *args, **kwargs):
        if hasattr(self, "timings"):
            self.timings.log()
        return close_method(self, *args, **kwargs)

    return wrapped


class _MaxOrMin(_Stats):
    def __init__(self, name: str, is_max: bool = True):
        self.name = name
        self.is_max = is_max
        self.max_or_min: Optional[float] = None
        self.reset()

    def reset(self):
        self.max_or_min = -math.inf if self.is_max else math.inf

    def act(self, value: float):
        assert self.max_or_min is not None
        if self.is_max:
            self.max_or_min = max(self.max_or_min, value)
        else:
            self.max_or_min = min(self.max_or_min, value)

    def rate_and_reset(self) -> float:
        max_ = self.max_or_min
        assert max_ is not None
        self.reset()
        return max_


class Max(_MaxOrMin):
    def __init__(self, name: str = ""):
        super().__init__(name, is_max=True)


class Min(_MaxOrMin):
    def __init__(self, name: str = ""):
        super().__init__(name, is_max=False)


class Count(_Stats):
    """
    Allow to take average of a value.
    """

    def __init__(self):
        self.count = 0

    def update(self, n: int = 1):
        self.count += n

    def set(self, n: int = 1):
        self.count = n

    def rate_and_reset(self) -> int:
        count = self.count
        self.count = 0
        return count


class Timer(_Stats):
    def __init__(self, cum_ratio: bool = False, overall_cum_time: bool = False):
        self._max: float = 0.0
        self._sum: float = 0.0
        self._count: int = 0
        self._start: Optional[float] = None

        self._add_overall_cum_time = overall_cum_time
        self._overall_cum_time: float = 0.0

        self.ratios_started = False
        self._add_cum_ratio = cum_ratio
        self._cum_ratio_start: Optional[float] = None
        self._overall_cum_ratio_start: Optional[float] = None

    def start(self):
        if self._is_started():
            raise RuntimeError("Timer was already started!")

        self._start = time.time()
        self._maybe_start_ratios()

    def start_if_not_started(self):
        if not self._is_started():
            self.start()

    def stop(self):
        if not self._is_started():
            raise RuntimeError("Timer was not started!")
        assert self._start is not None
        delta_t = time.time() - self._start
        self._start = None
        self.add_interval(delta_t)

    def stop_if_not_stopped(self):
        if self._is_started():
            self.stop()

    def _is_started(self):
        return self._start is not None

    def _maybe_start_ratios(self):
        if self.ratios_started:
            return

        if not self._add_cum_ratio:
            self.ratios_started = True
            return

        now = time.time()
        assert self._cum_ratio_start is None
        self._cum_ratio_start = now

        if self._add_overall_cum_time:
            assert self._overall_cum_ratio_start is None
            self._overall_cum_ratio_start = now

        self.ratios_started = True

    @contextmanager
    def timeit(self):
        self.start()
        try:
            yield
        finally:
            self.stop()

    def add_interval(self, interval: float):
        if self._is_started():
            raise RuntimeError("Timer was already started!")
        self._max = max(self._max, interval)
        self._sum += interval
        self._count += 1

    def rate_and_reset(self) -> Dict[str, float]:
        self._maybe_start_ratios()

        now = time.time()
        stats: Dict[str, float] = {}
        if self._count == 0:
            stats.update({"count": 0})
        else:
            stats.update(
                {
                    "count": self._count,
                    "avg_time": self._sum / self._count,
                    "max_time": self._max,
                    "cum_time": self._sum,
                }
            )

        if self._is_started():
            assert self._start is not None
            stats["current_elapsed"] = time.time() - self._start

        if self._add_overall_cum_time:
            self._overall_cum_time += self._sum
            stats["overall_cum_time"] = self._overall_cum_time

        if self._add_cum_ratio:
            assert self.ratios_started
            assert self._cum_ratio_start is not None
            stats["cum_ratio"] = self._sum / max(now - self._cum_ratio_start, 1e-4)
            self._cum_ratio_start = now
            if self._add_overall_cum_time:
                assert self._overall_cum_ratio_start is not None
                stats["overall_cum_ratio"] = self._overall_cum_time / max(
                    now - self._overall_cum_ratio_start, 1e-4
                )

        self._reset()
        return stats

    def _reset(self):
        self._max = 0.0
        self._sum = 0.0
        self._count = 0


class ActivityTracker(_Stats):
    def __init__(self):
        self.rates = defaultdict(Rate)
        self.tracked = set()

    def rate_and_reset(self) -> Dict[str, float]:
        rates = [y.rate_and_reset() for y in self.rates.values()]
        if len(rates) == 0:
            return {}
        self.rates = defaultdict(Rate)  # reset number of values in defaultdict
        return {
            "rate_min": np.min(rates),
            "rate_mean": np.mean(rates),
            "rate_max": np.max(rates),
            "n": len(rates),
            "n_total": len(self.tracked),
        }

    def act(self, id):
        self.rates[id].act()
        self.tracked.add(id)


@dataclass
class StatsCollection(_Stats):
    def __post_init__(self):
        dict_ = self._asdict()
        for k, v in dict_.items():
            assert isinstance(v, (_Stats, str, int, float)), f"{k}, {type(v)}"

    def rate_and_reset(self) -> Dict[str, SimpleStatValue]:
        dict_ = self._asdict()
        stats: Dict[str, SimpleStatValue] = {}
        for k, v in dict_.items():
            if not isinstance(v, _Stats):
                stats[k] = v
            else:
                inner_stats = v.rate_and_reset()
                if isinstance(inner_stats, dict):
                    for kk, vv in inner_stats.items():
                        stats[f"{k}/{kk}"] = vv
                else:
                    stats[k] = inner_stats
        return stats

    def _asdict(self):
        # not using as dict since recursive
        _dict = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if isinstance(getattr(self, field.name), _Stats)
        }
        return {k: v for k, v in _dict.items() if k != "__last_log"}

    def maybe_log(self, name: str, interval_s: float = 60.0) -> bool:
        now = time.time()
        if not hasattr(self, "__last_log"):
            setattr(self, "__last_log", now)
        if time.time() - getattr(self, "__last_log") > interval_s:
            stats = self.rate_and_reset()
            setattr(self, "__last_log", now)
            logging.info(f"{name}: {stats}")
            return True
        return False


@dataclass
class DistribStats(StatsCollection):
    avg: Avg = field(default_factory=lambda: Avg())
    max: Max = field(default_factory=lambda: Max())
    min: Min = field(default_factory=lambda: Min())
    count: Count = field(default_factory=lambda: Count())

    def update(self, value: float):
        self.avg.act(value)
        self.min.act(value)
        self.max.act(value)
        self.count.update(n=1)
