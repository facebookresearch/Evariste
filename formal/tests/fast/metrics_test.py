# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
from dataclasses import dataclass, field
from numbers import Number
from typing import cast

import numpy

from evariste import json
from evariste.metrics import Avg, AvgDict, Timer, StatsCollection, Max, Rate


def test_avg_single_value():
    avg = Avg()
    value = avg.stats_and_reset()
    assert isinstance(value, Number)
    for i in range(10):
        avg.act(i)

    value = avg.stats_and_reset()
    assert numpy.isclose(value, 4.5)

    for i in range(10, 20):
        avg.act(i)

    value = avg.stats_and_reset()
    assert numpy.isclose(value, 14.5)


def test_avg_dict():
    avg = AvgDict()
    value = avg.stats_and_reset()
    assert isinstance(value, dict)
    for i in range(10):
        item = {"1": i, "2": 2 * i}
        avg.act(item)

    res = avg.stats_and_reset()
    assert isinstance(res, dict)
    assert set(res.keys()) == {"1", "2"}
    assert numpy.isclose(res["1"], 4.5)
    assert numpy.isclose(res["2"], 9)

    for i in range(10, 20):
        item = {"1": i, "2": 2 * i}
        avg.act(item)

    res = avg.stats_and_reset()
    assert isinstance(res, dict)
    assert set(res.keys()) == {"1", "2"}
    assert numpy.isclose(res["1"], 14.5)
    assert numpy.isclose(res["2"], 29)


def test_timer():
    for overall in [True, False]:
        for ratio in [True, False]:
            print("overall", overall, "ratio", ratio)
            _test_timer(timer=Timer(overall_cum_time=overall, cum_ratio=ratio))
    # assert False


def _is_json_serializable(stats):
    json.dumps(stats)


def _test_timer(timer: Timer):
    stats = timer.rate_and_reset()
    _is_json_serializable(stats)

    for j in range(2):
        for i in range(10):
            timer.start()
            time.sleep(1e-2)
            timer.stop()
            time.sleep(1e-2)
        stats = timer.rate_and_reset()
        _is_json_serializable(stats)
        assert numpy.isclose(stats["avg_time"], 1e-2, atol=1e-2)
        assert numpy.isclose(stats["max_time"], 1e-2, atol=1e-2)
        assert numpy.isclose(stats["cum_time"], 1e-1, atol=5e-2)
        if "cum_ratio" in stats:
            assert numpy.isclose(stats["cum_ratio"], 0.5, atol=0.1)
        if "overall_cum_ratio" in stats:
            assert numpy.isclose(stats["overall_cum_ratio"], 0.5, atol=0.1)
    print(stats)


def test_stats_collection():
    @dataclass
    class Stats(StatsCollection):
        my_timer: Timer = field(default_factory=lambda: Timer())
        my_max: Max = field(default_factory=lambda: Max())
        my_avg: Avg = field(default_factory=lambda: Avg())
        my_rate: Rate = field(default_factory=lambda: Rate())

    stats = Stats()

    stats_ = stats.rate_and_reset()
    _is_json_serializable(stats_)

    for j in range(2):
        for i in range(10):
            stats.my_timer.start()
            stats.my_max.act(i)
            stats.my_avg.act(i)
            stats.my_timer.stop()
            stats.my_rate.act(1)
            stats.my_timer.add_interval(100.0)
        stats_ = stats.rate_and_reset()
        _is_json_serializable(stats_)
        assert stats_["my_avg"] == 4.5
        logged = stats.maybe_log(name="stats_test", interval_s=1e-3)
        assert logged == (j == 1)
        time.sleep(1e-2)
