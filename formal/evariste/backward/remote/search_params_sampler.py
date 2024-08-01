# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple, Optional, List, Dict
from dataclasses import asdict
from enum import unique, Enum
from copy import deepcopy
import math

import numpy as np
import nevergrad as ng

from evariste import json as json
from evariste.backward.graph import GoalParams
from evariste.logger import create_logger

logger = create_logger(None)


@unique
class HyperOptKind(str, Enum):
    Fixed = "fixed"
    Nevergrad = "nevergrad"
    Random = "random"


class HyperOptParams:
    UniformRange = {"length_penalty", "temperature"}
    LogRange = {"exploration", "n_expansions"}
    Range = UniformRange | LogRange
    FloatChoice = {"depth_penalty", "policy_temperature"}
    IntChoice = {"n_samples", "succ_expansions"}
    MultinomialChoice = {
        "beam",
        "policy",
        "policy_temp_level",
        "q_value_solved",
        "tactic_fill",
    }
    Choice = FloatChoice | IntChoice | MultinomialChoice
    WorkerParams = {"beam"}


def _parse_hyperopt_params(hyperopt_param_str: str):
    ng_goal_params = ng.p.Dict()
    ng_worker_params = ng.p.Dict()
    if not hyperopt_param_str:
        return ng_goal_params, ng_worker_params

    # parse hyper-opts
    params = json.loads(hyperopt_param_str.replace("'", '"'))

    # sanity check
    allowed_values: Dict[str, List] = {
        "beam": [True, False],
        "policy": ["alpha_zero", "other"],
        "policy_temp_level": ["global", "simutree", "node"],
        "q_value_solved": [2, 3, 4, 5],
        "tactic_fill": ["all", "errors", "none"],
    }
    for name, values in allowed_values.items():
        if name in params:
            assert all(v in values for v in params[name]), (name, params[name])

    for name, values in params.items():
        if name in HyperOptParams.WorkerParams:
            ng_params = ng_worker_params
        else:
            ng_params = ng_goal_params
        if name in HyperOptParams.Range:
            assert len(values) == 2, values
            assert values[0] <= values[1]
            if name in HyperOptParams.UniformRange:
                ng_params[name] = ng.p.Scalar(lower=values[0], upper=values[1])
            else:
                assert name in HyperOptParams.LogRange
                assert all(v > 0 for v in values)
                ng_params[name] = ng.p.Log(lower=values[0], upper=values[1])
        elif name in HyperOptParams.Choice:
            assert len(values) > 0
            ng_params[name] = ng.p.Choice(values)
        else:
            raise NotImplementedError(f"Hyper opt param {name} is not implemented.")
    return ng_goal_params, ng_worker_params


def maybe_make_search_params_samplers(
    hyperopt: HyperOptKind,
    hyperopt_param_str: str,
    n_machines: int,
    n_simultaneous_proofs: int,
) -> Tuple[Optional["GoalParamsSampler"], Optional["WorkerParamsSampler"]]:
    if hyperopt == HyperOptKind.Fixed:
        return None, None
    goal_params, worker_params = _parse_hyperopt_params(hyperopt_param_str)
    opt_class = {
        HyperOptKind.Nevergrad: ng.optimizers.OptimisticNoisyOnePlusOne,
        HyperOptKind.Random: ng.optimizers.RandomSearch,
    }[hyperopt]
    ng_goal_optimizer = opt_class(
        parametrization=goal_params,
        budget=None,
        num_workers=n_machines * n_simultaneous_proofs,
    )
    goal_params_sampler = GoalParamsSampler(ng_goal_optimizer=ng_goal_optimizer)
    worker_params_sampler = None
    if len(goal_params) + len(worker_params) > 0:
        # we also add goal_params in parametrization for
        # the setup with dynamic n_samples and beam=True
        ng_worker_optimizer = ng.optimizers.RandomSearch(
            parametrization=ng.p.Dict(**goal_params, **worker_params),
            budget=None,
            num_workers=n_machines,
        )
        if hyperopt == HyperOptKind.Nevergrad:
            logger.warning(
                f"Hyper opt worker params {list(worker_params.keys())} is Random"
            )
        worker_params_sampler = WorkerParamsSampler(ng_worker_optimizer)
    return goal_params_sampler, worker_params_sampler


class GoalParamsSampler:
    def __init__(self, ng_goal_optimizer: ng.optimizers.base.Optimizer):
        self.ng_goal_optimizer = ng_goal_optimizer
        self.name_to_ng: Dict[str, ng.p.Parameter] = {}

    def sample_goal_params(self, goal_name: str) -> GoalParams:
        job_params = self.ng_goal_optimizer.ask()
        goal_params = GoalParams(**job_params.value)
        if goal_params.tactic_fill == "all":
            goal_params.length_penalty = 0.0
        assert goal_name not in self.name_to_ng, goal_name
        self.name_to_ng[goal_name] = job_params
        # print(f"sending key: {goal_name}, name_to_ng: {self.name_to_ng.keys()}")
        return goal_params

    def get_recommendation(self) -> Dict:
        return self.ng_goal_optimizer.provide_recommendation().value

    def update(self, goal_name: str, success: bool):
        # print(f"key: {goal_name}, name_to_ng: {self.name_to_ng.keys()}")
        assert goal_name in self.name_to_ng, goal_name
        # nevergrad minimizes, so 1 if failure, 0 otherwise
        self.ng_goal_optimizer.tell(self.name_to_ng.pop(goal_name), float(not success))


class WorkerParamsSampler:
    def __init__(self, ng_worker_optimizer: ng.optimizers.base.Optimizer):
        self.ng_worker_optimizer = ng_worker_optimizer

    def sample_prover_params(self, params, verbose: bool = True):
        # circular imports
        from evariste.backward.prover.zmq_prover import ZMQProverParams

        assert isinstance(params, ZMQProverParams)

        new_params = deepcopy(params)
        ng_params = self.ng_worker_optimizer.ask()
        if "beam" in ng_params.value:
            new_params.decoding.use_beam = ng_params.value["beam"]
        if "n_samples" in ng_params.value and new_params.decoding.use_beam:
            new_params.decoding.n_samples = ng_params.value["n_samples"]
        if verbose:
            logger.info(f"Starting prover with decoding params: {new_params.decoding}")
        return new_params

    @staticmethod
    def rebuild_goal_and_worker_params(
        goal_params: Optional[GoalParams], use_beam: bool, n_samples: int
    ) -> Tuple[Dict, Dict]:
        goal_params_dict = (
            {k: v for k, v in asdict(goal_params).items() if v is not None}
            if goal_params is not None
            else {}
        )  # removing the None
        worker_params_dict = {"beam": use_beam, "n_samples": n_samples}
        if use_beam:
            goal_params_dict.pop("n_samples", None)  # remove if there
        else:
            worker_params_dict.pop("n_samples", None)  # if beam = False, n_samples is
            # set in goal_params
        return goal_params_dict, worker_params_dict


class MCTSSolvingStats:
    def __init__(self, hyperopt_param_str: str, n_buckets: int):
        """
        Accumulate and save all statistics about solved goals.
        """
        self.n_buckets = n_buckets
        assert n_buckets > 1

        # retrieve params
        if len(hyperopt_param_str) > 0:
            self.all_params = json.loads(hyperopt_param_str.replace("'", '"'))
        else:
            self.all_params = {}

        # stats
        self.solved = {}
        self.total = {}

        # buckets for params
        self.bucket_ranges: Dict = {}
        for name, values in self.all_params.items():
            if name in HyperOptParams.Choice:
                assert len(values) > 0
                self.solved[name] = {v: 0 for v in values}
                self.total[name] = {v: 0 for v in values}
                continue
            assert name in HyperOptParams.Range
            assert len(values) == 2
            a, b = values
            if name in HyperOptParams.UniformRange:
                r = np.linspace(a, b, n_buckets + 1)
            else:
                assert name in HyperOptParams.LogRange
                r = np.logspace(math.log2(a), math.log2(b), n_buckets + 1, base=2)
            self.bucket_ranges[name] = r
            self.solved[name] = {v: 0 for v in range(n_buckets)}
            self.total[name] = {v: 0 for v in range(n_buckets)}

        print(f"HyperOpt params: {self.all_params}")
        print(f"Bucket ranges: {self.bucket_ranges}")

    def get_bucket(self, param_name: str, param_value):
        """
        Use a bucket for values sampled within in range (uniform or logspace).
        """
        # there is a limited number of possible values
        if param_name not in self.bucket_ranges:
            assert param_value in self.all_params[param_name], (param_name, param_value)
            return param_value

        # the value has been sampled in a range, retrieve associated bucket
        bucket_id = np.searchsorted(self.bucket_ranges[param_name], param_value)
        assert 0 <= bucket_id <= self.n_buckets + 1
        bucket_id = min(max(bucket_id, 1), self.n_buckets)  # in case we sampled a or b
        bucket_id -= 1
        assert 0 <= bucket_id < self.n_buckets
        return bucket_id

    def update(self, solved: bool, g_params: Dict, w_params: Dict) -> None:
        assert type(solved) is bool

        # update goal / worker params
        for k, v in list(g_params.items()) + list(w_params.items()):
            if k not in self.all_params:
                assert k in HyperOptParams.WorkerParams, k
                continue
            self.solved[k][self.get_bucket(k, v)] += int(solved)
            self.total[k][self.get_bucket(k, v)] += 1

    def get_stats(self) -> Dict[str, float]:
        res = {}
        for pname, pvalues in self.all_params.items():
            is_range = pname in self.bucket_ranges
            values = range(self.n_buckets) if is_range else pvalues
            for value in values:
                if is_range:
                    vrange = self.bucket_ranges[pname]
                    vname = f"{vrange[value]:.2f}_{vrange[value + 1]:.2f}"
                else:
                    vname = value
                solved = self.solved[pname][value]
                total = self.total[pname][value]
                res[f"prop_solved__{pname}__{vname}"] = solved * 100.0 / max(total, 1)
        return res
