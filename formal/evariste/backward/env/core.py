# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, List, Set, Dict
from collections import defaultdict
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import torch
import numpy as np

from evariste.backward.prover.prover_args import ProverParams
from evariste.backward.graph import Theorem, Tactic, Token, ProofId, BackwardGoal
from evariste.backward.env.worker import (
    ExpanderEnv,
    TacticJob,
    TacticJobResult,
    ModelExpansion,
)
from lean_cluster.instance import LeanInstanceDied
from evariste.metrics import ActionCounter
from evariste.model.data.dictionary import (
    TACTIC_FILL_WORD,
    TACTIC_PARSE_ERROR_WORD,
    TACTIC_ENV_ERROR_WORD,
)


DAG = Dict[str, List[str]]  # label to list of directly dependent labels


SPECIAL_TACTIC_TOKENS = {
    "fill": TACTIC_FILL_WORD,
    "parse_error": TACTIC_PARSE_ERROR_WORD,
    "env_error": TACTIC_ENV_ERROR_WORD,
}


def sum_log_probs(values: List[float]) -> float:
    """
    Log probs here come from the same distribution.
    So the output should be <= 0. We use a min for numerical instabilities.
    """
    assert len(values) > 0
    assert all(v <= 0 for v in values)
    if len(values) == 1:
        return values[0]
    else:
        return min(torch.logsumexp(torch.DoubleTensor(values), 0).item(), 0)


class InvalidTactic(Exception):
    pass


class EnvHasDied(Exception):
    pass


@dataclass
class PreEnvExpansion:
    theorem: Theorem
    log_critic: float
    tactic_tokens: List[List[str]]  # before env
    log_priors: List[float]
    job_ids: List[int]
    job_results: Dict[int, TacticJobResult]
    exp_duration: float
    gpu_duration: float

    def __post_init__(self):
        assert type(self.log_critic) is float and self.log_critic <= 0
        assert type(self.tactic_tokens) is list, self.tactic_tokens
        assert type(self.log_priors) is list, self.log_priors
        assert len(self.tactic_tokens) == len(self.log_priors)
        assert all(
            type(p) is float for p in self.log_priors
        ), self.log_priors  # and p <= 0
        assert self.log_priors == sorted(self.log_priors, reverse=True)
        assert len(self.job_ids) == len(self.job_results) == 0

    @staticmethod
    def from_model_expansion(theorem: Theorem, expansion: ModelExpansion):
        assert not expansion.is_error
        assert expansion.log_critic is not None
        assert expansion.tactics is not None
        assert expansion.log_priors is not None
        return PreEnvExpansion(
            theorem=theorem,
            log_critic=expansion.log_critic,
            tactic_tokens=expansion.tactics,
            log_priors=expansion.log_priors,
            job_ids=[],
            job_results={},
            exp_duration=expansion.exp_duration,
            gpu_duration=expansion.gpu_duration,
        )

    def ready_applied_tactics(self):
        return len(self.job_results) == len(self.job_ids)


@dataclass
class EnvExpansion:

    """
    Sanitized expansion.
    `durations` contains the TacticJobResult processing time of initial tactics
    given by ModelExpansion, before filtering, so it can contain more elements
    than `tactics`.
    """

    theorem: Theorem
    exp_duration: float  # total time spent in the expander
    gpu_duration: float  # time spent to generate tactics in the expander
    env_durations: List[float]  # time spent by the environment to apply each tactic
    effects: List[Tuple[Tactic, Optional[List[Theorem]]]]
    error: Optional[str] = None
    log_critic: Optional[float] = None
    tactics: Optional[List[Tactic]] = None
    child_for_tac: Optional[List[List[Theorem]]] = None
    priors: Optional[List[float]] = None

    @property
    def is_error(self):
        return self.error is not None

    @staticmethod
    def build_error(
        theorem: Theorem,
        exp_duration: float,
        gpu_duration: float,
        env_durations: List[float],
        error: str,
    ):
        return EnvExpansion(
            theorem=theorem,
            exp_duration=exp_duration,
            gpu_duration=gpu_duration,
            env_durations=env_durations,
            error=error,
            effects=[],
        )

    def __post_init__(self):
        assert type(self.env_durations) is list
        is_none = [
            self.log_critic is None,
            self.tactics is None,
            self.child_for_tac is None,
            self.priors is None,
        ]
        if self.is_error:
            assert all(is_none)
        else:
            assert not any(is_none)
            assert len(self.tactics) == len(self.child_for_tac) == len(self.priors)
            assert type(self.log_critic) is float and self.log_critic <= 0
            assert type(self.priors) is list
            assert all(type(p) is float and 0 <= p <= 1 for p in self.priors)
            assert all(
                tac.is_valid or tac.error_msg in SPECIAL_TACTIC_TOKENS
                for tac in self.tactics
            )
            # assert self.priors == sorted(self.priors, reverse=True)

    @staticmethod
    def from_pre_env_expansion(
        holder: PreEnvExpansion, prover_params: ProverParams
    ) -> Tuple["EnvExpansion", Dict[str, float]]:
        assert holder.ready_applied_tactics()
        assert len(holder.job_ids) == len(holder.tactic_tokens) > 0, (
            len(holder.job_ids),
            len(holder.tactic_tokens),
        )
        assert len(holder.job_ids) == len(holder.log_priors), (
            len(holder.job_ids),
            len(holder.log_priors),
        )

        all_tactics: List[Tactic] = []
        durations: List[float] = []  # time to apply each tactic
        child_for_tac: List[List[Theorem]] = []  # subgoals

        # NOTE: tactics and tactic_tokens are not necessarily aligned
        for job_id in holder.job_ids:
            res = holder.job_results[job_id]
            assert isinstance(res.children, list)
            all_tactics.append(res.tactic)
            child_for_tac.append(res.children)
            durations.append(res.duration)

        # filter invalid tactics (i.e. tactics that failed in the environment)
        assert len(holder.log_priors) == len(all_tactics) == len(child_for_tac) > 0
        filtered = [
            (log_p, tac, children)
            for log_p, tac, children in zip(
                holder.log_priors, all_tactics, child_for_tac
            )
            if tac.is_valid
        ]
        effects: List[Tuple[Tactic, Optional[List[Theorem]]]] = [
            (tac, None if tac.is_error() else children)
            for tac, children in zip(all_tactics, child_for_tac)
        ]

        stats: Dict[str, float] = {
            "pre_env_exp_n_all_tactics": len(holder.tactic_tokens),
            "pre_env_exp_n_valid_tactics": len(filtered),
        }

        # sort tactics by prior, so that if there are duplicates,
        # we select the tactic highest score
        filtered = sorted(filtered, key=lambda x: x[0], reverse=True)
        if len(filtered) == 0:
            return (
                EnvExpansion.build_error(
                    theorem=holder.theorem,
                    exp_duration=holder.exp_duration,
                    gpu_duration=holder.gpu_duration,
                    env_durations=durations,
                    error="failed_no_tactic_after_env",
                ),
                stats,
            )

        # If one tactic solves return this tactic only
        # Also, remove duplicate tactics
        solving_tactics: List[Tactic] = []
        seen_tactics: Set[Tactic] = set()
        seen_children: List[Set[Theorem]] = []
        filtered_uniq = []
        for item in filtered:
            _, tactic, children = item
            assert isinstance(tactic, Tactic)
            if tactic in seen_tactics:  # ignore duplicate tactics
                continue
            seen_tactics.add(tactic)
            if len(children) == 0:
                solving_tactics.append(tactic)
                continue
            if set(children) in seen_children:
                continue
            seen_children.append(set(children))
            filtered_uniq.append(item)

        # O(n^2) filtering to make sure we keep only minimal subsets of children
        # We use strict issubset by checking for equality.x
        if len(solving_tactics) == 0:
            filtered_minimal = [
                (lp, t, c)
                for lp, t, c in filtered_uniq
                # we add children if no other is strictly smaller
                if not any(
                    other_children.issubset(c) and len(other_children) < len(set(c))
                    for other_children in seen_children
                )
            ]
            final_tactics = [t for _, t, _ in filtered_minimal]
            child_for_tac = [c for _, _, c in filtered_minimal]
            log_priors = [lp for lp, _, _ in filtered_minimal]
        else:
            if prover_params.only_keep_one_solving:
                solving_tactics = solving_tactics[:1]
            n = len(solving_tactics)
            final_tactics = solving_tactics
            child_for_tac = [[] for _ in range(n)]
            log_priors = [-math.log(n) for _ in range(n)]

        stats["pre_env_exp_n_uniq_tactics"] = len(filtered_uniq)
        stats["pre_env_exp_n_final_tactics"] = len(final_tactics)
        # stats["valid_tac_prob"] = math.exp(sum_log_probs(log_priors))  # removing because assert is wrong when summing

        def add_special_tac(tac_type: str, log_prob: float):
            """
            Add special tactics.
            NOTE: we could also merge the content of the failing tactics?
            """
            assert log_prob <= 0, log_prob
            token = SPECIAL_TACTIC_TOKENS[tac_type]
            tac = all_tactics[0].from_error(error_msg=tac_type, tokens=[token])
            final_tactics.append(tac)
            log_priors.append(log_prob)
            child_for_tac.append([])

        # add tactics to represent the ones that failed. separate parsing errors
        # from environment errors. only do this if we didn't find a solving tactic
        if prover_params.add_tactic_errors and len(solving_tactics) == 0:
            assert len(holder.log_priors) == len(all_tactics)
            parse_error_lp, env_error_lp = [], []
            for lp, tac in zip(holder.log_priors, all_tactics):
                if tac.is_valid:
                    continue
                if tac.malformed:
                    parse_error_lp.append(lp)
                else:
                    env_error_lp.append(lp)
            if len(parse_error_lp) > 0:
                add_special_tac("parse_error", sum_log_probs(parse_error_lp))
                stats["parse_error_tac_prob"] = math.exp(log_priors[-1])
            if len(env_error_lp) > 0:
                add_special_tac("env_error", sum_log_probs(env_error_lp))
                stats["env_error_tac_prob"] = math.exp(log_priors[-1])

        # add a filling tactic to cover the probability mass of the non generated ones.
        # this requires the length penalty to be 0. tactic scores must remain sorted
        if prover_params.add_tactic_fill and len(solving_tactics) == 0:
            probs = np.exp(np.array(log_priors, dtype=np.float64))
            assert 0 <= probs.sum() <= 1 + 1e-6, probs
            p_fill = max(1 - probs.sum(), 0)  # NOTE: could `continue` if 0
            lp_fill = -math.inf if p_fill == 0 else np.log(p_fill).item()
            add_special_tac("fill", lp_fill)
            stats["fill_tac_prob"] = math.exp(log_priors[-1])

        # kk = [  # TODO: remove
        #     "valid_tac_prob",
        #     "parse_error_tac_prob",
        #     "env_error_tac_prob",
        #     "fill_tac_prob",
        # ]
        # print(", ".join(f"{k}={stats[k]:6.3f}" for k in kk if k in stats))

        # priors must always sum to 1
        priors = torch.softmax(torch.DoubleTensor(log_priors), dim=-1).tolist()

        env_exp = EnvExpansion(
            theorem=holder.theorem,
            error=None,
            log_critic=holder.log_critic,
            tactics=final_tactics,
            child_for_tac=child_for_tac,
            priors=priors,
            effects=effects,
            exp_duration=holder.exp_duration,
            gpu_duration=holder.gpu_duration,
            env_durations=durations,  # not necessarily the same as number of tactics
        )

        return env_exp, stats


class BackwardEnv:
    """ The backward env is an interface that handles some book-keeping for the :class:`ExpanderEnv`. 
    It receives :class:`ModelExpansion` and apply all tactics in the :class:`ExpanderEnv`.

    Once results for all tactics have been received, an :class:`EnvExpansion` is returned.
    """

    def __init__(self, expander_env: ExpanderEnv):
        self.expander_env = expander_env
        self.pre_expansions: Dict[Tuple[ProofId, int], PreEnvExpansion] = {}
        self.job_id_to_holder: Dict[int, Tuple[ProofId, int]] = {}
        self.sum_stats: Dict[str, float] = defaultdict(float)
        self.avg_stats: Dict[str, ActionCounter] = defaultdict(
            lambda: ActionCounter(name="", is_rate=False, silent=True)
        )
        self.has_died: Set[ProofId] = set()

    def apply_tactics(
        self, theorem: Theorem, tactics: List[List[Token]], proof_id=None
    ) -> List[TacticJobResult]:
        """
        Used by the visualizer only.
        """
        qids = []
        for tactic_tokens in tactics:
            qids.append(
                self.expander_env.process(
                    TacticJob(
                        theorem=theorem, tactic_tokens=tactic_tokens, proof_id=proof_id
                    ),
                    batch_id=proof_id,
                )
            )
        results: Dict[int, TacticJobResult] = {}
        while len(results) < len(tactics):
            for pid, qid, res in self.expander_env.get_all_ready():
                results[qid] = res
        return [results[qid] for qid in qids]

    def apply_model_expansion(
        self,
        theorem: Theorem,
        model_expansion: ModelExpansion,
        proof_id: int,
        tid: int,
    ) -> None:
        holder = PreEnvExpansion.from_model_expansion(theorem, model_expansion)
        assert model_expansion.tactics is not None
        for tactic_tokens in model_expansion.tactics:
            job_id = self.expander_env.process(
                TacticJob(
                    theorem=theorem, tactic_tokens=tactic_tokens, proof_id=proof_id
                ),
                batch_id=proof_id,
            )
            holder.job_ids.append(job_id)
            self.job_id_to_holder[job_id] = (proof_id, tid)

        self.pre_expansions[proof_id, tid] = holder

    def ready_env_expansions(
        self, prover_params: ProverParams
    ) -> List[Tuple[ProofId, int, EnvExpansion]]:

        # receive results from expander_env
        res = self.expander_env.get_all_ready()

        for proof_id, job_id, tactic_job_result in res:
            try:
                pid, tid = self.job_id_to_holder.pop(job_id)
                assert proof_id == pid
            except KeyError:
                print(f"JOB_ID {job_id!r} unknown!", flush=True)
                raise
            holder = self.pre_expansions[pid, tid]
            holder.job_results[job_id] = tactic_job_result
            if tactic_job_result.tactic.is_error():
                error_code = tactic_job_result.tactic.get_error_code()
                self.sum_stats[error_code] += 1
                if error_code == LeanInstanceDied:
                    self.has_died.add(pid)
            else:
                self.sum_stats["GOOD"] += 1

        ready = []
        for pid, tid in list(self.pre_expansions.keys()):
            holder = self.pre_expansions[pid, tid]
            if holder.ready_applied_tactics():
                env_exp, stats = EnvExpansion.from_pre_env_expansion(
                    holder, prover_params
                )
                ready.append((pid, tid, env_exp))
                for k, v in stats.items():
                    self.avg_stats[k].act(v)
                del self.pre_expansions[pid, tid]
        return ready

    def finish_goal(self, goal: BackwardGoal) -> None:
        self.expander_env.finish_theorem(goal.theorem)

    def materialize_goal(self, goal: BackwardGoal) -> BackwardGoal:
        """Used to materialize lean theorems for example."""
        return self.expander_env.materialize_goal(goal)

    def get_stats(self) -> Dict[str, float]:
        # safe, stats is defaultdict so total > 0
        total = sum(self.sum_stats.values())
        stats = {k: v.rate_and_reset() for k, v in self.avg_stats.items()}
        stats.update({k: v / total for k, v in self.sum_stats.items()})
        self.sum_stats.clear()
        return stats

    def close(self):
        self.expander_env.close()


class EnvGen(ABC):
    @abstractmethod
    def __call__(self) -> BackwardEnv:
        pass

    @abstractmethod
    def close(self):
        pass
