# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, List, Tuple, Dict, Any, cast
import tempfile
import time
from pathlib import Path

import pytest
from evariste.backward.model.beam_search_kind import BeamSearchKind

from evariste.datasets.lean import (
    mk_lean_conf,
    LEAN_FULL_NAMES_DIR_V2_TACTIC_FULL_NAMES,
)
from evariste.model.data.envs.minproof_replay_buffer import MCTSSampleProof
from evariste.backward.graph import NonPicklableProof, Proof, BackwardGoal, ProofId
from evariste.backward.prover.args import MCTSParams
from evariste.backward.prover.prover_args import (
    ProverParams,
    CleaningLevel,
    ProofCleaningParams,
)
from evariste.backward.prover.mcts import MCTSResult
from evariste.backward.env.lean.graph import LeanContext, LeanTheorem, LeanTactic
from evariste.backward.env.lean.env import LeanExpanderEnv
from .cleaner import AsyncProofCleaner, InvalidStart, CmdResultError

LEAN_VERSION = "v1.1"
PROOF_CLEANER_VERSION = "v1"
StringProof = Tuple[str, str, List["StringProof"]]  # type: ignore ## mypy doesn't support recursive types


@pytest.fixture(scope="module")
def dump_path():
    with tempfile.TemporaryDirectory() as dump_path:
        yield Path(dump_path)


@pytest.fixture(scope="module")
def env(dump_path):
    dataset = mk_lean_conf(LEAN_VERSION)
    env = LeanExpanderEnv(dataset=dataset, dump_path=dump_path)
    yield env
    env.close()


@pytest.fixture(scope="function")
def make_async_proof_cleaner(env, dump_path):
    mcts = MCTSParams()
    prover_params = ProverParams(
        mcts=mcts,
        beam_path=dump_path,
        dump_path=dump_path,
        n_simultaneous_proofs=10,
        beam_kind=BeamSearchKind.Fixed,
    )
    old_params = prover_params.proof_cleaning_params

    def _make_async_proof_cleaner(params: Optional[ProofCleaningParams] = None):
        if params is None:
            params = ProofCleaningParams()
        params.version = PROOF_CLEANER_VERSION
        prover_params.proof_cleaning_params = params

        return AsyncProofCleaner(env, prover_params)

    yield _make_async_proof_cleaner
    prover_params.proof_cleaning_params = old_params


def _convert_str_proof(str_proof: StringProof) -> Proof:
    if not str_proof:
        return []
    theorem = LeanTheorem(
        conclusion=str_proof[0], context=LeanContext(namespaces=set()), state=None,
    )
    tac = LeanTactic(str_proof[1])
    return (theorem, tac, [_convert_str_proof(c) for c in str_proof[2]])


def _build_proof_result(label: str, spf: StringProof) -> MCTSResult:
    proof = _convert_str_proof(spf)
    assert not isinstance(proof, NonPicklableProof)
    theorem = LeanTheorem(
        conclusion=proof[0].conclusion,
        context=LeanContext(namespaces=set()),
        state=None,
    )
    goal = BackwardGoal(theorem=theorem, label=label)
    sample_proof = MCTSSampleProof(label=label, size=0, stype="size", samples=[])
    res = MCTSResult(
        proof=proof,
        goal=goal,
        exception=None,
        mcts_samples_critic=None,
        mcts_samples_tactic=None,
        mcts_samples_effect=None,
        sample_proof=sample_proof,
        simplified_state=None,
        stats={},
        hist_stats={},
    )
    return res


def _get_results(
    async_proof_cleaner, exp_len: int = 1, timeout: float = 600.0
) -> Tuple[
    List[Tuple[ProofId, Tuple[MCTSResult, Optional[Exception]]]], Dict[str, Any]
]:
    start = time.time()
    res: List[Tuple[ProofId, Tuple[MCTSResult, Optional[Exception]]]] = []
    while (time.time() - start < timeout) and len(res) < exp_len:
        to_ret = async_proof_cleaner.get_ready()
        if to_ret:
            res.extend(to_ret)
        time.sleep(0.1)
    return (res, async_proof_cleaner.get_stats())


# TODO: add logging to asyn proof cleaner


@pytest.fixture(scope="function")
def async_proof_cleaner(request, make_async_proof_cleaner):
    level = cast(
        CleaningLevel, request.node.get_closest_marker("cleaning_level").args[0]
    )
    return make_async_proof_cleaner(
        ProofCleaningParams(level=level, timeout_useless_tactics_removal=10)
    )


@pytest.fixture(scope="class")
def async_stats_keys(request):
    return set()


@pytest.fixture(scope="class")
def stat_keys(request):
    level = request.node.get_closest_marker("cleaning_level").args[0]
    if level == CleaningLevel.No:
        return set()
    if level == CleaningLevel.TacRemoval:
        return {
            "cleaning/total_time",
            "cleaning/invalid_start",
            "cleaning/tac_removal_time",
            "cleaning/tac_removal_success",
            "cleaning/tac_won_per_tac",
            "cleaning/tac_won",
        }
    if level in {
        CleaningLevel.TacRemoval_TacClean,
        CleaningLevel.TacRemoval_TacClean_SimpOnly,
    }:
        return {
            "cleaning/total_time",
            "cleaning/invalid_start",
            "cleaning/tac_removal_time",
            "cleaning/tac_removal_success",
            "cleaning/tac_won_per_tac",
            "cleaning/tac_won",
            "cleaning/tac_cleaning_time/count",
            "cleaning/tac_cleaning_time/avg_time",
            "cleaning/tac_cleaning_time/max_time",
            "cleaning/tac_cleaning_time/cum_time",
            "cleaning/tac_cleaning_success",
            "cleaning/tac_cleaning_success_per_tac",
            "cleaning/tac_cleaned",
            "cleaning/tac_cleaned_per_tac",
        }


@pytest.mark.slow
@pytest.mark.cleaning_level(CleaningLevel.No)
class TestProofCleaningNo:
    def test_one_goal(self, async_proof_cleaner, stat_keys, async_stats_keys):
        pr = _build_proof_result(
            label="mathd_numbertheory_101", spf=["⊢ 17 * 18 % 4 = 2", "norm_num", []],
        )
        async_proof_cleaner.process(0, pr)
        (res, timers) = _get_results(async_proof_cleaner)
        assert (
            len(res) == 1
            and res[0][0] == 0
            and res[0][1][1] == []
            and res[0][1][0].proof == pr.proof
            and set(res[0][1][0].stats) == stat_keys
        )
        assert set(timers) == async_stats_keys

    def test_multigoals(self, async_proof_cleaner, stat_keys, async_stats_keys):
        prs = [
            _build_proof_result(
                label="mathd_numbertheory_101",
                spf=["⊢ 17 * 18 % 4 = 2", "norm_num", []],
            ),
            _build_proof_result(
                label="mathd_numbertheory_149",
                spf=[
                    "⊢ ∑ (k : ℕ) in finset.filter (λ (x : ℕ), x % 8 = 5 ∧ x % 6 = 3) (finset.range 50), k = 66",
                    "refl",
                    [],
                ],
            ),
            _build_proof_result(
                label="algebra_2rootspoly_apatapbeq2asqp2ab",
                spf=[
                    "a b : ℂ\n⊢ (a + a) * (a + b) = 2 * a ^ 2 + 2 * (a * b)",
                    "ring",
                    [],
                ],
            ),
        ]
        for i, pr in enumerate(prs):
            async_proof_cleaner.process(i, pr)
        (res, timers) = _get_results(async_proof_cleaner, 3)
        assert len(res) == len(prs) and all(
            res[i][0] == i
            and res[i][1][1] == []
            and res[i][1][0].proof == pr.proof
            and set(res[i][1][0].stats) == stat_keys
            for i, pr in enumerate(prs)
        )
        assert set(timers) == async_stats_keys


@pytest.mark.slow
@pytest.mark.cleaning_level(CleaningLevel.TacRemoval)
class TestProofCleaningTacRemoval:
    def test_one_goal_no_cleaning(
        self, async_proof_cleaner, stat_keys, async_stats_keys
    ):
        pr = _build_proof_result(
            label="mathd_numbertheory_101", spf=["⊢ 17 * 18 % 4 = 2", "norm_num", []],
        )
        async_proof_cleaner.process(0, pr)
        (res, timers) = _get_results(async_proof_cleaner)
        assert (
            len(res) == 1
            and res[0][0] == 0
            and res[0][1][1] == []
            and res[0][1][0].proof == pr.proof
            and set(res[0][1][0].stats) == stat_keys
            and res[0][1][0].stats["cleaning/invalid_start"] == (0.0, 1)
            and res[0][1][0].stats["cleaning/tac_removal_success"] == (1.0, 1)
            and res[0][1][0].stats["cleaning/tac_won"] == (0.0, 1)
        )
        assert set(timers) == async_stats_keys

    def test_skip(self, async_proof_cleaner, stat_keys, async_stats_keys):
        proof = [
            "⊢ 17 * 18 % 4 = 2",
            "skip",
            [
                [
                    "⊢ 17 * 18 % 4 = 2",
                    "skip",
                    [
                        [
                            "⊢ 17 * 18 % 4 = 2",
                            "skip",
                            [["⊢ 17 * 18 % 4 = 2", "norm_num", []],],
                        ]
                    ],
                ]
            ],
        ]
        cleaned_proof = ["⊢ 17 * 18 % 4 = 2", "norm_num", []]
        pr = _build_proof_result(label="mathd_numbertheory_101", spf=proof)
        async_proof_cleaner.process(0, pr)
        (res, timers) = _get_results(async_proof_cleaner)
        assert (
            len(res) == 1
            and res[0][0] == 0
            and res[0][1][1] == []
            and res[0][1][0].proof == _convert_str_proof(cleaned_proof)
            and set(res[0][1][0].stats) == stat_keys
            and res[0][1][0].stats["cleaning/invalid_start"] == (0.0, 1)
            and res[0][1][0].stats["cleaning/tac_removal_success"] == (1.0, 1)
            and res[0][1][0].stats["cleaning/tac_won"] == (3.0, 1)
        )
        assert set(timers) == async_stats_keys

    def test_have_single_line(self, async_proof_cleaner, stat_keys, async_stats_keys):
        proof = [
            "⊢ 17 * 18 % 4 = 2",
            "have : 1 + 1 = 2 := by norm_num",
            [
                [
                    "this: 1 + 1 = 2\n⊢ 17 * 18 % 4 = 2",
                    "have h₀ : 2 + 2 = 4 := by norm_num",
                    [
                        [
                            "this: 1 + 1 = 2\nh₀ : 2 + 2 = 4\n⊢ 17 * 18 % 4 = 2",
                            "norm_num",
                            [],
                        ],
                    ],
                ]
            ],
        ]
        cleaned_proof = ["⊢ 17 * 18 % 4 = 2", "norm_num", []]
        pr = _build_proof_result(label="mathd_numbertheory_101", spf=proof)
        async_proof_cleaner.process(0, pr)
        (res, timers) = _get_results(async_proof_cleaner)
        assert (
            len(res) == 1
            and res[0][0] == 0
            and res[0][1][1] == []
            and res[0][1][0].proof == _convert_str_proof(cleaned_proof)
            and set(res[0][1][0].stats) == stat_keys
            and res[0][1][0].stats["cleaning/invalid_start"] == (0.0, 1)
            and res[0][1][0].stats["cleaning/tac_removal_success"] == (1.0, 1)
            and res[0][1][0].stats["cleaning/tac_won"] == (2.0, 1)
        )
        assert set(timers) == async_stats_keys

    def test_have_multi_lines(self, async_proof_cleaner, stat_keys, async_stats_keys):
        proof = [
            "⊢ 17 * 18 % 4 = 2",
            "have : 1 + 1 = 2",
            [
                ["⊢ 1 + 1 = 2", "norm_num", []],
                [
                    "this: 1 + 1 = 2\n⊢ 17 * 18 % 4 = 2",
                    "have h₀ : 2 + 2 = 4",
                    [
                        ["this: 1 + 1 = 2\n⊢ 2 + 2 = 4", "norm_num", []],
                        [
                            "this: 1 + 1 = 2,\nh₀ : 2 + 2 = 4\n⊢ 17 * 18 % 4 = 2",
                            "norm_num",
                            [],
                        ],
                    ],
                ],
            ],
        ]
        cleaned_proof = ["⊢ 17 * 18 % 4 = 2", "norm_num", []]
        pr = _build_proof_result(label="mathd_numbertheory_101", spf=proof)
        async_proof_cleaner.process(0, pr)
        (res, timers) = _get_results(async_proof_cleaner)
        assert (
            len(res) == 1
            and res[0][0] == 0
            and res[0][1][1] == []
            and res[0][1][0].proof == _convert_str_proof(cleaned_proof)
            and set(res[0][1][0].stats) == stat_keys
            and res[0][1][0].stats["cleaning/invalid_start"] == (0.0, 1)
            and res[0][1][0].stats["cleaning/tac_removal_success"] == (1.0, 1)
            and res[0][1][0].stats["cleaning/tac_won"] == (4.0, 1)
        )
        assert set(timers) == async_stats_keys

    def test_nested_haves(self, async_proof_cleaner, stat_keys, async_stats_keys):
        proof = [
            "⊢ 17 * 18 % 4 = 2",
            "have : 1 + 1 = 2",
            [
                [
                    "⊢ 1 + 1 = 2",
                    "have h₀ : 2 + 2 = 4",
                    [
                        [
                            "⊢ 2 + 2 = 4",
                            "have h₁ : 4 + 4 = 8",
                            [
                                ["⊢ 4 + 4 = 8", "norm_num", []],
                                ["h₁ : 4 + 4 = 8\n⊢ 2 + 2 = 4", "norm_num", []],
                            ],
                        ],
                        ["h₀ : 2 + 2 = 4\n⊢ 1 + 1 = 2", "norm_num", []],
                    ],
                ],
                ["this: 1 + 1 = 2\n⊢ 17 * 18 % 4 = 2", "norm_num", [],],
            ],
        ]
        cleaned_proof = ["⊢ 17 * 18 % 4 = 2", "norm_num", []]
        pr = _build_proof_result(label="mathd_numbertheory_101", spf=proof)
        async_proof_cleaner.process(0, pr)
        (res, timers) = _get_results(async_proof_cleaner)
        assert (
            len(res) == 1
            and res[0][0] == 0
            and res[0][1][1] == []
            and res[0][1][0].proof == _convert_str_proof(cleaned_proof)
            and set(res[0][1][0].stats) == stat_keys
            and res[0][1][0].stats["cleaning/invalid_start"] == (0.0, 1)
            and res[0][1][0].stats["cleaning/tac_removal_success"] == (1.0, 1)
            and res[0][1][0].stats["cleaning/tac_won"] == (6.0, 1)
        )
        assert set(timers) == async_stats_keys

    def test_tactic_splitting(self, async_proof_cleaner, stat_keys, async_stats_keys):
        proof = [
            """q p : ℝ,
h₀ : q = 2 - 4 + 6 - 8 + 10 - 12 + 14,
h₁ : p = 3 - 6 + 9 - 12 + 15 - 18 + 21
⊢ q / p = 2 / 3""",
            "simp only [h₀, h₁],norm_num",
            [],
        ]
        cleaned_proof = [
            """q p : ℝ,
h₀ : q = 2 - 4 + 6 - 8 + 10 - 12 + 14,
h₁ : p = 3 - 6 + 9 - 12 + 15 - 18 + 21
⊢ q / p = 2 / 3""",
            "simp only [h₀, h₁]",
            [
                [
                    """q p : ℝ,
h₀ : q = 2 - 4 + 6 - 8 + 10 - 12 + 14,
h₁ : p = 3 - 6 + 9 - 12 + 15 - 18 + 21
⊢ (2 - 4 + 6 - 8 + 10 - 12 + 14) / (3 - 6 + 9 - 12 + 15 - 18 + 21) = 2 / 3""",
                    "norm_num",
                    [],
                ]
            ],
        ]
        pr = _build_proof_result(label="mathd_algebra_55", spf=proof)
        async_proof_cleaner.process(0, pr)
        (res, timers) = _get_results(async_proof_cleaner)
        assert (
            len(res) == 1
            and res[0][0] == 0
            and res[0][1][1] == []
            and res[0][1][0].proof == _convert_str_proof(cleaned_proof)
            and set(res[0][1][0].stats) == stat_keys
            and res[0][1][0].stats["cleaning/invalid_start"] == (0.0, 1)
            and res[0][1][0].stats["cleaning/tac_removal_success"] == (1.0, 1)
            and res[0][1][0].stats["cleaning/tac_won"] == (0.0, 1)
        )
        assert set(timers) == async_stats_keys

    def test_junk_at_the_end(self, async_proof_cleaner, stat_keys, async_stats_keys):
        pr = _build_proof_result(
            label="algebra_2rootspoly_apatapbeq2asqp2ab",
            spf=[
                "a b : ℂ\n⊢ (a + a) * (a + b) = 2 * a ^ 2 + 2 * (a * b)",
                "ring ](( junk",
                [],
            ],
        )
        cleaned_proof = [
            "a b : ℂ\n⊢ (a + a) * (a + b) = 2 * a ^ 2 + 2 * (a * b)",
            "ring",
            [],
        ]
        async_proof_cleaner.process(0, pr)
        (res, timers) = _get_results(async_proof_cleaner)
        assert (
            len(res) == 1
            and res[0][0] == 0
            and res[0][1][1] == []
            and res[0][1][0].proof == _convert_str_proof(cleaned_proof)
            and set(res[0][1][0].stats) == stat_keys
            and res[0][1][0].stats["cleaning/invalid_start"] == (0.0, 1)
            and res[0][1][0].stats["cleaning/tac_removal_success"] == (1.0, 1)
            and res[0][1][0].stats["cleaning/tac_won"] == (0.0, 1)
        )
        assert set(timers) == async_stats_keys

    def test_goal_splitting(self, async_proof_cleaner, stat_keys, async_stats_keys):
        spf = [
            "p q : Prop,\nh : p ∧ q\n⊢ q ∧ p",
            "have hp",
            [
                [
                    "2 goals\np q : Prop,\nh : p ∧ q\n⊢ ?m_1\n\np q : Prop,\nh : p ∧ q,\nhp : ?m_1\n⊢ q ∧ p",
                    "from and.left h",
                    [
                        [
                            "p q : Prop,\nh : p ∧ q,\nhp : p\n⊢ q ∧ p",
                            "have hq",
                            [
                                [
                                    "2 goals\np q : Prop,\nh : p ∧ q,\nhp : p\n⊢ ?m_1\n\np q : Prop,\nh : p ∧ q,\nhp : p,\nhq : ?m_1\n⊢ q ∧ p",
                                    "from and.right h",
                                    [
                                        [
                                            "p q : Prop,\nh : p ∧ q,\nhp : p,\nhq : q\n⊢ q ∧ p",
                                            "exact and.intro hq hp",
                                            [],
                                        ]
                                    ],
                                ]
                            ],
                        ]
                    ],
                ]
            ],
        ]
        pr = _build_proof_result(label="EVARISTE_test_1", spf=spf,)
        async_proof_cleaner.process(0, pr)
        (res, timers) = _get_results(async_proof_cleaner)
        assert (
            len(res) == 1
            and res[0][0] == 0
            and res[0][1][1] == []
            and res[0][1][0].proof == _convert_str_proof(spf)
            and set(res[0][1][0].stats) == stat_keys
            and res[0][1][0].stats["cleaning/invalid_start"] == (0.0, 1)
            and res[0][1][0].stats["cleaning/tac_removal_success"] == (1.0, 1)
            and res[0][1][0].stats["cleaning/tac_won"] == (0.0, 1)
        )
        assert set(timers) == async_stats_keys

    def test_invalid_proof(self, async_proof_cleaner, stat_keys, async_stats_keys):
        pr = _build_proof_result(
            label="mathd_numbertheory_101", spf=["⊢ 17 * 18 % 4 = 2", "sorry", []],
        )
        async_proof_cleaner.process(0, pr)
        (res, timers) = _get_results(async_proof_cleaner)
        assert (
            len(res) == 1
            and res[0][0] == 0
            and len(res[0][1][1]) == 1
            and isinstance(res[0][1][1][0], InvalidStart)
            and str(res[0][1][1][0])
            == "ParsingError: Invalid proof, got exception: failed to validate"
            and res[0][1][0].proof == pr.proof
            and set(res[0][1][0].stats)
            == {
                "cleaning/invalid_start",
                "cleaning/total_time",
                "cleaning/tac_removal_time",
            }
            and res[0][1][0].stats["cleaning/invalid_start"] == (1.0, 1)
        )
        assert set(timers) == async_stats_keys

    def test_decl_not_found(self, async_proof_cleaner, stat_keys, async_stats_keys):
        pr = _build_proof_result(
            label="EVARISTE_test_nodecl", spf=["⊢ 17 * 18 % 4 = 2", "norm_num", []],
        )
        async_proof_cleaner.process(0, pr)
        (res, timers) = _get_results(async_proof_cleaner)
        assert (
            len(res) == 1
            and res[0][0] == 0
            and len(res[0][1][1]) == 1
            and isinstance(res[0][1][1][0], InvalidStart)
            and str(res[0][1][1][0])
            == "ParsingError: Invalid proof, got exception: unknown declaration 'EVARISTE_test_nodecl'"
            and res[0][1][0].proof == pr.proof
            and set(res[0][1][0].stats)
            == {
                "cleaning/invalid_start",
                "cleaning/total_time",
                "cleaning/tac_removal_time",
            }
            and res[0][1][0].stats["cleaning/invalid_start"] == (1.0, 1)
        )
        assert set(timers) == async_stats_keys

    def test_multi_goals(self, async_proof_cleaner, stat_keys, async_stats_keys):
        prs = [
            _build_proof_result(
                label="mathd_numbertheory_101",
                spf=[
                    "⊢ 17 * 18 % 4 = 2",
                    "skip",
                    [
                        [
                            "⊢ 17 * 18 % 4 = 2",
                            "skip",
                            [
                                [
                                    "⊢ 17 * 18 % 4 = 2",
                                    "skip",
                                    [["⊢ 17 * 18 % 4 = 2", "norm_num", []],],
                                ]
                            ],
                        ]
                    ],
                ],
            ),
            _build_proof_result(
                label="mathd_numbertheory_149",
                spf=[
                    "⊢ ∑ (k : ℕ) in finset.filter (λ (x : ℕ), x % 8 = 5 ∧ x % 6 = 3) (finset.range 50), k = 66",
                    "refl",
                    [],
                ],
            ),
            _build_proof_result(
                label="algebra_2rootspoly_apatapbeq2asqp2ab",
                spf=[
                    "a b : ℂ\n⊢ (a + a) * (a + b) = 2 * a ^ 2 + 2 * (a * b)",
                    "skip",
                    [
                        [
                            "a b : ℂ\n⊢ (a + a) * (a + b) = 2 * a ^ 2 + 2 * (a * b)",
                            "skip",
                            [
                                [
                                    "a b : ℂ\n⊢ (a + a) * (a + b) = 2 * a ^ 2 + 2 * (a * b)",
                                    "skip",
                                    [
                                        [
                                            "a b : ℂ\n⊢ (a + a) * (a + b) = 2 * a ^ 2 + 2 * (a * b)",
                                            "ring",
                                            [],
                                        ]
                                    ],
                                ]
                            ],
                        ]
                    ],
                ],
            ),
        ]
        cleaned_pfs = [
            ["⊢ 17 * 18 % 4 = 2", "norm_num", []],
            [
                "⊢ ∑ (k : ℕ) in finset.filter (λ (x : ℕ), x % 8 = 5 ∧ x % 6 = 3) (finset.range 50), k = 66",
                "refl",
                [],
            ],
            ["a b : ℂ\n⊢ (a + a) * (a + b) = 2 * a ^ 2 + 2 * (a * b)", "ring", [],],
        ]
        tac_won = [3.0, 0.0, 3.0]
        for i, pr in enumerate(prs):
            async_proof_cleaner.process(i, pr)
        (res, timers) = _get_results(async_proof_cleaner, 3)
        assert len(res) == len(prs)
        for i, pf in enumerate(cleaned_pfs):
            for r in res:
                if r[0] == i:
                    assert r[1][1] == []
                    assert r[1][0].proof == _convert_str_proof(pf)
                    assert set(r[1][0].stats) == stat_keys
                    assert r[1][0].stats["cleaning/invalid_start"] == (0.0, 1)
                    assert r[1][0].stats["cleaning/tac_removal_success"] == (1.0, 1)
                    assert r[1][0].stats["cleaning/tac_won"] == (tac_won[i], 1)
                    break
            else:
                raise AssertionError(f"no index {i} in {res}")
        assert set(timers) == async_stats_keys

    def test_all_goals(self, async_proof_cleaner, stat_keys, async_stats_keys):
        proof = [
            "c: ℝ,\nh₀: c / 3 ≤ 2 + c\nh₁: 2 + c < (-2) * (1 + c)\n⊢ -3 ≤ c ∧ c < (-4) / 3",
            "repeat{intro}",
            [
                [
                    "c: ℝ,\nh₀: c / 3 ≤ 2 + c\nh₁: 2 + c < (-2) * (1 + c)\n⊢ -3 ≤ c ∧ c < (-4) / 3",
                    "classical, split , all_goals { linarith}",
                    [],
                ]
            ],
        ]
        cleaned_proof = [
            "c : ℝ,\nh₀ : c / 3 ≤ 2 + c,\nh₁ : 2 + c < (-2) * (1 + c)\n⊢ -3 ≤ c ∧ c < (-4) / 3",
            "split",
            [
                [
                    "c : ℝ,\nh₀ : c / 3 ≤ 2 + c,\nh₁ : 2 + c < (-2) * (1 + c)\n⊢ -3 ≤ c",
                    "linarith",
                    [],
                ],
                [
                    "c : ℝ,\nh₀ : c / 3 ≤ 2 + c,\nh₁ : 2 + c < (-2) * (1 + c)\n⊢ c < (-4) / 3",
                    "linarith",
                    [],
                ],
            ],
        ]
        pr = _build_proof_result(label="mathd_train_algebra_71", spf=proof)
        async_proof_cleaner.process(0, pr)
        (res, timers) = _get_results(async_proof_cleaner)
        assert (
            len(res) == 1
            and res[0][0] == 0
            and res[0][1][1] == []
            and res[0][1][0].proof == _convert_str_proof(cleaned_proof)
            and set(res[0][1][0].stats) == stat_keys
            and res[0][1][0].stats["cleaning/invalid_start"] == (0.0, 1)
            and res[0][1][0].stats["cleaning/tac_removal_success"] == (1.0, 1)
            and res[0][1][0].stats["cleaning/tac_won"] == (2.0, 1)
        )
        assert set(timers) == async_stats_keys

    @pytest.mark.skip(reason="TODO")
    def test_annotations(self, async_proof_cleaner, stat_keys, async_stats_keys):
        pass


#     def test_mathlib(self, async_proof_cleaner, stat_keys, async_stats_keys):
#         # TODO: not working yet, why?
#         raise AssertionError

#     def test_timeout(self, async_proof_cleaner, stat_keys, async_stats_keys):
#         # TODO
#         raise AssertionError


@pytest.mark.slow
@pytest.mark.cleaning_level(CleaningLevel.TacRemoval_TacClean)
class TestProofCleaningTacRemoval_TacClean:
    def test_one_goal_no_cleaning(
        self, async_proof_cleaner, stat_keys, async_stats_keys
    ):
        pr = _build_proof_result(
            label="mathd_numbertheory_101", spf=["⊢ 17 * 18 % 4 = 2", "norm_num", []],
        )
        async_proof_cleaner.process(0, pr)
        (res, timers) = _get_results(async_proof_cleaner)
        assert (
            len(res) == 1
            and res[0][0] == 0
            and res[0][1][1] == []
            and res[0][1][0].proof == pr.proof
            and set(res[0][1][0].stats) == stat_keys
            and res[0][1][0].stats["cleaning/invalid_start"] == (0.0, 1)
            and res[0][1][0].stats["cleaning/tac_removal_success"] == (1.0, 1)
            and res[0][1][0].stats["cleaning/tac_won"] == (0.0, 1)
            and res[0][1][0].stats["cleaning/tac_cleaning_success"] == (1.0, 1)
            and res[0][1][0].stats["cleaning/tac_cleaned"] == (0.0, 1)
        )
        assert set(timers) == async_stats_keys

    def test_invalid_proof(self, async_proof_cleaner, stat_keys, async_stats_keys):
        pr = _build_proof_result(
            label="mathd_numbertheory_101", spf=["⊢ 17 * 18 % 4 = 2", "sorry", []],
        )
        async_proof_cleaner.process(0, pr)
        (res, timers) = _get_results(async_proof_cleaner)
        assert (
            len(res) == 1
            and res[0][0] == 0
            and len(res[0][1][1]) == 1
            and isinstance(res[0][1][1][0], InvalidStart)
            and str(res[0][1][1][0])
            == "ParsingError: Invalid proof, got exception: failed to validate"
            and res[0][1][0].proof == pr.proof
            and set(res[0][1][0].stats)
            == {
                "cleaning/invalid_start",
                "cleaning/total_time",
                "cleaning/tac_removal_time",
                "cleaning/tac_cleaning_time/count",
            }
            and res[0][1][0].stats["cleaning/invalid_start"] == (1.0, 1)
            and res[0][1][0].stats["cleaning/tac_cleaning_time/count"] == (0, 1)
        )
        assert set(timers) == async_stats_keys

    def test_timeout_on_removal(self, async_proof_cleaner, stat_keys, async_stats_keys):
        pr = _build_proof_result(
            label="amc12a_2011_p18",
            spf=[
                "x y: ℝ,\nh₀: abs (x + y) + abs (x - y) = 2\n⊢ x ^ 2 - 6 * x + y ^ 2 ≤ 9",
                "classical, ring, classical, delta max abs at h₀, "
                "erw sq , split_ifs at h₀ ,nlinarith ,nlinarith ,nlinarith! , nlinarith! []",
                [],
            ],
        )
        async_proof_cleaner.process(0, pr)
        (res, timers) = _get_results(async_proof_cleaner)
        assert (
            len(res) == 1
            and res[0][0] == 0
            and len(res[0][1][1]) == 1
            and isinstance(res[0][1][1][0], CmdResultError)
            and str(res[0][1][1][0]) == "eval_cmd timeout"
            and res[0][1][0].proof == pr.proof
            and set(res[0][1][0].stats)
            == {
                "cleaning/invalid_start",
                "cleaning/total_time",
                "cleaning/tac_removal_time",
                "cleaning/tac_cleaning_time/count",
            }
            and res[0][1][0].stats["cleaning/invalid_start"] == (1.0, 1)
            and res[0][1][0].stats["cleaning/tac_cleaning_time/count"] == (0, 1)
        )
        assert set(timers) == async_stats_keys


#     def test_norm_num(self, async_proof_cleaner, stat_keys, async_stats_keys):
#         # TODO
#         raise AssertionError

#     def test_simp(self, async_proof_cleaner, stat_keys, async_stats_keys):
#         # TODO
#         raise AssertionError

#     def test_field_simp(self, async_proof_cleaner, stat_keys, async_stats_keys):
#         # TODO
#         raise AssertionError

#     def test_tidy(self, async_proof_cleaner, stat_keys, async_stats_keys):
#         # TODO
#         raise AssertionError

#     def test_hyp_renaming(self, async_proof_cleaner, stat_keys, async_stats_keys):
#         # TODO: not implemented yet
#         raise AssertionError

#     def test_timeout(self, async_proof_cleaner, stat_keys, async_stats_keys):
#         # TODO
#         raise AssertionError

#     def test_no_valid(self, async_proof_cleaner, stat_keys, async_stats_keys):
#         # TODO
#         raise AssertionError

#     def test_removal_and_cleaning(self, async_proof_cleaner, stat_keys, async_stats_keys):
#         # TODO
#         raise AssertionError

#     def test_cleaning_with_curly_brackets(self, async_proof_cleaner, stat_keys, async_stats_keys):
#         # TODO
#         raise AssertionError


@pytest.mark.slow
@pytest.mark.cleaning_level(CleaningLevel.TacRemoval_TacClean_SimpOnly)
class TestProofCleaningTacRemoval_TacClean_SimpOnly:
    def test_simp_only(self, async_proof_cleaner, stat_keys, async_stats_keys):
        proof = [
            """x y : ℕ,
h₀ : x % 19 = 4,
h₁ : y % 19 = 7
⊢ (x + 1) ^ 2 * (y + 5) ^ 3 % 19 = 13""",
            "simp [h₀, h₁, nat.add_mod, nat.mul_mod, nat.mul_mod, pow_succ, nat.mul_mod],norm_num []",
            [],
        ]
        cleaned_proof = [
            """x y : ℕ,
h₀ : x % 19 = 4,
h₁ : y % 19 = 7
⊢ (x + 1) ^ 2 * (y + 5) ^ 3 % 19 = 13""",
            "simp only [h₁, h₀, mul_one, nat.mod_mod, nat.mul_mod, nat.add_mod, nat.one_mod, pow_succ, pow_zero]",
            [
                [
                    """x y : ℕ,
h₀ : x % 19 = 4,
h₁ : y % 19 = 7
⊢ (4 % 19 % 19 + 1 % 19) % 19 % 19 * ((4 % 19 % 19 + 1 % 19) % 19 % 19) % 19 % 19 *
        ((7 % 19 % 19 + 5 % 19 % 19) % 19 % 19 *
               ((7 % 19 % 19 + 5 % 19 % 19) % 19 % 19 * ((7 % 19 % 19 + 5 % 19 % 19) % 19 % 19) % 19 % 19) %
             19 %
           19) %
      19 =
    13""",
                    "norm_num",
                    [],
                ]
            ],
        ]
        pr = _build_proof_result(label="mathd_numbertheory_412", spf=proof,)
        async_proof_cleaner.process(0, pr)
        (res, timers) = _get_results(async_proof_cleaner)
        assert (
            len(res) == 1
            and res[0][0] == 0
            and res[0][1][1] == []
            and res[0][1][0].proof == _convert_str_proof(cleaned_proof)
            and set(res[0][1][0].stats) == stat_keys
            and res[0][1][0].stats["cleaning/invalid_start"] == (0.0, 1)
            and res[0][1][0].stats["cleaning/tac_removal_success"] == (1.0, 1)
            and res[0][1][0].stats["cleaning/tac_won"] == (0.0, 1)
            and res[0][1][0].stats["cleaning/tac_cleaning_success"] == (2.0, 1)
            and res[0][1][0].stats["cleaning/tac_cleaned"] == (2.0, 1)
        )
        assert set(timers) == async_stats_keys
