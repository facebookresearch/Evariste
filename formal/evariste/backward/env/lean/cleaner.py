# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
import os
import pickle
from abc import ABC
import shutil
from enum import Enum

from evariste import json as json
from evariste.logger import create_logger
from evariste.utils import get_tail_logger, this_job_id, NoneLogger
from evariste.metrics import Timer
from evariste.backward.graph import (
    NonPicklableProof,
    Proof,
    ProofId,
    fold_proof,
    get_tac,
    get_thm,
)
from evariste.backward.prover.core import ProofResult
from evariste.backward.prover.mcts import MCTSResult
from evariste.backward.prover.mcts_samples import MCTSSampleTactics
from evariste.backward.prover.prover_args import (
    ProverParams,
    CleaningLevel,
    ProofCleaningParams,
)
from evariste.backward.env.worker import ExpanderEnv
from evariste.backward.env.lean.graph import LeanTheorem, LeanTactic, LeanContext
from evariste.backward.env.lean.env import (
    LeanEvalCmdJob,
    LeanEvalCmdResult,
    LeanExpanderEnv,
)
from evariste.backward.env.lean.utils import (
    extract_tactics_from_proof,
    extract_actions_from_proof,
)


CleaningOutputProof = List[Tuple[str, Tuple[str, List[str]]]]


class ProofKey(Enum):
    StartProof = "proof"
    CleanedProof = "cleaned_proof"


@dataclass
class LeanCleaningOutput:
    valid_proof: bool
    proof: CleaningOutputProof
    valid_cleaned: bool
    cleaned_proof: CleaningOutputProof
    exc: str

    @staticmethod
    def from_json(d: Dict):
        return LeanCleaningOutput(
            valid_proof=d["valid_proof"],
            proof=d["proof"],
            valid_cleaned=d["valid_cleaned"],
            cleaned_proof=d["cleaned_proof"],
            exc=d["exc"],
        )


class _CleaningError(ABC, Exception):
    pass


class ProofReconstructionError(_CleaningError):
    pass


def clean_conclusion(conclusion: str) -> str:
    """Remove the extra `case or.inl` and such"""
    if "\n\n" in conclusion:
        return conclusion
    fst_line = conclusion.split("\n")[0]
    if ":" in fst_line or "⊢" in fst_line:
        return conclusion
    return conclusion[conclusion.index("\n") + 1 :]


def reconstruct_proof(
    cleaning_output_proof: CleaningOutputProof, ctx: Optional[LeanContext] = None,
) -> Proof:
    ctx = ctx if ctx is not None else LeanContext(namespaces=set())
    err = (
        "Not a well-structured recursive Proof object "
        "- it could be due to Lean splitting tactics on comma"
        f"\n{cleaning_output_proof}"
    )

    def _rec(idx: int) -> Tuple[Proof, int]:
        if idx >= len(cleaning_output_proof):
            raise ProofReconstructionError(err)

        pfl: List[Proof] = []
        conclusion, (tactic, children) = cleaning_output_proof[idx]
        assert ctx is not None
        conclusion = clean_conclusion(conclusion)
        thm = LeanTheorem(conclusion=conclusion, context=ctx, state=None)
        proof = (thm, LeanTactic(tactic), pfl)
        for child in children:
            pf, idx = _rec(idx + 1)
            actual_child = clean_conclusion(child)
            assert not isinstance(pf, NonPicklableProof)
            if pf[0].conclusion != actual_child:
                raise ProofReconstructionError(
                    f"Not a well-structured recursive Proof object "
                    f"- conclusion `{pf[0].conclusion}` differs from child `{actual_child}`"
                    f"\n{cleaning_output_proof}"
                )
            pfl.append(pf)
        return proof, idx

    proof, idx = _rec(0)
    if idx != len(cleaning_output_proof) - 1:
        # Typically this could happen on a tactic split, e.g.,
        # `split, all_goals { tac }` got split into `[split, all_goals { tac }]`
        # but in the proof object, it is applied at once to only one goal
        # so to Lean, there is only one "split tactic state" that contains one goal
        # hence `all_goals { tac }` is applied to this one goal, then only `tac` is applied
        # while actually, `tac` should have been applied to the top two goals
        raise ProofReconstructionError(err)
    return proof


class ParsingError(_CleaningError):
    pass


def extract_cleaning_output_proof_from_cleaning_output(
    output, proof_key: ProofKey
) -> CleaningOutputProof:
    try:
        idx = output.index("🦑")
    except ValueError as e:
        raise ParsingError(f'{e}: no "🦑" in\n"{output}"')
    try:
        lco = LeanCleaningOutput.from_json(json.loads(output[idx + 1 :]))
    except json.decoder.JSONDecodeError as e:
        raise ParsingError(f'Cannot json decode "{output}"')
    if proof_key == ProofKey.StartProof:
        if not lco.valid_proof:
            raise ParsingError(f"Invalid {proof_key.value}, got exception: {lco.exc}")
        return lco.proof
    if not lco.valid_cleaned:
        raise ParsingError(f"Invalid {proof_key.value}, got exception: {lco.exc}")
    return lco.cleaned_proof


def unique_name(name: str, pid: ProofId) -> str:
    return f"{name}__jid{this_job_id()}_pid{pid}"


def reconstruct_proof_from_lean_output(
    output: str, proof_key: ProofKey, ctx: Optional[LeanContext] = None,
) -> Proof:
    cleaning_output_proof = extract_cleaning_output_proof_from_cleaning_output(
        output, proof_key
    )
    return reconstruct_proof(cleaning_output_proof, ctx)


class CmdResultError(_CleaningError):
    pass


class InvalidStart(_CleaningError):
    pass


def apply_cmd_result(
    start_proof: Proof, cmd_result: LeanEvalCmdResult, update_start: bool = False
) -> Tuple[
    Optional[Tuple[Proof, Optional[Exception]]], Tuple[Proof, Optional[Exception]]
]:
    assert cmd_result.error is not None or cmd_result.output is not None
    new_start_proof = start_proof
    exc_start: Optional[Exception] = None
    end_proof: Proof
    exc_end: Optional[Exception] = None
    if cmd_result.error is not None:
        exc_end = CmdResultError(cmd_result.error)
        end_proof = start_proof
        if update_start:
            # TODO: typically it happens when there is a timeout in tac removal
            # it would be better then to juste `do_get_proof` on the start proof
            # in order to have the comma separation, etc
            exc_start = exc_end
    else:
        assert cmd_result.output is not None
        try:
            if update_start:
                new_start_proof = reconstruct_proof_from_lean_output(
                    cmd_result.output, proof_key=ProofKey.StartProof
                )
        except _CleaningError as e:
            exc_start = InvalidStart(f"{e.__class__.__name__}: {e}")
            end_proof = start_proof
        else:
            try:
                end_proof = reconstruct_proof_from_lean_output(
                    cmd_result.output, proof_key=ProofKey.CleanedProof
                )
            except _CleaningError as e:
                exc_end = e
                end_proof = new_start_proof
    if update_start:
        return ((new_start_proof, exc_start), (end_proof, exc_end))
    return (None, (end_proof, exc_end))


class Cleaner:
    """
    The Proof cleaner is a Python class that makes calls to the LeanAPI (or Lean cluster)
    to "clean" (i.e., removing unecessary tactics and arguments or simplifying complex
    tactics) proofs produced by the HPTS.

    Apart from these calls, most of the cleaning is implemented in lean,
    but because of some timeout issues for some cleaning steps, it was deemed more robust to
    have some of those steps individually called from the Python API (vs. running all steps
    in a single sequence in Lean).
    The lean code lies in the `formal/runtime_loaded_lean_files` folder which contains lean files
    that are **not** embedded in the lean checkpoint, but are loaded at runtime. Typically,
    it is faster to add new features to the lean environment by changing or adding lean files
    in this folder than to rebuild a new environment each time.
    The files related to proof cleaning itself are located in
    `formal/runtime_loaded_lean_files/cleaning_utils`, with one folder per version (e.g., `v0`, `v1`).
    The functions that the Python API calls are located in `do_clean_proof.lean`.

    The cleaning process is two-fold:

    1) `do_remove_useless_tactics` tries to remove tactics one by one, in the reverse order.
    The reason of using the reverse order is the following:
    if there are two unecessary tactics `tactic1` and `tactic2`, but `tactic2` depends on
    the result of `tactic1` to work, trying to remove `tactic1` first will fail,
    only `tactic2` will get removed; but if we first remove `tactic2`, then we can remove `tactic1`.

    It takes special care of the specific `have` case: when an unecessary hypothesis is introduced
    with a `have`, its full proof must be removed at once (trying to remove individual tactics
    within the proof of this hypothesis will fail)

    2) for each remaining tactic, a call to `do_clean_individual_tactic_at_idx` tries to clean
    the tactic at a given index (i.e., removing unecessary arguments or simplyifing a tactic).
    If the tactic belongs to a given set of identified potentially-cleanable tactics (e.g.,
    `norm_num`, `simp`, `field_simp`, `tidy`), it extract all arguments and creates a list of
    variants, from the simplest one to the most complex one (the initial tactic with all arguments).
    Conceptually these variants are just the "root" tactic applied to each one of the element of
    the power set of arguments (e.g., `tac [A,B,C]` has variants `tac A`, `tac B`, `tac C`,
    `tac [A,B]`, `tac [A,C]`, `tac [B,C]`, `tac [A,B,C]`).

    It has an extra flag `simp_only` so as to convert a `simp` tactic, that considers all theorems
    tagged with a `[simp]` flag, into a `simp_only` tactic that just uses the theorems passed as
    arguments.

    Each time a step is attempted (tactic removal or tactic cleaning), the resulting proof is ran
    with `check_and_get_proof` or `check_and_get_proof_from_steps` to see if the proof is still valid.
    A special care has been given on tactics that are applied on several goals at once: a tactic `tac`
    applied to the top goal of the goal stack is equivalent to `{ tac }`; a tactic `tac` applied to the
    top `N` goals of the goal stack is written as `solven N tac`.

    :param init_pr: proof to be cleaned
    :type init_pr: ProofResult
    :param params: cleaning params
    :type params: ProofCleaningParams
    :param pid: proof ID
    :type pid: ProofId
    :param env: environment
    :type env: LeanExpanderEnv
    """

    def __init__(
        self,
        init_pr: ProofResult,
        params: ProofCleaningParams,
        dump_path: Path,
        pid: ProofId,
        env: LeanExpanderEnv,
    ):
        self.init_pr = init_pr
        self.goal = init_pr.goal
        self.params = params
        assert self.level != CleaningLevel.No
        if not params.quiet:
            os.makedirs(dump_path, exist_ok=True)
            self.logger = create_logger(
                dump_path / f"{unique_name(init_pr.goal.name, pid)}.log",
                name=f"{unique_name(init_pr.goal.name, pid)}",
            )
        else:
            self.logger = NoneLogger()
        assert init_pr.proof is not None
        self.init_proof = init_pr.proof
        self.proof_after_tac_removal: Optional[Proof] = None
        self.cur_proof = init_pr.proof
        self.n_steps: int = 0
        self.done = self.invalid_init_pr
        self.invalid_uncleaned_proof: Optional[bool] = None
        self.idx: Optional[int] = None
        self.excs: List[Exception] = []
        self.tac_removal_success: Optional[bool] = None
        self.tac_cleaning_success: Optional[int] = None
        self.total_time = Timer()
        self.tac_removal_time = Timer()
        self.tac_cleaning_time: Optional[Timer] = (
            None if self.level == CleaningLevel.TacRemoval else Timer()
        )
        self.total_time.start()
        self.tac_removal_time.start()
        self.logger.info(
            f"Cleaner #{pid} for {init_pr.goal.name} started at level `{self.level}`"
        )

        self.pid = pid
        self.env = env  # used for get_label_full_name

    @property
    def level(self) -> CleaningLevel:
        return self.params.level

    def step(self, cmd_res: LeanEvalCmdResult) -> Optional[LeanEvalCmdJob]:
        assert not self.done
        self.n_steps += 1
        self.logger.info(f"received at step #{self.n_steps}: {cmd_res}")
        (start_res, (self.cur_proof, exc_end)) = apply_cmd_result(
            self.cur_proof, cmd_res, update_start=self.n_steps == 1
        )
        if self.n_steps == 1:
            self.tac_removal_time.stop()
            if self.level == CleaningLevel.TacRemoval:
                self.done = True
            assert start_res is not None
            start_proof, exc_start = start_res
            self.init_proof = start_proof
            if exc_start is not None:
                self.invalid_uncleaned_proof = True
                self.done = True
                self.logger.info(
                    "invalid initial (uncleaned) proof: "
                    f"{exc_start.__class__.__name__}: {exc_start}"
                )
                self.excs.append(exc_start)
            else:
                self.invalid_uncleaned_proof = False
                if exc_end is None:
                    self.tac_removal_success = True
                else:
                    self.logger.info(
                        "tac_removal failed: "
                        f"{exc_end.__class__.__name__}: {exc_start}"
                    )
                    self.tac_removal_success = False
                    self.excs.append(exc_end)
                if self.level in {
                    CleaningLevel.TacRemoval_TacClean,
                    CleaningLevel.TacRemoval_TacClean_SimpOnly,
                }:
                    assert self.tac_cleaning_time is not None
            if self.done:
                self.total_time.stop()
                return None
            self.proof_after_tac_removal = self.cur_proof
            actions_after_tac_removal = extract_actions_from_proof(
                self.proof_after_tac_removal, True
            )
            self.idx = len(actions_after_tac_removal)
        else:
            assert self.level in [
                CleaningLevel.TacRemoval_TacClean,
                CleaningLevel.TacRemoval_TacClean_SimpOnly,
            ]
            assert self.tac_cleaning_time is not None
            self.tac_cleaning_time.stop()
            if self.tac_cleaning_success is None:
                self.tac_cleaning_success = 0
            if exc_end is None:
                self.tac_cleaning_success += 1
            else:
                self.logger.info(
                    "tac_cleaning failed: " f"{exc_end.__class__.__name__}: {exc_end}"
                )
                self.excs.append(exc_end)
        assert self.proof_after_tac_removal is not None
        actions_after_tac_removal = extract_actions_from_proof(
            self.proof_after_tac_removal, True
        )
        assert self.idx is not None
        self.idx -= 1
        while self.idx >= 0 and actions_after_tac_removal[self.idx] in {
            "{",
            "}",
        }:
            self.idx -= 1
        if self.idx == -1:
            self.done = True
            self.total_time.stop()
            return None
        assert self.tac_cleaning_time is not None
        self.tac_cleaning_time.start()
        actions = extract_actions_from_proof(self.cur_proof, True)
        simp_only = self.level == CleaningLevel.TacRemoval_TacClean_SimpOnly
        label = self.env.get_label_full_name(self.goal.label)
        to_run = (
            f'do_clean_individual_tactic_at_idx {self.idx} "{label}" '
            f"{json.dumps(actions)} " + ("tt" if simp_only else "ff")
        )
        timeout = self.params.timeout_individual_tactic_cleaning
        module_path = self.env.get_cleaning_path(self.params.version)
        self.logger.info(
            f"next cmd LeanEvalCmdJob(to_run={to_run}, "
            f"module_path={module_path}, timeout=int({timeout} * 1000))"
        )
        return LeanEvalCmdJob(
            to_run=to_run, module_path=module_path, timeout=int(timeout * 1000)
        )

    @property
    def init_tactics(self) -> List[LeanTactic]:
        return extract_tactics_from_proof(self.init_proof)

    @property
    def invalid_init_pr(self) -> bool:
        if (
            not isinstance(self.init_pr, MCTSResult)
            or self.init_pr.sample_proof is None
        ):
            self.logger.info(f"invalid init_pr: {self.init_pr}")
            return True
        return False

    @property
    def init_cmd(self) -> LeanEvalCmdJob:
        actions = extract_actions_from_proof(self.init_proof, True)
        label = self.env.get_label_full_name(self.goal.label)
        to_run = f'do_remove_useless_tactics "{label}" {json.dumps(actions)}'
        timeout = self.params.timeout_useless_tactics_removal
        module_path = self.env.get_cleaning_path(self.params.version)
        self.logger.info(
            f"next cmd LeanEvalCmdJob(to_run={to_run}, "
            f"module_path={module_path}, timeout=int({timeout} * 1000))"
        )
        return LeanEvalCmdJob(
            to_run=to_run, module_path=module_path, timeout=int(timeout * 1000)
        )

    @property
    def final_proof(self) -> ProofResult:
        assert self.done
        if self.invalid_init_pr:
            return self.init_pr

        # here we assume that there is only one "minproof", which is the one we clean
        # this is typically the case with `self.mcts.params.proof_stype == "time"`
        mcts_samples_tactic: List[MCTSSampleTactics] = fold_proof(
            self.cur_proof,
            [],
            lambda p, l: l
            + [
                MCTSSampleTactics(
                    goal=get_thm(p),
                    tactics=[get_tac(p)],
                    target_pi=[1],
                    inproof=0,  # not used for training
                    q_tactics=None,  # not used for training
                    label=self.goal.label,
                    visit_count=0,  # not used for training
                )
            ],
        )
        assert isinstance(self.init_pr, MCTSResult)
        assert self.init_pr.sample_proof is not None
        sample_proof = deepcopy(self.init_pr.sample_proof)
        sample_proof.samples = (
            mcts_samples_tactic  # shouldn't we update `size` as well?
        )
        stats = self.init_pr.stats
        stats["cleaning/total_time"] = (self.total_time.rate_and_reset()["cum_time"], 1)
        stats["cleaning/tac_removal_time"] = (
            self.tac_removal_time.rate_and_reset()["cum_time"],
            1,
        )
        if self.tac_cleaning_time is not None:
            for k, v in self.tac_cleaning_time.rate_and_reset().items():
                stats[f"cleaning/tac_cleaning_time/{k}"] = (v, 1)
        if self.invalid_uncleaned_proof:
            stats["cleaning/invalid_start"] = (1.0, 1)
        else:
            stats["cleaning/invalid_start"] = (0.0, 1)
            assert self.tac_removal_success is not None
            stats["cleaning/tac_removal_success"] = (self.tac_removal_success, 1)
            stats["cleaning/tac_won"] = (self.tac_won, 1)
            stats["cleaning/tac_won_per_tac"] = (
                self.tac_won / len(self.init_tactics),
                1,
            )
            if self.level in {
                CleaningLevel.TacRemoval_TacClean,
                CleaningLevel.TacRemoval_TacClean_SimpOnly,
            }:
                assert self.tac_cleaning_success is not None
                stats["cleaning/tac_cleaning_success"] = (
                    self.tac_cleaning_success,
                    1,
                )
                assert self.proof_after_tac_removal is not None
                num_tactic_after_tac_removal = len(
                    extract_tactics_from_proof(self.proof_after_tac_removal)
                )
                stats["cleaning/tac_cleaning_success_per_tac"] = (
                    self.tac_cleaning_success / num_tactic_after_tac_removal,
                    1,
                )
                stats["cleaning/tac_cleaned"] = (self.tac_cleaned, 1)
                stats["cleaning/tac_cleaned_per_tac"] = (
                    self.tac_cleaned / num_tactic_after_tac_removal,
                    1,
                )
        for k in stats:
            if not k.startswith("cleaning/"):
                continue
            self.logger.info(f"STAT {k}: {stats[k]}")

        return MCTSResult(
            proof=self.cur_proof,
            goal=self.goal,
            exception=None,
            mcts_samples_critic=self.init_pr.mcts_samples_critic,
            mcts_samples_tactic=mcts_samples_tactic,
            mcts_samples_effect=self.init_pr.mcts_samples_effect,
            sample_proof=sample_proof,
            simplified_state=self.init_pr.simplified_state,
            stats=stats,
            hist_stats=self.init_pr.hist_stats,
        )

    @property
    def tac_won(self) -> int:
        """
        :return: Number of tactics "won" over the initial proof
        :rtype: int
        """
        assert self.n_steps >= 1
        return len(extract_tactics_from_proof(self.init_proof)) - len(
            extract_tactics_from_proof(self.cur_proof)
        )

    @property
    def tac_cleaned(self) -> int:
        """
        :return: Number of tactics "cleaned" over the initial proof
        :rtype: int
        """
        assert self.level in {
            CleaningLevel.TacRemoval_TacClean,
            CleaningLevel.TacRemoval_TacClean_SimpOnly,
        }
        assert self.done
        assert self.proof_after_tac_removal is not None
        acs_afer_tac_removal = extract_actions_from_proof(
            self.proof_after_tac_removal, True
        )
        last_acs = extract_actions_from_proof(self.cur_proof, True)
        if "tidy" not in acs_afer_tac_removal:  # tidy is expanded as if tidy?
            assert len(acs_afer_tac_removal) == len(
                last_acs
            ), f"{self.goal.label}: {acs_afer_tac_removal} {last_acs}"
        return sum(ac1 != ac2 for ac1, ac2 in zip(acs_afer_tac_removal, last_acs))


class AsyncProofCleaner:
    """
    Modeled on `AsyncProofStitcher`, it creates one `Cleaner` per proof
    when `AsyncProofCleaner.process` is called
    Then after each call to `AsyncProofCleaner.get_ready`, all `Cleaner` objects
    which is env has produced a result for is asked to do the next cleaning step
    The `Cleaner` objects themselves just produce the next command to be run by
    the env.
    `AsyncProofCleaner.get_ready` returns all the cleaned proofs whose `Cleaner`
    is done

    :param env: lean environment
    :type env: ExpanderEnv
    :param prover_params: prover params, which contain the cleaning params
    :type prover_params: ProverParams
    """

    def __init__(self, env: ExpanderEnv, prover_params: ProverParams):
        self.env: Optional[LeanExpanderEnv] = (
            env if isinstance(env, LeanExpanderEnv) else None
        )
        self.prover_params = prover_params
        self.processing: Dict[ProofId, Cleaner] = {}
        self.results: Dict[ProofId, Tuple[ProofResult, List[Exception]]] = {}
        self.dump_path = prover_params.dump_path / "cleaner"
        self.dump_path_ongoing = self.dump_path / "ongoing"
        self.dump_path_good = self.dump_path / "all_good"
        self.dump_path_errors = self.dump_path / "errors"
        if not self.quiet_cleaning:
            os.makedirs(self.dump_path, exist_ok=True)
            self.logger = get_tail_logger(
                self.dump_path / f"jid{this_job_id()}_async_proof_cleaner.log"
            )
        else:
            self.logger = NoneLogger()
        if self.dump_cleaning or not self.quiet_cleaning:
            os.makedirs(self.dump_path_ongoing, exist_ok=True)
            os.makedirs(self.dump_path_good, exist_ok=True)
            os.makedirs(self.dump_path_errors, exist_ok=True)
        self.logger.info(f"Start AsyncProofCleaner with level `{self.level}`")

    @property
    def params(self) -> ProofCleaningParams:
        return self.prover_params.proof_cleaning_params

    @property
    def level(self) -> CleaningLevel:
        if self.env is None:
            return CleaningLevel.No
        return self.params.level

    @property
    def quiet_cleaning(self) -> bool:
        if self.env is None:
            return True
        return self.params.quiet

    @property
    def dump_cleaning(self) -> bool:
        if self.level == CleaningLevel.No:
            return False
        return self.params.dump

    def process(self, pid: ProofId, to_process: ProofResult):
        self.logger.info(
            f"job_id {this_job_id()} process #{pid}: goal `{to_process.goal.label}` "
            f"({to_process.goal.name}) with proof: {to_process.proof}"
        )
        if to_process.proof is None or self.level == CleaningLevel.No:
            self.results[pid] = (to_process, [])
            return
        dump_path_ongoing = self.dump_path_ongoing / unique_name(
            to_process.goal.name, pid
        )
        if not self.quiet_cleaning or self.dump_cleaning:
            os.makedirs(dump_path_ongoing, exist_ok=True)

        assert self.env is not None
        cl = Cleaner(
            init_pr=to_process,
            params=self.params,
            dump_path=dump_path_ongoing,
            pid=pid,
            env=self.env,
        )
        self.env.process(
            task=cl.init_cmd, batch_id=f"cleaning_{pid}",
        )
        self.logger.info(
            f"PID #{pid} - Cleaner process "
            f"`LeanEvalCmdJob(to_run={cl.init_cmd.to_run}, timeout={cl.init_cmd.timeout})`"
        )

        self.processing[pid] = cl
        if self.dump_cleaning:
            with open(
                dump_path_ongoing
                / f"{unique_name(to_process.goal.name, pid)}_BEFORE_CLEANING.pkl",
                "wb",
            ) as f:
                pickle.dump(to_process.proof, f)

    def _dump_ret(
        self, to_ret: List[Tuple[ProofId, Tuple[ProofResult, List[Exception]]]]
    ):
        if (
            not self.quiet_cleaning or self.dump_cleaning
        ) and self.level != CleaningLevel.No:
            for pid, (pr, excs) in to_ret:
                if pr.proof is None:
                    continue
                dump_path_ongoing = self.dump_path_ongoing / unique_name(
                    pr.goal.name, pid
                )
                if self.dump_cleaning and pr.proof is not None:
                    with open(
                        dump_path_ongoing
                        / f"{unique_name(pr.goal.name, pid)}_AFTER_CLEANING.pkl",
                        "wb",
                    ) as f:
                        pickle.dump(pr.proof, f)
                try:
                    if excs:
                        for stre in {e.__class__.__name__ for e in excs}:
                            dump_path_err = (
                                self.dump_path_errors
                                / stre
                                / unique_name(pr.goal.name, pid)
                            )
                            shutil.copytree(dump_path_ongoing, dump_path_err)
                        shutil.rmtree(dump_path_ongoing)
                    else:
                        shutil.move(str(dump_path_ongoing), self.dump_path_good)
                except FileNotFoundError as e:
                    self.logger.info(f"PID #{pid} - {e.__class__.__name__}: {e}")

    def get_ready(self,) -> List[Tuple[ProofId, Tuple[ProofResult, List[Exception]]]]:
        # retrieve results from env
        log_processing = self.processing != {}
        if log_processing:
            self.logger.info(
                "before processing: "
                f"#{', '.join([str(pid)+'-'+cl.goal.name for pid, cl in self.processing.items()])}"
            )
        for pid, cl in self.processing.items():
            assert self.env is not None
            res = self.env.maybe_get_result(f"cleaning_{pid}")
            if res is None:
                continue
            assert isinstance(res, LeanEvalCmdResult)
            self.logger.info(f"PID #{pid} - Cleaner received: {res}")
            next_cmd = cl.step(res)
            if cl.done:
                assert next_cmd is None
                final_proof = cl.final_proof
                self.results[pid] = (final_proof, cl.excs)
                stats = (
                    ""
                    if not isinstance(final_proof, MCTSResult)
                    else f" - stats: {final_proof.stats}"
                )
                self.logger.info(
                    f"PID #{pid} - Cleaner done{stats} "
                    f"- `{final_proof.goal.label}` ({final_proof.goal.name}) "
                    f"with proof: {final_proof.proof}"
                )
                continue
            assert self.env is not None
            assert next_cmd is not None
            self.env.process(
                task=next_cmd, batch_id=f"cleaning_{pid}",
            )
            self.logger.info(
                f"PID #{pid} - Cleaner process "
                f"`LeanEvalCmdJob(to_run={next_cmd.to_run}, timeout={next_cmd.timeout})`"
            )
            self.processing[pid] = cl
        for pid in self.results:
            try:
                del self.processing[pid]
            except KeyError:
                pass
        if log_processing:
            self.logger.info(
                "after processing: "
                f"#{', '.join([str(pid)+'-'+cl.goal.name for pid, cl in self.processing.items()])}"
            )

        # return all results
        to_ret = list(self.results.items())
        self.results.clear()
        if to_ret:
            self.logger.info(f"to_ret: {to_ret}")
        self._dump_ret(to_ret)
        return to_ret

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        return stats
