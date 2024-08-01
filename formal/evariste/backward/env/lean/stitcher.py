# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Any, List, Dict, Tuple, Iterator, Set, Union, Optional
from copy import deepcopy
import re
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import pickle
import os

from leanml.parse_goal import parse_goal_structured, StructuredGoal
from evariste.backward.env.lean.env import DeclNotFound, LeanExpanderEnv, ExpanderEnv
from evariste.backward.prover.mcts import MCTSResult
from evariste.metrics import Timer, StatsCollection
from evariste.backward.env.lean.graph import (
    LeanTheorem,
    LeanTactic,
    LeanState,
    LeanContext,
)
from evariste.backward.graph import (
    BackwardGoal,
    Proof,
    NonPicklableProof,
    ProofId,
    UnMaterializedTheorem,
)
from evariste.logger import create_logger
from evariste.backward.prover.core import ProofResult
from evariste.backward.env.lean.env import LeanTacticOnState

logger = create_logger(None)


# regex for what's expected before or after a name in a lean tactic state
BEFORE = r"(?<=[\[\(\{⟨@, :←])"
AFTER = r"(?=[\]\)\}⟩,\. \n:\+]|$)"


def normalize_str(s: str) -> str:
    return re.sub("\s+", " ", s).strip()


def get_ordered_hyp_names(goals: List[List[StructuredGoal]]) -> List[str]:
    ordered_hyp_names = []
    for splitted in goals:
        for goal in splitted:
            for hyp in goal.hyps:
                names, _hyp_type = [
                    normalize_str(x) for x in hyp.hyp.split(":", maxsplit=1)
                ]
                for x in names.split(" "):
                    if not x.startswith("_inst"):
                        ordered_hyp_names.append(x)
    return ordered_hyp_names


def stitcher(
    base: List[List[StructuredGoal]], true: List[List[StructuredGoal]], tac: LeanTactic
) -> LeanTactic:
    assert len(base) == len(true)
    to_rename: Dict[str, str] = {}
    base_ordered_hyp_names = get_ordered_hyp_names(base)
    true_ordered_hyp_names = get_ordered_hyp_names(true)

    assert len(base_ordered_hyp_names) == len(true_ordered_hyp_names), (
        base_ordered_hyp_names,
        true_ordered_hyp_names,
    )

    for b, t in zip(base_ordered_hyp_names, true_ordered_hyp_names):
        if len(re.findall(rf"{BEFORE}{b}{AFTER}", str(tac))) > 0:
            if b in to_rename:
                assert to_rename[b] == t, f"multiple renamings ? {base} {true}"
            to_rename[b] = t
    assert len(set(to_rename.values())) == len(
        to_rename.values()
    ), f"multiple renames to ({to_rename})"

    tac_str = str(tac)

    all_substitutions: List[Tuple[int, int, str]] = []
    for base_name, true_name in to_rename.items():
        regex = rf"{BEFORE}{base_name}{AFTER}"
        for m in re.finditer(regex, tac_str):
            all_substitutions.append((*m.span(), true_name))
    all_substitutions = sorted(all_substitutions)

    # once all substitutions have been found, apply them
    if len(all_substitutions) > 0:
        prev: int = 0
        final_str: str = ""
        for begin, end, r in all_substitutions:
            final_str += tac_str[prev:begin] + r
            prev = end
        final_str += tac_str[prev:]
    else:
        final_str = tac_str

    to_ret = deepcopy(tac)
    to_ret._tactic = final_str
    return to_ret


def stitch_children(
    base: List[List[StructuredGoal]], true: List[List[StructuredGoal]], tac: LeanTactic
) -> Iterator[LeanTactic]:
    assert len(base) == len(true), "mismatched len children"
    to_rename: Dict[str, Set[str]] = defaultdict(set)
    base_ordered_hyp_names = get_ordered_hyp_names(base)
    true_ordered_hyp_names = get_ordered_hyp_names(true)

    assert len(base_ordered_hyp_names) == len(true_ordered_hyp_names), (
        "mismatched hyps children",
        base_ordered_hyp_names,
        true_ordered_hyp_names,
    )

    for b, t in zip(base_ordered_hyp_names, true_ordered_hyp_names):
        if len(re.findall(rf"{BEFORE}{t}{AFTER}", str(tac))) > 0:
            to_rename[t].add(b)

    if len(to_rename) == 0:
        yield tac
        return

    # Issue : for each renaming, there might be multiple occurence of the name
    # in the tactic. We don't know which one to change. For example `have h := h0`
    # which has been stitched to `have h := h`. We need to change the first h,
    # but leave the second intact.

    all_substitutions: List[Tuple[int, int, Set[str]]] = []
    for basename, trues in to_rename.items():
        tac_str = str(tac)

        regex = rf"{BEFORE}{basename}{AFTER}"
        for m in re.finditer(regex, tac_str):
            all_substitutions.append((*m.span(), trues))
        all_substitutions = sorted(all_substitutions)

    def walk_substitutions(substs, prev, cur_str, base_str):
        if len(substs) == 0:
            to_ret = deepcopy(tac)
            to_ret._tactic = cur_str + base_str[prev:]
            yield to_ret
            return
        begin, end, trues = substs[0]
        new_str = cur_str + base_str[prev:begin]
        # maybe no replace
        if base_str[begin:end] not in trues:
            yield from walk_substitutions(
                substs[1:], end, new_str + base_str[begin:end], base_str
            )
        for true in trues:
            yield from walk_substitutions(substs[1:], end, new_str + true, base_str)
        return

    yield from walk_substitutions(all_substitutions, 0, "", tac._tactic)


class StitchError(Exception):
    pass


class ProofStitcher:
    def __init__(self, do_stitch: bool = True):
        self.input: Union[None, LeanTheorem, dict] = None
        self.output: Optional[Proof] = None
        self.error: Optional[str] = None
        self.do_stitch = do_stitch
        self.session: str = ""

    def stitch_proof(
        self, proof: Proof, debug: bool = False
    ) -> Iterator[Tuple[LeanState, LeanTactic]]:
        assert isinstance(self.input, LeanTheorem), self.input
        assert self.input.state is not None, self.input
        root = self.input
        self.session = self.input.state.session

        def visit_proof(
            cur: LeanTheorem, p: Proof
        ) -> Iterator[Tuple[LeanState, LeanTactic]]:
            assert not isinstance(p, NonPicklableProof)
            assert cur.state is not None
            th, tac, sp = p
            assert isinstance(th, LeanTheorem)
            assert isinstance(tac, LeanTactic)

            stitched_tac = tac
            if self.do_stitch:
                true_goals = parse_goal_structured(cur.conclusion)
                base_goals = parse_goal_structured(th.conclusion)
                try:
                    stitched_tac = stitcher([base_goals], [true_goals], tac)
                except AssertionError as e:
                    if debug:
                        logger.info("~~~")
                        logger.info(e)
                        logger.info(cur.conclusion)
                        logger.info("\n\n\n")
                        logger.info(th.conclusion)
                    raise e

            expected_children = [c[0] for c in sp]  # type: ignore

            yield (cur.state, stitched_tac)

            res = self.input
            assert isinstance(res, dict), res
            if "error" in res:
                raise StitchError(res["error"])
            if len(res["nodes"]) != len(sp):
                raise StitchError(
                    f"Node length mismatch : got {res['nodes']} \n\n wanted {sp}"
                )

            # for tactics that create hypothesis (have, intro)
            # we match the expected name to avoid surprises with repeated hyp names

            children = [
                LeanTheorem(
                    conclusion=node["full_pp"],
                    context=LeanContext(set()),
                    state=LeanState(session=cur.state.session, node_id=node["node_id"]),
                )
                for node in res["nodes"]
            ]
            if not self.do_stitch:
                sps = []
                for child, subproof in zip(children, sp):
                    yield from visit_proof(child, subproof)
                    sps.append(self.output)
                self.output = (cur, stitched_tac, sps)  # type: ignore
                return
            else:
                # now check that we didn't introduce any bad names
                parsed_c = [parse_goal_structured(c.conclusion) for c in children]
                parsed_e = [
                    parse_goal_structured(c.conclusion) for c in expected_children
                ]
                stitched2 = list(stitch_children(parsed_e, parsed_c, stitched_tac))
                if debug and any([str(x) != str(tac) for x in stitched2]):
                    logger.info(f"STITCHING {tac} {stitched_tac} {stitched2}")
                for stitched_tac2 in stitched2:
                    try:
                        yield (cur.state, stitched_tac2)
                        res = self.input
                        assert isinstance(res, dict), res
                        assert "error" not in res, (tac, stitched_tac2, res)
                        assert len(res["nodes"]) == len(sp), (res, sp)

                        children2 = [
                            LeanTheorem(
                                conclusion=node["full_pp"],
                                context=LeanContext(set()),
                                state=LeanState(
                                    session=cur.state.session, node_id=node["node_id"]
                                ),
                            )
                            for node in res["nodes"]
                        ]

                        sps = []
                        for child, subproof in zip(children2, sp):
                            yield from visit_proof(child, subproof)
                            sps.append(self.output)
                        self.output = (cur, stitched_tac2, sps)  # type: ignore
                        return
                    except (AssertionError, StitchError) as e:
                        if debug:
                            logger.info("-----")
                            logger.info(
                                f"{type(e)}\n{e}\n{cur.conclusion}\n\n{th.conclusion}"
                            )
                        continue
                raise StitchError()

        try:
            yield from visit_proof(root, proof)
        except (AssertionError, StitchError) as e:
            self.output = None
            self.error = f"{type(e)} -- {e}"


@dataclass
class StitcherTimers(StatsCollection):
    next: Timer = field(default_factory=lambda: Timer())
    get_ready: Timer = field(default_factory=lambda: Timer())
    materialize: Timer = field(default_factory=lambda: Timer())
    get_from_env: Timer = field(default_factory=lambda: Timer())
    stitch: Timer = field(default_factory=lambda: Timer())
    theorems_ready: Timer = field(default_factory=lambda: Timer())
    maybe_get_results: Timer = field(default_factory=lambda: Timer())


class AsyncProofStitcher:
    def __init__(self, env: ExpanderEnv, try_stitch: bool, failed_stitch_path: Path):
        self.failed_stitch_path = failed_stitch_path
        self.env: Optional[LeanExpanderEnv] = None
        if try_stitch and isinstance(env, LeanExpanderEnv):
            self.env = env
            os.makedirs(self.failed_stitch_path, exist_ok=True)
        self.processing: Dict[ProofId, ProofResult] = {}
        self.materializing: Dict[ProofId, BackwardGoal] = {}
        self.stitchers: Dict[ProofId, ProofStitcher] = {}
        self.stitcher_it: Dict[ProofId, Iterator[Tuple[LeanState, LeanTactic]]] = {}
        self.waiting_for_env: Set[ProofId] = set()
        self.results: Dict[ProofId, Tuple[ProofResult, Optional[Exception]]] = {}

        self.timers = StitcherTimers()

    def process(self, proof_id: ProofId, to_process: ProofResult):
        # not stitching if not in lean
        if self.env is None or to_process.proof is None:
            self.results[proof_id] = (to_process, None)
            return
        self.processing[proof_id] = to_process
        # de-materialize goal because we need a new session without merge.
        new_goal = deepcopy(to_process.goal)
        new_goal.theorem = UnMaterializedTheorem(label=new_goal.label)
        new_goal.theorem.batch_id = self.env.submit_create_session(
            new_goal.label, merge_alpha_equiv=False
        )
        self.materializing[proof_id] = new_goal

    def get_ready(
        self,
    ) -> List[Tuple[ProofId, Tuple[ProofResult, Optional[Exception]]]]:
        self.timers.get_ready.start()

        # check if anything has been materialized
        if len(self.materializing) > 0:
            self.timers.materialize.start()
            # ignoring mypy here, would need to assert that goal.theorem is unmaterialized.
            pids, batch_ids = zip(
                *[
                    (pid, goal.theorem.batch_id)  # type: ignore
                    for pid, goal in self.materializing.items()
                ]
            )
            # zip mishandled by mypy
            self.timers.theorems_ready.start()
            ready = self.env.theorems_ready(batch_ids)  # type: ignore
            self.timers.theorems_ready.stop()
            for pid, thm in zip(pids, ready):
                if thm is None:
                    continue
                if isinstance(thm, str):
                    self.results[pid] = (self.processing.pop(pid), DeclNotFound(thm))
                    self.materializing.pop(pid)
                    continue

                ps = ProofStitcher()
                ps.input = thm
                self.stitchers[pid] = ps
                to_stitch = self.processing[pid].proof
                assert to_stitch is not None

                self.stitcher_it[pid] = self.stitchers[pid].stitch_proof(to_stitch)
            self.timers.materialize.stop()

        # check if we got anything from env
        self.timers.get_from_env.start()
        if len(self.waiting_for_env) > 0:
            # if necessary otherwise the zip breaks
            pids, batch_ids = zip(
                *[(pid, f"stitcher_{pid}") for pid in self.waiting_for_env]
            )
            self.timers.maybe_get_results.start()
            maybes = self.env.maybe_get_results(batch_ids)  # type: ignore
            self.timers.maybe_get_results.stop()
            assert len(maybes) == len(pids), (len(maybes), len(pids))
            for pid, maybe in zip(pids, maybes):
                if maybe is not None:
                    self.waiting_for_env.remove(pid)
                    self.stitchers[pid].input = maybe
        self.timers.get_from_env.stop()

        # for all received inputs, stitch
        self.timers.stitch.start()
        for pid, stitcher in list(self.stitchers.items()):
            assert self.env is not None
            if stitcher.input is not None:
                try:
                    self.timers.next.start()
                    state, tac = next(self.stitcher_it[pid])
                    self.timers.next.stop()
                    self.env.process(
                        LeanTacticOnState(state, tac), batch_id=f"stitcher_{pid}"
                    )
                    self.waiting_for_env.add(pid)
                    stitcher.input = None
                except StopIteration:
                    self.timers.next.stop()
                    res = self.processing.pop(pid)
                    assert isinstance(res, MCTSResult), res
                    ex: Optional[Exception] = None
                    if stitcher.output is not None:
                        res.proof = stitcher.output
                    else:
                        ex = StitchError(stitcher.error)
                        # dump errors for further debugging
                        with open(
                            self.failed_stitch_path / (res.goal.name + ".pkl"), "wb"
                        ) as f:
                            pickle.dump(res.proof, f)
                    res.stats["stitching_error"] = (ex is not None, 1)
                    self.results[pid] = (res, ex)
                    self.env.submit_del_session(self.stitchers[pid].session)
                    del self.stitcher_it[pid]
                    del self.stitchers[pid]
        self.timers.stitch.stop()

        # return all results
        to_ret = list(self.results.items())
        self.results.clear()
        self.timers.get_ready.stop()
        return to_ret

    def get_stats(self) -> Dict[str, Any]:
        return {f"stitcher/{x}": y for x, y in self.timers.rate_and_reset().items()}
