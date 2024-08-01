# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import defaultdict
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from typing import (
    List,
    Union,
    Dict,
    Optional,
    Tuple,
    Iterator,
)
from contextlib import closing
import os
import time
import random
import cProfile

from evariste import json as json
from evariste.async_workers.async_worker import AsyncWorker, RequestId
from evariste.async_workers.async_worker_helpers import make_iterator
from evariste.backward.prover.expander import MPExpander
from evariste.backward.prover.prover_args import (
    ProofStatus,
    ProofStatusBackwardRunner,
    ConditioningKind,
    ProverParams,
    ProverKind,
)
from evariste.utils import logged_closing
from evariste.backward.prover.utils import GPUMonitor
from evariste.backward.env.core import BackwardEnv, BackwardGoal, EnvExpansion
from evariste.backward.graph import Proof, Theorem, GoalParams, UnMaterializedTheorem
from evariste.backward.env.lean.stitcher import AsyncProofStitcher
from evariste.backward.env.lean.cleaner import AsyncProofCleaner
from evariste.backward.env.lean.graph import LeanTactic
from evariste.backward.env.lean.env import (
    DeclNotFound,
    LeanExpanderEnv,
)
from evariste.backward.prover.core import ProofResult
from evariste.datasets import DatasetConf
from evariste.metrics import Logger, ActionCounter, Timer, StatsCollection
from evariste.model.transformer_args import DecodingParams
from evariste.utils import PROFILE, wrap_timer, get_tail_logger, NoneLogger
from evariste.utils import rstr, this_job_id, get_mean, get_max
from lean_cluster.instance import LeanInstanceDied

from evariste.backward.prover_factory import (
    backward_init,
    get_prover_handler,
)
from evariste.backward.prover.core import ProofHandler
from evariste.backward.prover.mcts_prover import MCTSProofHandlerGenerator


ProofId_t = int


logger = getLogger("prover")


class TooLongInExpand(Exception):
    pass


# Use instead of the Tuple in curr proof for better typing
class ProofRecord:
    def __init__(self, proof_handler: ProofHandler):
        self.proof_handler: ProofHandler = proof_handler
        self._time_per_status: Dict[str, float] = defaultdict(float)
        self.s = ProofStatusBackwardRunner()
        self._last_status_change = time.time()
        self._creation_time = time.time()

        # Holds the label used for conditioning
        self.conditioning_label: Optional[str] = None


@dataclass
class ExpansionHandlerTimers(StatsCollection):
    expander_receive_expansions: Timer = field(default_factory=lambda: Timer())
    env_send_expansions: Timer = field(default_factory=lambda: Timer())
    env_receive_expansions: Timer = field(default_factory=lambda: Timer())

    env_receive_expansions__step: Timer = field(default_factory=lambda: Timer())


class BatchExpansionHandler:
    """
    input -> List[Theorem] to expand
    output -> List[EnvExpansion]
    replaces the old system which waited for a full GPU expansion to send expansions to the environment

    .. note::
      :class:`MPExpander` and :class:`BackwardEnv` return results on a theorem/theorem basis.
      This class returns on a proof (ie batch of theorem) basis.
      
      In particular, as soon as a single theorem is processed by the model, it is sent to be processed by the env.

    :param env: the backward env
    :type env: :class:`BackwardEnv`
    :param expander: the model expander
    :type expander: :class:`MPExpander`
    """

    def __init__(self, env: BackwardEnv, expander: MPExpander):
        self.env = env
        self.expander = expander
        # book-keeping
        self.processing: Dict[int, List[Theorem]] = {}
        self.received: Dict[int, Dict[int, EnvExpansion]] = defaultdict(dict)
        self.sum_stats: Dict[str, int] = defaultdict(int)
        self.avg_stats: Dict[str, ActionCounter] = defaultdict(
            lambda: ActionCounter(name="", is_rate=False, silent=True)
        )
        self.timers = ExpansionHandlerTimers()

    def send_theorems_to_expand(
        self,
        proof_id: int,
        to_expand: List[Theorem],
        params: Optional[GoalParams] = None,
    ):
        """
        Send a batch of :class:`Theorem` associated to a `proof_id` for expansion.
        
        :meth:`ready_env_expansions` will return when we have a complete expansions for all of theorems in the batch.

        :param proof_id: the proof id
        :type proof_id: int
        :param to_expand: list of theorems to expand
        :type to_expand: List[Theorem]
        :param params: some optional goal params for the model expander
        :type params: Optional[GoalParams]
        """
        start = time.time()
        assert proof_id not in self.processing
        self.processing[proof_id] = to_expand
        if len(self.processing[proof_id]) == 0:
            self.received[proof_id] = {}
        else:
            self.expander.process_async(proof_id, to_expand, params)
        self.avg_stats["send_theorems_to_expand"].act(time.time() - start)

    def ready_env_expansions(
        self, prover_params: ProverParams
    ) -> List[Tuple[int, List[EnvExpansion]]]:
        """
        First check if anything has been expanded by the model expander and sends these for expansions by the env.
        Then, goes through all current `EnvExpansion` and if any is complete, output it.

        Some filtering is done : 
        
        * if `prover_params.only_keep_one_solving` is set, only keep one solving tactic

        * call :meth:`ExpanderEnv.filter_model_expansion` which modifies the output of the model expander. See what's done in :class:`LeanExpanderEnv` for example.
        """
        # check if anything ready from the GPU
        start = time.time()
        self.timers.expander_receive_expansions.start()
        model_expansions = self.expander.ready_model_expansions()
        self.timers.expander_receive_expansions.stop()
        if len(model_expansions) > 0:
            self.avg_stats["received_model_expansions"].act(len(model_expansions))
            self.avg_stats["ready_env_expansions__got"].act(time.time() - start)
        self.timers.env_send_expansions.start()
        for pid, tid, model_expansion in model_expansions:
            self.avg_stats["expander_exp_time"].act(model_expansion.exp_duration)
            self.avg_stats["expander_gpu_time"].act(model_expansion.gpu_duration)
            if not model_expansion.is_error:
                assert model_expansion.tactics is not None
                self.avg_stats["tactics_from_model"].act(len(model_expansion.tactics))
                filtered = self.env.expander_env.filter_model_expansion(model_expansion)
                self.avg_stats["tactics_filtered"].act(filtered)
            self.avg_stats["model_expansion_errors"].act(int(model_expansion.is_error))

            theorem = self.processing[pid][tid]
            if model_expansion.is_error:
                assert model_expansion.error is not None
                self.received[pid][tid] = EnvExpansion.build_error(
                    theorem=theorem,
                    exp_duration=model_expansion.exp_duration,
                    gpu_duration=model_expansion.gpu_duration,
                    env_durations=[],
                    error=model_expansion.error,
                )
                continue
            assert model_expansion.tactics is not None
            self.timers.env_receive_expansions__step.start()
            self.env.apply_model_expansion(theorem, model_expansion, pid, tid=tid)
            self.timers.env_receive_expansions__step.stop()
            if isinstance(self.env.expander_env, LeanExpanderEnv):
                if self.env.expander_env.dataset.dump_tactic_batches:
                    tactics = [
                        repr(LeanTactic.from_tokens(tokens))
                        for tokens in model_expansion.tactics
                    ]
                    logger.info(f"TACTIC BATCH {json.dumps(tactics)}")

            self.sum_stats["n_expanded_goals"] += 1
            self.sum_stats["n_expanded_goal_tactics"] += len(model_expansion.tactics)
        self.timers.env_send_expansions.stop()

        self.timers.env_receive_expansions.start()
        for pid, tid, expansion_result in self.env.ready_env_expansions(prover_params):
            env_exp_time = sum(expansion_result.env_durations)
            self.avg_stats["env_expansion_time"].act(env_exp_time)
            self.avg_stats["env_over_exp_time_ratio"].act(
                env_exp_time / (expansion_result.exp_duration + 1e-3)
            )
            self.avg_stats["env_over_gpu_time_ratio"].act(
                env_exp_time / (expansion_result.gpu_duration + 1e-3)
            )
            self.received[pid][tid] = expansion_result
        self.timers.env_receive_expansions.stop()

        # add to ready when ready, i.e. return when we are done processing
        # all theorems returned by `MCTSHandler.get_theorems_to_expand`
        to_return = []
        for pid, expansions in self.received.items():
            if len(expansions) == len(self.processing[pid]):
                to_return.append(
                    (pid, [expansions[i] for i in range(len(self.processing[pid]))])
                )

        # clean-up
        pids_to_return = [x[0] for x in to_return]
        assert len(set(pids_to_return)) == len(pids_to_return)
        for pid, _ in to_return:
            del self.processing[pid]
            del self.received[pid]

        return to_return


@dataclass
class BackwardRunnerTimers(StatsCollection):
    expansion_handler: ExpansionHandlerTimers

    proof_cycles: Timer = field(default_factory=lambda: Timer())
    grab_proofs: Timer = field(default_factory=lambda: Timer())
    add_to_expander: Timer = field(default_factory=lambda: Timer())
    send_env_expansions: Timer = field(default_factory=lambda: Timer())
    stitch_proofs: Timer = field(default_factory=lambda: Timer())
    clean_proofs: Timer = field(default_factory=lambda: Timer())
    finish_proofs: Timer = field(default_factory=lambda: Timer())
    log_runner_stats: Timer = field(default_factory=lambda: Timer())
    hearbeat: Timer = field(default_factory=lambda: Timer())
    check_unmaterialized: Timer = field(default_factory=lambda: Timer())
    get_done: Timer = field(default_factory=lambda: Timer())

    # finish proof timers
    finish_proofs__get_result: Timer = field(default_factory=lambda: Timer())
    finish_proofs__yield: Timer = field(default_factory=lambda: Timer())
    finish_proofs__finish_goal: Timer = field(default_factory=lambda: Timer())


ProverOutput = Tuple[ProofResult, Dict[str, Tuple[float, float]]]


class BatchBackwardRunner(AsyncWorker[BackwardGoal, ProverOutput]):
    """
    This function handles the proving process of batched-theorems. It feeds to async process the goal to expand to the
    GPU, then it feeds the tactic to the env and then feeds those which are valid to the proof handler which decide how
    to continue the proof from this new state.

    :param dataset: The dataset configuration used to instantiate the `BackwardEnv`.
    :type dataset: `DatasetConf`
    :param decoding: The decoding params used to instantiate the `MPExpander`
    :type decoding: `DecodingParams`
    :param prover_params: The prover params used to instantiate the `ProofHandler`. Also contain some logging options, the dump_path and number of simultaneous proofs.
    :type prover_params: `ProverParams`
    """

    def __init__(
        self,
        dataset: DatasetConf,
        decoding: DecodingParams,
        prover_params: ProverParams,
        decoder_type: str,
    ):
        # this heavy initialisation should be more done in the start() but this would require to
        # have everything Optional (or to encapsulate the started stuff in another object ?)

        env_gen, env, expander = backward_init(
            dataset, decoding, prover_params, decoder_type
        )

        self.env_gen = env_gen

        self.proof_handler = get_prover_handler(prover_params)

        self.env = env
        self.stitcher = AsyncProofStitcher(
            env=env.expander_env,
            try_stitch=prover_params.try_stitch,
            failed_stitch_path=prover_params.dump_path / "stitch_failed",
        )
        self.cleaner = AsyncProofCleaner(
            env=env.expander_env, prover_params=prover_params
        )
        self.expander = expander
        self.params = prover_params
        self.expansion_handler = BatchExpansionHandler(env=env, expander=expander)

        # monitor / logger
        dirname = (
            f"batch_backward_runner_{this_job_id()}"
            if prover_params.only_jsonl
            else "batch_backward_runner"
        )
        metrics_dump_path = prover_params.dump_path / "runner_logs" / dirname
        os.makedirs(metrics_dump_path, exist_ok=True)
        self.metrics = Logger(
            outdir=metrics_dump_path,
            tag=ProofHandler.__name__,
            quiet=prover_params.quiet,
            only_jsonl=prover_params.only_jsonl,
        )

        self.env_metrics = Logger(
            outdir=metrics_dump_path,
            tag="env",
            quiet=prover_params.quiet,
            only_jsonl=prover_params.only_jsonl,
        )

        log_path = prover_params.dump_path / "mcts_logs" / f"batch_backward_runner.log"
        self.tail_logger = (
            get_tail_logger(log_path, loc_info=True)
            if not prover_params.quiet
            else NoneLogger()
        )

        # init stats
        self.avg_stats: Dict[str, ActionCounter] = defaultdict(
            lambda: ActionCounter(name="", is_rate=False, silent=True)
        )
        self.avg_stats["grabbed_rate"] = ActionCounter(
            name=f"grabbed_rate", is_rate=True, silent=True
        )

        self.start_time = time.time()
        self.last_log = time.time()
        self.log_freq = 30  # log frequency (in seconds)

        # Each we keep the global goal, the status and the proof handler we want to use
        self.cur_proofs: Dict[ProofId_t, ProofRecord] = {}
        self.iterate = 0
        self.last_iter = 0
        self.should_stop = False
        self.global_proof_id = 0
        self.finished_proofs = 0
        self.last_time_exp: List[float] = []
        self.last_time_wfe: List[float] = []
        self.last_time_wfa: List[float] = []
        self.last_time_stitch: List[float] = []
        self.last_time_tot: List[float] = []

        self.pr: Optional[cProfile.Profile] = None
        self.profile_path: Path = Path("")
        if PROFILE:
            profile_dir = prover_params.dump_path / "backward_runner_profile"
            os.makedirs(profile_dir, exist_ok=True)
            self.profile_path = profile_dir / f"{rstr()}.profile"
            self.pr = cProfile.Profile()
            self.pr.enable()

        self.dead_proofs = 0

        self.timers = BackwardRunnerTimers(
            expansion_handler=self.expansion_handler.timers
        )
        self.cycle_timer = self.timers.proof_cycles

        self.inner_loop: Optional[Iterator[Union[ProverOutput, None]]] = None
        self.input_it: Optional[Iterator[Union[None, ProofResult, ProofHandler]]] = None
        self.input_queue: List[BackwardGoal] = []

        self.inp_id = 0
        self.goal_name_to_rid: Dict[str, RequestId] = {}
        self._is_alive = False

        self.gpu_mon = GPUMonitor(delay=1.0)
        self.is_closed = False

    def start(self):
        logger.info(f"PROVER: START LOOP")
        self._is_alive = True
        self.cycle_timer.start()

    def submit(self, inp: BackwardGoal) -> RequestId:
        self.input_queue.append(inp)
        rid = self.inp_id
        self.goal_name_to_rid[inp.name] = rid
        self.inp_id += 1
        return rid

    def ready(self,) -> List[Tuple[RequestId, ProverOutput]]:
        assert self.is_alive()

        outputs = []
        self.iterate += 1

        # grab proofs
        outputs += self.grab_proofs()

        # check for un-materialized theorems being materialized
        outputs += self.check_unmaterialized()

        # add all outstanding goals to expander
        self.add_to_expander()

        # exp_results : List[EnvExpansion] (for each leaf to expand for a proof)
        # might yield if we received some LeanInstanceDied
        outputs += self.send_env_expansions()

        self.get_done()

        # outside of lean, or when stitching / cleaning are not activated
        # these are pass through
        self.handle_proof_stitching()
        outputs += self.handle_proof_cleaning()

        # start new cycle here to have complete cycles when logging
        self.cycle_timer.stop()
        self.cycle_timer.start()

        # log stats
        self.log_runner_stats()
        return outputs

    def is_alive(self) -> bool:
        return self._is_alive

    def stop(self):
        # not sure that it will be useful since we will never empty the prover before exiting
        self._is_alive = False
        # log env stats
        env_stats = self.env.get_stats()
        logger.info(f"PROVER: ALL DONE -- {json.dumps(env_stats)}")

    def _make_result(
        self, pr: ProofResult, stats: Dict[str, Tuple[float, float]]
    ) -> Tuple[RequestId, ProverOutput]:
        rid = self.goal_name_to_rid[pr.goal.name]
        return (rid, (pr, stats))

    @wrap_timer()
    def grab_proofs(self) -> List[Tuple[RequestId, ProverOutput]]:
        """
        Pop the input queue for goals and send them to the :class:`BackwardEnv` for materialization.
        Create the corresponding ProofHandler. Initial status is :enum:`ProofStatus.TO_EXPAND`

        In *lean*, ``env.materialize_goal`` might raise DeclNotFound in which case we return a failed proof result immediately.
        """
        outputs = []
        self.timers.grab_proofs.start()
        self.avg_stats["active_before_grab"].act(len(self.cur_proofs))

        while self.input_queue:
            start_get = time.time()
            goal = self.input_queue.pop(0)
            try:
                goal = self.env.materialize_goal(goal)
                ph = self.proof_handler(goal)
            except DeclNotFound as e:
                outputs.append(
                    self._make_result(
                        ProofResult(proof=None, goal=goal, exception=e), {}
                    )
                )
                continue

            self.avg_stats["grab_proofs__get"].act(time.time() - start_get)
            self.avg_stats["grabbed_rate"].act()

            # Now we can use different proof handler in this function
            self.cur_proofs[self.global_proof_id] = ProofRecord(proof_handler=ph)

            # If conditioning is on, condition once, at the beginning by choosing a conditioning label
            # It's done here to avoid doing it in MCTSHandler, ProverHandler and bwd_prove
            if self.params.conditioning_kind != ConditioningKind.No:
                labels = self.env.expander_env.conditioning_labels()
                assert labels is not None
                goal = self.cur_proofs[self.global_proof_id].proof_handler.goal
                if goal.params is None:
                    goal.params = GoalParams()
                if ph.goal.label not in labels:
                    goal.params.conditioning_label = random.choice(list(labels))
                else:
                    goal.params.conditioning_label = ph.goal.label

            self.global_proof_id += 1

        self.timers.grab_proofs.stop()
        return outputs

    @wrap_timer()
    def check_unmaterialized(self) -> List[Tuple[RequestId, ProverOutput]]:
        """
        Goes through all proof handlers and check if their goal theorem is materialized.
        If not, check for the corresponding *abtch_id* in :meth:`ExpanderEnv.theorem_ready`
        
        If materializing failed in *lean* -> return failed `ProofResult`.

        Otherwise, set the materialized theorem in the goal, and send it to the :class:`ProofHandler`
        """
        outputs = []
        self.timers.check_unmaterialized.start()
        for proof_id, proof_record in list(self.cur_proofs.items()):
            g = proof_record.proof_handler.goal
            if g.materialized:
                continue
            assert isinstance(g.theorem, UnMaterializedTheorem)
            assert g.theorem.batch_id is not None
            th: Optional[Theorem] = None
            try:
                th = self.env.expander_env.theorem_ready(g.theorem.batch_id)
            except DeclNotFound as e:
                print("DECL NOT FOUND")
                outputs.append(
                    self._make_result(ProofResult(proof=None, goal=g, exception=e), {})
                )
                proof_record.proof_handler.close()
                del self.cur_proofs[proof_id]
            if th is not None:
                proof_record.proof_handler.goal.theorem = th
                proof_record.proof_handler.send_materialized(th)
        self.timers.check_unmaterialized.stop()
        return outputs

    @wrap_timer()
    def add_to_expander(self) -> None:
        """
        For any proof in status :enum:`ProofStatus.TO_EXPAND`, check for theorems to expand and call :meth:`BatchExpansionHandler.send_theorems_to_expand`.
        Status then becomes :enum:`ProofStatus.WAITING_FOR_EXPAND`
        """
        self.timers.add_to_expander.start()
        for pid, pr in self.cur_proofs.items():
            if pr.s.status != ProofStatus.TO_EXPAND:
                continue
            start_q = time.time()
            th_to_expand = pr.proof_handler.get_theorems_to_expand()
            self.avg_stats["add_to_expander__get_all"].act(time.time() - start_q)
            # TODO: check that done if len(th_to_expand) == 0
            if th_to_expand is not None:
                self.avg_stats["add_to_expander__get"].act(time.time() - start_q)
                start_q = time.time()
                self.expansion_handler.send_theorems_to_expand(
                    pid, th_to_expand, params=pr.proof_handler.goal.params,
                )
                self.avg_stats["add_to_expander__send"].act(time.time() - start_q)
                pr.s.status = ProofStatus.WAITING_FOR_EXPAND
        self.timers.add_to_expander.stop()

    @wrap_timer()
    def send_env_expansions(self) -> List[Tuple[RequestId, ProverOutput]]:
        """
        For any proof in status :enum:`ProofStatus.WAITING_FOR_EXPAND`, check for ready expansions. Output failed results if the env died.

        Otherwise, send expansions to the proof handler with :meth:`ProofHandler.send_env_expansions`.

        Proof status becomes :enum:`ProofStatus.WAITING_FOR_STATUS`
        """
        outputs = []
        self.timers.send_env_expansions.start()
        for pid, expand_res in self.expansion_handler.ready_env_expansions(
            prover_params=self.params
        ):
            assert self.cur_proofs[pid].s.status == ProofStatus.WAITING_FOR_EXPAND
            proof_handler = self.cur_proofs[pid].proof_handler
            if pid not in self.expansion_handler.env.has_died:
                start_q = time.time()
                proof_handler.send_env_expansions(expand_res)
                self.avg_stats["send_env_expansions__queue"].act(time.time() - start_q)
                self.cur_proofs[pid].s.status = ProofStatus.WAITING_FOR_STATUS
            else:
                # proof dead means its lean instance died. Stop brutally, return no results.
                proof_handler.stop()
                self.dead_proofs += 1
                outputs.append(
                    self._make_result(
                        ProofResult(
                            proof=None,
                            goal=proof_handler.goal,
                            exception=RuntimeError(LeanInstanceDied),
                        ),
                        self.gpu_mon.stats[0].tuple_stats,
                    )
                )
                self.gpu_mon.stats[0].reset()
                proof_handler.close()
                del self.cur_proofs[pid]

        self.timers.send_env_expansions.stop()
        return outputs

    @wrap_timer()
    def get_done(self) -> None:
        """
        For any proof in status :enum:`ProofStatus.WAITING_FOR_STATUS`, if :meth:`ProofHandler.get_done()` and proof is done :
        
        * Change status to :enum:`ProofStatus.STITCHING` if :attr:`ProofHandler.done`
        
        * Change status to :enum:`ProofStatus.TO_EXPAND` otherwise
        """
        self.timers.get_done.start()
        for pid, pr in self.cur_proofs.items():
            if pr.s.status != ProofStatus.WAITING_FOR_STATUS:
                continue
            if not pr.proof_handler.get_done():
                continue
            if pr.proof_handler.done:
                pr.s.status = ProofStatus.STITCHING
            else:
                pr.s.status = ProofStatus.TO_EXPAND
            if pr.s.status == ProofStatus.STITCHING:
                self.timers.finish_proofs__get_result.start()
                res = pr.proof_handler.get_result()
                self.timers.finish_proofs__get_result.stop()
                self.stitcher.process(pid, res)
            else:
                assert pr.s.status == ProofStatus.TO_EXPAND
        self.timers.get_done.stop()

    def finish_proof(
        self, pid: ProofId_t, res: ProofResult
    ) -> Tuple[RequestId, ProverOutput]:
        """
        Gather stats, kill proof handler and close sessions in the env if necessary.
        """
        pr = self.cur_proofs[pid]
        time_exp = pr.s.time_per_status[ProofStatus.TO_EXPAND]
        time_wfe = pr.s.time_per_status[ProofStatus.WAITING_FOR_EXPAND]
        time_wfs = pr.s.time_per_status[ProofStatus.WAITING_FOR_STATUS]
        time_stitch = pr.s.time_per_status[ProofStatus.STITCHING]
        time_tot = time.time() - pr.s.creation_time
        self.last_time_exp.append(time_exp)
        self.last_time_wfe.append(time_wfe)
        self.last_time_wfa.append(time_wfs)
        self.last_time_stitch.append(time_stitch)
        self.last_time_tot.append(time_tot)
        if self.params.print_status:
            logger.info(
                f"\tPROVER RESULT {pid:>5}: goal={pr.proof_handler.goal.name[:40]:<40} "
                f"time_exp: {time_exp:.3f}\t"
                f"time_wfe: {time_wfe:.3f}\t"
                f"time_wfs: {time_wfs:.3f}\t"
                f"time_stitch: {time_stitch:.3f}\t"
                f"time_tot: {time_tot:.3f}"
            )
        res.goal = self.cur_proofs[pid].proof_handler.goal

        to_ret = self._make_result(res, self.gpu_mon.stats[0].tuple_stats)
        self.gpu_mon.stats[0].reset()

        self.timers.finish_proofs__finish_goal.start()
        pr.proof_handler.close()
        self.env.finish_goal(self.cur_proofs[pid].proof_handler.goal)
        del self.cur_proofs[pid]
        self.timers.finish_proofs__finish_goal.stop()
        self.finished_proofs += 1

        return to_ret

    @wrap_timer()
    def handle_proof_stitching(self):
        """
        Stitches if necessary. Once done, move to :enum:`ProofStatus.CLEANING`
        """
        self.timers.stitch_proofs.start()
        for pid, (res, _exc) in self.stitcher.get_ready():
            self.cur_proofs[pid].s.status = ProofStatus.CLEANING
            self.cleaner.process(pid, res)
        self.timers.stitch_proofs.stop()

    @wrap_timer()
    def handle_proof_cleaning(self) -> List[Tuple[RequestId, ProverOutput]]:
        """
        Clean if necessary. Once done, call :meth:`BatchBackwardRunner.finish_proof`.
        """
        outputs = []
        self.timers.clean_proofs.start()
        for pid, (res, _excs) in self.cleaner.get_ready():
            outputs.append(self.finish_proof(pid, res))
        self.timers.clean_proofs.stop()
        return outputs

    @wrap_timer()
    def log_runner_stats(self):

        now = time.time()
        if now - self.last_log <= self.log_freq:
            return

        self.timers.log_runner_stats.start()
        status_count = defaultdict(int)
        for pr in self.cur_proofs.values():
            status_count[str(pr.s.status)] += 1
        logger.info(
            f"PROVER STATS:\t"
            f"len(self.cur_proofs)={len(self.cur_proofs)}\t"
            f"{dict(status_count)}\t"
            f"iterate={self.iterate}\t"
            f"global_proof_id={self.global_proof_id}\t"
            f"finished_proofs={self.finished_proofs}\t"
            f"proof_rate={self.finished_proofs / (now - self.start_time):.4f} proofs/s\t"
            f"should_stop={self.should_stop}"
        )

        max_len = max(
            [len(pr.proof_handler.goal.name[:40]) for pr in self.cur_proofs.values()],
            default=0,
        )

        def pad_name(s: str):
            return s + (" " * (max_len - len(s)))

        for pid, pr in self.cur_proofs.items():
            materialized = pr.proof_handler.goal.materialized
            in_status = now - pr.s.last_status_change
            handler_status = pr.proof_handler.status()
            if (
                pr.s.status == ProofStatus.WAITING_FOR_EXPAND
                and self.params.prover_kind == ProverKind.BackwardMCTS
            ):
                assert (
                    len(self.expansion_handler.processing[pid])
                    == handler_status["last_goals"]
                )
                n_expansions = len(self.expansion_handler.received[pid])
                # assert n_expansions <= handler_status["last_goals"]
                assert "last_expansions" not in handler_status
                handler_status["last_expansions"] = n_expansions
            if self.params.print_status:
                logger.info(
                    f"\tPROVER PROOF {pid:>6}: goal={pad_name(pr.proof_handler.goal.name[:40])}\t"
                    f"status={pr.s.status:<24}"
                    f"last_update: {in_status:.3f}\t"
                    f"creation: {now - pr.s.creation_time:.3f}\t"
                    f"handler: {handler_status}\t"
                    f"materialized: {materialized}"
                )
        last_status = [now - pr.s.last_status_change for pr in self.cur_proofs.values()]
        total_ages = [now - pr.s.creation_time for pr in self.cur_proofs.values()]

        to_log = dict(
            global_proof_id=self.global_proof_id,
            finished_proofs=self.finished_proofs,
            proof_rate=self.finished_proofs / (now - self.start_time),
            iter_per_sec=(self.iterate - self.last_iter) / (now - self.last_log),
            proof_last_status_max=get_max(last_status),
            proof_last_status_mean=get_mean(last_status),
            proof_ages_max=get_max(total_ages),
            proof_ages_mean=get_mean(total_ages),
            time_exp_max=get_max(self.last_time_exp),
            time_wfe_max=get_max(self.last_time_wfe),
            time_wfa_max=get_max(self.last_time_wfa),
            time_stitch_max=get_max(self.last_time_stitch),
            time_tot_max=get_max(self.last_time_tot),
            time_exp_mean=get_mean(self.last_time_exp),
            time_wfe_mean=get_mean(self.last_time_wfe),
            time_wfa_mean=get_mean(self.last_time_wfa),
            time_stitch_mean=get_mean(self.last_time_stitch),
            time_tot_mean=get_mean(self.last_time_tot),
        )

        for k, v in list(self.expansion_handler.sum_stats.items()):
            assert k not in to_log, k
            to_log[k] = v
            self.expansion_handler.sum_stats[k] = 0

        for acs in [self.avg_stats, self.expansion_handler.avg_stats]:
            for name, ac in acs.items():
                assert name not in to_log, name
                to_log[name] = ac.rate_and_reset()

        timings = self.timers.rate_and_reset()
        to_log.update(timings)

        to_log.update(self.stitcher.get_stats())
        to_log.update(self.cleaner.get_stats())

        # log env stats
        env_stats = self.env.get_stats()
        to_log.update(env_stats)
        to_log["dead_proofs"] = self.dead_proofs

        self.last_time_exp.clear()
        self.last_time_wfe.clear()
        self.last_time_wfa.clear()
        self.last_time_tot.clear()
        self.metrics.log_metrics(to_log)
        env_stats = self.env.expander_env.get_stats()
        self.env_metrics.log_metrics(env_stats)
        logger.info(f"__log_env__:{json.dumps(env_stats)}")
        logger.info(f"__log_prover__:{json.dumps(to_log)}")

        self.last_log = now
        self.last_iter = self.iterate

        if self.pr is not None:
            self.pr.dump_stats(self.profile_path)
            self.pr.enable()
        self.timers.log_runner_stats.stop()

    def close(self):
        self.is_closed = True
        logger.info("Closing BatchBackwardRunner...")
        self.gpu_mon.close()
        self.metrics.close()
        self.env_metrics.close()

        self.expander.close()
        if self.env_gen is not None:
            self.env_gen.close()

        if isinstance(self.proof_handler, MCTSProofHandlerGenerator):
            self.proof_handler.close()

        self.env.close()
        logger.info("BatchBackwardRunner closed.")


def init_and_run_prover(
    dataset: DatasetConf,
    decoding: DecodingParams,
    prover_params: ProverParams,
    decoder_type: str,
    input_it: Iterator[Optional[BackwardGoal]],  # TODO: have special class
) -> Iterator[Tuple[ProofResult, Dict[str, Tuple[float, float]]]]:
    """
    Creates a :class:`BatchBackwardRunner` and feeds it `prover_params.n_simultaneous_proofs`.
    Returns an iterator over proof results and stats.

    :yield: Tuple[ProofResult, WeightedStats]
    """

    prover = BatchBackwardRunner(dataset, decoding, prover_params, decoder_type)
    with logged_closing(prover, "BatchBackwardRunner"):
        yield from make_iterator(
            async_worker=prover,
            input_it=input_it,
            max_in_worker=prover_params.n_simultaneous_proofs,
        )
