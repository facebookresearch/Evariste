# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from typing import Optional, Tuple, Union, List, Dict, Any, cast, TextIO, Set
from scipy.special import logsumexp
from collections import defaultdict
from dataclasses import dataclass
from subprocess import Popen
from pathlib import Path
import os
import re
import time
import pandas
import psutil
import random
import string


from evariste import json as json
import pickle

from evariste.backward.env.lean.filter_tactics import clean_tactic, TacticFilter
from evariste.datasets.lean import LeanDatasetConf, LEAN_CONDITIONING
from evariste.model.data.dictionary import B_CMD_WORD, E_CMD_WORD
from evariste.logger import create_logger
from evariste.metrics import ActionCounter, ActivityTracker
from evariste.backward.graph import Theorem, UnMaterializedTheorem
from evariste.backward.env.core import (
    BackwardGoal,
    ExpanderEnv,
    ProofId,
    ModelExpansion,
)
from evariste.backward.env.lean.graph import (
    LeanTactic,
    LeanTheorem,
    LeanState,
    LeanContext,
)
from evariste.backward.env.lean.tokenizer import LeanTokenizer
from evariste.backward.env.worker import (
    TacticJob,
    TacticJobResult,
)
from lean_cluster.client import LeanClusterClient
from lean_cluster.instance import LeanInstanceDied
from evariste.datasets.lean import LEAN_SPLITS


from leanml import get_api, WrongModule, LeanAPI


logger = getLogger()


class TooManyInstances(Exception):
    pass


@dataclass
class LeanCreateSession:
    label: str
    module_path: str
    merge_alpha_equiv: Optional[bool]


@dataclass
class LeanDelSession:
    session: str


@dataclass
class LeanEvalCmdJob:
    to_run: str
    module_path: Optional[str]
    timeout: int


@dataclass
class LeanTacticOnState:
    state: LeanState
    tactic: LeanTactic


@dataclass
class LeanEvalCmdResult:
    error: Optional[str]
    output: Optional[str]

    def __post_init__(self):
        assert self.error is not None or self.output is not None


class DeclNotFound(Exception):
    pass


@dataclass
class ForwardLeanJob:
    theorem: LeanTheorem
    tactic: LeanTactic
    session: str
    check_children: bool = False
    enforce_same_fp_for_parsed: bool = False


@dataclass
class ParseChildrenJob:
    task: ForwardLeanJob
    parsed_goal: LeanTheorem
    children: List[LeanTheorem]


LeanJob = Union[
    TacticJob,
    LeanDelSession,
    LeanCreateSession,
    ForwardLeanJob,
    ParseChildrenJob,
    LeanTacticOnState,
    LeanEvalCmdJob,
]
BatchId = Union[int, str]


class LeanExpanderEnv(ExpanderEnv):
    def __init__(
        self, dataset: LeanDatasetConf, dump_path: Path, debug: bool = False,
    ):
        self.dataset = dataset
        self.debug = debug
        self.label_to_module = {}
        self.label_to_namespace = {}
        self.label_to_context: Dict[str, LeanContext] = {}
        self.last_log = time.time()
        self.last_get_results = time.time()

        self.parsable_bwd = dataset.parsable_bwd

        # Stores a file open for each proof, where each expansion is dumped
        self.dump_files: Dict[str, TextIO] = {}
        self.dump_files_path: Dict[str, Path] = {}
        self.proof_log_path = None
        if dataset.dump_proof_logs:
            rstr = "".join(random.choice(string.ascii_lowercase) for _ in range(6))
            self.proof_log_path = dump_path / f"proof_logs_{rstr}"
            logger.info(f"Lean dumping all comms in {self.proof_log_path}")
            os.makedirs(self.proof_log_path, exist_ok=False)

        self.tactic_times: Optional[TextIO] = None
        if dataset.dump_tactic_times:
            rstr = "".join(random.choice(string.ascii_lowercase) for _ in range(6))
            self.tactic_times_path = dump_path / f"tactic_times_{rstr}"
            logger.info(f"Lean dumping tactic times in {self.tactic_times_path}")
            self.tactic_times = open(self.tactic_times_path, "w")

        self.all_decls: Dict[str, int] = {}

        if self.dataset.is_old:
            data_srcs = ["data.csv"] + [
                x for x in set(LEAN_SPLITS.values()) if x is not None
            ]

            # old style, load everything from .csv
            for to_load in data_srcs:
                if (Path(dataset.data_dir) / to_load).exists():
                    loaded_data = pandas.read_csv(Path(dataset.data_dir) / to_load)
                    print(
                        f"Loaded {len(loaded_data)} rows from {Path(dataset.data_dir) / to_load}"
                    )
                else:
                    assert (
                        to_load != "data.csv"
                    ), f"{Path(dataset.data_dir) / to_load} not found"
                    continue
                for row in loaded_data.iloc:  # type: ignore
                    self.label_to_module[row.decl_name] = row.filename
                    if to_load == "data.csv":
                        substr = row.decl_name.split(".")[-1]
                        if substr not in self.all_decls:
                            self.all_decls[substr] = len(self.all_decls)
                    if row.decl_name not in self.label_to_context:
                        ns = set()
                        if (
                            not self.dataset.pp_full_names
                            and "open_namespaces" in loaded_data.columns
                            and type(row.open_namespaces) is str
                        ):
                            ns = set(row.open_namespaces.split())
                        self.label_to_context[row.decl_name] = LeanContext(
                            namespaces=ns
                        )
        else:
            labels_and_splits = self.dataset.get_labels_and_splits()
            for _, data in labels_and_splits.items():
                for module_name, decl, namespace in data:
                    self.label_to_module[decl] = str(module_name)
                    self.label_to_namespace[decl] = namespace
            for _, decl, _ in labels_and_splits["train"]:
                self.all_decls[decl] = len(self.all_decls)

        self.tokenizer = LeanTokenizer.build(self.dataset.tokenizer)

        LeanTokenizer.build(dataset.tokenizer)
        if not dataset.lean_cluster:
            ckpt_path = dataset.get_materialized().checkpoint_path
            logger.info(f"Using lean checkpoint : {Path(ckpt_path)}")
            logger.info(f"Lean dataset: {dataset}")
            self.api: Union[LeanAPI, LeanClusterClient] = get_api(
                Path(ckpt_path),
                fast=dataset.fast,
                preload="ml_server" in ckpt_path,  # preload if not ckpt
                quiet=not debug,
                path_to_fifos=dataset.path_to_fifos,
                num_threads=dataset.num_threads,
                dump_comms=debug,
                old=dataset.is_old,
                additional_roots=None
                if dataset.is_old
                else [Path(self.dataset.statement_splits_path)],
            )
        else:
            rstr = "".join(random.choice(string.ascii_lowercase) for _ in range(6))
            print(f"Expander cluster ID : {rstr}")
            self.api = LeanClusterClient(
                name=f"{os.environ.get('SLURM_JOB_NAME', 'local')}_lean_cluster_{rstr}",
                expander_id=rstr,
                comm_folder=Path(dump_path) / "lean_cluster_comms",
                dataset=dataset,
            )

        self.conditioning_label_set = set()
        if dataset.conditioning != "":
            self.conditioning_label_set = set(
                pickle.load(open(LEAN_CONDITIONING[dataset.conditioning], "rb")).keys()
            )

        self.psutil_lean: Optional[psutil.Process] = None
        if isinstance(self.api, LeanAPI):
            assert isinstance(
                self.api._proc, Popen
            ), "Tried to get proc id of pre-existing ml server"
            self.psutil_lean = psutil.Process(self.api._proc.pid)

        self.task_for_req_id: Dict[str, Tuple[float, LeanJob]] = {}
        self.req_id_to_batch: Dict[str, BatchId] = {}
        # when some requests (e.g like parsable bwd) required two api calls, we
        # want to return the final result (so of the 2nd api call) with the req_id
        # of the first api call
        self.req_id_mapping: Dict[str, str] = {}

        self.results: List[Tuple[BatchId, str, TacticJobResult]] = []

        self.create_session_id = 0
        self.del_session_id = 0
        self.simple_results: Dict[
            Any, Any
        ] = {}  # for non batch requests (create session, delete session)

        self.tactic_timeout = ActionCounter("tactic_timeout", is_rate=False)
        self.tactic_time = ActionCounter("tactic_time", is_rate=False)
        self.outstanding_requests = ActionCounter("outstanding_requests", is_rate=False)
        self.cpu_usage = ActionCounter("cpu_usage", is_rate=False)
        self.active_threads = ActivityTracker()

        # check that there is no session name collision
        self.open_sessions: Set[str] = set()

    @staticmethod
    def do_filter_model_expansion(
        filter_tactics: TacticFilter, model_expansion: ModelExpansion
    ) -> int:
        assert model_expansion.tactics is not None
        assert model_expansion.log_priors is not None

        # detokenize tactics so we can clean them
        init_tactics = [
            repr(LeanTactic.from_tokens(tokens)) for tokens in model_expansion.tactics
        ]
        if len(init_tactics) == 0:
            return 0

        # cleaned tactics to scores
        n_init_tacs = len(set(init_tactics))
        new_tactics: Dict[str, List[float]] = defaultdict(list)

        # clean tactics
        for tactic, log_prior in zip(init_tactics, model_expansion.log_priors):
            cleaned_tac = clean_tactic(tactic, filter_tactics)
            if cleaned_tac is None:
                continue
            tactic = cleaned_tac
            new_tactics[tactic].append(log_prior)

        # everything was dirty
        if len(new_tactics) == 0:
            model_expansion.error = "failed_no_tactics_after_filtering"
            model_expansion.log_priors = None
            model_expansion.tactics = None
            model_expansion.log_critic = None
            return n_init_tacs

        # tokenize cleaned tactics. if multiple tactics lead to the same
        # cleaned tactic, either take the best log-prob, or sum the log-probs
        tactic_scores = [
            (
                (float(logsumexp(lp)) if filter_tactics.sum_tactic_scores else max(lp)),
                [B_CMD_WORD, *LeanTactic(tac).tokenize(), E_CMD_WORD],
            )
            for tac, lp in new_tactics.items()
        ]

        # update model expansion
        zipped: List[Tuple[float, List[str]]] = sorted(tactic_scores, reverse=True)
        res_log_priors, res_tokens = zip(*zipped)  # zipping kills mypy
        model_expansion.log_priors = list(res_log_priors)  # type: ignore
        model_expansion.tactics = list(res_tokens)  # type: ignore

        return n_init_tacs - len(model_expansion.tactics)

    def filter_model_expansion(self, model_expansion: ModelExpansion) -> int:
        return self.do_filter_model_expansion(
            self.dataset.filter_tactics, model_expansion
        )

    def conditioning_labels(self) -> Optional[Set[str]]:
        return self.conditioning_label_set

    def get_cleaning_path(self, version: str) -> Optional[str]:
        if self.dataset.is_old:
            return None
        return f"cleaning_utils/{version}/do_clean_proof.lean"

    def get_label_full_name(self, label: str) -> str:
        # special case for train / test / mathlib, which are already namespaced
        # get // None for v1.test_evariste which has no namespace
        if self.dataset.is_old or self.label_to_namespace.get(label, None) in {
            "train",
            "test",
            "valid",
            None,
        }:
            if not self.dataset.is_old and label not in self.label_to_namespace:
                assert "EVARISTE_test" in label, label
            return label
        return f"{self.label_to_namespace[label]}.{label}"

    # all these only refer to backward tactic jobs
    def process(self, task: LeanJob, batch_id: BatchId):
        if isinstance(task, TacticJob):
            tactic = LeanTactic.from_tokens(task.tactic_tokens)
            if self.proof_log_path:
                assert (
                    isinstance(task.theorem, LeanTheorem)
                    and task.theorem.state is not None
                )
                session = task.theorem.state.session
                self.dump_files[session].write(
                    json.dumps(
                        {
                            "subgoal": task.theorem.to_dict(light=False),
                            "tactic": tactic.to_dict(light=False),
                        }
                    )
                    + "\n"
                )
                self.dump_files[session].flush()

            if self.parsable_bwd:
                return self.process_parsable_bwd(task, batch_id)

            assert (
                isinstance(task.theorem, LeanTheorem) and task.theorem.state is not None
            )
            lean_state: LeanState = task.theorem.state
            this_req_id = self.api.send_tactic(
                session_name=lean_state.session,
                state_id=lean_state.node_id,
                tactic_str=repr(tactic),
                timeout=self.dataset.timeout,
                max_size=self.dataset.max_size,
                max_subgoals=self.dataset.max_subgoals,
                max_metavars=self.dataset.max_metavars,
                max_repeated_hyps=self.dataset.max_repeated_hyps,
                nosplit=self.dataset.nosplit,
            )
        elif isinstance(task, ForwardLeanJob):
            if self.proof_log_path:
                self.dump_files[task.session].write(
                    json.dumps({"goal_pp": task.theorem.to_dict(light=False)}) + "\n"
                )
                self.dump_files[task.session].flush()

            this_req_id = self.api.parse_goal_and_apply_tactic(
                goal_pp=task.theorem.conclusion,
                session_name=task.session,
                tactic_str=repr(task.tactic),
                timeout=self.dataset.parse_goal_timeout,
                max_size=self.dataset.max_size,
                max_subgoals=1,  # no multi-goal allowed
                max_metavars=self.dataset.max_metavars,
                max_repeated_hyps=self.dataset.max_repeated_hyps,
            )
        elif isinstance(task, LeanCreateSession):
            try:
                if self.debug:
                    print(
                        f"Create session {task.label}, {task.module_path}", flush=True
                    )
                merge_alpha_equiv = self.dataset.merge_alpha_equiv
                if task.merge_alpha_equiv is not None:
                    merge_alpha_equiv = task.merge_alpha_equiv

                this_req_id = self.api.new_session(
                    module_path=task.module_path,
                    decl_name=self.get_label_full_name(task.label),
                    merge_alpha_equiv=merge_alpha_equiv,
                    pp_opts={"pp.full_names": self.dataset.pp_full_names},
                )
                if self.debug:
                    print(
                        f"Create session {task.label}, "
                        f"{task.module_path} // {this_req_id} {batch_id}",
                        flush=True,
                    )
            except WrongModule as e:
                # mapped file doesn't exist
                self.simple_results[batch_id] = f"Mapped File doesn't exist: {e}"
                return
        elif isinstance(task, LeanDelSession):
            this_req_id = self.api.del_session(task.session)
            # done here and not in receive to avoid message ordering problems
            self.open_sessions.remove(task.session)
            if self.proof_log_path:
                self.dump_files[task.session].close()
                os.unlink(self.dump_files_path[task.session])
                self.dump_files.pop(task.session)
                self.dump_files_path.pop(task.session)
        elif isinstance(task, LeanTacticOnState):
            this_req_id = self.api.send_tactic(
                session_name=task.state.session,
                state_id=task.state.node_id,
                tactic_str=repr(task.tactic),
                timeout=self.dataset.timeout,
                max_size=self.dataset.max_size,
                max_subgoals=self.dataset.max_subgoals,
                max_metavars=self.dataset.max_metavars,
                max_repeated_hyps=self.dataset.max_repeated_hyps,
                nosplit=self.dataset.nosplit,
            )
        elif isinstance(task, LeanEvalCmdJob):
            this_req_id = self.api.eval_cmd(
                to_run=task.to_run, module_path=task.module_path, timeout=task.timeout
            )
        else:
            raise RuntimeError(f"Unknown task {task}")

        self.task_for_req_id[this_req_id] = (time.time(), task)
        self.req_id_to_batch[this_req_id] = batch_id
        return int(this_req_id)

    def process_parsable_bwd(self, task: TacticJob, batch_id: Any) -> int:
        # schedule a ForwardLeanJob instead of TacticJob to ensure that we can
        # use parse_goal_and_apply_tactic
        tactic = LeanTactic.from_tokens(task.tactic_tokens)
        theorem = cast(LeanTheorem, task.theorem)
        assert theorem.state is not None
        new_task = ForwardLeanJob(
            theorem=theorem,
            tactic=tactic,
            session=theorem.state.session,
            check_children=True,
            enforce_same_fp_for_parsed=self.dataset.parsable_bwd_same_fp_for_parsed,
        )
        ret = self.process(new_task, batch_id)
        assert ret is not None
        return ret

    def receive_res(
        self,
        task: Optional[LeanJob],
        res: Dict[str, Any],
        batch_id: Any,
        duration: float,
    ):
        def thm_from_dict(this_dict, session: str):
            assert isinstance(task, (ForwardLeanJob, TacticJob))
            assert isinstance(task.theorem, LeanTheorem)

            full_pp: str = this_dict["full_pp"]
            # try to catch old != ret.conclusion below where old is empty string
            assert len(full_pp) > 0, this_dict
            inst_count = full_pp.count("_inst_")
            if inst_count >= self.dataset.max_inst:
                raise TooManyInstances
            if self.dataset.nosplit:
                full_pp = re.sub(r"\d+ goal(s)?\n", "", full_pp)

            past_tactics = None
            if self.dataset.past_tactics > 0 and isinstance(task, TacticJob):
                # ordered from old to recent
                past_tactics = task.theorem.past_tactics + [
                    LeanTactic.from_tokens(task.tactic_tokens)
                ]
                past_tactics = past_tactics[-self.dataset.past_tactics :]

            fp = str(this_dict[self.dataset.fingerprint])
            to_ret = LeanTheorem(
                conclusion=full_pp,
                context=task.theorem.context,
                state=LeanState(
                    session=session,
                    node_id=this_dict.get("node_id", -1),  # no node_id for fwd
                    n_subgoals=this_dict["n_subgoals"],
                    n_metavars=this_dict.get(
                        "n_metavars", -1
                    ),  # for fwd, only n_metavars for subgoals
                ),
                # TODO: Should we recreate those to match minproof path once we extract data for training ?
                past_tactics=past_tactics,
                fingerprint=fp,
            )
            return to_ret

        if isinstance(task, TacticJob):
            tactic = LeanTactic.from_tokens(task.tactic_tokens)
            tactic.n_theorems = len(self.all_decls)
            if "error" in res:
                tactic.duration = self.dataset.timeout
                self.tactic_timeout.act(int(res["error"].startswith("tactic timeout")))
                if self.tactic_times is not None:
                    self.tactic_times.write(
                        json.dumps({"tactic": repr(tactic), "time": -1}) + "\n"
                    )
                    self.tactic_times.flush()
                if self.debug:
                    print(f"Lean got res : {res['error']} for {tactic}")

                if res["error"].startswith("Async constant:"):
                    logger.info(f"[ASYNC CONSTANT] {tactic._tactic}")

                tactic.is_valid = False
                tactic.error_msg = res["error"]
                return TacticJobResult(
                    tactic, [], theorem=task.theorem, duration=duration
                )
            else:
                tactic.duration = res["eval_time"]
                for substr in re.split("[^a-zA-Z0-9_]", str(tactic)):
                    if substr in self.all_decls:
                        tactic.uses_theorems.add(self.all_decls[substr])
                assert isinstance(task.theorem, LeanTheorem)
                self.tactic_timeout.act(0)
                self.tactic_time.act(res["eval_time"])
                if self.tactic_times is not None:
                    self.tactic_times.write(
                        json.dumps({"tactic": repr(tactic), "time": res["eval_time"]})
                        + "\n"
                    )
                    self.tactic_times.flush()
                base_th_state: Optional[LeanState] = task.theorem.state
                assert base_th_state is not None
                if self.debug:
                    print(f"Lean got res : {res} for {tactic}")
                try:
                    children = [
                        thm_from_dict(node, base_th_state.session)
                        for node in res["nodes"]
                    ]
                except TooManyInstances:
                    tactic.is_valid = False
                    tactic.error_msg = "Too many instances"
                    return TacticJobResult(
                        tactic, [], theorem=task.theorem, duration=duration
                    )

                return TacticJobResult(
                    tactic, children, theorem=task.theorem, duration=duration
                )
        elif isinstance(task, LeanCreateSession):
            if "error" in res:
                self.simple_results[batch_id] = res["error"]
            else:
                self.simple_results[batch_id] = LeanTheorem(
                    conclusion=res["initial_goal"]["full_pp"],
                    context=self.label_to_context.get(
                        task.label, LeanContext(namespaces=set())
                    ),
                    state=LeanState(
                        res["name"], 0
                    ),  # 1 was changed to 0 after intros change in ml_server
                    fingerprint=str(res["initial_goal"][self.dataset.fingerprint]),
                )
                session = res["name"]
                # session might have been deleted and the name re-used but results arrive in other order.
                assert session not in self.open_sessions
                self.open_sessions.add(session)

                if self.proof_log_path:
                    self.dump_files_path[session] = (
                        self.proof_log_path / f"{task.label}_{session}"
                    )
                    self.dump_files[session] = open(self.dump_files_path[session], "w")
            return None
        elif isinstance(task, LeanDelSession):
            self.simple_results[batch_id] = None
            return None
        elif isinstance(task, ForwardLeanJob):
            tactic = task.tactic
            if "error" in res:
                if self.debug:
                    print(f"Lean got res : {res['error']} for {tactic}")
                tactic.is_valid = False
                tactic.error_msg = res["error"]
                return TacticJobResult(tactic, [], theorem=task.theorem)

            # to delete when updated in cpp
            if res["parsed_goal"]["n_subgoals"] != 1 or any(
                child["n_subgoals"] != 1 for child in res["subgoals"]
            ):
                tactic.is_valid = False
                tactic.error_msg = "parse goal error: multi-goal"
                return TacticJobResult(tactic, [], theorem=task.theorem)

            if self.debug:
                print(f"Lean got res : {res} for {tactic}")

            try:
                children = [
                    thm_from_dict(node, task.session) for node in res["subgoals"]
                ]
                parsed_goal = thm_from_dict(res["parsed_goal"], task.session)
            except TooManyInstances:
                tactic.is_valid = False
                tactic.error_msg = "Too many instances"
                return TacticJobResult(tactic, [], theorem=task.theorem)

            if task.enforce_same_fp_for_parsed:
                assert task.theorem.fingerprint is not None, task.theorem
                if parsed_goal.fingerprint != task.theorem.fingerprint:
                    tactic.is_valid = False
                    tactic.error_msg = "parsed fingerprint different from fingerprint"
                    return TacticJobResult(tactic, [], theorem=task.theorem)

            if task.check_children:
                # schedule a parse_children job
                new_task = ParseChildrenJob(
                    task=task, children=children, parsed_goal=parsed_goal
                )
                this_req_id = self.api.parse_children(
                    children=[c.conclusion for c in children],
                    session_name=task.session,
                    timeout=self.dataset.parse_goal_timeout,
                    max_metavars=self.dataset.max_metavars,
                    max_repeated_hyps=self.dataset.max_repeated_hyps,
                )
                self.task_for_req_id[this_req_id] = (time.time(), new_task)
                self.req_id_to_batch[this_req_id] = batch_id
                self.req_id_mapping[this_req_id] = res["req_id"]
                return None

            return TacticJobResult(tactic, children, theorem=parsed_goal)
        elif isinstance(task, ParseChildrenJob):
            forward_task = task.task
            tactic = forward_task.tactic
            if "error" in res:
                if self.debug:
                    print(f"Lean got res : {res['error']} for parse_children")
                tactic.is_valid = False
                tactic.error_msg = f"parse_children_error:{res['error']}"
                return TacticJobResult(tactic, [], theorem=forward_task.theorem)
            received = [c["full_pp"] for c in res["parsed_children"]]
            expected = [c.conclusion for c in task.children]
            if received != expected:
                logger.info(
                    f"Mismatch between children and parsed children: "
                    f"{expected} != {received}"
                )
            return TacticJobResult(tactic, task.children, theorem=task.parsed_goal)
        elif isinstance(task, LeanTacticOnState):
            self.simple_results[batch_id] = res
            return None
        elif isinstance(task, LeanEvalCmdJob):
            self.simple_results[batch_id] = LeanEvalCmdResult(
                error=res.get("error", None), output=res.get("output", None)
            )
            return None
        else:
            raise RuntimeError("Misunderstood API result")

    def get_results(self):
        if time.time() - self.last_get_results < 0.05:
            return
        self.last_get_results = time.time()
        while True:
            try:
                res = self.api.recv(timeout=0.01)
                assert res is not None
                req_id = res["req_id"]
                if "thread_id" in res:
                    self.active_threads.act(res["thread_id"])
                batch_id = self.req_id_to_batch[req_id]
                start_time, task = self.task_for_req_id[req_id]
                res = self.receive_res(
                    task, res, batch_id=batch_id, duration=time.time() - start_time
                )
                if res is not None:
                    self.results.append((batch_id, req_id, res))
                self.req_id_to_batch.pop(req_id)
                self.task_for_req_id.pop(req_id)
            except TimeoutError:
                break

        self.outstanding_requests.act(len(self.req_id_to_batch))

    def get_stats(self) -> Dict[str, float]:
        stats = {}
        if not self.dataset.lean_cluster:
            assert self.psutil_lean is not None
            cpu = self.psutil_lean.cpu_percent(interval=None)
            stats["lean_cpu"] = cpu
        else:
            assert isinstance(self.api, LeanClusterClient)
            stats["lean_cpu"] = self.api.cpu_usage.rate()
            stats["lean_mem"] = self.api.mem_usage.rate()
            stats.update(self.api.timers.rate_and_reset())
        stats["lean_timeouts"] = self.tactic_timeout.rate_and_reset()
        stats["lean_tactic_time"] = self.tactic_time.rate_and_reset()
        stats["lean_outstanding_requests"] = self.outstanding_requests.rate_and_reset()

        for x, y in self.active_threads.rate_and_reset().items():
            stats[f"active_threads_{x}"] = y

        if isinstance(self.api, LeanClusterClient):
            self.api.alive()
        return stats

    def get_all_ready(self) -> List[Tuple[ProofId, int, TacticJobResult]]:
        self.get_results()
        to_ret: List[Tuple[ProofId, int, TacticJobResult]] = []
        for pid, req_id, result in self.results:
            # to return the req_id of the first api call when chained api calls
            # (for instance in parsable bwd when we first app[ly a tactic then check
            # if the children are parsable.
            if req_id in self.req_id_mapping:
                req_id = self.req_id_mapping.pop(req_id)
            assert isinstance(pid, int)
            to_ret.append((pid, int(req_id), result))
        self.results = []
        return to_ret

    def get_goal(self, label: str) -> BackwardGoal:
        batch_id = self.submit_create_session(label)
        return self.wait_for_goal(batch_id, label)

    def theorem_ready(self, batch_id: str) -> Optional[LeanTheorem]:
        res = self.theorems_ready([batch_id])[0]
        if isinstance(res, str):
            raise DeclNotFound(res)
        return res

    def theorems_ready(
        self, batch_ids: List[str]
    ) -> List[Optional[Union[LeanTheorem, str]]]:
        self.get_results()
        res = []
        for batch_id in batch_ids:
            res.append(self.simple_results.pop(batch_id, None))
        return res

    def maybe_get_result(self, batch_id: str) -> Optional[Dict]:
        return self.maybe_get_results([batch_id])[0]

    def maybe_get_results(self, batch_ids: List[str]) -> List[Optional[Dict]]:
        """Avoids calling get_results multiple time to check for several batch ids"""
        self.get_results()
        res = []
        for batch_id in batch_ids:
            res.append(self.simple_results.pop(batch_id, None))
        return res

    def wait_for_goal(self, batch_id: str, label: str):
        while batch_id not in self.simple_results:
            self.get_results()

        res = self.simple_results.pop(batch_id)
        if isinstance(res, str):
            raise DeclNotFound(res)
        assert isinstance(res, LeanTheorem)
        return BackwardGoal(theorem=res, label=label)

    def submit_create_session(
        self, label: str, merge_alpha_equiv: Optional[bool] = None
    ) -> str:
        batch_id = f"create_sess_{self.create_session_id}"
        self.create_session_id += 1
        try:
            self.process(
                LeanCreateSession(
                    label=label,
                    module_path=self.label_to_module[label],
                    merge_alpha_equiv=merge_alpha_equiv,
                ),
                batch_id,
            )
        except KeyError as e:
            raise DeclNotFound(f"{e} not in self.label_to_module")
        return batch_id

    def submit_del_session(self, session: str):
        batch_id = f"del_sess_{self.del_session_id}"
        self.del_session_id += 1
        self.process(LeanDelSession(session=session), batch_id)
        return batch_id

    def wait_for_del_session(self, batch_id):
        while batch_id not in self.simple_results:
            self.get_results()
        res = self.simple_results.pop(batch_id)
        if res == "NEVER_GOT_RESULT":
            pass
        elif res is not None:
            if res["error"] == LeanInstanceDied:
                logger.warning(f"Attempted to delete a session on dead instance ??")
            else:
                raise RuntimeError(f"While deleting session : {res}")

    def finish_theorem(self, th: Theorem):
        # we don't wait for the result here.
        assert isinstance(th, LeanTheorem) and th.state is not None
        self.submit_del_session(th.state.session)

    def materialize_goal(self, goal: BackwardGoal) -> BackwardGoal:
        if isinstance(goal.theorem, UnMaterializedTheorem):
            goal.theorem.batch_id = self.submit_create_session(goal.theorem.label)
            return goal
        return goal

    def close(self):
        logger.info("Closing LeanExpanderEnv ...")
        if self.dataset.lean_cluster:
            self.api.close()
        for x, y in self.dump_files.items():
            y.close()
        logger.info("Closed LeanExpanderEnv")
