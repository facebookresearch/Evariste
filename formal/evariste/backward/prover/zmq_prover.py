# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import abstractmethod
from typing import (
    Optional,
    Tuple,
    List,
    Set,
    Dict,
    Any,
    Generic,
    Iterator,
    Type,
    Callable,
)
from dataclasses import dataclass, field
from pathlib import Path
import os
import time
import pickle
import random
import submitit
from submitit.core.utils import FailedJobError

from evariste import json as json
from evariste.async_workers.async_worker import AsyncWorker
from evariste.async_workers.async_worker_helpers import (
    PostProcessedAsyncWorker,
    PostProcessor,
    make_iterator,
    RequestId,
)
from evariste.async_workers.worker_gang import AsyncWorkerGang
from evariste.async_workers.zmq_submitit_worker import (
    SubmititConfig,
    ZMQSubmititParams,
    ZmqSubmititWorkerLauncher,
)
from evariste.backward.prover.prior_search_prover import ProofResultWithStats
from evariste.backward.remote.search_params_sampler import (
    maybe_make_search_params_samplers,
    GoalParamsSampler,
    WorkerParamsSampler,
    HyperOptKind,
)
from params import Params, ConfStore, MISSING
from evariste.comms.store import Sender
from evariste.backward.remote.pool_training import PoolTrainingCfg, PoolTrainingSender
from evariste.datasets import (
    DatasetConf,
    LeanDatasetConf,
    EquationsDatasetConf,
    MetamathDatasetConf,
    HolLightDatasetConf,
)
from evariste.utils import (
    logged_closing,
    this_job_id,
    OurSignalHandler,
    set_TMPDIR,
)
from evariste.logger import create_logger, log_memory
from evariste.model.transformer_args import DecodingParams
from evariste.model.data.envs.mcts_loader import get_mcts_comms
from evariste.model.data.mcts_subproof import (
    MCTSSubProofArgs,
    MCTSProofSampler,
    ProofStepSample,
)
from evariste.comms.zmq import MultiWorkerZMQSender, S
from evariste.comms.comms import MultiSender
from evariste.backward.env.metamath.env import MMEnvWorker
from evariste.backward.goal_factory import get_goals_to_prove
from evariste.backward.prover.mcts import MCTSResult
from evariste.backward.prover.utils import (
    copy_model,
    WeightedAvgStats,
    set_MKL_env_vars,
)
from evariste.backward.prover.mcts_samples import ONLINE_MCTS_SUBTASKS
from evariste.backward.prover.prover import (
    ProverParams,
    ProverKind,
    ProofResult,
    BackwardGoal,
    ProverOutput,
    BatchBackwardRunner,
)
from evariste.clusters.utils import clusterify_partitions


@dataclass
class ZMQProverParams(Params):
    prover: ProverParams
    decoding: DecodingParams
    root_dir: Path

    eq_dataset: Optional[EquationsDatasetConf] = None
    lean_dataset: Optional[LeanDatasetConf] = None
    mm_dataset: Optional[MetamathDatasetConf] = None
    hl_dataset: Optional[HolLightDatasetConf] = None

    n_machines: int = 1
    max_attempts: int = 1
    partition: str = clusterify_partitions("Theorem_Proving")
    timeout_min: int = 3 * 24 * 60
    cpus_per_task: int = 10
    mem_gb: int = 60

    copy_model: bool = True  # otherwise, use a symlink to save memory
    dump_proofs: bool = False

    n_th_to_prove: Optional[int] = None
    shuffle_seed: int = 43

    max_restarts: int = 0
    n_trainers: int = 0
    decoder_type: str = "decoder"
    local: bool = False

    send_to_all: bool = True  # send MCTS data to all trainers
    retry_even_if_solved: bool = False

    shuffle_goals: bool = False

    pool_training: PoolTrainingCfg = field(default_factory=lambda: PoolTrainingCfg())

    hyperopt: HyperOptKind = HyperOptKind.Fixed
    hyperopt_param_str: str = ""

    exclude_nodes: str = ""  # nodes to exclude (both for training / inference)

    def __post_init__(self):
        if self.max_attempts > 1:
            assert (
                self.decoding.use_sampling
            ), "Useless to try multiple attempts with a beam."

        if self.prover.add_tactic_fill:
            assert self.decoding.length_penalty == 0.0

    def _check_and_mutate_args(self):
        assert (self.hyperopt_param_str == "") == (self.hyperopt == HyperOptKind.Fixed)

    @property
    def dataset(self) -> DatasetConf:
        datasets = [
            self.eq_dataset,
            self.lean_dataset,
            self.mm_dataset,
            self.hl_dataset,
        ]
        assert len([x for x in datasets if x is not None]) == 1
        for x in datasets:
            if x is not None:
                return x
        raise RuntimeError("unreachable due to assert")

    def set_dataset(self, lang: str, dataset: DatasetConf):
        if lang == "lean":
            assert isinstance(dataset, LeanDatasetConf)
            self.lean_dataset = dataset
        elif lang == "eq":
            assert isinstance(dataset, EquationsDatasetConf)
            self.eq_dataset = dataset
        elif lang == "mm":
            assert isinstance(dataset, MetamathDatasetConf)
            self.mm_dataset = dataset
        elif lang == "hl":
            assert isinstance(dataset, HolLightDatasetConf)
            self.hl_dataset = dataset
        else:
            raise RuntimeError("unknown lang")


ConfStore["zmq_fast"] = ZMQProverParams(
    root_dir=MISSING,
    prover=ProverParams(
        n_simultaneous_proofs=5,
        mcts=ConfStore["mcts_fast"],
        beam_path=MISSING,
        dump_path=MISSING,
        prover_kind=ProverKind.BackwardMCTS,
        beam_kind=MISSING,
        mcts_subproof_params=MCTSSubProofArgs(),
    ),
    decoding=ConfStore["decoding_bwd_eval"],
    n_machines=1,
    partition=clusterify_partitions("Theorem_Proving"),
)

ConfStore["zmq_slow"] = ZMQProverParams(
    root_dir=MISSING,
    prover=ProverParams(
        n_simultaneous_proofs=5,
        mcts=ConfStore["mcts_slow"],
        beam_path=MISSING,
        dump_path=MISSING,
        prover_kind=ProverKind.BackwardMCTS,
        beam_kind=MISSING,
        mcts_subproof_params=MCTSSubProofArgs(),
    ),
    decoding=ConfStore["decoding_slow"],
    n_machines=1,
)


logger = create_logger(None)


ZMQProverOutput = Dict[str, Any]  # make class in the future ?


class ZMQProver(
    PostProcessedAsyncWorker[BackwardGoal, ProverOutput, ZMQProverOutput],
):
    def __init__(self, params: ZMQProverParams):
        post_processor = ZMQProverPostProcessor(params=params)
        prover = BatchBackwardRunner(
            dataset=params.dataset,
            decoding=params.decoding,
            prover_params=params.prover,
            decoder_type=params.decoder_type,
        )
        super().__init__(worker=prover, post_processor=post_processor)


class ZMQProverPostProcessor(PostProcessor[ProverOutput, ZMQProverOutput]):
    def __init__(self, params: ZMQProverParams):
        self.params = params

        self.senders, self.env_worker = self.setup()
        self.total_sent = 0

    def __call__(self, inp: ProverOutput) -> Dict[str, Any]:
        res, gpu_stats = inp
        out = self.send_to_trainers_and_postprocess_result(res, gpu_stats)
        self.store_proof(res)
        return out

    def setup(self):
        job_id = this_job_id()
        params = self.params
        if params.prover.mcts_subproofs_online_training:
            mcts_subtasks = ["subproof"]
        else:
            mcts_subtasks = params.prover.mcts_subtasks_online_training
        logger.info(f"mcts_subtasks: {mcts_subtasks}")

        if (
            isinstance(params.dataset, MetamathDatasetConf)
            and params.prover.mcts_subproofs_online_training
        ):
            env_worker: Optional[MMEnvWorker] = MMEnvWorker(params.dataset)
            assert env_worker is not None
            env_worker.init()
        else:
            env_worker = None

        if params.pool_training.is_a_pool_training:
            logger.info(
                f"ZMQ PROVER: Waiting for connection with {params.pool_training.n_models} "
                f"models (each one with {params.n_trainers} trainers) "
            )
        else:
            logger.info(
                f"ZMQ PROVER: Waiting for connection with {params.n_trainers} trainers"
            )

        senders: Dict[str, Sender[Tuple[Label, Any]]] = {}
        for mcts_subtask in mcts_subtasks:
            if params.pool_training.is_a_pool_training:
                senders[mcts_subtask] = PoolTrainingSender.from_zmq_prover_params(
                    client_id=job_id,
                    mcts_subtask=mcts_subtask,
                    params=params,
                )
            else:
                senders[mcts_subtask] = MCTSSubTaskSender(
                    client_id=job_id,
                    root_dir=params.root_dir,
                    mcts_subtask=mcts_subtask,
                    params=params,
                )

        # TODO: do not store this for async evals
        os.makedirs(params.prover.dump_path / "first_proofs", exist_ok=True)
        if params.dump_proofs:
            os.makedirs(params.prover.dump_path / "proofs", exist_ok=True)

        return senders, env_worker

    def send_to_trainers_and_postprocess_result(
        self, res: ProofResult, res_gpu_stats: Dict[str, Tuple[float, float]]
    ) -> Dict[str, Any]:
        params = self.params
        env_worker = self.env_worker
        senders = self.senders
        # TODO: FIX
        # _p, model_version = get_latest_checkpoint(params.prover.beam_path)
        logger.info(f"ZMQ PROVER: Done with {res.goal.name}")
        if params.prover.mcts_subproofs_online_training:
            mcts_subtasks = {"subproof"}
        else:
            mcts_subtasks = ONLINE_MCTS_SUBTASKS
        res_train_samples: Dict[str, List] = {task: [] for task in mcts_subtasks}
        res_mcts_hist_stats, res_mcts_stats = {}, {}
        proof_stats = {}
        # extract a simplified MCTS state and sample subproofs
        if isinstance(res, MCTSResult):
            if params.prover.mcts_subproofs_online_training:
                if res.simplified_state is not None:
                    sampler = MCTSProofSampler(
                        res.simplified_state.root,
                        res.simplified_state.nodes,
                        env_worker=env_worker,
                    )
                    sampling_params = params.prover.mcts_subproof_params
                    for _ in range(sampling_params.n_sample_per_proofs):
                        goal, proofs, hyps = sampler.sample(sampling_params)
                        samples: List[ProofStepSample] = sampler.build_steps(
                            goal, proofs, hyps
                        )
                        res_train_samples["subproof"].extend(samples)
                res_mcts_stats = res.stats
            else:
                assert res.mcts_samples_critic is not None
                assert res.mcts_samples_tactic is not None
                assert res.mcts_samples_effect is not None
                res_train_samples["critic"] = res.mcts_samples_critic
                res_train_samples["tactic"] = res.mcts_samples_tactic
                res_train_samples["effect"] = res.mcts_samples_effect
                res_train_samples["minproof"] = (
                    [res.sample_proof] if res.sample_proof is not None else []
                )
                res_mcts_stats = res.stats
                res_mcts_hist_stats = res.hist_stats
        elif isinstance(res, ProofResultWithStats):
            proof_stats = res.proof_stats
            proof_stats["label"] = res.goal.label
            proof_stats["name"] = res.goal.name

        sent: Dict[str, int] = {}
        for mcts_subtask in mcts_subtasks:
            if mcts_subtask not in params.prover.mcts_subtasks_online_training:
                continue
            logger.info(
                f"ZMQ PROVER: Sending {len(res_train_samples[mcts_subtask])} "
                f"{mcts_subtask} samples for {res.goal.name}"
            )
            sent[mcts_subtask] = len(res_train_samples[mcts_subtask])

            for sample in res_train_samples[mcts_subtask]:
                if params.n_trainers > 0:
                    assert res.goal.label is not None, res.goal
                    senders[mcts_subtask].store((res.goal.label, sample))

        (
            goal_params_dict,
            worker_params_dict,
        ) = WorkerParamsSampler.rebuild_goal_and_worker_params(
            goal_params=res.goal.params,
            use_beam=params.decoding.use_beam,
            n_samples=params.decoding.n_samples,
        )

        # This message has to be sent otherwise we'll have bad accounting on the controller side.
        # Try to send and if we fail, warn and wait. This shouldn't happen
        exception: Optional[str] = None
        if isinstance(res.exception, RuntimeError):
            exception = repr(res.exception)
        elif res.exception is not None:
            exception = str(type(res.exception))

        self.total_sent += sum(sent.values())
        return {
            "type": "result",
            "label": res.goal.label,
            "name": res.goal.name,
            "model": "MODEL_NAME",
            "success": res.proof is not None,
            "src": "prover",  # not used
            "n_samples_sent": sent,
            "mcts_stats": res_mcts_stats,
            "mcts_hist_stats": res_mcts_hist_stats,
            "proof_stats": proof_stats,
            "gpu_stats": res_gpu_stats,
            "exception": exception,
            "goal_params": goal_params_dict,
            "worker_params": worker_params_dict,
        }

    def store_proof(self, result: ProofResult):
        params = self.params

        # store the proof
        if result.proof is not None:

            # dump all proofs
            if params.dump_proofs:
                fname = f"{result.goal.name}.pkl"
                fpath = params.prover.dump_path / "proofs" / fname
                with open(fpath, "wb") as f:
                    pickle.dump(result.proof, f)

            # dump only first proofs (during online MCTS)
            assert result.goal.split is not None
            dirpath = params.prover.dump_path / "first_proofs" / result.goal.split
            os.makedirs(dirpath, exist_ok=True)
            final_path = dirpath / f"{result.goal.label}.pkl"
            tmp_path = dirpath / f"{result.goal.name}.pkl"
            if not final_path.exists():
                with open(tmp_path, "wb") as f:
                    pickle.dump(result.proof, f)
                try:
                    os.rename(tmp_path, final_path)
                except OSError:
                    os.unlink(tmp_path)
                    logger.warning(
                        f"Couldn't store proof for {result.goal.label}. "
                        f"Maybe another worker dumped it?"
                    )

    def close(self):
        for sender in self.senders.values():
            sender.close()
        if self.env_worker is not None:
            self.env_worker.close()


def make_prover_factory(
    params: ZMQProverParams,
    prover_fn: Callable[
        [
            ZMQProverParams,
        ],
        AsyncWorker,
    ],
    worker_params_sampler=None,
) -> Callable[[int, bool], AsyncWorker]:
    if worker_params_sampler is not None:
        params = worker_params_sampler.sample_prover_params(params=params)

    def fn(worker_id: int, full_logging: bool) -> AsyncWorker:
        # here worker_id to respect typing, but in case of prover do not need worker_id actually
        params.prover.only_jsonl = not full_logging
        return prover_fn(params)

    return fn


def prover_launcher(
    prover_fn: Callable[
        [
            ZMQProverParams,
        ],
        AsyncWorker,
    ],
    params: ZMQProverParams,
    name: str = "",
    worker_params_sampler: Optional[WorkerParamsSampler] = None,
) -> ZmqSubmititWorkerLauncher:

    params = params
    name = name

    # Submitit params for ZMQSubmitit Workers
    submitit_cfg = SubmititConfig(
        str(params.root_dir / "provers"), f"prover_gang_{name}"
    )
    if params.local:
        submitit_cfg.local = True
        assert params.max_restarts == 0, "local restarts won't work"
    else:
        submitit_cfg.local = False
        submitit_cfg.slurm_timeout_min = params.timeout_min
        submitit_cfg.slurm_gpus_per_task = 1
        submitit_cfg.slurm_cpus_per_task = params.cpus_per_task
        submitit_cfg.slurm_ntasks_per_node = 1
        submitit_cfg.slurm_partition = clusterify_partitions(params.partition)
        submitit_cfg.slurm_job_name = f"prover_gang_{name}"
        submitit_cfg.slurm_array_parallelism = params.n_machines
        submitit_cfg.slurm_srun_args = ["-vv"]
        submitit_cfg.slurm_mem_gb = params.mem_gb
        if params.exclude_nodes:
            submitit_cfg.slurm_exclude = params.exclude_nodes

    assert params.prover.heartbeats_freq > 0

    # ZMQSubmitit worker params
    zmq_submitit_params: ZMQSubmititParams = ZMQSubmititParams(
        submitit_cfg,
        heartbeat_freq=params.prover.heartbeats_freq,
        max_heartbeat_missed=5,
        check_status_freq=60,
    )

    return ZmqSubmititWorkerLauncher(
        make_prover_factory(params, prover_fn, worker_params_sampler),
        zmq_submitit_params,
    )


class ProverHandler(AsyncWorkerGang):
    """
    Main class responsible to run a MCTS evaluation:
     - creating and the distributed pool of provers (via the :class:`evariste.async_workers.worker_gang.AsyncWorkerGang`).
     - load and select goals to send to the prover gang, and keep track of the one solved or not.

    :param params:
    :param folder_exists_ok:
    :param split:
    :param name:
    :param prover_fn:
    """

    def __init__(
        self,
        params: ZMQProverParams,
        folder_exists_ok: bool = False,
        split: str = "valid",
        name: str = "",
        prover_fn: Callable[[ZMQProverParams], AsyncWorker] = ZMQProver,  # For testing
    ):

        if not params.prover.mcts.early_stop:
            logger.warning("ProverHandler: SETTING EARLY STOP TO TRUE")
            params.prover.mcts.early_stop = True

        # create dump path / dump params
        os.makedirs(params.root_dir, exist_ok=folder_exists_ok)
        with open(os.path.join(params.root_dir, "params.json"), "w") as f:
            f.write(params.to_json())

        # copying / linking model
        copy_model(
            src_path=params.prover.beam_path,
            tgt_dir=params.root_dir,
            hard_copy=params.copy_model,
        )

        self.goal_params_sampler: Optional[GoalParamsSampler] = None
        worker_params_sampler: Optional[WorkerParamsSampler] = None
        (
            self.goal_params_sampler,
            worker_params_sampler,
        ) = maybe_make_search_params_samplers(
            hyperopt=params.hyperopt,
            hyperopt_param_str=params.hyperopt_param_str,
            n_machines=params.n_machines,
            n_simultaneous_proofs=params.prover.n_simultaneous_proofs,
        )

        super().__init__(
            worker_launcher=prover_launcher(
                prover_fn,
                params=params,
                name=name,
                worker_params_sampler=worker_params_sampler,
            ),
            n_workers=params.n_machines,
            max_restarts=params.max_restarts,
            max_queued_inputs=params.prover.n_simultaneous_proofs,
            check_alive_freq=60,
        )

        self.workdir = Path(params.root_dir)
        self.results = open(self.workdir / "results.json", "w+")
        self.failed: Set[str] = set()
        self.proved: Set[str] = set()
        self.split = split

        self.proof_stats = open(self.workdir / "proof_stats.json", "w+")

        self.mcts_stats = WeightedAvgStats()
        self.gpu_stats = WeightedAvgStats()
        self.attempts_remaining: Dict[str, int] = {}

        self.to_prove: List[BackwardGoal] = get_goals_to_prove(
            dataset=params.dataset,
            split=split,
            n_to_prove=params.n_th_to_prove,
            shuffle_seed=params.shuffle_seed,
        )
        self.label2goal: Dict[str, BackwardGoal] = {}
        for goal in self.to_prove:
            self.label2goal[goal.label] = goal
            self.attempts_remaining[goal.label] = params.max_attempts

        if params.shuffle_goals:
            random.shuffle(self.to_prove)

        logger.info(
            f"Will attempt to prove {len(self.to_prove)} theorems "
            f"with at most {params.max_attempts} attempts."
        )
        self.params = params

    def run(self):
        # For tim-> Even if it deduplicates a little bit the code I find it lighter to have it here
        self.last_log = time.time()
        for _ in make_iterator(
            self,
            max_in_worker=self.max_queued_inputs * self.n_workers,
            input_it=self.input_it(),
        ):
            pass

    def input_it(self) -> Iterator[BackwardGoal]:
        while sum(self.attempts_remaining.values()) > 0:
            keys = sorted(
                [(n, label) for label, n in self.attempts_remaining.items()],
                reverse=True,
            )
            for n, label in keys:
                if n == 0:
                    break
                self.attempts_remaining[label] -= 1
                goal: BackwardGoal = self.label2goal[label]
                if self.goal_params_sampler is not None:
                    # create a new goal to have a new name and avoid key collision
                    # and to change the params of the goal
                    goal = BackwardGoal.create_unmat(goal.label, split=self.split)
                    goal.params = self.goal_params_sampler.sample_goal_params(
                        goal_name=goal.name
                    )
                yield goal
        # there is no more goals to send, stop workers that are done.
        self.stop_workers_if_no_inputs()

    def ready(self) -> List[Tuple[RequestId, ZMQProverOutput]]:
        if time.time() - self.last_log > 60:
            self.log_stats()
            self.last_log = time.time()
        return super().ready()

    def handle_result(self, result):
        label = result["label"]
        result["attempt"] = (
            self.params.max_attempts - self.attempts_remaining[result["label"]]
        )
        result.pop("mcts_hist_stats", None)
        self.results.write(json.dumps(result) + "\n")
        self.results.flush()
        self.mcts_stats.update(result["mcts_stats"])
        self.gpu_stats.update(result["gpu_stats"])

        if result["proof_stats"]:
            proof_stats = result["proof_stats"]
            try:
                self.proof_stats.write(json.dumps(proof_stats) + "\n")
            except TypeError:
                logger.error(f"{self.proof_stats}{proof_stats}")
                raise
            self.proof_stats.flush()

        if result["success"]:
            self.proved.add(label)
            if label in self.failed:
                self.failed.remove(label)
            if not self.params.retry_even_if_solved:
                self.attempts_remaining[label] = 0  # do not reprocess this
        elif label not in self.proved:
            self.failed.add(label)

        if self.goal_params_sampler is not None:
            self.goal_params_sampler.update(
                goal_name=result["name"], success=result["success"]
            )
        return result

    def log_stats(self):
        log_memory(logger)
        total = len(self.proved) + len(self.failed)
        assert len(self.proved.intersection(self.failed)) == 0
        attempts_remaining = 0
        for label, n in self.attempts_remaining.items():
            if label not in self.proved:
                attempts_remaining += n
        if total > 0:
            logger.info(
                f"ZMQ PROVER -- Proved {len(self.proved)}/{total} "
                f"({100 * len(self.proved) / total:.2f}%). "
                f"{attempts_remaining} total attempts remaining."
            )
            logger.info(self.mcts_stats.stats)
            logger.info(self.gpu_stats.stats)
        else:
            logger.info(f"No answer received yet.")


def run_async(
    params: ZMQProverParams,
    split: str,
    name: str,
) -> float:
    """
    Run async eval and return proving accuracy.
    """
    set_TMPDIR()
    set_MKL_env_vars()
    OurSignalHandler.start()

    # initialize and run the prover
    handler = ProverHandler(
        params,
        folder_exists_ok=True,  # folder created by submitit
        split=split,
        name=name,
    )
    with logged_closing(handler, "prover_handler"):
        handler.run()

    # create empty `done` file if prover finished successfully
    remaining = len(handler.to_prove) - (len(handler.proved) + len(handler.failed))
    if remaining == 0:
        logger.info(f"run_async: done with {len(handler.to_prove)} goals")
        (handler.params.root_dir / "done").touch()
        if sum(handler.attempts_remaining.values()) > 0:
            logger.error(
                f"run_async: processed all goals but found "
                f"{sum(handler.attempts_remaining.values())} attempts remaining!"
            )
    else:
        logger.error(
            f"run_async: only processed {len(handler.proved) + len(handler.failed)} "
            f"out of {len(handler.to_prove)} goals!"
        )

    # compute and return proving accuracy
    total = len(handler.proved) + len(handler.failed)
    if total > 0:
        logger.info(f"run_async: accuracy {100 * len(handler.proved) / total} %")
        return 100 * len(handler.proved) / total
    else:
        return 0.0


def launch_async(
    params: ZMQProverParams, split: str, name: str, timeout_min: int = 240
) -> submitit.Job:
    """
    Used for async evaluation during training, or in eval_one.
    """
    set_MKL_env_vars()
    if (
        not name.startswith("sweep_eval_")
        and params.max_restarts != 0
        and not isinstance(params.dataset, LeanDatasetConf)
    ):
        raise RuntimeError(f"No restart for async eval but got {params.max_restarts}!")
    params.decoding.__post_init__()

    # copy model to evaluate
    params.prover.beam_path = copy_model(
        src_path=params.prover.beam_path,
        tgt_dir=params.root_dir,
        hard_copy=params.copy_model,
    )

    # initialize executor (controller job)
    folder = params.root_dir / "provers"
    if params.local:
        os.makedirs(folder, exist_ok=True)
        executor: submitit.Executor = submitit.LocalExecutor(folder=folder)
        executor.update_parameters(timeout_min=timeout_min, gpus_per_node=0)
    else:
        executor = submitit.AutoExecutor(folder=folder, slurm_max_num_timeout=-1)
        executor.update_parameters(
            slurm_timeout_min=timeout_min,
            slurm_gpus_per_node=0,
            slurm_cpus_per_task=1,
            slurm_ntasks_per_node=1,
            slurm_partition=clusterify_partitions(params.partition),
            slurm_job_name=f"prover_handler_{name}",
            slurm_srun_args=["-vv"],
            slurm_mem_gb=25,
        )

    # submit job
    backoff = 60 * 5  # wait 5 minutes if the slurm scheduler times out
    max_attempts = 3
    for attempt_id in range(max_attempts):
        try:
            job = executor.submit(run_async, params, split, name)
            if not params.local:
                is_custom = (
                    "#SBATCH --gpus-per-node=0" in job.paths.submission_file.read_text()
                )
                if not is_custom:
                    logger.warning(
                        "Not using custom version of submitit! If encountering gres errors, use: "
                        "pip install git+https://github.com/facebookincubator/submitit@main#egg=submitit"
                    )
            return job
        except FailedJobError as e:
            logger.warning(
                f"Error submitting async eval job! "
                f"({attempt_id + 1}/{max_attempts}): {e}"
            )
            time.sleep(backoff)
    raise FailedJobError


Label = str


class MCTSSubTaskSender(Generic[S], Sender[Tuple[Label, S]]):
    def __init__(
        self, client_id: str, root_dir: Path, mcts_subtask: str, params: ZMQProverParams
    ):
        inner_senders: List[MultiWorkerZMQSender] = [
            MultiWorkerZMQSender(
                client_id=client_id,
                socket_file_root=Path(get_mcts_comms(root_dir))
                / f"mcts_{mcts_subtask}_sample_store_sockets",
                socket_file_pattern=rf"^{trainer_id}_(\d+).addr$",
            )
            for trainer_id in range(params.n_trainers)
        ]
        for inner_sender in inner_senders:
            logger.info(
                f"{self.__class__.__name__}: "
                f"Connecting to {inner_sender.socket_file_root}..."
            )
            inner_sender.wait_for_store()
            logger.info(
                f"{self.__class__.__name__}: "
                f"Connected to {inner_sender.socket_file_root}!"
            )
        self.sender: MultiSender[S] = MultiSender(
            senders=inner_senders, send_to_all=params.send_to_all
        )

    def store(self, seq: Tuple[Label, S]):
        _, sample = seq
        self.sender.store(sample)

    def close(self):
        self.sender.close()

    def rate_and_reset(self) -> float:
        return self.sender.rate_and_reset()
