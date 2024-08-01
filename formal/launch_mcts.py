# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Any, List, Set, Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import asdict
from pprint import pformat
from copy import deepcopy
from pathlib import Path
import os
import zmq
from zmq import ZMQError
import time
import random
import shutil
import socket
import string
import submitit
import numpy as np

from configs.online_mcts_configs import (
    register_online_mcts_configs,
    MCTSOnlineParams,
)
from evariste.async_workers.async_worker_helpers import make_iterator, RequestId
from evariste.async_workers.worker_gang import AsyncWorkerGang
from evariste.backward.remote.prioritized_label_sampler import (
    PrioritizedLabelSampler,
    prioritized_label_sampler_factory,
    PLSStats,
)
from evariste.backward.remote.search_params_sampler import (
    HyperOptKind,
    MCTSSolvingStats,
    maybe_make_search_params_samplers,
    GoalParamsSampler,
    WorkerParamsSampler,
)
from evariste.comms.zmq import ManagedZmqContext
from params.params import cfg_from_cli
from evariste import json as json
from evariste.utils import prepare, logged_closing, OurSignalHandler, set_TMPDIR
from evariste.logger import create_logger, log_memory
from evariste.metrics import Logger as MetricsLogger
from evariste.metrics import ActionCounter
from evariste.model.utils import get_path_bwd_proving_eval
from evariste.backward.goal_factory import get_labels
from evariste.backward.graph import BackwardGoal
from evariste.backward.prover.mcts_samples import ONLINE_MCTS_SUBTASKS
from evariste.backward.prover.prover_args import BeamSearchKind
from evariste.backward.prover.utils import HistStats, WeightedAvgStats, copy_model
from evariste.backward.prover.zmq_prover import (
    ZMQProverOutput,
    set_MKL_env_vars,
    ZMQProverParams,
    ZMQProver,
    prover_launcher,
)
from evariste.backward.remote.state_dump_utils import StateDumper
from evariste.trainer.launch import launch_train
from evariste.clusters.utils import clusterify_path, get_max_timeout


logger = create_logger(None)


# Used for testing with slurm LocalExecutor
AVAILABLE_GPU = 0
EXP_NAME = "mcts_training"
RANDOM_SEED = 0


class MCTSGangHandler(AsyncWorkerGang):
    """
    Main class responsible to run a MCTS online training:
     - creating and the distributed pool of provers (via the :class:`evariste.async_workers.worker_gang.AsyncWorkerGang`).
     - launching and watching the trainers (single `submitit` job with many workers).
     - load and select goals to send to the prover gang, and keep track of the one solved or not.


    If the trainer die, this gang handler should die and close all jobs

    :param params: cfg for the MCTS handler
    """

    def __init__(self, params: MCTSOnlineParams):

        root_dump_path = Path(clusterify_path(params.train_params.root_dump_path))
        assert root_dump_path.is_dir(), root_dump_path
        dump_path = root_dump_path / params.exp_name

        if params.pre_existing_run is not None:
            self.dump_path = Path(params.pre_existing_run)
            assert self.dump_path.is_dir()
            # destroy all previous socket infos to avoid sending messages to nowhere
            shutil.rmtree(self.dump_path / "trainer_sockets", ignore_errors=True)
            shutil.rmtree(self.dump_path / "provers", ignore_errors=True)
            shutil.rmtree(self.dump_path / "prover_dumps", ignore_errors=True)
            params.train_params.dump_path = params.pre_existing_run
            # params.train_params.override_dump_path = params.pre_existing_run

        if not params.debug and not params.local:
            logger.info(f"======== Online MCTS Params ========")
            logger.info(params.to_json())

        # Socket to communicate with trainer that will send stats to MCTSGangHandler
        self.trainer_controller_context: ManagedZmqContext = ManagedZmqContext()
        self.trainer_sock = self.trainer_controller_context.socket(zmq.ROUTER)
        self.trainer_port = self.trainer_sock.bind_to_random_port(f"tcp://*")
        trainer_hostname = socket.gethostname()
        params.train_params.stats_socket_addr = (
            f"tcp://{trainer_hostname}:{self.trainer_port}"
        )
        # check and mutate before launching anything
        params.check_and_mutate_args(avoid={type(params.train_params.slurm_conf)})
        params.check_type()

        # TODO rmv that and do it cleanly, for debug now
        to_mutate = deepcopy(params)
        to_mutate.check_and_mutate_args(avoid={type(to_mutate.train_params.slurm_conf)})
        assert to_mutate == params

        logger.info(f"Starting from {params.train_params.reload_checkpoint}")
        job_id = os.environ.get(
            "SLURM_JOB_ID",
            "".join(random.choice(string.ascii_lowercase) for _ in range(6)),
        )
        if params.pre_existing_run is None:
            self.dump_path = dump_path / job_id
            os.makedirs(self.dump_path, exist_ok=True)

        increasing_quality = (
            params.zmq_prover.prover.beam_kind == BeamSearchKind.IncreasingQuality
        )

        self.trainer_job: Optional[submitit.Job] = None

        if not params.no_trainer:

            logger.info(
                f"MCTSGangHandler: Launching trainer on {params.n_gpu_trainer} GPUs"
            )

            # # if this was part of a sweep, use jobid to disambiguate between all trainers
            # params.train_params.override_dump_path = str(self.dump_path)

            # Trainer job params
            if params.n_gpu_trainer > 8:
                assert params.n_gpu_trainer % 8 == 0, params.n_gpu_trainer
                gpus_per_node = 8
                n_nodes = params.n_gpu_trainer // 8
                n_tasks_per_node = 8
            else:
                assert params.n_gpu_trainer > 0, params.n_gpu_trainer
                gpus_per_node = params.n_gpu_trainer
                n_nodes = 1
                n_tasks_per_node = params.n_gpu_trainer

            # Set trainer dump path. The trainer dump path should be set and overriden
            # here, otherwise it will be named after the trained job ID which is not
            # known at this moment, but that we need to provide to the provers.
            assert params.train_params.dump_path == ""
            params.train_params.override_dump_path = str(self.dump_path)

            assert params.train_params.num_workers <= 8
            cpus_per_task = 8  # adapt to your cluster

            self.trainer_job = launch_train(
                dump_path=self.dump_path / "trainer",
                slurm_job_name=f"trainer_{params.exp_name}_{job_id}",
                exp_name=params.exp_name,
                partition=params.train_partition,
                trainer_args=params.train_params,
                gpus_per_node=gpus_per_node,
                cpus_per_task=cpus_per_task,
                ntasks_per_node=n_tasks_per_node,
                n_nodes=n_nodes,
                timeout_min=get_max_timeout(params.train_partition),
                mem_per_gpu=60,
                local=params.local,
                exclude_nodes=params.zmq_prover.exclude_nodes,
            )
            try:  # try / except to make sure trainer_job is killed if init crash # TODO improve that e.g in start
                logger.info(
                    f"MCTSGangHandler: Launched trainer job {self.trainer_job.job_id}\n"
                    f"MCTSGangHandler: Dump path {self.dump_path} // {params.train_params.dump_path}"
                )

                if increasing_quality and params.pre_existing_run is None:
                    eval_split = params.zmq_prover.prover.eval_split
                    assert params.lang is not None
                    eval_proving_path = get_path_bwd_proving_eval(
                        root=Path(self.dump_path),
                        lang=params.lang,
                        split=params.zmq_prover.prover.eval_split,
                        decoder_type="decoder",
                        epoch=0,
                    )
                    os.makedirs(eval_proving_path, exist_ok=True)
                    # beam path format is env/split/epoch
                    beam_path = params.zmq_prover.prover.beam_path
                    *prefix, beam_split, epoch = str(beam_path).split("/")
                    if beam_split != eval_split:
                        folder_for_eval = "/".join(prefix + [eval_split, epoch])
                        if not os.path.exists(folder_for_eval):
                            raise RuntimeError(
                                f"Cannot use increasing quality beam search for split "
                                f"{eval_split}, because {folder_for_eval} does not exist."
                            )
                        logger.info(
                            f"Increasing quality beam: Using {folder_for_eval} for first eval."
                        )
                    else:
                        folder_for_eval = str(beam_path)
                    shutil.copytree(
                        folder_for_eval, eval_proving_path, dirs_exist_ok=True
                    )
                    # set done in case it wasn't present in src folder
                    (Path(eval_proving_path) / "done").touch()

                logger.info(f"MCTSGangHandler: Waiting for trainer job to start")
                ckpt_path = self.dump_path / "checkpoint.-1.pth"
                while not (self.trainer_job.state == "RUNNING"):
                    logger.info(
                        f"Trainer job state: {self.trainer_job.state} -- "
                        f"Sleeping 10 seconds..."
                    )
                    time.sleep(10)
                    if "unk" in self.trainer_job.state.lower() and ckpt_path.exists():
                        logger.info(f"{ckpt_path} exists, beginning training")
                        break
                    else:
                        logger.info("unk state but no ckpt")
                logger.info(f"MCTSGangHandler: Trainer job started.")
            except:
                self.close_trainer()
                raise
        else:
            if increasing_quality:
                raise RuntimeError("increasing_quality makes no sense without trainer")
            assert self.trainer_job is None
            copy_model(
                params.zmq_prover.prover.beam_path / "checkpoint.-1.pth",
                self.dump_path,
            )

        try:  # try / except to make sure trainer_job is killed if init crash # TODO improve that e.g in start
            # copy launcher_params parameters
            with open(os.path.join(self.dump_path, "launcher_params.json"), "w") as f:
                json.dump(asdict(params), f, sort_keys=True, indent=4)

            # dump path
            params.zmq_prover.root_dir = self.dump_path
            params.zmq_prover.prover.dump_path = self.dump_path / "prover_dumps"
            if params.zmq_prover.prover.dump_path.exists():
                if params.pre_existing_run is None:
                    raise RuntimeError(
                        f"Pre-existing run {params.pre_existing_run} is None"
                    )
                shutil.rmtree(params.zmq_prover.prover.dump_path)
            os.makedirs(params.zmq_prover.prover.dump_path, exist_ok=False)
            params.zmq_prover.prover.beam_path = self.dump_path
            if params.zmq_prover.prover.beam_kind == BeamSearchKind.Fixed:
                params.zmq_prover.prover.beam_path = (
                    self.dump_path / "checkpoint.-1.pth"
                )

            self.params = params
            # worker / goal params sampler
            self.goal_params_sampler: Optional[GoalParamsSampler] = None
            worker_params_sampler: Optional[WorkerParamsSampler] = None
            (
                self.goal_params_sampler,
                worker_params_sampler,
            ) = maybe_make_search_params_samplers(
                hyperopt=params.hyperopt,
                hyperopt_param_str=params.hyperopt_param_str,
                n_machines=params.zmq_prover.n_machines,
                n_simultaneous_proofs=params.zmq_prover.prover.n_simultaneous_proofs,
            )
            super().__init__(
                worker_launcher=prover_launcher(
                    ZMQProver,
                    params=params.zmq_prover,
                    name=f"{params.exp_name}_{job_id}",
                    worker_params_sampler=worker_params_sampler,
                ),
                n_workers=params.zmq_prover.n_machines,
                max_restarts=params.zmq_prover.max_restarts,
                max_queued_inputs=params.zmq_prover.prover.n_simultaneous_proofs,
                check_alive_freq=60,
            )

            # split labels
            self.splits, props = params.splits_props
            self.props = np.array(props) / sum(props)
            self.all_labels: Dict[str, List[str]] = {}
            self.label_to_split: Dict[str, str] = {}
            for split in self.splits:
                if params.lang == "eq" and split in [
                    "eq_bwd_rwalk_seq2seq",
                    "eq_bwd_graph_seq2seq",
                ]:
                    labels = [split]  # will be generated on the fly
                else:
                    labels = get_labels(
                        params.zmq_prover.dataset, split, n_to_prove=1_000_000_000
                    )
                for label in labels:
                    assert "__" not in label
                    assert (
                        label not in self.label_to_split
                    ), f"{label} in multiple splits"
                    self.label_to_split[label] = split
                self.all_labels[split] = labels
                logger.info(f"Loaded {len(labels)} labels for {split}")

            self.label_sampler: Optional[PrioritizedLabelSampler] = (
                prioritized_label_sampler_factory(
                    cfg=params.label_sampler,
                    splits=self.splits,
                    split_probs=self.props,
                    label_to_split=self.label_to_split,
                    split_to_labels=self.all_labels,
                )
            )
            self.rng = np.random.RandomState(
                params.launcher_seed if params.launcher_seed > -1 else None
            )

            os.makedirs(self.dump_path, exist_ok=True)
            self.result_dump = open(self.dump_path / "mcts_results.jsonl", "a+")
            self.mcts_hists_dump = open(self.dump_path / "mcts_hists.jsonl", "a+")

            self.trainer_dump = open(self.dump_path / "trainer_stats.json", "a+")
            self.proved: Dict[str, Set[str]] = {split: set() for split in self.splits}
            self.total_recv_results = 0
            self.total_recv_samples = 0

            self.solving_stats: MCTSSolvingStats = MCTSSolvingStats(
                params.hyperopt_param_str, n_buckets=10
            )

            self.mcts_stats = WeightedAvgStats()
            self.mcts_hist_stats = HistStats()
            self.gpu_stats = WeightedAvgStats()
            self.produced_stats: Dict[str, ActionCounter] = defaultdict(
                lambda: ActionCounter(name="", is_rate=True, silent=True)
            )

            self.exceptions: Dict[str, int] = defaultdict(int)
            # tb stats loggers
            self.launch_metrics_logger = MetricsLogger(
                outdir=self.dump_path, quiet=False, tag="launcher_stats"
            )
            self.mcts_metrics_logger = MetricsLogger(
                outdir=self.dump_path, quiet=False, tag="mcts_stats"
            )
            self.label_sampler_metrics_logger = MetricsLogger(
                outdir=self.dump_path, quiet=False, tag="label_sampler_stats"
            )
            self.label_sampler_state_dumper = StateDumper(
                folder=self.dump_path / "label_sampler", n_states_max=100
            )

            self.has_been_closed = False
            self.last_dump: float = -1
        except:
            self.close_trainer()
            raise

    def run(self):
        # For tim-> Even if it duplicates a little bit the code I find it lighter to have it here
        self.last_log = time.time()
        for _ in make_iterator(
            self,
            max_in_worker=self.max_queued_inputs * self.n_workers,
            input_it=self.input_it(),
        ):
            pass

    def ready(self) -> List[Tuple[RequestId, ZMQProverOutput]]:
        if time.time() - self.last_log > 60:
            self.log_stats()
            if self.trainer_job is not None and self.trainer_job.done():
                try:
                    _ = self.trainer_job.result()
                except Exception as e:
                    logger.error(
                        f"Trainer job {self.trainer_job.job_id} died with: \n {str(e)}."
                    )
                raise RuntimeError("Trainer job {self.trainer_job.job_id} died.")
            self.last_log = time.time()
        # receive stats from trainer
        try:
            _, trainer_stats_json = self.trainer_sock.recv_multipart(zmq.NOBLOCK)
        except ZMQError:
            # no message -> busy wait
            pass
        else:
            # todo fix CI (local mypy works)
            trainer_stats = json.loads(trainer_stats_json)  # type: ignore
            self.handle_message(trainer_stats)
        return super().ready()

    def log_info(self, msg: str):
        logger.info(f"{self.__class__.__name__}: {msg}")

    def close_trainer(self):
        if self.trainer_job is not None:
            self.log_info(f"Canceling trainer_job {self.trainer_job.job_id}")
            self.trainer_job.cancel(check=False)
            self.log_info(f"Canceled trainer_job {self.trainer_job.job_id}")

    def close(self):
        if not self.has_been_closed:
            self.log_info("Closing MCTSGangHandler ...")
            self.has_been_closed = True
            self.launch_metrics_logger.close()
            self.mcts_metrics_logger.close()
            self.label_sampler_metrics_logger.close()
            self.close_trainer()
            self.result_dump.close()
            self.mcts_hists_dump.close()
            self.trainer_dump.close()
            self.trainer_controller_context.close()
            super().close()
            self.log_info("MCTSGangHandler closed.")
        else:
            self.log_info("MCTSGangHandler already closed!")

    def input_it(self):
        while True:
            if self.label_sampler:
                label = self.label_sampler.sample_label(self.rng)
            else:
                next_split = self.rng.choice(self.splits, size=1, p=self.props)[0]
                label = str(self.rng.choice(self.all_labels[next_split]))
            assert isinstance(label, str), type(label)
            goal = BackwardGoal.create_unmat(label, split=self.label_to_split[label])
            if self.goal_params_sampler is not None:
                goal.params = self.goal_params_sampler.sample_goal_params(
                    goal_name=goal.name
                )
            yield goal

    def handle_result(self, result: Dict):
        assert result["type"] == "result"
        label = result["label"]
        # name = result["name"]

        assert "split" not in result
        result["split"] = self.label_to_split[label]

        to_dump = dict(result)  # shallow copy
        mcts_hist_stats = to_dump.pop("mcts_hist_stats", None)
        # dump result
        self.result_dump.write(json.dumps(to_dump) + "\n")
        self.result_dump.flush()
        self.mcts_hists_dump.write(json.dumps(mcts_hist_stats) + "\n")
        self.mcts_hists_dump.flush()

        if params.hyperopt != HyperOptKind.Fixed:
            self.solving_stats.update(
                solved=result["success"],
                g_params=result["goal_params"],
                w_params=result["worker_params"],
            )
        if self.goal_params_sampler is not None:
            self.goal_params_sampler.update(
                result["name"], success=result["success"]
            )

        if self.label_sampler:
            pls_stats = PLSStats.from_mcts_stats(result["mcts_stats"])
            self.label_sampler.update_score(label, pls_stats=pls_stats)
        if result["success"]:
            split = self.label_to_split[label]
            if label not in self.proved[split]:
                self.proved[split].add(label)
                solved = len(self.proved[split])
                total = len(self.all_labels[split])
                logger.info(
                    f"Solved label {label} for the first time. "
                    f"{split} accuracy: {100 * solved / total:.2f}% ({solved}/{total})"
                )

        # TODO: factor this with ProofHandler?
        self.mcts_stats.update(result["mcts_stats"])
        self.mcts_hist_stats.update(result["mcts_hist_stats"])
        self.gpu_stats.update(result["gpu_stats"])

        # stats on received results / samples / exceptions
        n_recv = sum(result["n_samples_sent"].values())
        self.produced_stats["recv_results/s"].act()
        self.produced_stats["recv_samples/s"].act(n_recv)
        self.total_recv_results += 1
        self.total_recv_samples += n_recv
        for subtask, count in result["n_samples_sent"].items():
            self.produced_stats[f"recv_samples__{subtask}/s"].act(count)

        if result["exception"] is None:
            self.produced_stats["recv_valid_results/s"].act()
        else:
            self.exceptions["exception__" + result["exception"]] += 1

        return result

    def handle_message(self, message: Dict[str, Any]):
        if message["type"] == "trainer_stat":
            logger.info(message["message"])
        elif message["type"] == "trainer_eval":
            logger.info(message["scores"])
            self.trainer_dump.write(json.dumps(message["scores"]) + "\n")
            self.trainer_dump.flush()
        elif message["type"] == "trainer_epoch":
            logger.info(f"-------- END OF TRAINER EPOCH {message['epoch']} ---------")
        else:
            raise RuntimeError(f"Unexpected message {message}")

    def log_stats(self):
        log_memory(logger)
        produced_stats = {x: y.rate_and_reset() for x, y in self.produced_stats.items()}
        produced_stats["total_recv_results"] = self.total_recv_results
        produced_stats["total_recv_samples"] = self.total_recv_samples
        produced_stats["n_dead_goals"] = self.n_dead_inputs
        produced_stats["n_input_buffer"] = len(self.input_buffer)
        produced_stats.update({k: v for k, v in self.jobs_stats.items()})

        solving_stats = self.solving_stats.get_stats()
        logger.info(pformat(produced_stats))
        logger.info(pformat(self.mcts_stats.stats))
        logger.info(pformat(self.gpu_stats.stats))
        logger.info(pformat(dict(self.exceptions)))
        logger.info(pformat(solving_stats))

        assert self.mcts_stats.stats.keys().isdisjoint(
            self.mcts_hist_stats.stats.keys()
        )
        all_metrics = dict(self.mcts_stats.stats, **self.mcts_hist_stats.stats)
        all_metrics.update(solving_stats)
        self.mcts_metrics_logger.log_metrics(all_metrics)
        self.mcts_metrics_logger.log_histograms(
            self.mcts_hist_stats.hist, restore_hist_to_values=True
        )

        launch_metrics = {}
        if self.goal_params_sampler is not None:
            rec = self.goal_params_sampler.get_recommendation()
            logger.info(pformat(rec))
            launch_metrics.update({f"ng_{x}": y for x, y in rec.items()})
        launch_metrics.update(produced_stats)
        launch_metrics.update(self.gpu_stats.stats)
        launch_metrics.update(self.exceptions)
        for split, solved in self.proved.items():
            acc = 100 * len(solved) / len(self.all_labels[split])
            launch_metrics.update(
                {
                    f"solved_labels_acc_{split}": acc,
                    f"solved_labels_n_{split}": len(solved),
                }
            )
        self.launch_metrics_logger.log_metrics(launch_metrics)

        self.mcts_stats.reset()
        self.mcts_hist_stats.reset()
        self.gpu_stats.reset()

        if self.label_sampler:
            self.label_sampler_metrics_logger.log_metrics(self.label_sampler.stats())
            if time.time() - self.last_dump > 3600:
                self.label_sampler_state_dumper.maybe_dump_state(
                    self.label_sampler.state()
                )
                self.last_dump = time.time()


if __name__ == "__main__":
    set_TMPDIR()
    set_MKL_env_vars()
    register_online_mcts_configs()

    params: MCTSOnlineParams = cfg_from_cli(schema=MCTSOnlineParams)

    if params.exp_name == EXP_NAME:
        logger.info("Need to prepare the run.")
        prepare(exp_name=EXP_NAME)

    logger.info(f"Experiment workdir: {os.getcwd()}")

    OurSignalHandler.start()

    handler = MCTSGangHandler(params)
    with logged_closing(handler, "MCTSGangHandler"):
        handler.run()
