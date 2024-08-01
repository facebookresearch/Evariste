# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Optional, Dict, Tuple
from collections import defaultdict
from dataclasses import asdict
from logging import getLogger
from pathlib import Path
from evariste.comms.zmq import ZMQNotReadySample, ManagedZmqContext
from evariste.model.data.envs.hol_light import HOLLightDataEnvironment
from evariste.model.data.envs.latex import LatexDataEnvironment
from zmq import ZMQError
import os
import zmq
import time
import math
import psutil
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from evariste.refac.utils import safe_load
from evariste.model.kl_teacher_model import KLTeacherModel
from evariste.model.optim import get_optimizer
from evariste.model.transformer import (
    entropy_loss,
    kl_div_loss,
    get_decoder_scores,
    distillation_hidden_states_loss,
)
from evariste.model.transformer_utils import get_clm_mask_target
from evariste.model.utils import (
    to_cuda,
    batch_sequences,
    get_embs,
    load_module_state_dict,
    reload_optimizer_state,
    # show_batch,
)

from evariste.backward.prover.utils import copy_model
from evariste.trainer.args import TrainerArgs
from evariste.model.checkpoints import get_latest_checkpoint, keep_latest_checkpoints
from evariste.model.data.dictionary import Dictionary
from evariste.model.transformer_args import DecodingParams
from evariste.metrics import Logger as MetricsLogger
from evariste.backward.prover.utils import GPUMonitor
from evariste.utils import timeout


logger = getLogger()


class Trainer(object):
    def __init__(
        self, modules: Dict, envs: Dict, dico: Dictionary, params: TrainerArgs
    ):
        """
        Initialize trainer.
        """
        self.MODULE_NAMES = sorted(list(modules.keys()))
        assert sorted(params.module_names) == self.MODULE_NAMES

        # modules / environments / params
        self.modules = modules
        self.envs = envs
        self.params = params

        self.metrics_logger = MetricsLogger(
            outdir=params.dump_path, quiet=not params.slurm_conf.is_master
        )
        if params.log_network_stats_freq > 0:
            self.network_metrics_logger = MetricsLogger(
                outdir=params.dump_path,
                quiet=not params.slurm_conf.is_master,
                tag="network",
            )

        # dictionary
        self.dico = dico
        assert all(env.dico == self.dico for env in self.envs.values())

        # epoch / iteration size
        self.epoch_size = params.epoch_size
        assert self.epoch_size > 0

        # set parameters
        self.set_parameters()

        # float16 / distributed (no AMP)
        assert params.model.fp16 or params.accumulate_gradients == 1
        if params.slurm_conf.multi_gpu:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            for name in self.MODULE_NAMES:
                logger.info(f"Wrapping {name} ...")
                self.modules[name] = nn.parallel.DistributedDataParallel(
                    self.modules[name],
                    device_ids=[params.slurm_conf.local_rank],
                    output_device=params.slurm_conf.local_rank,
                    broadcast_buffers=True,
                    find_unused_parameters=True,
                )

        # set optimizer
        self.set_optimizer_scaler()

        self.kl_teacher_model: Optional[KLTeacherModel] = KLTeacherModel.create(
            self.params.kl,
            device="cuda",
            fp16=self.params.model.fp16,
            student_dico=self.dico,
        ) if self.params.kl.should_use_kl() else None

        # stopping criterion used for early stopping
        if params.stopping_criterion != "":
            split = params.stopping_criterion.split(",")
            assert len(split) == 2 and split[1].isdigit(), params.stopping_criterion
            assert not split[0].endswith(
                "-proven"
            ), "bwd_prover_eval not supported for stopping criterion"
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == "_":
                self.stopping_criterion: Optional[Tuple[str, bool]] = (
                    split[0][1:],
                    False,
                )
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion: Optional[float] = (
                -1e12 if self.stopping_criterion[1] else 1e12
            )
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(",") if m != ""]
        for metric in metrics:
            metric_tuple = (metric[1:], False) if metric[0] == "_" else (metric, True)
            assert not metric[0].endswith(
                "-proven"
            ), "bwd_prover_eval not supported validation metric"
            self.metrics.append(metric_tuple)
        self.best_metrics = {
            metric_name: (-1e12 if biggest else 1e12)
            for (metric_name, biggest) in self.metrics
        }

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sentences = 0
        self.last_time = time.time()
        self.int_stats: Dict[str, int] = defaultdict(int)
        self.stats: Dict[str, List] = defaultdict(list)
        self.cond_stats: Dict[str, List] = defaultdict(list)
        self.batch_time: List[float] = []

        # reload potential checkpoints
        self.reload_checkpoint()

        # probability of masking out / randomize / not modify words to predict
        self.pred_probs = torch.FloatTensor(
            [params.mlm.word_mask, params.mlm.word_keep, params.mlm.word_rand]
        )

        # probability for a word to be masked out
        counts = np.array(
            [self.dico.counts[self.dico[i]] for i in range(len(self.dico))],
            dtype=np.float64,
        )
        self.mask_scores = np.maximum(counts, 1) ** -params.mlm.sample_alpha
        self.mask_scores[:4] = 0  # do not predict bos, eos, pad and unk words
        self.mask_scores[counts == 0] = 0  # do not predict special tokens

        # connect to stats_socket if defined
        self.stat_socket: Optional[zmq.Socket] = None
        self.context: Optional[ManagedZmqContext] = None
        if params.stats_socket_addr != "":
            self.context = ManagedZmqContext()
            self.stat_socket = self.context.socket(zmq.DEALER)
            self.stat_socket.identity = b"trainer"
            self.stat_socket.connect(params.stats_socket_addr)

        # negative samplers
        self.negative_samplers = {}
        if "mm_goal2rule_seq2emb" in params.parsed_tasks("mm"):
            self.negative_samplers["mm"] = self.envs["mm"].build_negative_sampler()

        # export checkpoint for adversarial
        if (
            params.online_fwd_generation
            or params.rl_distributed.is_adversarial_training
            or (
                params.has_mcts() and get_latest_checkpoint(params.dump_path)[1] == -2
            )  # mcts but not checkpoint is there
        ):
            # this allow generation workers to start early and have good GPU usage stats
            self.save_checkpoint("checkpoint.-1", include_optimizer=True)

        # data iterators
        logger.info("========== Creating Iterators ... ==========")
        self.iterators = {}
        for task in self.params.parsed_tasks():
            logger.info(f"===== Creating iterator for {task} ...")
            env_name = task.split("_")[0]
            env = self.envs[env_name]
            if "mcts" in task:
                self.iterators[task] = env.create_data_iterator("train", task)
                logger.info(f"Opening socket for ... {task}")
                sample = next(self.iterators[task])
                assert isinstance(sample, ZMQNotReadySample), repr(sample)
                logger.info("Socket Opened.")
            elif task.endswith("_rl"):
                self.iterators[task] = env.create_data_iterator("train", task)
                logger.info(f"Fetching example for ... {task}")
                sample = next(self.iterators[task])
                assert isinstance(sample, ZMQNotReadySample), repr(sample)
                logger.info("Socket Opened.")
            # to be sure that all workers load PACT data at the beginning, query one batch.
            # indeed, if we are training with N GPUs, choose_and_run_task may select this task
            # at N different timesteps, which would make the overall loading time N slower.
            # also do it for other tasks to be sure that all iterators are working properly.
            else:
                if isinstance(env, LatexDataEnvironment) or isinstance(
                    env, HOLLightDataEnvironment
                ):  # create data iterator not implemented
                    self.iterators[task] = iter(env.create_data_loader("train", task))
                else:
                    self.iterators[task] = env.create_data_iterator(
                        "train",
                        task,
                        try_fetch=True,
                        seconds=900,  # higher now to load data
                    )

        logger.info("========== Created Iterators. ==========")

        # Now that all sockets are opened, make sure we are able to receive one sample for each task before starting.
        for task in self.params.parsed_tasks():
            if "mcts" in task or task.endswith("_rl"):
                logger.info(f"Waiting for sample for {task}...")
                if isinstance(next(self.iterators[task]), ZMQNotReadySample):
                    raise RuntimeError(f"ZMQ uninitialized for task {task}")
                else:
                    logger.info(f"Sample received for {task}")

        # log each processed task
        f_name = f"tasks_{params.slurm_conf.global_rank}"
        self.f_task = open(os.path.join(params.dump_path, f_name), "a")
        self.f_task.write("===== Init f_task logging =====")
        self.f_task.flush()

    def get_optimizer_parameters(self):
        """
        Return parameters for optimizer.
        If the optimizer params are the same for all modules, return list of parameters.
        Else, return [
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ]
        Params should be given in the same order as self.parameters so that
        there is no silent bug in reloading optimizer state.
        """
        if len(self.params.per_module_optimizer_params) == 0:
            return self.parameters
        else:
            parameters: List[Dict] = []
            for name, module in self.modules.items():
                params = {"params": [p for p in module.parameters() if p.requires_grad]}
                if name in self.params.per_module_optimizer_params:
                    params.update(self.params.per_module_optimizer_params[name])
                start = sum(len(p["params"]) for p in parameters)
                parameters.append(params)
                assert self.parameter_indices[name] == (
                    start,
                    start + len(params["params"]),
                )
            assert len(self.parameters) == sum(len(p["params"]) for p in parameters)
            return parameters

    def set_parameters(self):
        """
        Set parameters.
        """
        # flattened module parameters
        self.parameters: List[torch.nn.Parameter] = []

        # for each module, store the corresponding parameters indices for the optimizer
        self.parameter_indices: Dict[str, Tuple[int, int]] = {}

        start = 0
        for name, module in self.modules.items():
            params = [p for p in module.parameters() if p.requires_grad]
            self.parameters += params
            self.parameter_indices[name] = (start, start + len(params))
            start += len(params)
        logger.info(
            f"Found {len(self.parameters)} parameters in {len(self.modules)} modules."
        )

    def set_optimizer_scaler(self):
        """
        Set optimizer and scaler.
        """
        params = self.params
        parameters = self.get_optimizer_parameters()
        self.optimizer = get_optimizer(parameters, params.optimizer)
        self.scaler = None
        if params.model.fp16:
            logger.info("Using torch.cuda.amp GradScaler for fp16 optimization")
            self.scaler = torch.cuda.amp.GradScaler()
            assert params.accumulate_gradients >= 1
        else:
            assert params.accumulate_gradients == 1
            # TODO: accumulation should be possible with:
            # https://github.com/pytorch/pytorch/pull/21736

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).any():
            logger.warning("NaN detected")

        params = self.params
        parameters = self.parameters
        optimizer = self.optimizer

        # regular optimization / torch.amp fp16 optimization
        if params.model.fp16 is False:
            optimizer.zero_grad()
            loss.backward()
            if params.clip_grad_norm > 0:
                # norm_check_a = (sum([p.grad.norm(p=2).item() ** 2 for p in parameters])) ** 0.5
                clip_grad_norm_(parameters, params.clip_grad_norm)
                # norm_check_b = (sum([p.grad.norm(p=2).item() ** 2 for p in parameters])) ** 0.5
                # print(norm_check_a, norm_check_b)
            optimizer.step()
        else:
            self.scaler.scale(loss).backward()
            if (self.n_iter + 1) % params.accumulate_gradients == 0:
                if params.clip_grad_norm > 0:
                    # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
                    # Unscales the gradients of optimizer's assigned params in-place
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters, params.clip_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

    def iter(self, gpu_mon: GPUMonitor):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        stats = self.print_stats(gpu_mon)
        if stats and self.stat_socket is not None:
            try:
                self.stat_socket.send_json(
                    {"src": "trainer", "message": stats, "type": "trainer_stat"},
                    zmq.NOBLOCK,
                )
            except ZMQError as e:
                logger.warning(f"Couldn't send stats back! {e}")

    @timeout(seconds=900)
    def get_batch(self, task: str):
        """
        Return a batch, and compute the time required to generate it.
        """
        self.f_task.write(f"GET_BATCH {self.n_iter} {task} {time.time()}\n")
        self.f_task.flush()

        start = time.time()
        batch = next(self.iterators[task])
        b_time = time.time() - start
        self.batch_time.append(b_time)

        self.f_task.write(
            f"GOT_BATCH {self.n_iter} {task} {time.time()} batch_time={b_time}\n"
        )
        self.f_task.flush()

        return batch

    @torch.no_grad()
    def log_network_stats(self):
        """
        Log statistics about network modules.
        """
        if not (
            self.params.slurm_conf.is_master
            and self.params.log_network_stats_freq > 0
            and self.n_total_iter > 0
            and self.n_total_iter % self.params.log_network_stats_freq == 0
        ):
            return
        stats: Dict[str, float] = {}
        for module_name, module in self.modules.items():
            for param_name, param_weights in module.state_dict().items():
                # if "bias" in param_name:
                #     continue
                stats[f"{module_name}__{param_name}"] = param_weights.norm(2).item()
        self.network_metrics_logger.log_metrics(stats)

    def print_stats(self, gpu_mon: GPUMonitor):
        """
        Print statistics about the training.
        """
        self.log_network_stats()

        if self.n_total_iter % 20 != 0:
            return ""

        stat_dict: Dict[str, float] = {
            f"trainer/{k}": float(np.mean(v))
            for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        }
        s_stat = " || ".join(
            f"{k}: {float(np.mean(v)):7.4f}"
            for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        )

        # conditioning statistics
        for k, values in self.cond_stats.items():
            n_classes = self.params.cond.n_classes
            counter = torch.zeros(n_classes)
            for x in values:
                counter[x] += 1
            counter /= counter.sum()
            counter = counter[counter > 0]
            entropy = -float((np.log(counter) * counter).sum().item())
            usage = 100 * float((counter > 0).sum()) / n_classes
            stat_dict[f"trainer/{k}_entropy"] = entropy
            stat_dict[f"trainer/{k}_usage"] = usage
            s_stat += f" || {k}: ent={entropy:.3f} use={usage:.1f}%"
        self.cond_stats.clear()

        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                self.stats[k].clear()

        # time since last log of stats
        now = time.time()
        diff = now - self.last_time

        # batch generation time
        batch_time = sum(self.batch_time)
        s_batch = f"Batch time: {batch_time:6.3f}s ({100. * batch_time / diff:4.1f}%)"
        self.batch_time.clear()

        # optimizer learning rate
        s_lr = "LR: " + " / ".join(
            f"{group['lr']:.4e} ({group.get('num_updates', None)})"
            for group in self.optimizer.param_groups
        )
        for i, group in enumerate(self.optimizer.param_groups):
            suffix = "" if i == 0 else f"_{i}"
            stat_dict[f"trainer/optim/lr{suffix}"] = group["lr"]

        # GPU Util
        gpu_stats = gpu_mon.stats[torch.cuda.current_device()]
        gpu_mem = gpu_stats.stats.get("mem", -1)
        gpu_usage = gpu_stats.stats.get("gpu", -1)
        s_gpu = f"GPU mem={gpu_mem:5.2f}% usage={gpu_usage:5.2f}%"
        stat_dict[f"trainer/gpu/mem"] = gpu_mem
        stat_dict[f"trainer/gpu/usage"] = gpu_usage
        gpu_stats.reset()

        # processing speed
        s_speed = " - ".join(f"{v / diff:7.1f} {k}" for k, v in self.int_stats.items())
        for k, v in self.int_stats.items():
            stat_dict[f"speed/{k.replace('/', '_')}"] = v / diff
            self.int_stats[k] = 0

        self.metrics_logger.log_metrics(stat_dict)
        self.last_time = now

        # log speed + stats + learning rate
        to_print = (
            f"{self.n_total_iter:<7} - "
            f"epoch {self.epoch} ({100 * self.n_sentences / self.epoch_size:.1f}%) - "
            f"{s_speed} - "
            f"{s_stat} - "
            f"{s_lr} - "
            f"{s_batch} - "
            f"{s_gpu}"
        )
        logger.info(to_print)
        return to_print

    def save_checkpoint(self, name: str, include_optimizer: bool = True) -> None:
        """
        Save the model / checkpoints.
        """
        if not self.params.slurm_conf.is_master:
            return
        temp_path = os.path.join(self.params.dump_path, "tmp.pth")
        path = os.path.join(self.params.dump_path, "%s.pth" % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            "epoch": self.epoch,
            "n_total_iter": self.n_total_iter,
            "best_metrics": self.best_metrics,
            "best_stopping_criterion": self.best_stopping_criterion,
        }

        for name in self.MODULE_NAMES:
            logger.warning(f"Saving {name} parameters ...")
            data[name] = self.modules[name].state_dict()

        if include_optimizer:
            logger.warning("Saving optimizer ...")
            data[f"optimizer"] = self.optimizer.state_dict()
            data[f"optimizer_parameter_indices"] = self.parameter_indices
            if self.scaler is not None:
                data["scaler"] = self.scaler.state_dict()

        data["dico_id2word"] = self.dico.id2word
        data["dico_word2id"] = self.dico.word2id
        data["dico_counts"] = self.dico.counts
        data["params"] = asdict(self.params)

        # first write to tmp file, then move to final destination
        # this avoids weird errors on load for the MCTS provers.
        torch.save(data, temp_path)
        os.rename(temp_path, path)

    def reload_checkpoint(self) -> None:
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path, _ = get_latest_checkpoint(self.params.dump_path)
        if checkpoint_path is None:
            if (
                self.params.reload_checkpoint == ""
                and self.params.reload_partial_checkpoint == ""
            ):
                return
            elif len(self.params.reload_partial_checkpoint) > 0:
                assert self.params.reload_checkpoint == ""
                return self.reload_partial_checkpoint()
            else:
                checkpoint_path = self.params.reload_checkpoint
                assert os.path.isfile(checkpoint_path)
                # if we start from a checkpoint, copy it to our own folder as epoch -1
                if self.params.slurm_conf.is_master:
                    copy_model(
                        Path(self.params.reload_checkpoint), Path(self.params.dump_path)
                    )

        logger.info(f"Reloading checkpoint from {checkpoint_path} ...")
        data = safe_load(checkpoint_path, map_location="cpu")

        # check dictionaries
        _dico = Dictionary(
            data["dico_id2word"], data["dico_word2id"], data["dico_counts"], frozen=True
        )
        if not self.params.use_checkpoint_dico:
            if not _dico == self.dico:
                print(set(_dico.word2id.keys()) - set(self.dico.word2id.keys()))
                print(set(self.dico.word2id.keys()) - set(_dico.word2id.keys()))
            assert _dico == self.dico, (len(_dico), len(self.dico))

        # reload model parameters
        for name in self.MODULE_NAMES:
            logger.info(f"Reloading module {name} from checkpoint ...")
            load_module_state_dict(self.modules[name], name, data, self.params)

        # reload optimizer
        logger.info("Reloading checkpoint optimizer ...")
        if self.params.reload_checkpoint_optimizer_state_only:
            logger.warning("Reloading the optimizer state only!")
            tgt_state = self.optimizer.state_dict()["state"]
            src_state = data["optimizer"]["state"]
            assert tgt_state.keys() == src_state.keys()
            n_params = len(tgt_state.keys())
            reload_optimizer_state(tgt_state, src_state, n_params)
        else:
            self.optimizer.load_state_dict(data["optimizer"])
        for group_id, param_group in enumerate(self.optimizer.param_groups):
            for k in ["num_updates", "lr"]:
                if k in param_group:
                    logger.info(
                        f"Optimizer parameter group (ID={group_id}) - {k}: "
                        f"{param_group[k]}"
                    )

        # reload gradient scaler
        if self.params.model.fp16:
            logger.info("Reloading gradient scaler ...")
            self.scaler.load_state_dict(data["scaler"])
        else:
            assert self.scaler is None and "scaler" not in data

        # reload main metrics
        self.epoch = data["epoch"] + 1
        self.n_total_iter = data["n_total_iter"]
        self.best_metrics.update(data["best_metrics"])
        self.best_stopping_criterion = data["best_stopping_criterion"]
        logger.info(
            f"Checkpoint reloaded. Resuming at epoch {self.epoch} / "
            f"iteration {self.n_total_iter} ..."
        )

    def reload_partial_checkpoint(self) -> None:
        """
        For each module in params.parsed_reload_partial_checkpoint,
        will reload the associated model weights and optimizer state of the given checkpoint.
        """

        for name, path in self.params.parsed_reload_partial_checkpoint.items():
            logger.info(f"Reloading {name} module checkpoint from {path} ...")
            data = safe_load(path, map_location="cpu")
            if name == "big_decoder" and "big_decoder" not in data:
                src_name = "decoder"
                logger.warning(f"For big decoder, reloading decoder.")
            elif name in data:
                src_name = name
            else:
                raise RuntimeError(f"Module {name} not in {path}")
            # check dictionaries
            _dico = Dictionary(
                data["dico_id2word"],
                data["dico_word2id"],
                data["dico_counts"],
                frozen=True,
            )
            assert _dico == self.dico, (len(_dico), len(self.dico))

            # reload model parameters
            logger.info(f"Reloading {name} weights from {src_name} checkpoint ...")
            load_module_state_dict(self.modules[name], src_name, data, self.params)

            # retrieve indices for this module.
            # for now reload partial checkpoint only supports reloading optimizer state only.
            # TODO: if needed supports more optimizer reloading, not sure it will be useful at some point?
            logger.info(
                f"Reloading {name} optimizer state from {src_name} checkpoint ..."
            )
            if src_name not in data["optimizer_parameter_indices"]:
                raise RuntimeError(
                    f"Could not reload optimizer state safely for {name}, "
                    f"no parameters index available in {path}."
                )
            src_indices = data["optimizer_parameter_indices"][src_name]
            tgt_indices = self.parameter_indices[name]
            n_src_params = src_indices[1] - src_indices[0]
            n_tgt_params = tgt_indices[1] - tgt_indices[0]
            assert n_src_params == n_tgt_params, (n_src_params, n_tgt_params)

            # reload optimizer
            reload_optimizer_state(
                tgt_state=self.optimizer.state_dict()["state"],
                src_state=data["optimizer"]["state"],
                n_params=n_tgt_params,
                tgt_ind=tgt_indices[0],
                src_ind=src_indices[0],
            )

    def save_best_model(self, scores) -> None:
        """
        Save best models according to given validation metrics.
        """
        if not self.params.slurm_conf.is_master:
            return
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.warning(f'Metric "{metric}" not found in scores!')
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info(f"New best score for {metric}: {scores[metric]:.6f}")
                self.save_checkpoint(f"best-{metric}", include_optimizer=True)

    def save_and_roll_checkpoint(self) -> None:
        if not self.params.slurm_conf.is_master:
            return
        self.save_checkpoint(f"checkpoint.{self.epoch}", include_optimizer=True)
        keep_latest_checkpoints(
            self.params.dump_path, to_keep=self.params.n_kept_checkpoints
        )

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        if (
            self.stopping_criterion is not None
            and scores is not None
            and (
                self.params.slurm_conf.is_master
                or not self.stopping_criterion[0].endswith("-bleu")
                and not self.stopping_criterion[0].endswith("-proven")
                and not self.stopping_criterion[0].endswith("-proven-greedy")
            )
        ):
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            # best_stopping_criterion can be none if reloaded
            if (
                self.best_stopping_criterion is None
                or factor * scores[metric] > factor * self.best_stopping_criterion
            ):
                self.best_stopping_criterion = scores[metric]
                logger.info(
                    f"New best validation score: {self.best_stopping_criterion}"
                )
                self.decrease_counts = 0
            else:
                logger.info(
                    "Not a better validation score (%i / %i)."
                    % (self.decrease_counts, self.decrease_counts_max)
                )
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info(
                    f"Stopping criterion has been below its best value for more "
                    f"than {self.decrease_counts_max} epochs. Ending the experiment ..."
                )
                if self.params.slurm_conf.multi_gpu and "SLURM_JOB_ID" in os.environ:
                    os.system("scancel " + os.environ["SLURM_JOB_ID"])
                exit()

        if self.stat_socket is not None:
            try:
                self.stat_socket.send_json(
                    {"type": "trainer_epoch", "epoch": self.epoch}, zmq.NOBLOCK
                )
            except ZMQError as e:
                logger.warning(
                    f"Got {e} while sending trainer_epoch msg to stat_socket!"
                )

        # Reset counters of max memory used at end of epoch.
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)

        self.epoch += 1

    def log_module_weights(self):
        """
        Print module weights to be sure they are still synchronized across GPUs.
        """
        logger.info("===== Module weights =====")
        for name, module in self.modules.items():
            x = {k: v.sum().item() for k, v in module.named_parameters()}
            logger.info(f"{name}: {sum(x.values())}")

    def get_mlm_mask_target(self, x: torch.Tensor, lengths: torch.Tensor, rng=None):
        """
        Get masked prediction mask with target tokens to predict.
        Decide of random words to mask out, and what target they get assigned to.
        """
        rng = np.random if rng is None else rng
        params = self.params
        bs, slen = x.size()

        # define target words to predict
        if params.mlm.sample_alpha == 0:
            pred_mask = rng.rand(bs, slen) <= params.mlm.word_pred
            pred_mask = torch.from_numpy(pred_mask)
        else:
            n_tgt = math.ceil(params.mlm.word_pred * lengths.sum().item())
            x_prob = self.mask_scores[x.view(-1)]
            tgt_ids = rng.choice(
                len(x_prob), n_tgt, replace=False, p=x_prob / x_prob.sum()
            )
            pred_mask = torch.zeros(bs * slen, dtype=torch.bool)
            pred_mask[tgt_ids] = 1
            pred_mask = pred_mask.view(bs, slen)

        # do not predict padding
        pred_mask[x == params.dico.pad_index] = 0

        # do not predict the first token
        pred_mask[:, 0] = 0

        # # mask a number of words == 0 [8] (faster with fp16)  # TODO: fix
        # if params.fp16:
        #     pred_mask = pred_mask.view(-1)
        #     n1 = pred_mask.sum().item()
        #     n2 = max(n1 % 8, 8 * (n1 // 8))
        #     if n2 != n1:
        #         pred_mask[torch.nonzero(pred_mask).view(-1)[:n1 - n2]] = 0
        #     pred_mask = pred_mask.view(slen, bs)
        #     assert pred_mask.sum().item() % 8 == 0

        # generate possible targets / update x input
        x_real = x[pred_mask]
        x_rand = rng.choice(params.dico.n_words, size=len(x_real))
        x_mask = np.full(len(x_real), params.dico.mask_index, dtype=np.int64)
        cat = rng.choice(
            len(self.pred_probs), size=len(x_real), p=self.pred_probs, replace=True
        )
        new_x = x_mask * (cat == 0) + x_real.numpy() * (cat == 1) + x_rand * (cat == 2)
        x = x.masked_scatter(pred_mask, torch.from_numpy(new_x))

        # sanity check
        assert 0 <= x.min() <= x.max() < params.dico.n_words
        assert x.size() == (bs, slen)
        assert pred_mask.size() == (bs, slen)

        return x, pred_mask, x_real

    def round_batch(self, x, xlen):
        """
        For float16 only.
        Sub-sample sentences in a batch, and add padding,
        so that each dimension is a multiple of 8.
        """
        params = self.params
        if not params.model.fp16:  # or len(xlen) < 8
            return x, xlen, None

        # number of sentences == 0 [8]
        bs1 = len(xlen)
        bs2 = 8 * (bs1 // 8)
        assert bs2 > 0 and bs2 % 8 == 0
        if bs1 != bs2:
            idx = torch.randperm(bs1)[:bs2]
            xlen = xlen[idx]
            slen = xlen.max().item()
            x = x[idx, :slen]
        else:
            idx = None

        # sequence length == 0 [8]
        ml1 = x.size(1)
        if ml1 % 8 != 0:
            pad = 8 - (ml1 % 8)
            ml2 = ml1 + pad
            x = torch.cat(
                [x, torch.LongTensor(bs2, pad).fill_(params.dico.pad_index)], 1
            )
            assert x.size() == (bs2, ml2)

        # sanity check
        assert x.shape[0] % 8 == 0
        assert x.shape[1] % 8 == 0
        assert x.shape[0] == xlen.shape[0]

        return x, xlen, idx

    def get_masked_tokens(self, x, rng=None):
        """
        Create a masked version of a tensor (BERT-like).
        """
        rng = np.random if rng is None else rng
        params = self.params
        n_cat = len(self.pred_probs)

        x = x.clone().numpy()
        x_real = x
        x_rand = rng.choice(params.dico.n_words, size=x.shape)
        x_mask = np.full_like(x, params.dico.mask_index)
        cat = rng.choice(n_cat, size=x.shape, p=self.pred_probs, replace=True)
        new_x = x_mask * (cat == 0) + x_real * (cat == 1) + x_rand * (cat == 2)
        return torch.from_numpy(new_x)

    def get_mass_batch(self, x, xlen, rng=None):
        """
        Create MASS batch (sequence filling).
        """
        rng = np.random if rng is None else rng

        def get_start_pos(slen, masked_len):
            """
            Get the start position of a masked segment in a sequence.
                20% -> masked segment is at the beginning
                20% -> masked segment is at the end
                60% -> masked segment is at a random position
            https://github.com/microsoft/MASS/blob/master/MASS-unsupNMT/src/trainer.py#L765
            """
            assert masked_len <= slen
            p = rng.random()
            if p <= 0.2:
                return 0
            elif p >= 0.8:
                return slen - masked_len
            else:
                return rng.randint(slen - masked_len + 1)

        x_batch = []
        y_batch = []
        ylen = xlen.float().mul(0.5).ceil().long()
        start_pos = []

        for i, (xl, yl) in enumerate(zip(xlen.tolist(), ylen.tolist())):

            # position of the masked sequence
            # a = rng.randint(0, xl - yl + 1)
            # a = rng.randint(-yl, xl + 1)
            # a = max(min(a, xl - yl), 0)
            a = get_start_pos(xl, yl)
            b = a + yl
            start_pos.append(a)

            # masked input / output
            new_x = x[i].clone()
            new_x[a:b] = self.get_masked_tokens(new_x[a:b])
            x_batch.append(new_x)
            y_batch.append(x[i, a:b])

        # batch sequences
        x_batch = torch.cat([item[None] for item in x_batch], 0)
        y_batch, ylen_ = batch_sequences(y_batch, self.params.dico.pad_index)
        assert ylen.eq(ylen_).all()

        # masked sequence token positions
        positions = torch.arange(0, ylen.max(), dtype=torch.long)
        positions = positions[None] + torch.LongTensor(start_pos)[:, None]

        return (x_batch, xlen), (y_batch, ylen), positions

    @property
    def conditining_temperature(self):
        params = self.params
        T = 1 - (self.n_total_iter / (params.cond.n_epoch_decrease * params.epoch_size))
        return max(T, 0)

    def clm_mlm_step(self, task: str, causal: bool):
        """
        CLM mode (causal = True)
            - Causal Language Modeling (CLM) objective

        MLM mode (causal = False)
            - Masked Language Modeling (MLM) objective
        """
        assert type(causal) is bool
        params = self.params
        encoder = self.modules["encoder"]
        encoder.train()

        # generate batch / select words to predict
        batch = self.get_batch(task)
        x = batch["x"]
        xlen = batch["xlen"]
        langs = batch.get("langs", None)

        # tokens to predict / cuda
        if causal:
            pred_mask, target = get_clm_mask_target(x, xlen)
        else:
            x, pred_mask, target = self.get_mlm_mask_target(x, xlen)
        x, xlen, pred_mask, target, langs = to_cuda(x, xlen, pred_mask, target, langs)

        # forward
        tensor = encoder("fwd", causal=causal, tokens=x, lengths=xlen, langs=langs)

        # compute loss / optimize
        _, loss = encoder(
            "compute_loss",
            tensor=tensor,
            pred_mask=pred_mask,
            target=target,
            epsilon=params.label_smoothing_eps,
        )
        self.optimize(loss)

        # update stats
        self.n_sentences += params.batch.size
        self.stats[task].append(loss.item())
        self.int_stats["sent/s"] += xlen.size(0)
        self.int_stats["word/s"] += int(pred_mask.sum().item())

    def mass_step(self, task: str):
        """
        MASS pretraining step.
        https://arxiv.org/pdf/1905.02450.pdf
        """
        params = self.params
        encoder = self.modules["encoder"]
        decoder = self.modules["decoder"]
        encoder.train()
        decoder.train()

        # generate batch
        batch = self.get_batch(task)
        (x, xlen), (y, ylen), positions = self.get_mass_batch(batch["x"], batch["xlen"])

        # tokens to predict
        pred_mask, target = get_clm_mask_target(y, ylen)

        # decoder does not attend masked tokens
        enc_mask = x.ne(params.dico.mask_index)

        # cuda
        x, xlen, y, ylen, positions = to_cuda(x, xlen, y, ylen, positions)
        enc_mask, pred_mask, target = to_cuda(enc_mask, pred_mask, target)

        # encode x
        encoded = encoder("fwd", causal=False, tokens=x, lengths=xlen)

        multi_gpu = params.slurm_conf.multi_gpu
        unwrapped_encoder = encoder.module if multi_gpu else encoder
        if unwrapped_encoder.is_changing_input_lengths():
            encoded_len = unwrapped_encoder.new_input_lengths(xlen)
            # TODO: maybe adapt enc_mask to match kernel conv
            enc_mask = None  # shape mismatch between encoded and enc_mask
        else:
            encoded_len = xlen

        # decode y
        decoded = decoder(
            "fwd",
            causal=True,
            tokens=y,
            lengths=ylen,
            src_enc=encoded,
            src_len=encoded_len,
            positions=positions,
            enc_mask=enc_mask,
        )

        # compute loss / optimize
        _, loss = decoder(
            "compute_loss",
            tensor=decoded,
            pred_mask=pred_mask,
            target=target,
            epsilon=params.label_smoothing_eps,
        )
        self.optimize(loss)

        # update stats
        self.n_sentences += params.batch.size
        self.stats[task].append(loss.item())
        self.int_stats["sent/s"] += xlen.size(0)
        self.int_stats["input_word/s"] += xlen.sum().item()
        self.int_stats["output_word/s"] += int(pred_mask.sum().item())

    @torch.no_grad()
    def generate_samples(self, score: float):
        """
        Generate sentences for discriminative training.
        """
        assert 0 <= score <= 1
        params = self.params
        encoder = self.modules["encoder"].eval()
        encoder = encoder.module if params.slurm_conf.multi_gpu else encoder

        # decoding parameters
        decoding_params = DecodingParams(
            max_gen_len=params.batch.max_len,
            n_samples=params.batch.size,
            use_beam=False,
            fixed_temperature=1.0,  # TODO: add hyper-parameter
            top_k=None,  # TODO: add hyper-parameter
        )

        # generate
        discr = to_cuda(torch.FloatTensor(params.batch.size).fill_(score))
        x, xlen, _ = encoder.generate(
            src_enc=None, src_len=None, decoding_params=decoding_params, discr=discr
        )

        return x, xlen

    @torch.no_grad()
    def get_disc_scores(self, x, xlen):
        """
        Get discriminator scores.
        """
        params = self.params
        discriminator = self.modules["discriminator"]
        classifier = self.modules["classifier"]

        # get modules
        if params.slurm_conf.multi_gpu:
            discriminator = discriminator.module
            classifier = classifier.module

        discriminator.eval()
        classifier.eval()

        x, xlen = to_cuda(x, xlen)
        bs = len(xlen)

        # classify
        tensor = discriminator("fwd", causal=False, tokens=x, lengths=xlen)
        predicted = classifier(tensor[:, 0])
        assert predicted.size() == (bs, 1)

        return torch.sigmoid(predicted.view(-1))

    def cclm_step(self, task: str, sample: bool):
        """
        Conditional CLM mode
        """
        params = self.params
        encoder = self.modules["encoder"]
        encoder.train()

        # generate fake samples and label them, or use real samples
        if sample:
            x, xlen = self.generate_samples(score=1)
            disc_scores = self.get_disc_scores(x, xlen)
            # if self.params.slurm_conf.is_master:
            #     print(disc_scores.tolist())
        else:
            batch = self.get_batch(task)
            x = batch["x"]
            xlen = batch["xlen"]
            disc_scores = torch.full(size=(len(xlen),), fill_value=1)

        # tokens to predict / cuda
        pred_mask, target = get_clm_mask_target(x, xlen)
        x, xlen, pred_mask, target = to_cuda(x, xlen, pred_mask, target)
        disc_scores = to_cuda(disc_scores)

        # forward
        tensor = encoder("fwd", causal=True, tokens=x, lengths=xlen, discr=disc_scores)

        # compute loss / optimize
        _, loss = encoder(
            "compute_loss",
            tensor=tensor,
            pred_mask=pred_mask,
            target=target,
            epsilon=params.label_smoothing_eps,
        )
        self.optimize(loss)

        # update stats
        self.n_sentences += params.batch.size
        self.stats[f"{task}-gen-{'fake' if sample else 'real'}"].append(loss.item())
        self.int_stats["sent/s"] += xlen.size(0)
        self.int_stats["word/s"] += (xlen - 1).sum().item()

    def cclm_disc_step(self, task: str, sample: bool):
        """
        Conditional CLM mode
        """
        params = self.params

        # generate fake samples
        if sample:
            x, xlen = self.generate_samples(score=1)
            y = torch.FloatTensor(len(xlen)).fill_(params.gan.smooth_eps)
        # or use real samples
        else:
            batch = self.get_batch(task)
            x = batch["x"]
            xlen = batch["xlen"]
            y = torch.FloatTensor(len(xlen)).fill_(1 - params.gan.smooth_eps)

        loss = self.discriminator_loss(x, xlen, y)
        self.optimize(loss)

        # update stats
        self.n_sentences += params.batch.size
        self.stats[f"{task}-disc-{'fake' if sample else 'real'}"].append(loss.item())
        self.int_stats["sent/s"] += xlen.size(0)
        self.int_stats["word/s"] += (xlen - 1).sum().item()

    def seq2seq_disc_step(self, task: str):
        """
        This method is used to train a discriminator on a batch.
        Batch is expected to have keys:
         - "disc_tgt" indicating the target class of the sample.
         - "disc_inp" the input of discriminator
         - "disc_inp_len" the input lengths of discriminator
        """
        params = self.params

        # note we use
        batch = self.get_batch(task)
        tgt = batch["disc_tgt"]
        tgt[tgt == 0] = params.gan.smooth_eps
        tgt[tgt == 1] = 1 - params.gan.smooth_eps

        # we use y (command) as input of discriminator
        disc_inp = batch["disc_inp"]
        disc_inp_len = batch["disc_inp_len"]

        loss = self.discriminator_loss(disc_inp, disc_inp_len, tgt)
        self.optimize(loss)

        # update stats
        self.n_sentences += params.batch.size
        self.stats[f"{task}/critic"].append(loss.item())
        self.int_stats["sent/s"] += disc_inp_len.size(0)
        self.int_stats["word/s"] += (disc_inp_len - 1).sum().item()

    def discriminator_loss(self, disc_inp, disc_inp_len, tgt):
        """
        Small factorisation to make pycharm happy.
        """
        discriminator = self.modules["discriminator"]
        classifier = self.modules["classifier"]
        discriminator.train()
        classifier.train()

        # tokens to predict / cuda
        disc_inp, disc_inp_len, tgt = to_cuda(disc_inp, disc_inp_len, tgt)
        bs = len(disc_inp)

        # forward
        tensor = discriminator(
            "fwd", causal=False, tokens=disc_inp, lengths=disc_inp_len
        )
        predicted = classifier(tensor[:, 0])
        assert predicted.size() == (bs, 1)

        # loss
        loss = F.binary_cross_entropy_with_logits(predicted.view(-1), tgt)
        return loss

    def seq2seq_step(
        self, task: str, rng: np.random.RandomState, use_dicriminator: bool = False
    ):
        """
        Sequence-to-sequence training objective. Can be used for:
            - Machine Translation (MT)
            - Denoising Auto-Encoding (DAE)

        if use_dicriminator is True, then it computes the discriminator score and
        condition the encoder with it
        """
        seq2seq_time = time.time()
        params = self.params
        encoder = self.modules["encoder"]
        decoder = self.modules["decoder"]
        encoder.train()
        decoder.train()

        # generate batch
        # start = time.time()
        batch = self.get_batch(task)

        # batch_time = time.time() - start
        # n_tokens = batch["xlen"].sum().item() + batch["ylen"].sum().item()
        x = batch["x"]
        y = batch["y"]
        xlen = batch["xlen"]
        ylen = batch["ylen"]
        langs2 = batch.get("langs2", None)
        input_conditioning = batch.get("input_conditioning", None)
        bs = len(x)

        q_tactics = batch.get("q_tactics", None)
        q_conditioning = params.mcts_train.q_conditioning  # in ["", "sum", "prefix"]
        decoder_conditioning: Optional[torch.Tensor] = None
        if q_conditioning and q_tactics is None:
            decoder_conditioning = torch.rand(len(xlen), dtype=torch.float)
        elif q_conditioning:
            decoder_conditioning = q_tactics

        encoder_conditioning = (
            self.disc_scores_for_seq2seq_step(batch) if use_dicriminator else None
        )
        if input_conditioning is not None:
            assert encoder_conditioning is None
            encoder_conditioning = input_conditioning

        # show_batch(
        #     logger, [("source", x), ("target", y)], self.dico, "S2S train",
        # )

        # n_pad_x = (x == self.dico.pad_index).float().sum()
        # n_pad_y = (y == self.dico.pad_index).float().sum()
        # if self.n_total_iter % 100 == 0:
        #     logger.info(
        #         f"Padding x {n_pad_x / x.numel():.2f} - "
        #         f"Padding y {n_pad_y / y.numel():.2f} - "
        #         f"Padding x & y {(n_pad_x + n_pad_y) / (x.numel() + y.numel()):.2f} - "
        #         # f"Time {batch_time:.2f} - N tokens {n_tokens} - N real {len(x) * xlen.max().item() + len(y) * ylen.max().item()} - "
        #         f"bs_x={len(x)} - max_len_x={xlen.max().item()} - "
        #         f"bs_y={len(y)} - max_len_y={ylen.max().item()} -"
        #         f"xlen: {xlen.tolist()} - ylen: {ylen.tolist()}"
        #         f"xlen: {[int(x ** 0.25) for x in xlen.tolist()]} - "
        #         f"ylen: {[int(y ** 0.25) for y in ylen.tolist()]}"
        #     )

        # tokens to predict
        pred_mask, target = get_clm_mask_target(y, ylen)

        # cuda
        x, xlen, y, ylen = to_cuda(x, xlen, y, ylen)
        pred_mask, target, langs2 = to_cuda(pred_mask, target, langs2)
        encoder_conditioning, decoder_conditioning = to_cuda(
            encoder_conditioning, decoder_conditioning
        )

        try:
            # encode x
            encoded = encoder(
                "fwd",
                causal=False,
                tokens=x,
                lengths=xlen,
                discr=encoder_conditioning,
                discr_mode="" if encoder_conditioning is None else "sum",
            )

            multi_gpu = params.slurm_conf.multi_gpu
            unwrapped_encoder = encoder.module if multi_gpu else encoder
            if unwrapped_encoder.is_changing_input_lengths():
                xlen = unwrapped_encoder.new_input_lengths(xlen)
            assert xlen.max().item() <= encoded.size(1), (
                task,
                xlen.max().item(),
                encoded.size(1),
                xlen,
                encoded,
            )

            reg_loss = None
            if (
                "x2y" in task
                and params.cond.n_classes > 0
                and rng.random() < params.cond.prob_cond
            ):
                n_classes = params.cond.n_classes

                # conditioning incompatible with variety
                assert q_conditioning == "" and decoder_conditioning is None
                q_conditioning = params.cond.input_mode

                if params.cond.enc_tgt:
                    assert rng is not None
                    tgt_encoder = self.modules["target_encoder"]
                    tgt_encoder.train()
                    T = self.conditining_temperature
                    hard = T < 1e-3 or rng.rand() < params.cond.proba_hard
                    weights, best_classes, reg_loss = tgt_encoder(
                        "fwd_tgt",
                        tokens=y,
                        lengths=ylen,
                        n_classes=n_classes,
                        hard=hard,
                        T=T,
                    )
                else:
                    class_embs = self.modules["cond_embeddings"]
                    class_embs = class_embs.module if multi_gpu else class_embs
                    decoder.eval()
                    best_classes = decoder(
                        "best_classes_fast_split",
                        tokens=y,
                        lengths=ylen,
                        src_enc=encoded,
                        src_len=xlen,
                        n_classes=n_classes,
                        class_embs=class_embs.weight,
                        max_classes=params.cond.max_classes,
                    )
                    weights = F.one_hot(best_classes, num_classes=n_classes).float()
                    decoder.train()

                assert weights.size() == (bs, n_classes)
                assert best_classes.size() == (bs,)
                with torch.cuda.amp.autocast(enabled=params.model.fp16):
                    decoder_conditioning = self.modules["cond_embeddings"](weights)
                self.cond_stats[f"{task}/cond"] += best_classes.tolist()

            # decode y
            decoded = decoder(
                "fwd",
                causal=True,
                tokens=y,
                lengths=ylen,
                src_enc=encoded,
                src_len=xlen,
                langs=langs2,
                discr=decoder_conditioning,
                discr_mode=q_conditioning,
            )

            # compute loss / optimize
            scores, loss = decoder(
                "compute_loss",
                tensor=decoded,
                pred_mask=pred_mask,
                target=target,
                epsilon=params.label_smoothing_eps,
            )

            if params.kl.should_use_kl():
                assert self.kl_teacher_model is not None
                loss = self.kl_teacher_model.add_kl_loss_to_loss_and_update_stats(
                    batch,
                    student_scores=scores,
                    task=task,
                    loss_before_kl=loss,
                    stats=self.stats,
                )
            if params.entropy_cost:
                ent_loss = entropy_loss(scores)
                loss += params.entropy_cost * ent_loss

            if reg_loss is not None:
                loss += params.cond.reg * reg_loss

            if (loss != loss).any():
                logger.warning("NaN detected ---- printing batch")
                logger.info(batch)

            self.optimize(loss)

        except RuntimeError as err:
            if "out of memory" in str(err):
                logger.exception(
                    f"OOM detected in seq2seq, "
                    f"batch_size: {len(xlen)}, max_len: {xlen.max().item()}"
                )
            # not doing the retry here
            raise err

        # update stats
        n_x = xlen.sum().item()
        n_y = (ylen - 1).sum().item()
        self.n_sentences += params.batch.size
        self.stats[task].append(loss.item())
        self.int_stats["sent/s"] += ylen.size(0)
        self.int_stats["word/s"] += n_x + n_y
        self.int_stats["input_word/s"] += n_x
        self.int_stats["output_word/s"] += n_y
        self.stats[f"{task}/time"].append(time.time() - seq2seq_time)

    def distillation_step(self, task: str):
        """
        Train big decoder with seq2seq loss and critic loss if online distillation.
        If offline, big decoder stays unchanged.
        Train decoder with distillation loss on big decoder.
        """
        params = self.params

        encoder = self.modules["encoder"]
        decoder = self.modules["decoder"]
        big_decoder = self.modules["big_decoder"]

        if params.distillation.hidden_states_loss:
            hidden_states_linear = self.modules["hidden_states_linear"]
            hidden_states_linear.train()
        if params.distillation.embedding_loss:
            embedding_linear = self.modules["embedding_linear"]
            embedding_linear.train()

        online = params.distillation.online
        if online:
            encoder.train()
            big_decoder.train()
        else:
            encoder.eval()
            big_decoder.eval()
        decoder.train()

        kl_temp = params.distillation.kl_temperature
        assert 0 < kl_temp

        # generate batch
        batch = self.get_batch(task)
        # if MCTS data
        if "q" in batch:
            q = batch["q"]
        else:
            q = torch.ones(batch["x"].shape[0])
        x = batch["x"]
        y = batch["y"]
        xlen = batch["xlen"]
        ylen = batch["ylen"]
        langs2 = batch.get("langs2", None)

        # tokens to predict / cuda
        pred_mask, target = get_clm_mask_target(y, ylen)
        x, xlen, y, ylen, q = to_cuda(x, xlen, y, ylen, q)
        pred_mask, target, langs2 = to_cuda(pred_mask, target, langs2)

        try:

            with torch.set_grad_enabled(online):
                # encode x
                encoded = encoder("fwd", causal=False, tokens=x, lengths=xlen)
                multi_gpu = params.slurm_conf.multi_gpu
                unwrapped_encoder = encoder.module if multi_gpu else encoder
                if unwrapped_encoder.is_changing_input_lengths():
                    xlen = unwrapped_encoder.new_input_lengths(xlen)
                assert xlen.max().item() <= encoded.size(1)

            # decode y and get scores from decoders: {'score_name' : score}
            big_decoder_scores, big_decoder_losses = get_decoder_scores(
                big_decoder,
                encoded,
                xlen,
                y,
                ylen,
                langs2,
                pred_mask,
                target,
                params.label_smoothing_eps,
                grad_enabled=online,
                get_critic=params.distillation.critic,
                q=q,
                output_hidden_states=params.distillation.hidden_states_loss
                or params.distillation.embedding_loss,
            )
            decoder_scores, decoder_losses = get_decoder_scores(
                decoder,
                encoded.detach(),
                xlen,
                y,
                ylen,
                langs2,
                pred_mask,
                target,
                params.label_smoothing_eps,
                grad_enabled=True,
                get_critic=params.distillation.critic,
                q=q,
                output_hidden_states=params.distillation.hidden_states_loss
                or params.distillation.embedding_loss,
            )

            assert len(decoder_scores) > 0
            assert big_decoder_scores.keys() == decoder_scores.keys()
            #  add seq2seq and critic big decoder losses to losses if we are in online distillation mode
            losses = (
                {k + "_bigdecoder": v for k, v in big_decoder_losses.items()}
                if online
                else {}
            )
            if params.distillation.task_loss:
                losses.update(decoder_losses)
            # compute soft distillation loss with KL div and add distillation losses

            for score_name, score in decoder_scores.items():
                if score_name == "hidden_states":
                    if params.distillation.embedding_loss:
                        losses[
                            f"distillation_embeddings"
                        ] = distillation_hidden_states_loss(
                            [score.embeddings],
                            [big_decoder_scores[score_name].embeddings],
                            linear_mapping=embedding_linear,
                        )
                    if params.distillation.hidden_states_loss:
                        losses[
                            f"distillation_{score_name}"
                        ] = distillation_hidden_states_loss(
                            score.hiddens,
                            big_decoder_scores[score_name].hiddens,
                            linear_mapping=hidden_states_linear,
                        )
                else:
                    if params.distillation.cross_entropy_loss:
                        losses[f"distillation_{score_name}"] = kl_div_loss(
                            score, big_decoder_scores[score_name].detach(), kl_temp
                        )
            loss = sum(losses.values())
            self.optimize(loss)

        except RuntimeError as err:
            if "out of memory" in str(err):
                logger.exception(
                    f"OOM detected in distillation, "
                    f"batch_size: {len(xlen)}, max_len: {xlen.max().item()}"
                )
            # not doing the retry here
            raise err

        # update stats
        self.n_sentences += params.batch.size
        for k, v in losses.items():
            self.stats[f"{task}/{k}"].append(v.item())
        self.int_stats["sent/s"] += ylen.size(0)
        self.int_stats["word/s"] += int(pred_mask.sum().item())

    def rl_step(self, task: str):
        params = self.params
        encoder = self.modules["encoder"]
        decoder = self.modules["decoder"]
        critic = None
        if "critic" in task:
            critic = self.modules["critic"]
            critic.train()
        encoder.train()
        decoder.train()

        # generate batch
        # start = time.time()
        batch = self.get_batch(task)

        x = batch["x"]
        y = batch["y"]
        xlen = batch["xlen"]
        ylen = batch["ylen"]
        langs2 = batch.get("langs2", None)
        returns = batch["return"].unsqueeze(-1)

        # tokens to predict / cuda
        pred_mask, target = get_clm_mask_target(y, ylen)
        x, xlen, y, ylen = to_cuda(x, xlen, y, ylen)
        pred_mask, target, langs2, returns = to_cuda(pred_mask, target, langs2, returns)

        try:
            # encode x
            encoded = encoder("fwd", causal=False, tokens=x, lengths=xlen)
            if critic is not None:
                critic_inputs = encoded[:, 0]

                if self.params.rl_params.detach_critic:
                    critic_inputs = critic_inputs.detach()
                critics = torch.tanh(
                    critic(critic_inputs)
                )  # tanh to control critic magnitude
                self.stats[task + "/returns"].append(returns.float().mean().item())
                self.stats[task + "/estimated_returns"].append(critics.mean().item())
                advantages = returns - critics
                self.stats[task + "/advantages"].append(advantages.mean().item())
            else:
                advantages = returns

            advantages_detached = advantages.detach() * y.new_ones(size=y.size())

            multi_gpu = params.slurm_conf.multi_gpu
            unwrapped_encoder = encoder.module if multi_gpu else encoder
            if unwrapped_encoder.is_changing_input_lengths():
                xlen = unwrapped_encoder.new_input_lengths(xlen)
            assert xlen.max().item() <= encoded.size(1)

            # decode y
            decoded = decoder(
                "fwd",
                causal=True,
                tokens=y,
                lengths=ylen,
                src_enc=encoded,
                src_len=xlen,
                langs=langs2,
            )

            # compute loss / optimize
            scores, loss = decoder(
                "compute_loss",
                tensor=decoded,
                pred_mask=pred_mask,
                target=target,
                epsilon=params.label_smoothing_eps,
            )
            if critic is not None:
                self.stats[task + "/s2s_no_adv"].append(loss.mean().item())

            advantages_detached = advantages_detached[pred_mask]
            assert advantages_detached.size() == loss.size()
            loss = (loss * advantages_detached).mean()
            if critic is not None:
                self.stats[task + "/s2s"].append(loss.item())

            if critic is not None:
                critics_loss = torch.mean(advantages ** 2)
                self.stats[task + "/critic_loss"].append(critics_loss.item())
                loss += self.params.rl_params.critic_weight * critics_loss
            if params.entropy_cost:
                ent_loss = entropy_loss(scores)
                loss += params.entropy_cost * ent_loss
                self.stats[task + "/ent"].append(loss.item())
            self.stats[task].append(loss.item())
            self.optimize(loss)
        except RuntimeError as err:
            if "out of memory" in str(err):
                logger.exception(
                    f"OOM detected in rl step, "
                    f"batch_size: {len(xlen)}, max_len: {xlen.max().item()}"
                )
            # not doing the retry here
            raise err

        # update stats
        self.n_sentences += params.batch.size
        self.int_stats["sent/s"] += ylen.size(0)
        self.int_stats["word/s"] += int(pred_mask.sum().item())

    def bt_step(self, task: str):
        """
        Sequence-to-sequence training objective. Can be used for:
            - Machine Translation (MT)
            - Denoising Auto-Encoding (DAE)
        """
        params = self.params
        encoder = self.modules["encoder"]
        decoder = self.modules["decoder"]

        # retrieve languages
        assert task.startswith("multi_") and task.endswith("_bt")
        lang1, lang2, lang3 = task[6:-3].split("-")
        assert lang1 == lang3 and lang1 != lang2

        # generate batch
        batch = self.get_batch(task)
        x1 = batch["x"]
        xlen1 = batch["xlen"]
        langs1 = batch["langs1"]
        langs2 = batch["langs2"]

        # tokens to predict / cuda
        pred_mask, target = get_clm_mask_target(x1, xlen1)
        x1, xlen1, pred_mask, target = to_cuda(x1, xlen1, pred_mask, target)
        langs1, langs2 = to_cuda(langs1, langs2)

        try:
            # eval mode
            encoder.eval()
            decoder.eval()

            with torch.no_grad():

                multi_gpu = params.slurm_conf.multi_gpu
                _encoder = encoder.module if multi_gpu else encoder
                _decoder = decoder.module if multi_gpu else decoder

                # encode
                encoded1 = _encoder("fwd", causal=False, tokens=x1, lengths=xlen1)

                # decoding parameters
                max_len = int(3 * xlen1.max().item() + 10)
                decoding_params = DecodingParams(
                    max_gen_len=min(params.batch.max_len, max_len),
                    n_samples=1,
                    use_beam=False,
                )

                # translate
                x2, xlen2, _ = _decoder.generate(
                    src_enc=encoded1,
                    src_len=xlen1,
                    decoding_params=decoding_params,
                    langs=langs2,
                )

                # free CUDA memory
                del encoded1

            # train mode
            encoder.train()
            decoder.train()

            # encode x
            encoded2 = encoder("fwd", causal=False, tokens=x2, lengths=xlen2)

            # TODO: handle this
            # unwrapped_encoder = (
            #     encoder.module if params.slurm_conf.multi_gpu else encoder
            # )
            # if unwrapped_encoder.is_changing_input_lengths():
            #     xlen = unwrapped_encoder.new_input_lengths(xlen)
            # assert xlen.max().item() <= encoded.size(1)

            # decode y
            decoded = decoder(
                "fwd",
                causal=True,
                tokens=x1,
                lengths=xlen1,
                src_enc=encoded2,
                src_len=xlen2,
                langs=langs1,
            )

            # compute loss / optimize
            _, loss = decoder(
                "compute_loss",
                tensor=decoded,
                pred_mask=pred_mask,
                target=target,
                epsilon=params.label_smoothing_eps,
            )
            self.optimize(loss)
        except RuntimeError as err:
            if "out of memory" in str(err):
                logger.exception(
                    f"OOM detected in bt, "
                    f"x1: {len(xlen1)}, max_len: {xlen1.max().item()} -- "
                    f"x2: {len(xlen2)}, max_len: {xlen2.max().item()}"
                )
            # not doing the retry here
            raise err

        # update stats
        n_x = int(xlen1.sum().item())
        n_y = int(pred_mask.sum().item())
        self.n_sentences += params.batch.size
        self.stats[task].append(loss.item())
        self.int_stats["sent/s"] += xlen1.size(0)
        self.int_stats["word/s"] += n_x + n_y
        self.int_stats["input_word/s"] += n_x
        self.int_stats["output_word/s"] += n_y

    def seq2tok_seq2seqtok_step(self, task: str, with_decoder: bool):
        """
        Sequence-to-token binary classification task.
        """
        params = self.params
        encoder = self.modules["encoder"]
        encoder.train()
        if with_decoder:
            decoder = self.modules["decoder"]
            decoder.train()
        else:
            classifier = self.modules["classifier"]
            classifier.train()

        # generate batch
        batch = self.get_batch(task)
        x = batch["x"]
        y = batch["y"]
        xlen = batch["xlen"]
        bs, slen = x.size()
        assert y.size() == (bs, 1)

        # cuda
        x, xlen, y = to_cuda(x, xlen, y)

        # encode x
        encoded = encoder("fwd", causal=False, tokens=x, lengths=xlen)
        assert encoded.size() == (bs, slen, params.model.enc_emb_dim)

        # classify
        if with_decoder:
            predicted, _ = decoder(
                "compute_critic", src_enc=encoded, src_len=xlen, q=None,
            )
        else:
            predicted = classifier(encoded[:, 0])
            predicted = predicted.log_softmax(-1)
            # sanity checks
        assert predicted.size() == (bs, 2)
        assert predicted.exp().sum(1).sub(1).abs().lt(1e-5).all()
        # compute loss: y.log x + (1-y) * log (1-x)
        target = torch.cat([y, 1 - y], 1).float()
        loss = -torch.sum(target * predicted, 1).sum()

        # optimize
        self.optimize(loss)

        # update stats
        self.n_sentences += params.batch.size
        self.stats[task].append(loss.item())
        self.int_stats["sent/s"] += bs
        self.int_stats["word/s"] += xlen.sum().item()

    def seq2emb_step(self, task: str):
        """
        Sequence-to-embedding classification task.
        """
        params = self.params
        embedder = self.modules["embedder"]
        encoder = self.modules["encoder"]
        embedder.train()
        encoder.train()

        # generate batch
        batch = self.get_batch(task)
        x = batch["x"]
        y = batch["y"]
        xlen = batch["xlen"]
        ylen = batch["ylen"]

        # update statement embeddings
        neg_sampler = self.negative_samplers["mm"]
        freq = params.mm.neg.update_frequency
        if freq > 0 and self.n_iter > 0 and self.n_iter % freq == 0:
            if params.slurm_conf.multi_gpu:
                neg_sampler.update_all_embeddings_distributed(embedder.module)
            else:
                neg_sampler.update_all_embeddings(embedder)

        # retrieve target sequences / negative sampling on target theorems
        tgt_seqs = [
            [self.dico[wid] for wid in y[i, : ylen[i]].tolist()]
            for i in range(len(ylen))
        ]
        _batch = neg_sampler.get_negative_batch(
            sequences=tgt_seqs, sample_size=params.mm.neg.sample_size
        )
        local_tgt_ids, global_tgt_ids, seq_ids, neg_batch, neg_len = _batch

        # sanity checks
        bs, slen = x.size()
        n_trm, nlen = neg_batch.size()
        assert len(y) == len(ylen) == bs
        assert len(local_tgt_ids) == bs
        assert len(global_tgt_ids) == bs
        assert len(neg_len) == n_trm
        for i, j in enumerate(local_tgt_ids.tolist()):
            assert ylen[i].item() == neg_len[j].item()
            assert (y[i, : ylen[i]] == neg_batch[j, : neg_len[j]]).all()
        for i, j in enumerate(global_tgt_ids.tolist()):
            assert ylen[i].item() == len(neg_sampler.id2tensor[j])
            assert (y[i, : ylen[i]] == neg_sampler.id2tensor[j]).all()

        # cuda
        x, xlen, y, seq_ids = to_cuda(x, xlen, y, seq_ids)
        local_tgt_ids, global_tgt_ids = to_cuda(local_tgt_ids, global_tgt_ids)
        neg_batch, neg_len = to_cuda(neg_batch, neg_len)

        # embed theorems
        embedded = embedder("fwd", causal=False, tokens=neg_batch, lengths=neg_len)
        assert embedded.size() == (n_trm, nlen, params.model.enc_emb_dim)
        embedded = get_embs(embedded, neg_len, params.model.tf_build_emb)

        # update embeddings
        neg_sampler.update_embeddings(seq_ids, embedded)

        # include all embeddings
        if params.mm.neg.softmax_all:
            temp = neg_sampler.embeddings.float()
            temp[seq_ids] = embedded
            embedded = temp
            n_trm = neg_sampler.size
            tgt_ids = global_tgt_ids
        else:
            tgt_ids = local_tgt_ids

        # encode x
        encoded = encoder("fwd", causal=False, tokens=x, lengths=xlen)
        assert encoded.size() == (bs, slen, params.model.enc_emb_dim)
        encoded = encoded[:, 0]

        # compute theorem scores
        trm_scores = embedded.mm(encoded.transpose(0, 1)).transpose(0, 1)
        assert trm_scores.shape == (bs, n_trm)

        # compute loss / optimize
        loss = F.cross_entropy(trm_scores, tgt_ids)
        self.optimize(loss)

        # update stats
        self.n_sentences += params.batch.size
        self.stats[task].append(loss.item())
        self.int_stats["sent/s"] += ylen.size(0)
        self.int_stats["word/s"] += ylen.sum().item()

    def mcts_critic_step(self, task: str, seq2seq_batch=None):
        """
        MCTS Objective : given a list of goals, predict tactics and steps estimates.
        """
        total = time.time()
        params = self.params
        encoder = self.modules["encoder"]
        decoder = self.modules["decoder"]
        encoder.train()
        decoder.train()
        before_batch = time.time()

        if seq2seq_batch is None:
            batch = self.get_batch(task)
        waiting = time.time() - before_batch

        before_process = time.time()
        x = batch["x"]
        xlen = batch["xlen"]
        q = batch["q"]

        # cuda
        x, xlen, q, = to_cuda(x, xlen, q)

        encoded = encoder("fwd", causal=False, tokens=x, lengths=xlen)

        if params.mcts_train.hard_critic_estimates:  # 0 or 1 for the critic
            q[q < 1] = 0
        _, loss = decoder(
            "compute_critic",
            src_enc=encoded,
            src_len=xlen,
            freeze_params=params.critic_freeze,
            q=q,
        )

        # optimize
        self.optimize(loss)

        # update stats
        processing = time.time() - before_process
        self.n_sentences += params.batch.size
        self.stats[task].append(loss.item())
        self.stats[f"{task}"].append(loss.item())
        self.stats[f"{task}/time"].append(time.time() - total)
        self.int_stats[f"{task}/s"] += x.size(0)
        self.int_stats["word/s"] += xlen.sum().item()
        self.stats["ratio_waiting"].append(waiting / processing)

    def disc_scores_for_seq2seq_step(self, batch):
        """
        Computing discriminator scores for a batch

        we set the discriminator score to one where the target is 1 (real data)
        """

        disc_inp, disc_inp_len, disc_tgt = (
            batch["disc_inp"],
            batch["disc_inp_len"],
            batch["disc_tgt"],
        )
        disc_scores = self.get_disc_scores(disc_inp, disc_inp_len)
        # we set the discriminator score to one where the target is 1 (real data)
        disc_scores[disc_tgt == 1] = 1
        return disc_scores

    def update_stats(self, other_stats: Dict[str, float]):
        for key, value in other_stats.items():
            self.stats[key].append(value)

    def close(self):
        logger.info("Closing Trainer ...")
        self.metrics_logger.close()
        if self.params.log_network_stats_freq > 0:
            self.network_metrics_logger.close()
        if self.context:
            self.context.close()
        self.f_task.close()
        logger.info("Closed Trainer")


def _mem() -> str:
    return f"mem_usage: {psutil.virtual_memory().used / 1024 ** 3:.01f}GB"
