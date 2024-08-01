# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, List, Dict, cast
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import closing
from datetime import datetime
from logging import getLogger
from pathlib import Path
from queue import Empty
import os
import sys
import time
import cProfile
import traceback
import subprocess
import numpy as np
import torch
from torch import multiprocessing as mp

from evariste import json as json
from evariste.backward.prover.args import ExpanderParams
from evariste.datasets import DatasetConf
from evariste.datasets.lean import LeanDatasetConf
from evariste.metrics import Logger, Timer, StatsCollection, SimpleStatValue
from evariste.utils import PicklingQueue, get_mean, find_descendents, this_job_id
from evariste.backward.dag_factory import get_dag
from evariste.backward.env.core import ModelExpansion
from evariste.backward.graph import Theorem, ProofId, GoalParams, NodeInfo
from evariste.backward.model.beam_search import (
    BeamSearchModel,
    FAILED_GPU_OOM,
    FAILED_UNK,
    FAILED_BIG,
)
from evariste.backward.prover.utils import GPUMonitor
from evariste.model.data.dictionary import Dictionary


@dataclass
class RequestId:
    proof_id: int
    id_in_request: int
    timestamp: float


@dataclass
class ExpanderInput:
    req_id: RequestId
    label: Optional[str]
    tokens: List[str]
    info: NodeInfo
    params: Optional[GoalParams]
    chunk_id: Optional[int] = None


@dataclass
class GPUInput:
    req_ids: List[RequestId]
    tokens: torch.Tensor
    lengths: torch.Tensor
    forbiddens: Optional[List[torch.Tensor]]
    infos: List[NodeInfo]
    params: Optional[List[GoalParams]]

    def __post_init__(self):
        bs = len(self.req_ids)
        assert len(self.tokens) == len(self.lengths) == bs
        assert self.forbiddens is None or len(self.forbiddens) == bs
        assert len(self.infos) == bs
        assert self.params is None or len(self.params) == bs


@dataclass
class ExpanderOutput:
    req_id: RequestId
    model_expansion: ModelExpansion


@dataclass
class DoneStatus:
    n_seqs: int
    oom: bool


class ExpanderDied(Exception):
    pass


class ExpanderStopped(Exception):
    pass


def tensorize_batch(
    th_token_ids: List[List[int]],
    pad_index: int,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    bs = len(th_token_ids)
    lengths = [len(toks) for toks in th_token_ids]
    src_len = torch.LongTensor(lengths, device=device)
    token_ids = src_len.new_full(size=(bs, max(lengths)), fill_value=pad_index)
    for i, toks in enumerate(th_token_ids):
        token_ids[i, : len(toks)] = torch.tensor(toks, dtype=torch.int64)
    return src_len, token_ids


def sort_sequences(
    inputs: List[ExpanderInput], params: ExpanderParams
) -> List[ExpanderInput]:

    if params.sorting_strategy == "none":
        return inputs

    # set received IDs
    received_id: Dict[int, int] = {}
    for x in inputs:
        proof_id = x.req_id.proof_id
        if proof_id not in received_id:
            received_id[proof_id] = len(received_id)

    def sort_fn(x: ExpanderInput) -> Tuple[int, int]:
        if params.sorting_strategy == "pid_slen":
            return received_id[x.req_id.proof_id], len(x.tokens)
        else:
            assert params.sorting_strategy == "chunk_slen"
            assert x.chunk_id is not None
            return x.chunk_id, len(x.tokens)

    return sorted(inputs, key=sort_fn)


def filter_and_create_batch(
    inputs: List[ExpanderInput], expander_params: ExpanderParams, dico: Dictionary
) -> Tuple[
    Optional[Tuple[List[int], torch.Tensor, torch.Tensor, Optional[List[GoalParams]]]],
    List[int],
    List[int],
    List[Tuple[int, List[str]]],
]:
    """
    Returns pre-filled results and an iterator over batches
        - Optional batch: (ID of theorems, lengths, tokens)
        - remaining: List[int] = ID of theorems we still need to process
        - too_long: List[int] = ID of theorems with too long sequences
    """
    tokens_per_batch = expander_params.tokens_per_batch
    allow_mix = expander_params.allow_mix
    assert tokens_per_batch > 0

    if len(inputs) == 0:
        return None, [], [], []

    # IDs and lengths of sequences to process / IDs of too long sequences
    ids: List[int] = []
    lengths: List[int] = []
    too_long: List[int] = []
    unk: List[Tuple[int, List[str]]] = []
    tokens: List[List[int]] = []

    # filter out theorems with too long sequences
    for i, x in enumerate(inputs):
        toks = [dico.eos_word, *x.tokens, dico.eos_word]
        slen = len(toks)
        if slen <= expander_params.max_input_len:
            try:
                tokens.append([dico.index(tok) for tok in toks])
            except KeyError:
                unk.append((i, [tok for tok in toks if tok not in dico]))
            else:
                ids.append(i)
                lengths.append(slen)
        else:
            too_long.append(i)

    # nothing to process
    if len(ids) == 0:
        return None, [], too_long, unk

    def sid(x: ExpanderInput) -> int:
        if expander_params.sorting_strategy == "pid_slen":
            return x.req_id.proof_id
        else:
            assert expander_params.sorting_strategy == "chunk_slen"
            assert x.chunk_id is not None
            return x.chunk_id

    # cut the batch into smaller sub-batch with at most self.tokens_per_batch tokens (including padding)
    assert len(ids) == len(lengths)
    bs, max_len = 0, 0
    while bs < len(ids) and max_len * bs < tokens_per_batch:
        # do not mix sequences with different proof ID or chunk ID
        if bs > 0 and not allow_mix and sid(inputs[bs - 1]) != sid(inputs[bs]):
            break
        max_len = max(max_len, lengths[bs])
        bs += 1
    if max_len * bs > tokens_per_batch:
        bs -= 1
    assert bs > 0 and max(lengths[:bs]) * bs <= tokens_per_batch, (bs, lengths)

    src_len, src_tokens = tensorize_batch(tokens[:bs], dico.pad_index)

    # sanity check
    assert len(src_tokens) == len(src_len) == bs, (src_tokens.shape, src_len.shape, bs)
    assert 0 < src_len.max().item() <= expander_params.max_input_len, src_len

    batch_params = cast(
        Optional[List[GoalParams]],
        None if inputs[0].params is None else [inputs[i].params for i in ids[:bs]],
    )
    assert batch_params is None or all(b is not None for b in batch_params)

    return (
        (ids[:bs], src_tokens, src_len, batch_params),
        ids[bs:],
        too_long,
        unk,
    )


@dataclass
class GPUTimers(StatsCollection):
    receiving: Timer = field(
        default_factory=lambda: Timer(cum_ratio=True, overall_cum_time=True)
    )
    sending: Timer = field(default_factory=lambda: Timer())
    in_model: Timer = field(
        default_factory=lambda: Timer(cum_ratio=True, overall_cum_time=True)
    )
    in_reloading: Timer = field(default_factory=lambda: Timer())


class MPExpander:
    """Spawn two processes:
    
    * 1 batcher to receive tokenized (in the main process) theorems, pool them and sending GPU batches after sorting by length
    
    * 1 GPU process which runs beam_search.do_batch() on received batches.

    Batch size is automatically adapted depending on CUDA OOM errors. Increased if none, decreased when some.

    .. warning::
        Since CUDA is spawned in another process, memory management is more difficult.
        Make sure that no other process (even the main one) is using the same cuda device, otherwise you **will** run into cuda OOM.

    .. note::
        Debugging tips: If one of the two process hangs for some reason (badly handled exception shouldn't happen but we never know...). Then
        it will stop outputting logs. This allows to locate the issue.

    :param beam_search: the :class:`BeamSearchModel` its load method will be run when the first batch is processed, to avoid spawning a cuda context in the main process.
    :type beam_search: :class:`BeamSearchModel`
    :param profile: whether to run everything in cProfile and dump results in profiles/..., defaults = True.
    :type profile: bool 
    :param n_gpus: use more `n_gpus` GPUs. never useful so far, default = 1
    :type n_gpus: int
    :param quiet: passed to all Logger along with only_jsonl to either log nowhere, to tensorboard+jsonl or only in jsonl files.
    :type quiet: bool
    :param only_jsonl:
    :type only_jsonl: bool
    """

    def __init__(
        self,
        beam_search: BeamSearchModel,
        expander_params: ExpanderParams,
        dataset: DatasetConf,
        prover_dump_path: Path,
        profile: bool = False,
        n_gpus: int = 1,
        quiet: bool = False,
        only_jsonl: bool = False,
    ):
        self.beam_search: Optional[BeamSearchModel] = None  # will be set in processes
        if "cuda" in beam_search.device:
            assert n_gpus <= torch.cuda.device_count()
        else:
            assert n_gpus == 1, n_gpus
        self.device = beam_search.device
        self.dico = beam_search.dico
        self.decoding_params = beam_search.decoding_params
        assert self.dico is not None, "beam search should have a dictionary!"
        self.profile = profile
        self.dag = None
        self.forbidden: Dict[str, List[str]] = {}
        self.dataset = dataset
        self.quiet = quiet
        self.only_jsonl = only_jsonl

        self.step = 0
        self.last_tpb_update = 0
        self.skipped_too_long = 0
        self.skipped_unk = 0
        self.unk_words: Dict[str, int] = defaultdict(int)

        self.expander_params = expander_params

        # for tensorboard log
        self.prover_dump_path = prover_dump_path
        self.metrics: Optional[Logger] = None

        ctx = mp.get_context("spawn")

        # queue of tokenized theorems for which we want tactics
        self.input: PicklingQueue[ExpanderInput] = PicklingQueue(ctx.Queue())
        # queue of results (ModelExpansion)
        self.results: PicklingQueue[ExpanderOutput] = PicklingQueue(ctx.Queue())
        # holds input tensors for the beam search (Optional because None means we should stop)
        self._gpu_queue: PicklingQueue[Optional[GPUInput]] = PicklingQueue(ctx.Queue())
        # signals the batcher that a batch has been processed (and whether we had an OOM)
        self._batch_done_queue: PicklingQueue[DoneStatus] = PicklingQueue(ctx.Queue())

        # set if an exception is raised somewhere
        self.died = ctx.Event()
        # set if we want to exit cleanly
        self.stop_signal = ctx.Event()
        # set when we want to reload model weights
        self.reload_weights = [ctx.Event() for _ in range(n_gpus)]
        # set by GPU process once CUDA gpu_queue is empty
        self._stop_batcher = ctx.Event()

        self.name = "batcher"
        try:
            batcher = ctx.Process(
                name="batcher", target=self._batcher_loop, args=(beam_search.dico,),
            )
            batcher.start()
            _gpus = []
            for i in range(n_gpus):
                self.name = f"gpu_{i}"
                if "cuda" in beam_search.device and n_gpus > 1:
                    self.device = f"cuda:{i}"
                    beam_search.device = f"cuda:{i}"
                else:
                    self.device = beam_search.device
                getLogger().warning(f"Backward prover {i} using device {self.device}")
                _gpus.append(
                    ctx.Process(
                        name=f"gpu_{i}", target=self._gpu_loop, args=(beam_search, i)
                    )
                )
                _gpus[-1].start()
        except Exception:
            traceback.print_exc()
            raise

        self.name = "main"
        self.last_alive_check = time.time()
        self._batcher = batcher
        self._gpus = _gpus
        self.cur_client = 0
        self.closed = False

    def log_metrics(self, data: Dict[str, SimpleStatValue]):
        if self.metrics is None:
            return
        self.metrics.log_metrics(data)

    def alive(self) -> bool:
        alive = not (self.stop_signal.is_set() or self.died.is_set())
        if self.name == "main" and time.time() - self.last_alive_check > 10:
            alive &= self._batcher.is_alive() and all(
                [p.is_alive() for p in self._gpus]
            )
            self.last_alive_check = time.time()
        return alive

    def process_async(
        self,
        proof_id: ProofId,
        theorems: List[Theorem],
        params: Optional[GoalParams] = None,
    ):
        for i, th in enumerate(theorems):
            request_id = RequestId(
                proof_id=proof_id, id_in_request=i, timestamp=time.time()
            )
            self.input.put(
                ExpanderInput(
                    req_id=request_id,
                    label=th.train_label,
                    tokens=th.tokenize(),
                    info=th.info,
                    params=params,
                )
            )

    def ready_model_expansions(self) -> List[Tuple[int, int, ModelExpansion]]:
        ready = []
        try:
            while True:
                assert self.alive()
                res = self.results.get_nowait()
                ready.append(
                    (
                        res.req_id.proof_id,
                        res.req_id.id_in_request,
                        res.model_expansion,
                    )
                )
        except Empty:
            pass
        return ready

    def reload_model_weights(self) -> None:
        for signal in self.reload_weights:
            signal.set()

    def get_forbidden(self, label: Optional[str]) -> List[str]:
        if isinstance(self.dataset, LeanDatasetConf):
            to_filter: List[str] = []
            if self.dataset.filter_tactics.no_try:
                to_filter.append("▁try")
            if self.dataset.filter_tactics.no_clear:
                to_filter.append("▁clear")
            if self.dataset.filter_tactics.no_repeat:
                to_filter.append("▁repeat")
            return to_filter
        if label is None:
            return []
        if label not in self.forbidden:
            self.forbidden[label] = find_descendents(self.dag, label)
        return self.forbidden[label]

    def send_one_batch(
        self, to_send: List[ExpanderInput], dico: Dictionary
    ) -> Tuple[int, List[ExpanderInput]]:
        """
        Send a list of theorems to process.
        Create a batch, filter too long theorems.
        Return a tuple with:
        * the number of theorems sent for processing
        * the number of invalid theorems (i.e. too long sequences)
        * the updated list of theorems yet to send
        """

        # create a batch / filter long sequences
        batch, remaining, too_long, unks = filter_and_create_batch(
            to_send, self.expander_params, dico=dico
        )
        batch_size = 0 if batch is None else len(batch[0])
        assert batch_size + len(remaining) + len(too_long) + len(unks) == len(to_send)

        # update results for theorems with too many tokens
        self.skipped_too_long += len(too_long)
        for tid in too_long:
            expansion = ModelExpansion(
                exp_duration=time.time() - to_send[tid].req_id.timestamp,
                gpu_duration=0.0,
                error=FAILED_BIG,
            )
            self.results.put(
                ExpanderOutput(req_id=to_send[tid].req_id, model_expansion=expansion)
            )

        # update results for theorems with unknown words
        self.skipped_unk += len(unks)
        for tid, unk_words in unks:
            assert len(unk_words) > 0
            for w in unk_words:
                self.unk_words[w] += 1
            expansion = ModelExpansion(
                exp_duration=time.time() - to_send[tid].req_id.timestamp,
                gpu_duration=0.0,
                error=FAILED_UNK,
            )
            start_time_in_put = 0.0
            self.results.put(
                ExpanderOutput(req_id=to_send[tid].req_id, model_expansion=expansion)
            )

        if batch is None:
            return 0, []

        # compute forbiddens
        tids, tokens, lengths, batch_params = batch
        forbiddens_list = [
            [
                dico.word2id[x]
                for x in self.get_forbidden(to_send[tid].label)
                if x in dico.word2id  # this happens in inequality2
            ]
            for tid in tids
        ]
        forbiddens = (
            None
            if all(len(f) == 0 for f in forbiddens_list)
            else [torch.tensor(f, dtype=torch.int64) for f in forbiddens_list]
        )
        # Send the batch for processing
        self._gpu_queue.put(
            GPUInput(
                req_ids=[to_send[tid].req_id for tid in tids],
                tokens=tokens,
                lengths=lengths,
                forbiddens=forbiddens,
                infos=[to_send[tid].info for tid in tids],
                params=batch_params,
            )
        )

        # return the number of theorems sent to process, the number of too long,
        # and the list of remaining theorems to process
        return len(tids), [to_send[tid] for tid in remaining]

    def update_batch_size(self, oom: bool):
        """
        Reduce tokens_per_batch if we had an OOM. Increase it if no OOM for a while.
        Allows provers to run efficiently on both 16GB and 32GB GPUs.
        """
        exp_params = self.expander_params
        if not exp_params.resize_with_oom:
            return
        logger = getLogger()
        STEPS_TO_INCR = 100
        DELTA = 1000
        MIN_DELAY_UPDATE = 10  # prevent the model from reducing too much too quickly
        init_tpb = exp_params.tokens_per_batch
        self.step += 1
        if oom:
            if self.step - self.last_tpb_update >= MIN_DELAY_UPDATE:
                self.last_tpb_update = self.step
                exp_params.tokens_per_batch = max(exp_params.min_tpb, init_tpb - DELTA)
                logger.warning(
                    f"GPU OOM (step {self.step}) -- Reducing tokens per batch "
                    f"from {init_tpb} to {exp_params.tokens_per_batch}"
                )
        else:
            if self.step - self.last_tpb_update >= STEPS_TO_INCR:
                self.last_tpb_update = self.step
                exp_params.tokens_per_batch = min(exp_params.max_tpb, init_tpb + DELTA)
                logger.warning(
                    f"NO GPU OOM for {STEPS_TO_INCR} steps (step {self.step}) -- "
                    f"Increasing tokens per batch from {init_tpb} "
                    f"to {exp_params.tokens_per_batch}"
                )

    def __batcher_loop(self, dico: Dictionary):

        logger = getLogger()

        max_batches_in_queue = self.expander_params.max_batches_in_queue
        chunk_size = self.expander_params.chunk_size

        self.dag = get_dag(self.dataset)
        self.forbidden = {}

        sequences: List[ExpanderInput] = []
        batch_in_queue = 0
        seqs_in_queue = 0

        total_sent_batches = 0
        total_input_seqs = 0
        last_recv_seqs = 0
        last_sent_seqs = 0
        last_recv_slen: List[int] = []
        last_log = 0.0

        try:
            while (not self.died.is_set()) and (not self.stop_signal.is_set()):
                try:
                    recv = self._batch_done_queue.get(block=False)
                    batch_in_queue -= 1
                    seqs_in_queue -= recv.n_seqs
                    assert batch_in_queue >= 0
                    assert seqs_in_queue >= 0
                    self.update_batch_size(recv.oom)  # dynamic batch size
                except Empty:
                    pass

                try:
                    sequences.append(self.input.get(timeout=0.01))
                    last_recv_seqs += 1
                    last_recv_slen.append(len(sequences[-1].tokens))
                    total_input_seqs += 1
                    sequences[-1].chunk_id = total_input_seqs // chunk_size
                except Empty:
                    if batch_in_queue < max_batches_in_queue and len(sequences) > 0:
                        total_sent_batches += 1
                        sequences = sort_sequences(
                            sequences, params=self.expander_params
                        )
                        n_sent, sequences = self.send_one_batch(
                            to_send=sequences, dico=dico
                        )
                        last_sent_seqs += n_sent
                        if n_sent > 0:
                            batch_in_queue += 1
                            seqs_in_queue += n_sent

                # print stats
                diff = time.time() - last_log
                if diff > 60:
                    last_recv_max_slen = max(last_recv_slen, default=0)
                    last_recv_avg_slen = (
                        float(np.mean(last_recv_slen)) if len(last_recv_slen) > 0 else 0
                    )
                    unk_str = ""
                    if self.skipped_unk > 0:
                        unk_str = f" - Found {len(self.unk_words)} different unknown words: {self.unk_words}"
                    logger.info(
                        f"BATCHER -- Received a total of {total_input_seqs} input sequences, and "
                        f"sent a total of {total_sent_batches} batches. Received {last_recv_seqs} "
                        f"and sent {last_sent_seqs} sequences over the last {diff:.2f} seconds "
                        f"(Sent {last_sent_seqs / diff:.2f} seqs/s). "
                        f"Average last received sequence lengths: {last_recv_avg_slen:.2f} - "
                        f"(max: {last_recv_max_slen}) - "
                        f"Batch in queue: {batch_in_queue} - "
                        f"Sequences in queue: {seqs_in_queue} - "
                        f"Awaiting sequences: {len(sequences)} - "
                        f"Sequences skipped because too long: {self.skipped_too_long} - "
                        f"Sequences skipped because unknown words: {self.skipped_unk}{unk_str}"
                    )
                    log_data: Dict[str, SimpleStatValue] = {
                        "last_recv_seqs/s": last_recv_seqs / diff,
                        "last_sent_seqs/s": last_sent_seqs / diff,
                        "last_recv_max_slen": last_recv_max_slen,
                        "last_recv_avg_slen": last_recv_avg_slen,
                        "batch_in_queue": batch_in_queue,
                        "seqs_in_queue": seqs_in_queue,
                        "awaiting_sequences": len(sequences),
                        "skipped_too_long": self.skipped_too_long,
                        "skipped_unk": self.skipped_unk,
                        "total_uniq_unks": len(self.unk_words),
                        "total_unks": sum(self.unk_words.values()),
                        "max_tokens_per_batch": self.expander_params.tokens_per_batch,
                    }
                    logger.info(f"__batcher_log__:{json.dumps(log_data)}")
                    self.log_metrics(log_data)

                    # check that theorems have not been waiting for too long
                    took_too_long = 0
                    for x in sequences:
                        wait_time = time.time() - x.req_id.timestamp
                        if wait_time > 5 * 60:
                            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            logger.warning(
                                f"{now} - SPENDING TOO LONG IN EXPANDER: "
                                f"{wait_time:.2f}s - slen: {len(x.tokens)}"
                            )
                            took_too_long += 1
                        if took_too_long >= 10:
                            break

                    # reset stats
                    last_recv_seqs = 0
                    last_sent_seqs = 0
                    last_recv_slen = []
                    last_log = time.time()

        except:
            traceback.print_exc()
            sys.stdout.flush()
            self.died.set()

        logger.warning("None in GPU_Queue")
        self._gpu_queue.put(None)
        logger.warning("Emptying batch done queue")
        while batch_in_queue > 0:
            try:
                recv = self._batch_done_queue.get_nowait()
            except Empty:
                break
            batch_in_queue -= 1
            seqs_in_queue -= recv.n_seqs
        logger.warning("Emptied batch done queue")

    def _batcher_loop(self, dico: Dictionary):

        # NOTE: metrics might be aggregated across workers
        dirname = (
            f"expander_batcher_{this_job_id()}"
            if self.only_jsonl
            else "expander_batcher"
        )
        metrics_dump_path = self.prover_dump_path / "runner_logs" / dirname
        os.makedirs(metrics_dump_path, exist_ok=True)
        self.metrics = Logger(
            outdir=metrics_dump_path,
            tag="expander_batcher",
            quiet=self.quiet,
            only_jsonl=self.only_jsonl,
        )

        self.input.cancel_join_thread()
        self._batch_done_queue.cancel_join_thread()
        self._gpu_queue.cancel_join_thread()

        pr = None if not self.profile else cProfile.Profile()
        if pr is not None:
            pr.enable()
        try:
            self.__batcher_loop(dico)
        except Exception:
            self.died.set()
            raise
        finally:
            self.metrics.close()

        if pr is not None:
            pr.disable()
            pr.dump_stats("profiles/batcher_loop.profile")
        logger = getLogger()

        logger.warning("waiting for stop batcher to be set")
        while not self._stop_batcher.is_set():
            time.sleep(1)
            logger.warning("STOP BATCHER NOT SET")
        logger.warning("Byeeeeeee batcher")

    def __gpu_loop(self, gpu_id: int):
        """
        Main GPU loop. Receive batches and process then.
        Split batches if they raise an OOM.
        """
        assert isinstance(self.beam_search, BeamSearchModel)
        logger = getLogger()
        logger.info(f"GPU {gpu_id} IS {os.getpid()}, device: {self.device}")

        begin_time = time.time()
        total_recv_batches = 0
        total_recv_seqs = 0
        total_proc_batches = 0
        total_proc_seqs = 0
        total_proc_toks = 0
        total_pad_toks = 0
        total_toks = 0
        total_oom_count = 0
        total_oom_failures = 0  # OOM with batch size 1
        last_proc_batches = 0
        last_proc_seqs = 0
        last_proc_toks = 0
        last_pad_toks = 0
        last_tot_toks = 0
        last_proc_slen: List[int] = []
        last_proc_time: List[float] = []
        last_dec_slen: List[int] = []
        last_log = 0.0

        gpu_mon = GPUMonitor(delay=1.0)

        timers = GPUTimers()

        with closing(gpu_mon):
            try:
                while (not self.died.is_set()) and (not self.stop_signal.is_set()):
                    # do this here otherwise it's not printed if the queue is empty.
                    diff = time.time() - last_log
                    if diff > 60:
                        if "cuda" in self.device:
                            device_id = int(self.device.split(":")[-1])
                            gpu_util = gpu_mon.stats[device_id].stats.get("gpu", -1)
                            gpu_mem = gpu_mon.stats[device_id].stats.get("mem", -1)
                        else:
                            gpu_util = 0
                            gpu_mem = 0
                        overall_time = time.time() - begin_time
                        logger.info(
                            f"GPU -- Model index: {self.beam_search.model_id} -- "
                            f"Received {total_recv_batches} batches ({total_recv_seqs} sequences) and "
                            f"processed {total_proc_batches} batches ({total_proc_seqs} sequences) in "
                            f"{overall_time:.2f}s ({total_proc_seqs / overall_time:.2f} seqs/s). "
                            f"Caught {total_oom_count} OOM errors ({total_oom_failures} with bs=1). "
                            f"Padding ratio: {total_pad_toks / max(total_toks, 1):.2f}. "
                            f"Over the last {diff:.2f}s, processed {last_proc_toks} tokens "
                            f"({last_proc_toks / diff:.2f} toks/s) and {last_proc_seqs} sequences "
                            f"({last_proc_seqs / diff:.2f} seqs/s), with a padding ratio of "
                            f"{last_pad_toks / max(last_tot_toks, 1):.2f}. "
                            f"Decoded {len(last_dec_slen)} sequences ({len(last_dec_slen) / diff:.2f} seqs/s) "
                            f"with {sum(last_dec_slen)} tokens ({sum(last_dec_slen) / diff:.2f} toks/s), "
                            f"and a longest generation of {max(last_dec_slen, default=-1)} tokens. "
                            f"Avg batch processing time: {get_mean(last_proc_time):.3f}s. "
                            f"Avg GPU util/mem: {gpu_util:.2f}%/{gpu_mem:.2f}"
                        )
                        log_data: Dict[str, SimpleStatValue] = {
                            "model_id": self.beam_search.model_id,
                            "gpu_util": gpu_util,
                            "total_proc_batches": total_proc_batches,
                            "total_proc_toks": total_proc_toks,
                            "total_proc_seqs": total_proc_seqs,
                            "total_pad_ratio": total_pad_toks / max(total_toks, 1),
                            "total_oom_count": total_oom_count,
                            "total_oom_failures": total_oom_failures,
                            "last_pad_ratio": last_pad_toks / max(last_tot_toks, 1),
                            "last_proc_toks/s": last_proc_toks / diff,
                            "last_proc_seqs/s": last_proc_seqs / diff,
                            "last_proc_slen": get_mean(last_proc_slen),
                            "last_proc_time": get_mean(last_proc_time),
                            "last_dec_seqs": len(last_dec_slen),
                            "last_dec_toks": sum(last_dec_slen),
                            "last_dec_slen_max": max(last_dec_slen, default=-1),
                            "last_proc_batches": last_proc_batches,
                            "last_avg_batch_size": last_proc_seqs
                            / max(1, last_proc_batches),
                        }

                        log_data.update(timers.rate_and_reset())

                        logger.info(f"__gpu_log__:{json.dumps(log_data)}")
                        self.log_metrics(log_data)
                        gpu_mon.stats[torch.cuda.current_device()].reset()
                        last_pad_toks = 0
                        last_tot_toks = 0
                        last_proc_batches = 0
                        last_proc_toks = 0
                        last_proc_seqs = 0
                        last_proc_slen = []
                        last_proc_time = []
                        last_dec_slen = []
                        last_log = time.time()

                    timers.receiving.start()
                    try:
                        to_process = self._gpu_queue.get(timeout=0.1)
                    except Empty:
                        continue
                    finally:
                        timers.receiving.stop()

                    if to_process is None:
                        break

                    if total_recv_batches == 0:
                        logger.info("First batch received!")

                    total_recv_batches += 1

                    if self.reload_weights[gpu_id].is_set():
                        timers.in_reloading.start()
                        self.reload_weights[gpu_id].clear()
                        self.beam_search.maybe_load()
                        timers.in_reloading.stop()

                    # input data
                    req_ids = to_process.req_ids
                    tokens = to_process.tokens
                    lengths = to_process.lengths
                    forbiddens = to_process.forbiddens
                    infos = to_process.infos
                    params = to_process.params

                    batch_size = len(lengths)
                    total_recv_seqs += batch_size

                    # will hold smaller and smaller batch_size recursively
                    size_queue = [(0, len(req_ids))]

                    # avoids undefined sub_tokens in except
                    sub_tokens = tokens

                    # while we have mini-batches to process. not expected to loop
                    # over more than 1 iteration, except if we have an OOM. in that
                    # case, split the batch and iterate over mini-batches
                    n_processed = 0
                    n_failed = 0
                    start_time = time.time()
                    dec_lengths: List[int] = []

                    had_oom_once = False

                    while len(size_queue) > 0:
                        oom = False
                        begin, end = size_queue.pop()
                        assert begin < end
                        try:
                            # take sub-batch
                            sub_forbiddens = (
                                None
                                if forbiddens is None
                                else [f.to(self.device) for f in forbiddens[begin:end]]
                            )
                            sub_lengths = lengths[begin:end].to(self.device)
                            slen = sub_lengths.max().item()
                            sub_tokens = tokens[begin:end].to(self.device)
                            sub_tokens = sub_tokens[:, :slen]  # type: ignore
                            sub_infos = infos[begin:end]
                            sub_params = None if params is None else params[begin:end]

                            with timers.in_model.timeit():  # do_batch can raise
                                # call the model
                                tactics, priors, critics = self.beam_search.do_batch(
                                    src_len=sub_lengths,
                                    src_tokens=sub_tokens,
                                    forbiddens=sub_forbiddens,
                                    infos=sub_infos,
                                    params=sub_params,
                                )

                            timers.sending.start()
                            # send out the results
                            for local_id, global_id in enumerate(range(begin, end)):
                                assert len(tactics[local_id]) > 0
                                expansion = ModelExpansion(
                                    exp_duration=time.time()
                                    - req_ids[global_id].timestamp,
                                    gpu_duration=time.time() - start_time,
                                    log_critic=critics[local_id],
                                    tactics=tactics[local_id],
                                    log_priors=priors[local_id],
                                )
                                n_processed += 1
                                self.results.put(
                                    ExpanderOutput(
                                        req_id=req_ids[global_id],
                                        model_expansion=expansion,
                                    )
                                )
                                dec_lengths.extend([len(x) for x in tactics[local_id]])
                            timers.sending.stop()

                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                oom = True
                                had_oom_once = True
                                logger.warning(e)
                            else:
                                raise e

                        # OOM. split the batch in two and retry
                        if oom:
                            mem_a = torch.cuda.memory_reserved()
                            assert self.beam_search.decoder is not None
                            self.beam_search.decoder.empty_cache()
                            torch.cuda.empty_cache()
                            mem_b = torch.cuda.memory_reserved()
                            total_oom_count += 1
                            logger.warning(
                                f"CUDA OOM on {sub_tokens.numel()} tokens "
                                f"(bs={sub_tokens.shape[0]}, slen={sub_tokens.shape[1]}) - "
                                f"Memory after empty cache: {mem_a} -> {mem_b}"
                            )
                            print(torch.cuda.list_gpu_processes(), flush=True)
                            if end - begin > 1:
                                middle = int((begin + end) / 2)
                                size_queue.append((begin, middle))
                                size_queue.append((middle, end))
                            else:
                                total_oom_failures += 1
                                print("OOM WITH BS=1!!", file=sys.stderr, flush=True)
                                logger.warning("Returning empty tactics.")
                                expansion = ModelExpansion(
                                    exp_duration=time.time() - req_ids[begin].timestamp,
                                    gpu_duration=time.time() - start_time,
                                    error=FAILED_GPU_OOM,
                                )
                                n_failed += 1
                                self.results.put(
                                    ExpanderOutput(
                                        req_id=req_ids[begin],
                                        model_expansion=expansion,
                                    )
                                )

                    last_proc_time.append(time.time() - start_time)

                    timers.sending.start()
                    # sent signal that we are done with this batch.
                    # get by self.__batcher_loop
                    self._batch_done_queue.put(
                        DoneStatus(n_seqs=len(req_ids), oom=had_oom_once)
                    )
                    timers.sending.stop()

                    # sanity check / update stats
                    assert n_processed + n_failed == batch_size, (
                        n_processed,
                        n_failed,
                        batch_size,
                        total_oom_count,
                        total_oom_failures,
                    )
                    assert len(size_queue) == 0
                    pad_toks = tokens == self.beam_search.dico.pad_index  # type: ignore
                    batch_pad_toks: int = pad_toks.long().sum().item()  # type: ignore
                    batch_tot_toks: int = tokens.nelement()
                    batch_proc_toks: int = lengths.sum().item()  # type: ignore
                    # overall stats
                    total_proc_batches += 1
                    total_proc_seqs += batch_size
                    total_proc_toks += batch_proc_toks
                    total_pad_toks += batch_pad_toks
                    total_toks += batch_tot_toks
                    # recent stats
                    last_proc_batches += 1
                    last_proc_seqs += batch_size
                    last_proc_toks += batch_proc_toks
                    last_pad_toks += batch_pad_toks
                    last_tot_toks += batch_tot_toks
                    last_proc_slen.extend(lengths.tolist())
                    last_dec_slen.extend(dec_lengths)
            except:
                traceback.print_exc()
                sys.stdout.flush()
                self.died.set()
                # empty queue of CUDA tensors
                while True:
                    try:
                        x = self._gpu_queue.get_nowait()
                    except Empty:
                        break
                    if x is None:
                        break

        logger.info("Setting stop batcher!")
        self._stop_batcher.set()

        # log statistics
        logger.info(
            f"GPU stopping: "
            f"Received {total_recv_batches} batches ({total_recv_seqs} sequences). "
            f"Processed {total_proc_batches} batches ({total_proc_seqs} sequences). "
            f"Caught {total_oom_count} OOM errors ({total_oom_failures} with bs=1). "
            f"Padding ratio: {total_pad_toks / max(total_proc_toks, 1):.2f}"
        )

    def _gpu_loop(self, beam: BeamSearchModel, gpu_id: int):
        if gpu_id == 0:
            dirname = (
                f"expander_gpu_{this_job_id()}" if self.only_jsonl else "expander_gpu"
            )
            metrics_dump_path = self.prover_dump_path / "runner_logs" / dirname
            os.makedirs(metrics_dump_path, exist_ok=True)
            self.metrics = Logger(
                outdir=metrics_dump_path,
                tag="expander_gpu",
                quiet=self.quiet,
                only_jsonl=self.only_jsonl,
            )

        logger = getLogger()
        logger.info("===== Starting GPU loop =====")
        sp = subprocess.Popen(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out_str = sp.communicate()
        logger.info(out_str[0].decode("utf-8"))

        assert beam is not None, f"{beam} {gpu_id}"
        pr = None if not self.profile else cProfile.Profile()
        if pr is not None:
            pr.enable()
        try:
            self._gpu_queue.cancel_join_thread()
            self.results.cancel_join_thread()
            self.beam_search = beam
            self.__gpu_loop(gpu_id)
        except Exception:
            self.died.set()
            raise
        finally:
            if pr is not None:
                pr.disable()
                pr.dump_stats("profiles/gpu_loop.profile")
            if self.metrics is not None:
                self.metrics.close()

    def close(self):
        if self.name == "main" and not self.closed:
            self.closed = True
            print(f"Killing {self.name}", flush=True)
            self.stop_signal.set()
            print("Waiting on batcher", flush=True)
            self._batcher.join()
            print("Waiting on gpu", flush=True)
            for gpu_proc in self._gpus:
                gpu_proc.join()
            # If call has been closed, we don't expect anymore input.
            self.input.cancel_join_thread()
            print("Expander exited")
