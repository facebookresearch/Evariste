# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from queue import Empty
from typing import List, Tuple, Optional, NamedTuple, Iterable, Dict

import torch
from torch import nn, multiprocessing as mp, Tensor
from torch._utils import ExceptionWrapper
from multiprocessing.synchronize import Event as EventClass

from evariste.forward.model.batching import EncoderInputs
from evariste.model.checkpoints import get_latest_checkpoint
from evariste.model.transformer import DecodingParams, TransformerModel

logger = getLogger()


Commands = List[List[int]]
Scores = List[float]


class DecoderOutputs(NamedTuple):
    cmds: List[Commands]
    scores: List[Scores]
    critics: Optional[List[float]]

    def unbatch(self) -> Iterable[Tuple[Commands, Scores]]:
        return zip(self.cmds, self.scores)

    def get_critic(self, idx: int) -> Optional[float]:
        return self.critics[idx] if self.critics is not None else None


@dataclass
class EncoderOutput:
    enc_len: Tensor
    enc_max_len: int
    encoded: Tensor
    forbidden: Optional[List[torch.Tensor]]
    enc_inp: Optional[Tensor] = None
    ptr_candidates: Optional[Tensor] = None
    ptr_candidates_len: Optional[Tensor] = None
    max_ptr_candidates_len: Optional[int] = None


class Seq2SeqModel(nn.Module):
    def __init__(
        self,
        encoder: TransformerModel,
        decoder: TransformerModel,
        critic: Optional[torch.nn.Module],
        decoding_params: DecodingParams,
        max_len: int,
        use_ptrs: bool = False,
        train_dir: Optional[str] = None,  # Needed for model reload
        discr_conditioning: bool = False,
        fp16: bool = False,
    ):
        super(Seq2SeqModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.critic = critic
        self.use_critic = critic is not None

        self.decoding_params = decoding_params
        self.device = next(self.decoder.parameters()).device
        self.use_ptrs = use_ptrs
        self.max_len = max_len
        if self.use_ptrs:
            raise NotImplementedError
        self.train_dir = train_dir
        self.discr_conditioning = discr_conditioning
        self.fp16 = fp16

    def _forward(self, batch: EncoderInputs) -> DecoderOutputs:
        with torch.cuda.amp.autocast(enabled=self.fp16):
            encoded = self.encode(batch)
            decoded = self.decode(encoded)
        return decoded

    def encode(self, batch: EncoderInputs) -> EncoderOutput:
        n_graphs = batch.batch_size

        # batching
        enc_inp = batch.enc_inp
        enc_len = batch.enc_len
        enc_max_len = enc_len.max().item()

        discr = enc_len.clone().float() if self.discr_conditioning else None

        encoded = self.encoder(
            "fwd",
            causal=False,
            tokens=enc_inp,
            lengths=enc_len,
            types=None,
            discr=discr,
            eval_to_keep_str=self.decoding_params.enc_gen_to_keep,
        )
        if self.encoder.is_changing_input_lengths():
            enc_len = self.encoder.new_input_lengths(enc_len)
            enc_max_len = enc_len.max().item()
        assert encoded.size() == (n_graphs, enc_max_len, self.encoder.dim)
        if self.use_ptrs:
            raise NotImplementedError

        return EncoderOutput(
            enc_max_len=int(enc_max_len),
            encoded=encoded,
            enc_len=enc_len,
            forbidden=batch.forbidden,
        )

    def decode(self, encoder_out: EncoderOutput) -> DecoderOutputs:
        assert not self.use_ptrs
        bs = len(encoder_out.enc_len)

        encoded, enc_len, enc_max_len, forbidden_tokens = (
            encoder_out.encoded,
            encoder_out.enc_len,
            encoder_out.enc_max_len,
            encoder_out.forbidden,
        )

        # decode substitutions of the next node
        #
        batched_cmds: List[Commands] = []
        batched_scores: List[Scores] = []
        if self.decoding_params.use_beam:
            generated_hyps = self.decoder.generate_beam(
                encoded,
                enc_len,
                decoding_params=self.decoding_params,
                forbidden_tokens=forbidden_tokens,
            )
            # select hypotheses
            for beam_hyps in generated_hyps:
                # keys are scores, values are tactic token_ids
                beam_hyps_dict = dict(beam_hyps.hyp)
                scores = sorted(beam_hyps_dict.keys(), reverse=True)[
                    : self.decoding_params.n_samples
                ]
                these_cmds, these_scores = [], []
                for score in scores:
                    cmd = beam_hyps_dict[score].tolist() + [
                        self.decoder.dico.word2id[self.decoding_params.stop_symbol]
                    ]
                    these_cmds.append(cmd)
                    these_scores.append(score)
                batched_cmds.append(these_cmds)
                batched_scores.append(these_scores)
        else:
            n_samples = self.decoding_params.n_samples
            forbiddens_repeated: Optional[List[Tensor]]
            if forbidden_tokens is not None:
                empty_list: List[Tensor] = []
                # mypy
                # https://stackoverflow.com/questions/58906541/incompatible-types-in-assignment-expression-has-type-listnothing-variabl
                forbiddens_repeated = sum(
                    [[x] * n_samples for x in forbidden_tokens], empty_list
                )
            else:
                forbiddens_repeated = None
            encoded = (
                encoded.unsqueeze(1)
                .repeat(1, n_samples, 1, 1)
                .view(bs * n_samples, enc_max_len, self.encoder.dim)
            )
            input_len = enc_len.unsqueeze(1).repeat(1, n_samples).view(bs * n_samples)

            generated, gen_len, scores = self.decoder.generate(
                encoded,
                input_len,
                decoding_params=self.decoding_params,
                forbidden_tokens=forbiddens_repeated,
            )

            assert len(gen_len) == len(encoder_out.encoded) * n_samples
            assert generated.size() == (
                len(encoder_out.encoded) * n_samples,
                gen_len.max(),
            )
            assert isinstance(scores, Tensor)

            generated = generated.cpu()
            gen_len = gen_len.cpu()
            scores = scores.cpu()
            for i in range(bs):
                these_cmds, these_scores = [], []
                for j in range(n_samples):
                    idx = i * n_samples + j
                    these_cmds.append(generated[idx, : gen_len[idx]].tolist())
                    these_scores.append(scores[idx].item())
                batched_cmds.append(these_cmds)
                batched_scores.append(these_scores)

        if self.use_critic:
            assert self.critic is not None
            critics = self.critic(encoded[:, 0]).cpu().tolist()
        else:
            critics = None

        return DecoderOutputs(cmds=batched_cmds, scores=batched_scores, critics=critics)

    def forward(
        self, batch: EncoderInputs, depth: int = 0, retrying_size_one: bool = False
    ) -> DecoderOutputs:
        """
        retrying_size_one: we observed than even sometimes we OOM for batch one
        inputs. In this case we clean the cache and retry. We set retrying_size_one
        in this case.
        """
        try:
            return self._forward(batch)
        except RuntimeError as e:
            if retrying_size_one:
                logger.warning("Already retried with this batch of size one")
                raise e
            elif "out of memory" in str(e):
                logger.warning(
                    f"CUDA OOM on {batch.size} tokens. depth: {depth}. Retrying."
                )
            elif "current device" in str(e):
                logger.warning("Detected RuntimeError with device error. Retrying")
                logger.info(
                    f"device debug: model device {self.device},"
                    f"embeddings device: {self.encoder.embeddings.weight.device} "
                    f"enc_inp device: {batch.enc_inp.device} "
                    f"enc_len device: {batch.enc_len.device} "
                )
                logger.info(f"batch: {batch}")
                logger.info(f"embeddings: {self.encoder.embeddings}")
            else:
                msg = (
                    f"CUDA RunTimeError on {batch.size} tokens, "
                    f"bs:{batch.batch_size} max_len {batch.max_len}. "
                    f"depth: {depth}. "
                    f"No retry."
                )
                logger.error(msg)
                raise e

        return self.redo_batch(batch, depth=depth)

    def redo_batch(self, batch: EncoderInputs, depth: int) -> DecoderOutputs:
        logger.info(
            f"Redo batch called on batch of size {batch.batch_size},"
            f" max_len {batch.max_len} (depth: {depth})"
        )

        if batch.batch_size == 1:
            msg = (
                f"Retrying with batch size of {batch.batch_size} "
                f"shape: {batch.enc_inp.shape}, "
                f"n toks: {batch.size}, "
                f"max len {batch.max_len} "
                f"(GPU {self.device})"
            )
            logger.info(msg)
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
            return self.forward(batch, depth=depth + 1, retrying_size_one=True)

        len_split = int(math.ceil(batch.batch_size / 2))
        begin = 0
        cmds, scores = [], []
        critics: Optional[List[float]]
        if self.use_critic:
            critics = []
        else:
            critics = None
        while begin < batch.batch_size:
            new_batch = batch.sub_batch(begin, begin + len_split)
            msg = (
                f"Trying with batch size {new_batch.batch_size} "
                f"(b: {begin}, e: {begin+len_split}), {new_batch.enc_inp.shape} "
                f"n toks: {new_batch.size}, max len {new_batch.max_len} "
                f"(GPU {self.device})"
            )
            logger.info(msg)
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
            dec_outs = self.forward(new_batch, depth=depth + 1)
            logger.info("-> succeed")

            cmds += dec_outs.cmds
            scores += dec_outs.scores
            if critics is not None:
                assert dec_outs.critics is not None
                critics += dec_outs.critics
            begin += len_split
        return DecoderOutputs(cmds=cmds, scores=scores, critics=critics)

    def reload(self):
        from evariste.model.utils import reload_ckpt

        if self.train_dir is None:
            raise ValueError("Missing train_dir for model reload")
        path, _ = get_latest_checkpoint(self.train_dir)
        logger.info(f"Model reloading from {path}")
        _, _, reloaded = reload_ckpt(path)
        state_dict = {}
        modules = ["encoder", "decoder"]
        for name in modules:
            state_dict[name] = {
                (k[len("module.") :] if k.startswith("module.") else k): v
                for k, v in reloaded[name].items()
            }
        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])


def batch_sequences(x, pad_index):
    """
    Create a batch of padded sequences.
    """
    assert type(x) is list
    assert all(type(xs) is list for xs in x)

    # sequence lengths
    bs = len(x)
    xlen = torch.LongTensor([len(s) for s in x])

    # merge sequences into a batch
    x_batch = torch.full((bs, max(xlen)), pad_index, dtype=torch.long)
    for sid, (xl, xs) in enumerate(zip(xlen, x)):
        assert len(xs) == xl
        x_batch[sid, :xl] = torch.LongTensor(xs)

    return x_batch, xlen


def _worker_loop(
    model: Seq2SeqModel,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    stop: EventClass,
    reload: EventClass,
):
    torch.cuda.set_device(model.device)
    logging.basicConfig(level=logging.INFO)
    logger_ = getLogger(__name__)
    logger_.info(
        f"Starting process: {mp.current_process().name}, "
        f"model.device: {model.device}"
    )
    stats: Dict[str, float] = defaultdict(float)
    activated = False
    try:
        while True:
            if reload.is_set():
                try:
                    start = time.time()
                    model.reload()
                    reload.clear()
                    stats["worker_reload"] += time.time() - start
                except Exception:
                    logger_.error("Error in async model, in reloading")
                    err = ExceptionWrapper(
                        where="in AsyncModel Process, model reloading"
                    )
                    out_queue.put((-1, err))

            start = time.time()
            try:
                r = in_queue.get(timeout=0.01)
                activated = True
            except Empty:
                continue
            finally:
                if activated:
                    stats["worker_get"] += time.time() - start
            if r is None:
                assert stop.is_set()
                # Sentinel, we exit
                break
            elif stop.is_set():
                # we are just emptying the queue
                del r
                continue
            idx, inp = r

            start = time.time()
            try:
                with torch.no_grad():
                    output = model(inp)
            except Exception:
                logger_.error("Error in async model in processing")
                output = ExceptionWrapper(where="in AsyncModel Process")
            stats["model"] += time.time() - start
            start = time.time()
            out_queue.put((idx, output))
            stats["worker_put"] += time.time() - start
    except KeyboardInterrupt:
        # will raise in main
        pass

    if stop.is_set():
        # note: if this process crash with unexpected error, we dont cancel_join_thread
        # on the out_queue, since:
        #       - (1) the out_queue will be emptied by 'main' process
        #       - (2) so this process will exit
        #       - (3) 'main' will detect that this process is not alive,
        #       and raise RunTimeError
        # if 'main' crashes while doing (1), 'main' will call .join() on this process
        # this join will timeout because of this out_queue, so 'main' will call
        # terminate() on this process.
        out_queue.cancel_join_thread()
        out_queue.close()
    else:
        # should not happen (or maybe with race condition on KeyboardInterrupt between
        # main and this process)
        logger.info("Stop was not set!")

    logger_.info(
        f"Stopping process: {mp.current_process().name} - stats: {dict(stats)}"
    )


class AsyncSeq2SeqModel:
    def __init__(self, model: Seq2SeqModel, name="async_model"):
        self.model = model
        ctx = mp.get_context("spawn")
        self.in_queue = ctx.Queue()
        self.out_queue = ctx.Queue()
        self.stop = ctx.Event()
        self.reload_event = ctx.Event()
        self._closed = True
        logger.info(
            f"Starting background process for AsyncModel on GPU: {model.device}"
        )
        self.process = ctx.Process(
            name=name,
            target=_worker_loop,
            args=(
                self.model,
                self.in_queue,
                self.out_queue,
                self.stop,
                self.reload_event,
            ),
        )
        self.process.start()
        # don't need to clean if we crash before the starting on process
        self._closed = False

    def send_batch(self, idx: List[int], inp: EncoderInputs):
        if self._closed:
            raise RuntimeError("AsyncModel already closed")
        self.in_queue.put((idx, inp))

    def receive_batch(self) -> Tuple[List[int], DecoderOutputs]:
        """
        Block until receiving next batch
        """
        received = False
        while not received:
            try:
                idx, data = self.out_queue.get(timeout=0.1)
                if isinstance(data, ExceptionWrapper):
                    data.reraise()
                return idx, data
            except Empty:
                if not self.process.is_alive():
                    raise RuntimeError(
                        f"AsyncModel worker (pid {self.process.pid}) exited unexpectedly"
                    )
                continue
        raise RuntimeError("Not reachable")

    def reload(self):
        self.reload_event.set()

    def close(self):
        if not self._closed:
            self._closed = True
            self.stop.set()
            # Signal termination to worker.
            self.in_queue.put(None)
            self.process.join(timeout=5.0)
            if self.process.is_alive():
                logger.info("Worker process didn't join after 5s")
                # Existing mechanisms try to make the workers exit
                # peacefully, but in case that we unfortunately reach
                # here, which we shouldn't, (e.g., pytorch/pytorch#39570),
                # we kill the worker.
                self.process.terminate()

            self.in_queue.cancel_join_thread()
            self.in_queue.close()

    def __del__(self):
        self.close()
