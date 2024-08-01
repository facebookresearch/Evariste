# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from logging import getLogger
from typing import (
    List,
    Tuple,
    Optional,
    Union,
    Iterator,
    Set,
    Generator,
    Any,
    Dict,
    cast,
)

import torch

from evariste.forward.common import (
    ForwardGraph,
    ForwardTactic,
    StopTactic,
    GenerationError,
    PolicyOutputBeam,
    PolicyOutput,
)
from evariste.forward.env_specifics.prover_env_specifics import FwdTokenizer
from evariste.forward.core.generation_errors import (
    InputDicoError,
    MaxLenReached,
    Duplicated,
    NotAllowedStop,
)
from evariste.forward.core.maybe import Maybe, Fail, Ok
from evariste.forward.model import batching
from evariste.forward.model.batching import EncoderInputs, group_samples_by_size
from evariste.forward.model.seq2seq_model import (
    Seq2SeqModel,
    DecoderOutputs,
    AsyncSeq2SeqModel,
)
from evariste.model.data.dictionary import Dictionary
from params import Params

PREFETCH = 3
MAX_BATCH_SIZE = 256
MAX_BATCH_MEM = 40_000_000

logger = getLogger()


class ForwardPolicy(ABC):
    @abstractmethod
    def submit_graph(self, graph_id: int, graph: ForwardGraph):
        pass

    @abstractmethod
    def ready_beams(self) -> List[Tuple[int, PolicyOutputBeam]]:
        pass

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def reload_model_weights(self):
        pass

    @abstractmethod
    def close(self):
        pass


@dataclass
class BatchConfig(Params):
    max_batch_mem: int = MAX_BATCH_MEM
    max_batch_size: int = MAX_BATCH_SIZE


class SyncForwardPolicy(ForwardPolicy):
    def __init__(
        self,
        fwd_tokenizer: FwdTokenizer,
        model: Seq2SeqModel,
        batch_cfg: BatchConfig,
        dico: Dictionary,
        max_len: int,
        allow_stop: bool = False,
        stop_command: Optional[List[str]] = None,
    ):
        self.tokenizer = fwd_tokenizer
        self.model = model
        self.cur_stats: Dict[str, Any] = defaultdict(float)
        self.max_batch_mem = batch_cfg.max_batch_mem
        self.max_batch_size = batch_cfg.max_batch_size
        self.dico = dico
        self.max_len = max_len
        self.allow_stop = allow_stop

        if self.allow_stop:
            assert stop_command is not None

        self.stop_command_indices: Optional[List[int]] = None
        if stop_command:
            self.stop_command_indices = [self.dico.index(w) for w in stop_command]
            assert (
                self.stop_command_indices[0] == dico.eos_index
                and self.stop_command_indices[-1] == dico.eos_index
            ), stop_command

        # for dummy async env api
        self.waiting: List[Tuple[int, ForwardGraph]] = []

    def submit_graph(self, graph_id: int, graph: ForwardGraph):
        self.waiting.append((graph_id, graph))

    def ready_beams(self) -> List[Tuple[int, PolicyOutputBeam]]:
        graphs, self.waiting = self.waiting, []
        return self._get_beams(graphs)

    def reload_model_weights(self):
        self.model.reload()

    @torch.no_grad()
    def _get_beams(
        self, graphs: List[Tuple[int, ForwardGraph]]
    ) -> List[Tuple[int, PolicyOutputBeam]]:
        id2graph = {i: g for i, g in graphs}

        start_pre = time.time()
        for_batching, too_long = self.make_samples(graphs)

        # wrapping error in ForwardStepBeam
        too_long_wrapped: List[Tuple[int, PolicyOutputBeam]] = [
            (id_, [Fail(err)]) for id_, err in too_long
        ]
        if not for_batching:
            assert len(too_long) == len(graphs)
            return too_long_wrapped

        step_beams: List[Tuple[int, PolicyOutputBeam]] = []
        for idxs, batch in self.batch_samples(for_batching):
            self.cur_stats["time_in_policy.preprocessing"] += time.time() - start_pre
            start_model = time.time()
            decoder_outputs = self.model(batch)
            self.cur_stats["time_in_policy.model"] += time.time() - start_model
            start = time.time()
            these_graphs = [id2graph[graph_id] for graph_id in idxs]
            these_beams = self.make_policy_output_beams(
                idxs, decoder_outputs, these_graphs
            )
            self.cur_stats["time_in_policy.post_processing"] += time.time() - start
            start_pre = time.time()
            step_beams.extend(these_beams)

        all_step_beams = step_beams + too_long_wrapped
        assert len(all_step_beams) == len(
            graphs
        ), f"{len(all_step_beams)} != {len(graphs)}"
        return all_step_beams

    def make_policy_output_beams(
        self,
        graph_ids: List[int],
        decoder_outputs: DecoderOutputs,
        graphs: List[ForwardGraph],
    ) -> List[Tuple[int, PolicyOutputBeam]]:
        assert len(graph_ids) == len(graphs)
        steps: List[Tuple[int, PolicyOutputBeam]] = []
        for i, (cmds, scores) in enumerate(decoder_outputs.unbatch()):
            graph_id = graph_ids[i]
            graph = graphs[i]
            beam: PolicyOutputBeam = []
            seen: Set[Tuple[int, ...]] = set()
            for id_in_beam, (cmd, score) in enumerate(zip(cmds, scores)):
                try:
                    forward_tactic, command_str = self.parse_forward_tactic(
                        cmd, seen, graph=graph
                    )
                except GenerationError as err:
                    beam.append(Fail(err))
                else:
                    beam.append(
                        Ok(
                            PolicyOutput(
                                graph=graph,
                                command=cmd,
                                command_str=command_str,
                                fwd_tactic=forward_tactic,
                                score=score,
                                normalized_score=score,
                                critic=None,
                            )
                        )
                    )
            steps.append((graph_id, beam))
        return steps

    def make_samples(
        self, graphs: List[Tuple[int, ForwardGraph]]
    ) -> Tuple[List[Tuple[int, EncoderInputs]], List[Tuple[int, GenerationError]]]:

        for_batching = []
        failed: List[Tuple[int, GenerationError]] = []
        for graph_idx, graph in graphs:
            try:
                sample = self.make_encoder_input(graph)
            except KeyError as err:
                gen_err = InputDicoError(str(err))
                failed.append((graph_idx, gen_err))
                continue
            if sample.max_len > self.max_len:
                ml_err = MaxLenReached(f"enc_len: {sample.max_len} > {self.max_len}")
                failed.append((graph_idx, ml_err))
            else:
                for_batching.append((graph_idx, sample))

        return for_batching, failed

    def make_encoder_input(self, graph: ForwardGraph) -> EncoderInputs:
        enc_inp = self.tokenizer.tokenize_graph(graph)
        assert enc_inp[0] == enc_inp[-1] == self.dico.eos_word, enc_inp
        enc_inp_ids = [self.dico.index(t) for t in enc_inp]

        enc_len = len(enc_inp_ids)
        enc_inp_tensor = torch.tensor([enc_inp_ids], dtype=torch.long)
        enc_len_tensor = torch.tensor([enc_len], dtype=torch.long)

        forbidden_tensors: Optional[List[torch.Tensor]] = None
        if graph.forbidden is not None:
            forbidden_ids = {
                self.dico.index(t) for t in graph.forbidden if t in self.dico
            }
            forbidden_tensors = [torch.tensor(list(forbidden_ids))]

        return EncoderInputs(
            enc_inp=enc_inp_tensor, enc_len=enc_len_tensor, forbidden=forbidden_tensors,
        )

    def batch_samples(
        self, samples: List[Tuple[int, EncoderInputs]]
    ) -> Iterator[Tuple[List[int], EncoderInputs]]:
        assert samples
        sample_idxs, enc_samples = zip(*samples)  # type: ignore
        sample_idxs = cast(Tuple[int, ...], sample_idxs)
        enc_samples = cast(Tuple[EncoderInputs, ...], enc_samples)
        for idxs, batch_samples in group_samples_by_size(
            list(sample_idxs),
            list(enc_samples),
            max_size=self.max_batch_mem,
            max_batch_size=self.max_batch_size,
            size_fn=batching.estimate_mem_per_batch,
        ):
            batch = batching.batch(batch_samples, self.dico.pad_index)
            max_retries = 1000
            i = 0
            while True:
                try:
                    # blocking to be sure that we catch the OOM
                    batch = batch.to(device=self.model.device, non_blocking=False)
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if i >= max_retries:
                            logger.error(
                                "Too many OOM detected when putting batch on GPU"
                            )
                            raise e
                        if i % 10 == 0:
                            logger.warning(
                                f"OOM while putting batch on GPU. "
                                f"Retry [{i}/{max_retries}]. "
                                f"Device: {self.model.device}"
                            )
                        with torch.cuda.device(self.model.device):
                            torch.cuda.empty_cache()
                        time.sleep(0.1)
                    else:
                        raise e
                    i += 1

            assert (
                batch.enc_inp.device == self.model.device
            ), f"{batch.enc_inp.device} != {self.model.device}"
            assert (
                batch.enc_len.device == self.model.device
            ), f"{batch.enc_len.device} != {self.model.device}"

            yield idxs, batch

    def parse_forward_tactic(
        self, command: List[int], seen: Set[Tuple[int, ...]], graph: ForwardGraph
    ) -> Tuple[Union[ForwardTactic, StopTactic], str]:
        key = tuple(command)
        if key in seen:
            raise Duplicated("Duplicated command")
        seen.add(key)
        tactic: Union[StopTactic, ForwardTactic]
        if self.stop_command_indices and command == self.stop_command_indices:
            if not self.allow_stop:
                raise NotAllowedStop("Not allowed stop!")
            tactic = StopTactic()
            command_toks = [self.dico[i] for i in command]
        else:
            command_toks = [self.dico[i] for i in command]
            if self.model.decoding_params.prefix is not None:
                # command toks is </s>, *prefix, ...
                # we should remove the prefix
                command_toks = [command_toks[0]] + command_toks[
                    1 + len(self.model.decoding_params.prefix) :
                ]
            tactic = self.tokenizer.detokenize_command(command_toks, graph=graph)
        return tactic, " ".join(command_toks)

    def close(self):
        pass

    def stats(self) -> Dict[str, Any]:
        return dict(self.cur_stats)


class AsyncForwardPolicy(ForwardPolicy):
    def __init__(self, policy: SyncForwardPolicy):
        self._async = _AsyncState(
            waiting_inputs=[], ready_outputs=[], waiting_for_batch=[]
        )
        self._async_loop = _async_policy_loop(policy=policy, state=self._async)
        self.policy = policy
        self.closed = False

    def submit_graph(self, graph_id: int, graph: ForwardGraph):
        self._async.waiting_inputs.append((graph_id, graph))

    def ready_beams(self) -> List[Tuple[int, PolicyOutputBeam]]:
        next(self._async_loop)
        result = self._async.ready_outputs
        self._async.ready_outputs = []
        return result

    def reload_model_weights(self):
        self.policy.reload_model_weights()

    def close(self):
        self._async_loop.close()
        self.closed = True

    def stats(self) -> Dict[str, Any]:
        return self.policy.stats()

    def __del__(self):
        if not self.closed:
            raise RuntimeError(f"{self} was not closed properly")


@dataclass
class _AsyncState:
    waiting_inputs: List[Tuple[int, ForwardGraph]]
    ready_outputs: List[Tuple[int, PolicyOutputBeam]]
    waiting_for_batch: List[Tuple[int, EncoderInputs]]


@torch.no_grad()
def _async_policy_loop(policy: SyncForwardPolicy, state: _AsyncState) -> Generator:
    id2graph: Dict[int, ForwardGraph] = {}
    in_model = 0
    async_model = AsyncSeq2SeqModel(model=policy.model)
    batch_iterator = _async_batcher_loop(policy, state)
    with closing(batch_iterator), closing(async_model):
        while True:
            start = time.time()
            received = state.waiting_inputs
            state.waiting_inputs = []
            for idx, graph in received:
                assert idx not in id2graph, f"{idx} in {id2graph.keys()}"
                id2graph[idx] = graph

            for_batching, too_long = policy.make_samples(received)
            state.waiting_for_batch.extend(for_batching)
            # wrapping error in ForwardStepBeam
            too_long_wrapped: List[Tuple[int, PolicyOutputBeam]] = [
                (id_, [Fail(err)]) for id_, err in too_long
            ]

            # we try to have PREFETCH batches in async model queue
            while in_model < PREFETCH:
                next_batch = next(batch_iterator)
                if next_batch is None:
                    # batcher is empty
                    break
                in_idxs, batch = next_batch
                assert all(i in id2graph for i in in_idxs)
                async_model.send_batch(in_idxs, batch)
                in_model += 1

            policy.cur_stats["time_in_policy.preprocessing"] += time.time() - start
            idxs: Optional[List[int]] = None
            decoder_outputs: Optional[DecoderOutputs] = None
            if in_model > 0:
                start = time.time()
                idxs, decoder_outputs = async_model.receive_batch()
                policy.cur_stats["time_in_policy.waiting_model"] += time.time() - start
                assert all(i in id2graph for i in idxs)
                in_model -= 1

            start = time.time()
            if decoder_outputs is not None:
                assert idxs is not None
                beams: List[
                    Tuple[int, PolicyOutputBeam]
                ] = policy.make_policy_output_beams(
                    graph_ids=idxs,
                    decoder_outputs=decoder_outputs,
                    graphs=[id2graph[idx] for idx in idxs],
                )
            else:
                beams = []

            beam_and_failed = beams + too_long_wrapped
            for idx, _ in beam_and_failed:
                id2graph.pop(idx)
            state.ready_outputs.extend(beam_and_failed)
            policy.cur_stats["time_in_policy.postprocessing"] += time.time() - start
            yield


def _async_batcher_loop(
    policy: SyncForwardPolicy, state: _AsyncState
) -> Generator[Optional[Tuple[List[int], EncoderInputs]], None, None]:
    while True:
        for_batching = state.waiting_for_batch
        state.waiting_for_batch = []
        if len(for_batching) != 0:
            yield from policy.batch_samples(for_batching)
        else:
            yield None
