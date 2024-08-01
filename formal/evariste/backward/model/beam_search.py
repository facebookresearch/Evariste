# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, List, Set, Dict, cast
from json import JSONDecodeError
from functools import lru_cache
from enum import Enum, unique
from logging import getLogger
from pathlib import Path
from torch import nn
import os
import math
import time
import torch
import random
import numpy as np

from evariste import json as json
from evariste.backward.graph import GoalParams, NodeInfo
from evariste.model.checkpoints import get_latest_checkpoint
from evariste.model.transformer import DecodingParams, TransformerModel
from evariste.model.data.dictionary import Dictionary
from evariste.model.data.dictionary import B_CMD_WORD, E_CMD_WORD, SUCCESS_WORD
from evariste.model.utils import (
    reload_ckpt,
    load_from_checkpoint,
    get_path_bwd_proving_eval,
)
from evariste.utils import load_and_project_conditioning_vectors


logger = getLogger()


FAILED_UNK = "failed_unk"
FAILED_BIG = "failed_big"
FAILED_GPU_OOM = "failed_gpu_oom"


TacticTokens = List[str]
BeamTokens = List[TacticTokens]
Priors = List[float]
Critic = float


@lru_cache()
def _get_n_samples(method: str, alpha: float, init_n_samples: int, depth: int) -> int:
    assert alpha > 0
    if method == "linear":
        n = init_n_samples - depth * alpha
    elif method == "exponential":
        n = init_n_samples / (depth + 1) ** alpha
    elif method == "cosine":
        w = (math.pi / 2) / alpha  # alpha & depth have the same dimension (e.g. period)
        n = init_n_samples * math.cos(w * depth) if depth < alpha else 1
    else:
        raise RuntimeError(f"Unknown method: {method}")
    return max(int(math.ceil(n)), 1)


def get_n_samples(decoding_params: DecodingParams, infos: List[NodeInfo]) -> List[int]:
    """
    Optionally return a different n_samples per sequence.
    """
    n_samples_per_depth = decoding_params.n_samples_per_depth

    if n_samples_per_depth is None:
        return [decoding_params.n_samples for _ in range(len(infos))]
    if any(info.depth is None for info in infos):
        assert all(info.depth is None for info in infos)
        return [decoding_params.n_samples for _ in range(len(infos))]
    method, alpha = n_samples_per_depth
    return [
        _get_n_samples(
            method=method,
            alpha=alpha,
            init_n_samples=decoding_params.n_samples,
            depth=info.depth,
        )
        for info in infos
    ]


def get_class_ids(n_classes: int, n_samples: int) -> List[int]:
    """
    Split classes as much as possible.
    If 5 classes and 22 samples, use all classes 4 times, then sample 2 randomly.
    """
    assert n_samples > 0
    assert n_classes > 0
    class_ids: List[int] = [
        i % n_classes for i in range(n_classes * (n_samples // n_classes))
    ]
    extra = n_samples - len(class_ids)
    if extra > 0:
        class_ids += np.random.choice(n_classes, size=(extra,), replace=False).tolist()
    return class_ids


def get_cond_mask(n_samples: int, cond_prop: float) -> List[bool]:
    n_cond = math.floor(n_samples * cond_prop)
    cond = [True] * n_cond + [False] * (n_samples - n_cond)
    random.shuffle(cond)
    return cond


class BeamSearchModel:
    def __init__(
        self,
        decoding_params: DecodingParams,
        device: str,
        decoder_type: str,
        conditioning_vectors: Optional[Path] = None,
    ):
        self.encoder: Optional[TransformerModel] = None
        self.decoder: Optional[TransformerModel] = None
        self.cond_embeddings: Optional[nn.Linear] = None
        self.dico: Optional[Dictionary] = None
        device = "cpu" if decoding_params.cpu else device
        self.device = device
        assert decoder_type in ["decoder", "big_decoder"], decoder_type
        self.decoder_type = decoder_type

        self.decoding_params = decoding_params
        if self.decoding_params.prefix_success:
            assert self.decoding_params.has_success_token
            self.decoding_params.prefix = [SUCCESS_WORD]

        # Q conditioning
        self.q_conditioning = self.decoding_params.q_conditioning
        self.q_conditioning_ts = self.decoding_params.q_conditioning_inference_ts
        assert self.q_conditioning in ["", "sum", "prefix"]
        assert 0 <= self.q_conditioning_ts <= 1

        # keep track of model index
        self.model_id = -1

        # Lazily loaded, requires encoder dim
        self.conditioning_vectors_path = conditioning_vectors
        self.conditioning_vectors: Optional[Dict[str, np.ndarray]] = None

    @torch.no_grad()
    def _do_batch(
        self,
        src_len: torch.Tensor,
        src_tokens: torch.Tensor,
        forbiddens: Optional[List[torch.Tensor]],
        infos: Optional[List[NodeInfo]],
        params: Optional[List[GoalParams]],
    ) -> Tuple[List[BeamTokens], List[Priors], List[Critic]]:

        assert (
            self.encoder is not None
            and self.decoder is not None
            and self.dico is not None
        )

        dec_params = self.decoding_params
        b_cmd = self.dico.word2id[B_CMD_WORD]
        e_cmd = self.dico.word2id[E_CMD_WORD]

        assert len(src_tokens) == len(src_len)
        assert infos is None or len(infos) == len(src_len)
        assert params is None or len(params) == len(src_len)

        input_conditioning = None
        if self.conditioning_vectors is not None:
            assert params is not None
            input_conditioning = torch.from_numpy(
                np.vstack(
                    [
                        self.conditioning_vectors[param.conditioning_label]
                        for param in params
                        if param.conditioning_label is not None
                    ]
                )
            ).to(self.device)
            assert input_conditioning.shape == (src_tokens.shape[0], self.encoder.dim)

        src_enc = self.encoder(
            "fwd",
            causal=False,
            tokens=src_tokens,
            lengths=src_len,
            eval_to_keep_str=dec_params.enc_gen_to_keep,
            discr=input_conditioning,
            discr_mode="" if input_conditioning is None else "sum",
        )
        bs, slen, dim = src_enc.size()
        n = dec_params.n_samples
        infos = [NodeInfo() for _ in range(bs)] if infos is None else infos
        assert len(infos) == bs

        # First, compute the critic score
        critic_scores, _ = self.decoder(
            "compute_critic", src_enc=src_enc, src_len=src_len,
        )
        assert critic_scores.size() == (bs, 2)
        log_critics: List[float] = critic_scores[:, 0].tolist()  # log probas

        # Q conditioning
        discr: Optional[torch.Tensor] = None
        discr_mode: str = ""
        if self.q_conditioning:
            discr = ((self.q_conditioning_ts - 1.0) * torch.rand(size=(bs,)) + 1.0).to(
                device=src_enc.device
            )
            discr_mode = self.q_conditioning

        batched_tactics: List[List[List[str]]] = []
        batched_log_priors: List[List[float]] = []

        # no class conditioning for beam search (TODO: implement?)
        if dec_params.n_conditioning_classes > 0:
            assert not dec_params.use_beam

        if self.decoding_params.use_beam:
            generated_hyps = self.decoder.generate_beam(
                src_enc=src_enc,
                src_len=src_len,
                decoding_params=self.decoding_params,
                forbidden_tokens=forbiddens,
                discr=discr,
                discr_mode=discr_mode,
                goal_params=params,
            )
            # select hypotheses
            for beam_hyps in generated_hyps:
                these_tactics: List[List[str]] = []
                these_scores: List[float] = []
                assert len(beam_hyps.hyp) == n
                for score, hyp in beam_hyps.hyp:
                    assert type(score) is float
                    to_ignore = 1  # remove eos
                    if self.decoding_params.has_success_token:
                        to_ignore += 1  # first token is success token

                    tok_ids = hyp.tolist()[to_ignore:]

                    tok_ids = [b_cmd, *tok_ids, e_cmd]
                    tactic = [self.dico.id2word[x] for x in tok_ids]
                    these_tactics.append(tactic)
                    these_scores.append(score)
                batched_tactics.append(these_tactics)
                batched_log_priors.append(these_scores)
                assert len(batched_tactics[-1]) == n
                assert len(batched_log_priors[-1]) == n
        else:
            # number of samples per node
            if params is not None and params[0].n_samples is not None:
                assert all(p.n_samples is not None for p in params)
                n_gen = cast(List[int], [p.n_samples for p in params])
            else:
                n_gen = get_n_samples(self.decoding_params, infos)
            tot_gen = sum(n_gen)
            idx = src_len.new(sum([[i] * n_gen[i] for i in range(bs)], []))

            # duplicate goal params
            if params is not None:
                empty_gp: List[GoalParams] = []  # for mypy
                params = sum([[params[i]] * n_gen[i] for i in range(bs)], empty_gp)

            # conditioning
            n_classes = dec_params.n_conditioning_classes
            if n_classes > 0:
                assert discr is None
                assert self.cond_embeddings is not None
                class_embs = self.cond_embeddings.weight
                assert class_embs.size(1) == n_classes
                # use class IDs provided in GoalParams
                if dec_params.conditioning_strategy == "fixed":
                    assert params is not None
                    class_ids = cast(List[int], [gp.cond_id for gp in params])
                # use fully random class IDs
                elif dec_params.conditioning_strategy == "random":
                    class_ids = np.random.randint(
                        n_classes, size=(tot_gen,), dtype=np.int64
                    ).tolist()
                # try to split the class usage uniformly
                else:
                    assert dec_params.conditioning_strategy == "split"
                    class_ids = sum([get_class_ids(n_classes, n) for n in n_gen], [])
                discr = class_embs.T[idx.new(class_ids)]
                discr_mode = dec_params.conditioning_input_mode
                if dec_params.conditioning_prop < 1:
                    cond_mask = [
                        get_cond_mask(n, dec_params.conditioning_prop) for n in n_gen
                    ]
                    discr[~torch.BoolTensor(sum(cond_mask, []))] = 0

            # forbidden tokens
            forbiddens_rep: Optional[List[torch.Tensor]] = None
            if forbiddens is not None:
                empty_t: List[torch.Tensor] = []  # for mypy
                assert len(forbiddens) == bs
                forbiddens_rep = sum(
                    [[x] * n_ for x, n_ in zip(forbiddens, n_gen)], empty_t
                )
                assert forbiddens_rep is not None and len(forbiddens_rep) == tot_gen

            # repeat inputs times
            encoded = src_enc[idx]
            input_len = src_len[idx]
            assert encoded.shape == (tot_gen, slen, dim)
            assert input_len.shape == (tot_gen,)

            if self.q_conditioning:
                assert discr is not None and len(discr) == bs
                discr = discr[idx]
                assert discr.shape == (tot_gen,)

            dec_tok, dec_len, scores = self.decoder.generate(
                encoded,
                input_len,
                decoding_params=self.decoding_params,
                forbidden_tokens=forbiddens_rep,
                discr=discr,
                discr_mode=discr_mode,
                goal_params=params,
            )
            dec_tok = dec_tok.tolist()
            dec_len = dec_len.tolist()
            scores = scores.tolist()

            bid = 0  # batch global ID
            for i in range(bs):
                these_tactics, these_scores = [], []
                seen: Set[Tuple[str, ...]] = set()
                for j in range(n_gen[i]):
                    # remove eos tokens
                    tok_ids = dec_tok[bid][1 : dec_len[bid] - 1]
                    if self.decoding_params.has_success_token:
                        tok_ids = tok_ids[1:]  # first token is success token
                    tok_ids = [b_cmd, *tok_ids, e_cmd]
                    tactic = [self.dico.id2word[x] for x in tok_ids]
                    if tuple(tactic) not in seen:
                        seen.add(tuple(tactic))
                        these_tactics.append(tactic)
                        these_scores.append(scores[bid])
                    bid += 1
                # re-order by score
                best_ids = np.argsort(these_scores)[::-1]
                these_tactics = [these_tactics[j] for j in best_ids]
                these_scores = [these_scores[j] for j in best_ids]
                # add tactics
                batched_tactics.append(these_tactics)
                batched_log_priors.append(these_scores)
            assert bid == len(scores) == tot_gen

        # sanity check
        assert len(batched_tactics) == len(batched_log_priors) == len(log_critics) == bs
        assert all(
            lp <= 0 for log_priors in batched_log_priors for lp in log_priors
        ), batched_log_priors
        assert all(
            lp == sorted(lp, reverse=True) for lp in batched_log_priors
        ), batched_log_priors
        assert all(x <= 0 for x in log_critics), log_critics

        return batched_tactics, batched_log_priors, log_critics

    def do_batch(
        self,
        src_len: torch.Tensor,
        src_tokens: torch.Tensor,
        forbiddens: Optional[List[torch.Tensor]],
        infos: Optional[List[NodeInfo]],
        params: Optional[List[GoalParams]],
    ) -> Tuple[List[BeamTokens], List[Priors], List[Critic]]:
        return self._do_batch(src_len, src_tokens, forbiddens, infos, params)

    def load(self, path: Path):
        self.model_id += 1
        logger.info(
            f"Loading beam search model from {path} to {self.device} "
            f"(model ID: {self.model_id})"
        )

        # reload model from checkpoint
        to_reload = ["encoder", self.decoder_type]
        if self.decoding_params.n_conditioning_classes > 0:
            to_reload.append("cond_embeddings")
        modules, dico, _ = load_from_checkpoint(
            path=path,
            module_names=to_reload,
            device=self.device,
            fp16=(self.decoding_params.precision == "half"),
        )
        # ignoring types to avoid 3 extra lines...
        self.encoder = modules["encoder"]  # type: ignore
        self.decoder = modules[self.decoder_type]  # type: ignore
        if "cond_embeddings" in modules:
            assert isinstance(modules["cond_embeddings"], nn.Linear)
            self.cond_embeddings = modules["cond_embeddings"]

        if self.decoding_params.n_conditioning_classes > 0:
            assert self.cond_embeddings is not None, list(modules.keys())

        self.dico = dico

        # modules should already be in eval mode
        assert self.encoder is not None and self.decoder is not None
        assert self.encoder.training is False
        assert self.decoder.training is False

        # useful to reproduce MCTS scores
        if self.decoding_params.precision == "double":
            self.encoder.double()
            self.decoder.double()

        # sanity check (embeddings should match)
        n_words = len(dico)
        for module in [self.encoder, self.decoder]:
            n_emb = module.embeddings.num_embeddings
            assert len(dico) == n_emb, f"dico / emb mismatch: {n_words}!={n_emb}"

        # reload conditioning vectors if needed
        if (
            self.conditioning_vectors_path is None
            or self.conditioning_vectors is not None
        ):
            return
        self.conditioning_vectors = load_and_project_conditioning_vectors(
            self.conditioning_vectors_path, self.encoder.dim
        )

    def maybe_load(self) -> None:
        assert (
            self.encoder is not None
        ), "maybe_load called on BeamSearchModel. Use FixedBeamSearch instead."

    def load_dico(self) -> None:
        assert (
            self.dico is not None
        ), "load_dico called on BeamSearchModel. Use FixedBeamSearch instead."


class AutomaticallyReloadingBeamSearch(BeamSearchModel):
    """
    Uses the latest checkpoint in the directory path.
    Check for new checkpoint before each batch is processed
    """

    def __init__(
        self,
        path: Path,
        decoding_params: DecodingParams,
        device: str,
        decoder_type: str = "decoder",
        conditioning_vectors: Optional[Path] = None,
        min_reload_time: int = 60,
    ):
        super().__init__(decoding_params, device, decoder_type, conditioning_vectors)
        self.path = path
        self.model_version = -2
        self.last_load = 0.0
        self.min_reload_time = min_reload_time

    def maybe_load(self) -> None:
        if (
            time.time() - self.last_load <= self.min_reload_time
            and self.encoder is not None
        ):
            return
        self.last_load = time.time()
        once = True
        # Blocking while we haven't loaded anything, then just check once
        while self.encoder is None or once:
            path, version = get_latest_checkpoint(self.path)
            if path is None:
                time.sleep(5)
                continue
            if version > self.model_version:
                self.load(Path(path))
                self.model_version = version
            once = False

    def load_dico(self) -> None:
        while self.dico is None:
            path, version = get_latest_checkpoint(self.path)
            if path is None:
                time.sleep(5)
                continue
            else:
                _, self.dico, _ = reload_ckpt(Path(path))

    def do_batch(
        self,
        src_len: torch.Tensor,
        src_tokens: torch.Tensor,
        forbiddens: Optional[List[torch.Tensor]],
        infos: Optional[List[NodeInfo]],
        params: Optional[List[GoalParams]],
    ):
        self.maybe_load()
        return self._do_batch(src_len, src_tokens, forbiddens, infos, params)


class IncreasingQualityBeamSearch(AutomaticallyReloadingBeamSearch):
    """
    Holds a path to a folder and will check before each GPU call if a new, *better* model should be loaded
    """

    def __init__(
        self,
        path: Path,
        split: str,
        decoding_params: DecodingParams,
        device: str,
        lang: str,
        decoder_type: str = "decoder",
        conditioning_vectors: Optional[Path] = None,
    ):
        super().__init__(
            path, decoding_params, device, decoder_type, conditioning_vectors
        )
        self.last_best = 0
        self.already_evaled: Set = set()
        self.lang = lang
        self.split = split
        self.last_load = 0
        self.path = path

    def get_best_async_evaled_checkpoint(
        self,
    ) -> Tuple[Optional[Path], Optional[int], Optional[float]]:
        root = Path(
            get_path_bwd_proving_eval(
                root=self.path, lang=self.lang, split=self.split, decoder_type="decoder"
            )
        )
        assert root.exists(), "bwd_prover_eval folder doesn't exist"
        files = os.listdir(root)
        best_epoch, best_path, best_result = -2, None, -1.0
        for f in files:
            if not f.isnumeric() or f in self.already_evaled:
                continue
            results = root / f / "results.json"
            if not results.exists() or not (root / f / "done").exists():
                continue
            try:
                with open(results) as res:
                    lines = [json.loads(line.rstrip()) for line in res]
            except JSONDecodeError:
                continue
            self.already_evaled.add(f)
            success = len([True for line in lines if line["success"]])
            total = len(lines)
            if total == 0:
                continue
            result = success / total
            if result > best_result:
                best_epoch = int(f)
                best_path = root / f / "checkpoint.-1.pth"
                logger.info(
                    f"MODEL - Found new model: {best_path} ({result} > {best_result})"
                )
                best_result = result
            else:
                logger.info(f"MODEL - {f} was not better ({result} <= {best_result})")
        return best_path, best_epoch, best_result

    def maybe_load(self):
        # once a model is loaded, try to reload at most every 60s
        if time.time() - self.last_load <= 60 and self.encoder is not None:
            return
        self.last_load = time.time()
        path, epoch, result = self.get_best_async_evaled_checkpoint()
        if path is None:
            assert (
                self.encoder is not None
            ), "path is None but no beam_search is loaded!"
            return
        assert result is not None and epoch is not None
        if result >= self.last_best:
            logger.info(
                f"MODEL - Loading better model {path} (epoch {epoch}): "
                f"{result} >= {self.last_best}"
            )
            self.last_best = result
            self.load(path)
            self.model_version = epoch
        else:
            assert (
                self.encoder is not None
            ), f"{result} < {self.last_best} but no beam_search is loaded!"
            logger.info(
                f"MODEL - Ignoring new model with perfs {result} < {self.last_best}"
            )


class ManuallyReloadingBeamSearch(BeamSearchModel):
    """
    Uses the latest checkpoint in the directory containing the *file* in path.
    Only reload latest checkpoint when asked by the expander to do so, not before processing each batch
    """

    def __init__(
        self,
        path: Path,
        decoding_params: DecodingParams,
        device: str,
        decoder_type: str = "decoder",
        conditioning_vectors: Optional[Path] = None,
    ):
        super().__init__(decoding_params, device, decoder_type, conditioning_vectors)

        self.path = path.parent
        _, self.dico, _ = reload_ckpt(path)
        self.model_version = -2

    def maybe_load(self):
        path, version = get_latest_checkpoint(self.path)
        if version > self.model_version:
            logger.info(f"Manually reloading model in expander to version {version}!")
            self.load(Path(path))
            self.model_version = version


class FixedBeamSearch(BeamSearchModel):
    def __init__(
        self,
        path: Path,
        decoding_params: DecodingParams,
        device: str,
        decoder_type: str = "decoder",
        conditioning_vectors: Optional[Path] = None,
    ):
        super().__init__(decoding_params, device, decoder_type, conditioning_vectors)
        self.path = path
        _, self.dico, _ = reload_ckpt(path)

    def maybe_load(self):
        if self.encoder is None:
            logger.info(f"Loading model in expander from path: {self.path} !")
            self.load(self.path)


def load_beam_search_from_checkpoint(
    path: Path,
    decoding_params: DecodingParams,
    device: str,
    decoder_type: str = "decoder",
) -> BeamSearchModel:

    model = BeamSearchModel(decoding_params, device, decoder_type)
    model.load(path)
    return model
