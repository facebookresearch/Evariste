# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from typing import Any, List, Optional, Tuple, cast
from numpy import isin
from evariste.forward.core.forward_policy import ForwardPolicy, SyncForwardPolicy
from evariste.forward.common import FwdEnvOutput
import torch
from evariste.backward.graph import Tactic, Theorem
from evariste.backward.model.beam_search import (
    BeamSearchModel,
    FixedBeamSearch,
    load_beam_search_from_checkpoint,
)
from evariste.backward.model.beam_search_kind import BeamSearchKind
from evariste.backward.prover.prover_args import ProverParams
from evariste.forward.common import (
    ForwardGoal,
    ForwardStep,
    ForwardStepBeam,
    MaybeForwardStep,
)
from evariste.forward.env_specifics.prover_env_specifics import (
    AsyncForwardEnv,
    FwdTokenizer,
    FwdTrainParams,
    ProverEnvSpecifics,
)
from evariste.forward.forward_prover import ForwardProver, ProverConfig
from evariste.forward.proof_search import (
    ForwardProofSearch,
    SearchType,
    StandardProofSearch,
    StopConfig,
)
from evariste.model.data.dictionary import EOS_WORD, Dictionary
from evariste.model.transformer import TransformerModel
from evariste.model.transformer_args import DecodingParams
from evariste.model.transformer_utils import get_clm_mask_target
from evariste.model.utils import to_cuda


class ProverLossSearch(StandardProofSearch):
    """TODO(@tim) REVIEW"""

    def __init__(
        self,
        beam_search: BeamSearchModel,
        tokenizer: FwdTokenizer,
        dico: Dictionary,
        stop_config: StopConfig,
        goal: ForwardGoal,
        max_tokens: int,
        use_critic: bool = False,
        is_generation: bool = False,
    ):
        super().__init__(stop_config, goal, is_generation)
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.dico = dico
        self.max_tokens = max_tokens
        self.use_critic = use_critic

    def finish(self):
        # delete large beam search model
        self.beam_search = None

    def _prover_losses(self, steps: List[ForwardStep]) -> List[float]:
        losses = []
        batches = self._make_batches(steps)
        logging.info(f"Need {len(batches)} batches for prover loss/critic")
        for batch in batches:
            losses.extend(self._prover_loss(batch))
            logging.info(f"len losses: {len(losses)}")
        return losses

    @torch.no_grad()
    def _prover_loss(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> List[float]:
        assert self.beam_search is not None
        encoder = self.beam_search.encoder
        decoder = self.beam_search.decoder
        assert isinstance(encoder, TransformerModel)
        assert isinstance(decoder, TransformerModel)
        x, xlen, y, ylen = batch
        pred_mask, target = get_clm_mask_target(y, ylen)
        x, xlen, y, ylen, pred_mask, target = to_cuda(
            x, xlen, y, ylen, pred_mask, target
        )
        x_enc = encoder(
            "fwd",
            causal=False,
            tokens=x,
            lengths=xlen,
            eval_to_keep_str=self.beam_search.decoding_params.enc_gen_to_keep,
        )
        if encoder.is_changing_input_lengths():
            xlen = encoder.new_input_lengths(xlen)

        if self.use_critic:
            critic_scores, _ = decoder("compute_critic", src_enc=x_enc, src_len=xlen)
            assert critic_scores.size() == (len(x), 2)
            unsolvable_scores = critic_scores[:, 1]
            return unsolvable_scores.tolist()

        # decode y
        decoded = decoder(
            "fwd", causal=True, tokens=y, lengths=ylen, src_enc=x_enc, src_len=xlen
        )

        # loss
        _, losses = decoder(
            "compute_loss",
            tensor=decoded,
            pred_mask=pred_mask,
            target=target,
            reduction="none",
            # epsilon=params.label_smoothing_eps,
        )

        # retrieve the losses of the individual items in the batch
        # ylen[i] - 1 loss elements belonging to y[i]
        y_losses: List[float] = []
        idx: int = 0
        for i in range(len(y)):
            assert ylen[i] > 0
            y_losses.append(sum(losses[idx : (idx + int(ylen[i].item()) - 1)]))
            idx += int(ylen[i].item() - 1)
        assert idx == len(losses)

        assert len(y_losses) == len(xlen), (len(losses), xlen.tolist(), ylen.tolist())
        return y_losses

    def _make_x(self, goal: Theorem) -> List[int]:
        x = [EOS_WORD, *goal.tokenize(), EOS_WORD]
        return [self.dico.index(t) for t in x]

    def _make_y(self, tactic: Tactic) -> List[int]:
        y = [EOS_WORD, *tactic.tokenize(), EOS_WORD]
        return [self.dico.index(t) for t in y]

    def _make_batches(
        self, steps: List[ForwardStep]
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Split into batches of padded sequences.
        """
        xs = [self._make_x(s.generated) for s in steps]
        ys = [self._make_y(s.tactic) for s in steps]

        batches = []
        while xs:
            # fill batch until max_tokens is reached
            max_xlen, max_ylen = 0, 0
            too_long = False
            for i, (x, y) in enumerate(zip(xs, ys)):
                max_xlen = max(max_xlen, len(x))
                max_ylen = max(max_ylen, len(y))
                # length of all items up to and including i
                if self.use_critic:
                    toks = (i + 1) * max_xlen
                else:
                    toks = (i + 1) * (max_xlen + max_ylen)
                if toks > self.max_tokens:
                    too_long = True
                    break
            if not too_long:
                # all elements fit, we can include the last one
                i += 1
            assert (
                i > 0
            ), f"item too long: {i}, {max_xlen}, {max_ylen}, {self.max_tokens}, {xs[i]}, {ys[i]}"
            ys_batch = ys[:i]
            xs_batch = xs[:i]
            xs = xs[i:]
            ys = ys[i:]
            xlen: torch.Tensor = torch.LongTensor([len(s) for s in xs_batch])
            ylen: torch.Tensor = torch.LongTensor([len(s) for s in ys_batch])
            # merge sequences into a batch
            x_batch = torch.full((i, max(xlen)), self.dico.pad_index, dtype=torch.long)
            for idx, (xl, x) in enumerate(zip(xlen, xs_batch)):
                assert len(x) == xl
                x_batch[idx, :xl] = torch.LongTensor(x)
            y_batch = torch.full((i, max(ylen)), self.dico.pad_index, dtype=torch.long)
            for idx, (yl, y) in enumerate(zip(ylen, ys_batch)):
                assert len(y) == yl
                y_batch[idx, :yl] = torch.LongTensor(y)

            batches.append((x_batch, xlen, y_batch, ylen))

        assert sum(len(xlen) for _, xlen, _, _ in batches) == len(steps)
        assert sum(len(ylen) for _, _, _, ylen in batches) == len(steps)
        return batches

    def sample_valid_candidate(
        self, maybe_steps: ForwardStepBeam
    ) -> Tuple[int, MaybeForwardStep]:
        valid_steps: List[Tuple[int, MaybeForwardStep]] = [
            (i, s) for i, s in enumerate(maybe_steps) if s.step
        ]
        steps = cast(List[ForwardStep], [s.step for _, s in valid_steps])
        losses = self._prover_losses(steps)
        assert len(losses) == len(valid_steps)

        step_losses = list(
            sorted(zip(valid_steps, losses), key=lambda x: x[1], reverse=True)
        )
        # DEBUG:
        for step, loss in step_losses:
            assert isinstance(step[1].step, ForwardStep)
            logging.info(f"Step {step[0]}: {step[1].step.tactic}, prover loss: {loss}")

        # select step with maximal loss, if possible
        if not valid_steps:
            return 0, maybe_steps[0]
        else:
            return valid_steps[0]


class ProverLossForwardProver(ForwardProver):
    def __init__(
        self,
        cfg: ProverConfig,
        sync_policy: ForwardPolicy,
        fwd_env: AsyncForwardEnv,
        bwd_prover_params: ProverParams,
        bwd_decoding_params: DecodingParams,
        max_tokens: int,
        use_critic: bool = False,
        train_params: Optional[FwdTrainParams] = None,
    ):
        super().__init__(cfg, sync_policy, fwd_env, train_params)

        assert bwd_prover_params.beam_kind == BeamSearchKind.Fixed
        device = (
            "cpu" if bwd_decoding_params.cpu else f"cuda:{torch.cuda.current_device()}"
        )
        logging.info(f"Setting GPU to device {device} for prover-loss prover!")
        self.beam_search: BeamSearchModel = load_beam_search_from_checkpoint(
            path=bwd_prover_params.beam_path,
            decoding_params=bwd_decoding_params,
            device=device,
        )
        self.bwd_prover_params = bwd_prover_params
        self.max_tokens = max_tokens
        self.use_critic = use_critic

    def create_proof(self, goal: ForwardGoal) -> ForwardProofSearch:
        assert (
            self.search_cfg.proof_search_type == SearchType.PROVER_LOSS
        ), self.search_cfg.proof_search_type
        assert self.is_generator
        stop_cfg = StopConfig(
            max_nodes=self.search_cfg.max_nodes,
            max_generations=self.search_cfg.max_generations,
            max_cons_inv_allowed=self.search_cfg.max_cons_inv_allowed,
        )
        return ProverLossSearch(
            beam_search=self.beam_search,
            tokenizer=self.tokenizer,
            dico=self.dico,
            stop_config=stop_cfg,
            goal=goal,
            is_generation=self.is_generator,
            max_tokens=self.max_tokens,
            use_critic=self.use_critic,
        )

    @staticmethod
    def from_forward_prover(
        fwd_prover: ForwardProver,
        bwd_prover_params: ProverParams,
        bwd_decoding_params: DecodingParams,
        max_tokens: int,
        use_critic: bool = False,
    ) -> "ProverLossForwardProver":

        return ProverLossForwardProver(
            cfg=fwd_prover.cfg,
            sync_policy=fwd_prover.stepper.sync_policy,
            fwd_env=fwd_prover.fwd_env,
            train_params=fwd_prover._train_params,  # allow None
            bwd_prover_params=bwd_prover_params,
            bwd_decoding_params=bwd_decoding_params,
            max_tokens=max_tokens,
            use_critic=use_critic,
        )
