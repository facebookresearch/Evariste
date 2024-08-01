# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Any, List, Dict
from dataclasses import dataclass
from pathlib import Path
import torch

from params import Params
from evariste.clusters.utils import clusterify_path
from evariste.model.data.dictionary import Dictionary
from evariste.model.transformer import TransformerModel, kl_div_loss


Loss = torch.Tensor
Stats = Dict[str, float]


@dataclass
class KLCfg(Params):
    kl_coef: float = 0.0
    kl_temp: float = 1.0  # todo: check
    teacher_model_path: str = ""
    debug_only_kl_loss: bool = False  # if True only use kl loss to train model

    def should_use_kl(self):
        return self.kl_coef > 0

    def _check(self):
        assert (self.teacher_model_path != "") == (self.kl_coef > 0)

    def _check_and_mutate_args(self):
        self._check()


@dataclass
class KLTeacherModel:
    """Class to load, store and compute kl loss given a KLCfg"""

    kl_cfg: KLCfg
    encoder: TransformerModel
    decoder: TransformerModel

    @classmethod
    def create(
        cls, kl_cfg: KLCfg, device, fp16: bool, student_dico: Dictionary
    ) -> "KLTeacherModel":
        from evariste.model.utils import load_from_checkpoint  # circular imports

        assert kl_cfg.should_use_kl()

        # reload model from checkpoint
        path = Path(clusterify_path(kl_cfg.teacher_model_path))
        modules, dico, _ = load_from_checkpoint(
            path=path, module_names=["encoder", "decoder"], device=device, fp16=fp16,
        )
        encoder = modules["encoder"]
        decoder = modules["decoder"]
        if dico.id2word != student_dico.id2word:
            raise RuntimeError("Student and teacher dico are not equal!")
        assert isinstance(encoder, TransformerModel) and isinstance(
            decoder, TransformerModel
        )
        return cls(kl_cfg=kl_cfg, encoder=encoder, decoder=decoder)

    def add_kl_loss_to_loss_and_update_stats(
        self,
        batch: Dict[str, Any],
        student_scores: torch.Tensor,
        task: str,
        stats: Dict[str, List],
        loss_before_kl: Loss,
    ) -> Loss:
        """Adding all logic here not to avoid polluting trainer.py"""
        from evariste.model.transformer_utils import get_clm_mask_target
        from evariste.model.utils import to_cuda  # circular imports

        with torch.no_grad():
            x = batch["x"]
            y = batch["y"]
            xlen = batch["xlen"]
            ylen = batch["ylen"]
            langs2 = batch.get("langs2", None)

            # tokens to predict / cuda
            pred_mask, target = get_clm_mask_target(y, ylen)
            x, xlen, y, ylen = to_cuda(x, xlen, y, ylen)
            pred_mask, target, langs2 = to_cuda(pred_mask, target, langs2)

            # encode x
            encoded = self.encoder("fwd", causal=False, tokens=x, lengths=xlen)
            if self.encoder.is_changing_input_lengths():
                xlen = self.decoder.new_input_lengths(xlen)
            assert xlen.max().item() <= encoded.size(1)

            # decode y
            decoded = self.decoder(
                "fwd",
                causal=True,
                tokens=y,
                lengths=ylen,
                src_enc=encoded,
                src_len=xlen,
                langs=langs2,
            )

            # compute loss / optimize
            teacher_scores, teacher_loss = self.decoder(
                "compute_loss",
                tensor=decoded,
                pred_mask=pred_mask,
                target=target,
                epsilon=0.0,  # no label smoothing for teacher ?
                # (we only care about scores, not loss)
            )

        kl_loss = kl_div_loss(
            student_scores, teacher_scores.detach(), self.kl_cfg.kl_temp
        )

        if self.kl_cfg.debug_only_kl_loss:
            total_loss = self.kl_cfg.kl_coef * kl_loss
        else:
            total_loss = loss_before_kl + self.kl_cfg.kl_coef * kl_loss

        new_stats = {
            f"{task}/loss_before_kl": loss_before_kl.cpu().item(),
            f"{task}/kl_loss": kl_loss.cpu().item(),
            f"{task}/teacher_loss": teacher_loss.cpu().item(),
        }

        for key, value in new_stats.items():
            stats[key].append(value)

        return total_loss
