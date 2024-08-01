# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, List, Tuple
import itertools
import time
import math
import numpy as np
import torch
from evariste.backward.graph import GoalParams

from evariste.model.transformer_args import DecodingParams


def get_clm_mask_target(
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    masked_out_seqs: Optional[torch.BoolTensor] = None,
    ignore: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get causal prediction mask with target tokens to predict.
    Optionally ignore tokens (e.g. tokens that correspond to emb_mask).
    if q is not None, ignore all rows of tokens that have q != 1
    """
    bs, slen = tokens.size()
    assert lengths.size() == (bs,)
    assert lengths.max() == slen
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    pred_tok_mask = alen[None] < lengths[:, None] - 1
    if masked_out_seqs is not None:
        assert masked_out_seqs.size() == (bs,)
        pred_tok_mask[masked_out_seqs] = 0

    pred_tok_mask[:, :ignore] = False
    n_targets = pred_tok_mask.sum().item()
    target_tok = tokens[:, 1:].masked_select(pred_tok_mask[:, :-1])
    assert target_tok.size() == (n_targets,), (target_tok, n_targets)
    assert n_targets > 0

    return pred_tok_mask, target_tok


def compute_entropy(p: np.ndarray, is_scores: bool) -> float:
    """
    Compute the entropy of a distribution.
    Take probabilities or scores as input.
    """
    assert p.ndim == 1
    if is_scores:
        p = np.exp(p - p.max())
        p = p / p.sum()
    else:
        assert np.all(p >= 0)
    p[p == 0] = 1
    entropy = -(np.log(p) * p).sum()
    return float(entropy)


@torch.no_grad()
def compute_entropy_batched(p: torch.Tensor, is_scores: bool) -> torch.Tensor:
    """
    Batched version of `compute_entropy`.
    """
    assert p.dim() == 2
    if is_scores:
        p = p.softmax(dim=-1)
    else:
        assert (p >= 0).all()
    p[p == 0] = 1
    entropy = -(p.log() * p).sum(1)
    return entropy


def set_target_entropy(
    scores: np.ndarray,
    tgt_ent: float,
    ignore_higher: bool,
    tolerance=1e-3,
    max_iter=100,
):
    """
    Given scores or probabilities, compute a temperature that moves the entropy of the
    distribution within a `tolerance` range of a target entropy `tgt_ent`.
    Optionally, do not update distributions with an initial entropy larger than the target.
    """
    (n,) = scores.shape
    assert 0 < tgt_ent < np.log(n)
    assert type(ignore_higher) is bool
    assert tolerance > 0
    assert max_iter > 0

    init_ent = compute_entropy(scores, is_scores=True)
    if ignore_higher and init_ent > tgt_ent:
        return 1, init_ent

    t_min = 1.0
    while compute_entropy(scores / t_min, is_scores=True) > tgt_ent:
        t_min /= 2

    t_max = 1.0
    while compute_entropy(scores / t_max, is_scores=True) < tgt_ent:
        t_max *= 2

    ent = None
    t = (t_min + t_max) / 2

    for i in range(max_iter):
        ent = compute_entropy(scores / t, is_scores=True)
        # print(f"{i:3} {t_min:.4f} {T:.4f} {t_max:.4f} --- {ent:.4f}")
        if abs(ent - tgt_ent) < tolerance:
            break
        elif ent < tgt_ent:
            t_min, t = t, (t + t_max) / 2
        elif ent > tgt_ent:
            t_max, t = t, (t + t_min) / 2
        else:
            print(t_min, t, t_max, ent, tgt_ent)
            raise

    return t, ent


@torch.no_grad()
def set_target_entropy_batched(
    scores: torch.Tensor,
    tgt_ent: float,
    ignore_higher: bool = False,
    tolerance=1e-3,
    max_iter=100,
):
    """
    Batched version of `set_target_entropy`.
    """
    bs, n = scores.size()
    assert 0 < tgt_ent < np.log(n)
    assert type(ignore_higher) is bool
    assert tolerance > 0
    assert max_iter > 0

    init_ent = compute_entropy_batched(scores, is_scores=True)
    todo = (init_ent < tgt_ent) if ignore_higher else None

    t_min = scores.new_ones((bs,), dtype=torch.float32)
    t_max = scores.new_ones((bs,), dtype=torch.float32)

    ent: Optional[torch.Tensor] = None

    for i in itertools.count():
        ent = (
            init_ent
            if i == 0
            else compute_entropy_batched(scores / t_min[:, None], is_scores=True)
        )
        mask = ent > tgt_ent
        mask = (mask & todo) if ignore_higher else mask
        if not mask.any():
            break
        t_min[mask] /= 2

    for i in itertools.count():
        ent = (
            init_ent
            if i == 0
            else compute_entropy_batched(scores / t_max[:, None], is_scores=True)
        )
        mask = ent < tgt_ent
        mask = (mask & todo) if ignore_higher else mask
        if not mask.any():
            break
        t_max[mask] *= 2

    ent = None
    t = (t_min + t_max) / 2

    for i in range(max_iter):
        ent = compute_entropy_batched(scores / t[:, None], is_scores=True)
        remaining = (ent - tgt_ent).abs() >= tolerance
        if ignore_higher:
            remaining &= todo
        # print(f"{i:3} {sum(remaining)} {t_min} {T} {t_max} --- {ent}")
        if not remaining.any():
            break

        mask = (ent < tgt_ent) & remaining
        if mask.any():
            t_min[mask] = t[mask]
            t[mask] = (t[mask] + t_max[mask]) / 2

        mask = (ent > tgt_ent) & remaining
        if mask.any():
            t_max[mask] = t[mask]
            t[mask] = (t[mask] + t_min[mask]) / 2
    assert ent is not None
    return t, ent


@torch.no_grad()
def filter_top_k(scores: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Set score to -inf for tokens not in the top-k.
    """
    assert 0 < top_k < scores.shape[-1]
    kth_values = torch.topk(scores, k=top_k, dim=-1).values[..., -1:]
    indices_to_remove = scores < kth_values
    return scores.masked_fill(indices_to_remove, -math.inf)


@torch.no_grad()
def filter_top_p(
    scores: torch.Tensor,
    top_p: float,
    min_tokens: Optional[int],
    max_tokens: Optional[int] = None,
) -> torch.Tensor:
    """
    Set score to -inf for tokens that do not contribute to a fraction `top_p`
    of the probability mass. Optionally keep a minimum number of tokens.

    Optionally, only consider the `max_tokens` tokens with the highest probabilities,
    which significantly speeds up sorting. This can result in slightly different
    results when the `scores` distribution is extremely flat. This makes the standalone
    function ~10x faster, and ~3 times faster when decoding with a reloaded model.
    """
    assert 0 < top_p < 1
    assert min_tokens is None or min_tokens >= 2
    assert max_tokens is None or min_tokens is None or min_tokens <= max_tokens

    # sort scores
    if max_tokens is None:
        # consider all tokens
        sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
        cumulative_probs = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
    else:
        # do not consider anything beyond `max_tokens`
        probs = scores.softmax(dim=-1)
        sorted_probs, sorted_indices = torch.topk(
            probs, k=max_tokens, dim=-1, largest=True, sorted=True
        )
        cumulative_probs = sorted_probs.cumsum(dim=-1)

    # keep tokens with cumulative probabilities below the threshold
    sorted_indices_to_keep = cumulative_probs < top_p

    # shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1].clone()
    sorted_indices_to_keep[..., 0] = True

    # keep at least `min_tokens` tokens
    if min_tokens is not None:
        sorted_indices_to_keep[..., :min_tokens] = True

    # scatter sorted tensors to original indexing
    if max_tokens is None:
        indices_to_keep = sorted_indices_to_keep.scatter(
            -1, sorted_indices, sorted_indices_to_keep
        )
    else:
        indices_to_keep = torch.zeros_like(scores, dtype=torch.bool).scatter(
            -1, sorted_indices, sorted_indices_to_keep
        )

    return scores.masked_fill(~indices_to_keep, -math.inf)


@torch.no_grad()
def update_scores(
    scores: torch.Tensor,
    dec_params: DecodingParams,
    extra_temperatures: Optional[torch.Tensor],
) -> torch.Tensor:

    # temperature
    if (
        extra_temperatures is None
        and dec_params.fixed_temperature is not None
        and dec_params.fixed_temperature != 1
    ):
        scores = scores / dec_params.fixed_temperature
    elif extra_temperatures is not None:
        assert extra_temperatures.shape[0] == scores.shape[0], (
            extra_temperatures.shape,
            scores.shape,
        )
        scores = scores / extra_temperatures[:, None]
    elif dec_params.target_entropy_temperature is not None:
        assert dec_params.ignore_higher_entropy is not None
        temperature, _ = set_target_entropy_batched(
            scores=scores,
            tgt_ent=dec_params.target_entropy_temperature,
            tolerance=1e-3,
            ignore_higher=dec_params.ignore_higher_entropy,
            max_iter=20,
        )
        scores = scores / temperature[:, None]

    # top-k
    if dec_params.top_k is not None:
        scores = filter_top_k(scores, top_k=dec_params.top_k)

    # top-p
    if dec_params.top_p is not None:
        scores = filter_top_p(
            scores=scores,
            top_p=dec_params.top_p,
            min_tokens=dec_params.top_p_min_tokens,
            max_tokens=1000,
        )

    return scores


if __name__ == "__main__":

    def _test_set_entropy():

        print("========== Running set_entropy tests")

        rng = np.random.RandomState(1)
        bs = 8
        tgt_ent_ = [0.1, 1, 3]
        ignore_higher_ = [True, False]
        tolerance_ = [1e-3, 1e-6]
        n_ = [1000, 10000]
        std_ = [0.01, 1, 100]

        for tgt_ent, ignore_higher, tolerance, n, std in itertools.product(
            tgt_ent_, ignore_higher_, tolerance_, n_, std_
        ):
            print(
                f"===== tgt_ent={tgt_ent} ignore_higher={ignore_higher} "
                f"tolerance={tolerance} n={n} std={std}"
            )
            scores = rng.randn(bs, n) * std
            temp_ent_1 = [
                set_target_entropy(
                    scores=scores[i],
                    tgt_ent=tgt_ent,
                    ignore_higher=ignore_higher,
                    tolerance=tolerance,
                )
                for i in range(bs)
            ]
            temp1 = torch.Tensor([temp for temp, _ in temp_ent_1])
            ent1 = torch.Tensor([ent for _, ent in temp_ent_1])
            temp2, ent2 = set_target_entropy_batched(
                scores=torch.from_numpy(scores),
                tgt_ent=tgt_ent,
                ignore_higher=ignore_higher,
                tolerance=tolerance,
            )
            # error1 = (ent1 - tgt_ent).abs().max().item()
            # error2 = (ent2 - tgt_ent).abs().max().item()
            # print(temp1.tolist())
            # print(temp2.tolist())
            # print(ent1.tolist())
            # print(ent2.tolist())
            # print("ent1 - ent2", (ent1 - ent2).abs().max().item())
            # print("temp1 - temp2", (temp1 - temp2).abs().max().item())
            # print("|ent1 - tgt_ent|", error1)
            # print("|ent2 - tgt_ent|", error2)
            if ignore_higher:
                assert all(ent >= tgt_ent - tolerance for ent in ent1.tolist())
                assert all(ent >= tgt_ent - tolerance for ent in ent2.tolist())
            else:
                assert all(abs(ent - tgt_ent) < tolerance for ent in ent1.tolist())
                assert all(abs(ent - tgt_ent) < tolerance for ent in ent2.tolist())

    def _test_filter_top_p():

        print("========== Running filter_top_p tests")

        bs = 16
        n_tokens = 50_000

        for std in [0.1, 1, 3, 5, 10]:
            print(f"===== std={std}")
            scores = (torch.randn(bs, n_tokens) * std).cuda()
            scores1 = filter_top_p(scores, top_p=0.9, min_tokens=10, max_tokens=None)
            scores2 = filter_top_p(scores, top_p=0.9, min_tokens=10, max_tokens=1000)
            diff = scores1 == scores2
            print(f"Selected exact: {(scores1 != -math.inf).sum(-1).tolist()}")
            print(f"Selected approx: {(scores2 != -math.inf).sum(-1).tolist()}")
            print(diff.sum(-1).tolist())
            print(f"{diff.sum()}/{diff.nelement()} identical elements.")

        bs = 512
        n_tokens = 100_000
        n_iter = 50
        scores = (torch.randn(bs, n_tokens) * 5).cuda()

        # exact run
        start = time.time()
        for _ in range(n_iter):
            _ = filter_top_p(scores, top_p=0.9, min_tokens=10, max_tokens=None)
            torch.cuda.synchronize()
        print(f"Exact run in {time.time() - start:.3f} secs")

        # approx run
        start = time.time()
        for _ in range(n_iter):
            _ = filter_top_p(scores, top_p=0.9, min_tokens=10, max_tokens=1000)
            torch.cuda.synchronize()
        print(f"Approx run in {time.time() - start:.3f} secs")

    _test_set_entropy()
    _test_filter_top_p()
