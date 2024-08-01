# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import time
from typing import Optional
from logging import getLogger

import torch
from torch import nn, Tensor

import torch.autograd.profiler as profiler
from torch.cuda.amp import autocast
from torch.nn import Parameter, init
import torch.nn.functional as F


logger = getLogger(__name__)


def get_causal_mask(lengths):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    slen = lengths.max().long().item()
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)

    # attention mask is the same as mask, or triangular inferior attention (causal)
    return alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]


class RFA(nn.Module):
    def __init__(
        self,
        n_heads: int,
        dim: int,
        phi_size_ratio: int = 1,
        phi_type: str = "arccos",
        n_train_projs: int = 200,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        assert phi_type in ["arccos", "gaussian", "learnt"]
        self.phi_type = phi_type

        assert self.dim % self.n_heads == 0
        self.dim_per_head = self.dim // self.n_heads

        self.n_projs = n_train_projs

        self.phi_dim = self.dim_per_head * phi_size_ratio

        # remarks on sigma:
        # 1. not sure for the range of init of sigma
        # 2. should sigma be the same across different layers (I don't think so)
        # 3. one different sigma per head
        self.log_sigma = Parameter(torch.ones(self.n_heads, 1, self.dim_per_head))
        self.log_sigma.data.fill_(math.log(1.0))
        self.min_log_std = math.log(1e-5)

        # two remarks on random projections:
        # 1. in the papers they say it's working best if we sample 200 of them
        # and sample from one of the 200 at each pass.
        # 2. I sample the same proj for key and query.
        # I don't know if I am supposed to do this
        # 3. I sample different proj for each head
        self.random_projs = Parameter(
            torch.Tensor(self.n_projs, self.phi_dim, self.dim_per_head),
            requires_grad=False,
        )
        init.normal_(self.random_projs, 0.0, 1.0)

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)

        if self.phi_type == "learnt":
            # quick test to see if I see a difference between random proj or learnt proj
            # for the moment learning the proj seems to work better
            # but still far from baseline
            self.proj: Optional[nn.Module] = nn.Linear(
                self.dim_per_head, self.phi_dim, bias=False
            )
        else:
            self.proj = None

        self.i = 0

    def forward(
        self,
        input: Tensor,
        mask: Tensor,
        kv: Optional[Tensor] = None,
        use_cache: bool = False,
    ):
        if kv is None:
            src = input
        else:
            src = kv
        assert not use_cache, "not implemented"
        bs, q_len, dim = input.size()
        k_len = src.size(1)

        if mask.dim() == 3:
            causal = True
            assert k_len == q_len, "not implemented"
            assert mask.size() == (bs, k_len, q_len)
            mask_2d = mask[:, :, 0]
            # we should check that the matrix is what we expect
            # a triangular inferior since causal rfa is not implemented for
            # generic 3d attn_mask
            expected_mask = get_causal_mask(lengths=mask_2d.sum(-1).long())
            assert torch.equal(mask, expected_mask)
            mask = mask_2d
        else:
            causal = False
        assert mask.size() == (bs, k_len)

        def _shape(x):
            return x.view(bs, -1, self.n_heads, self.dim_per_head)

        def _unshape(x):
            return x.view(bs, -1, self.n_heads * self.dim_per_head)

        with profiler.record_function("proj"):
            q = _shape(self.q_lin(input))
            k = _shape(self.k_lin(src))
            v = _shape(self.v_lin(src))

        if causal:
            c = self.causal_rfa(q, k, v, mask)
        else:
            c = self.cross_rfa(q, k, v, mask)
        with profiler.record_function("proj"):
            return self.out_lin(_unshape(c))

    def cross_rfa(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        mask = mask.unsqueeze(-1).unsqueeze(-1).float()
        assert q.dim() == k.dim() == v.dim() == 4
        assert q.size(0) == k.size(0)
        assert k.size() == v.size()

        with profiler.record_function("phi computation"):
            proj_ids = self.sample_proj_ids()
            phi_q = self.phi(q, proj_ids).transpose(
                1, 2
            )  # (bs, n_heads, qlen, phi_dim)
            phi_k = self.phi(k, proj_ids) * mask  # (bs, klen, n_heads, phi_dim)

        # Problem with Nan in fp16
        with autocast(enabled=False):
            phi_k = phi_k.float()
            phi_q = phi_q.float()
            v = v.float()
            with profiler.record_function("S z computation"):
                # We can probably use something more efficient (memory, time) than einsum
                S = torch.einsum(
                    "bthi,bthj->bhij", phi_k, v
                )  # (bs, n_heads, phi_dim, head_dim)
                z = phi_k.sum(1).unsqueeze(-1)  # (bs, n_heads, phi_dim, 1)

            with profiler.record_function("h computation"):
                # WARNING: no gradient when clamped, maybe find something better?
                normalisation = phi_q @ z  # (bs, n_heads, qlen, 1)
                h = (
                    (phi_q @ S / torch.clamp(normalisation, min=1e-5))
                    .transpose(1, 2)
                    .contiguous()
                )  # (bs, q_len, n_heads, head_dim)
        if self.i % 1000 == 0:
            logger.info(
                f"log_sigma.min() {self.log_sigma.min().item()} "
                f"log_sigma.mean() {self.log_sigma.mean().item()} "
                f"normalisation.min() {normalisation.min().item()} "
                f"normalisation.mean() {normalisation.mean().item()} "
                f"normalisation.abs().min() {normalisation.abs().min().item()} "
                f"h.abs().max() {h.abs().max().item()} "
                f"S.abs().mean() {S.abs().mean().item()}"
            )
        self.i += 1
        return h

    def sample_proj_ids(self):
        # we sample a different random proj for each head
        if self.training:
            proj_id = torch.randint(0, self.n_projs, (self.n_heads,))
        else:
            proj_id = [0] * self.n_heads
        return proj_id

    def phi(self, x: Tensor, proj_ids: Tensor) -> Tensor:
        if self.phi_type == "learnt":
            assert self.proj is not None
            return F.relu(self.proj(x))
        random_proj = self.random_projs[proj_ids]  # (n_heads, head_dim, phi_dim)
        bs, xlen, _, _ = x.size()

        scale = torch.exp(torch.clamp(self.log_sigma, min=self.min_log_std))
        random_proj = scale * random_proj
        x = F.normalize(x, p=2, dim=-1)
        proj_x = torch.einsum("blhj,hkj->blhk", x, random_proj)

        if self.phi_type == "arccos":
            # x was normalized before, but i'm not sure we need to it
            phi = F.relu(proj_x)
        elif self.phi_type == "gaussian":
            phi = torch.cat([torch.sin(proj_x), torch.cos(proj_x)], dim=-1)
        else:
            raise NotImplementedError

        # Don't know why we need to normalize if everything is linear
        phi = phi / math.sqrt(self.phi_dim)
        return phi

    def causal_rfa(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        # very slow
        mask = mask.unsqueeze(-1).unsqueeze(-1).float()
        assert q.dim() == k.dim() == v.dim() == 4
        assert q.size(0) == k.size(0)
        assert k.size() == v.size()
        with profiler.record_function("phi computation"):
            proj_id = self.sample_proj_ids()
            phi_k = self.phi(k, proj_id) * mask  # (bs, k_len, n_heads, D)
            phi_q = self.phi(q, proj_id)  # (bs, q_len, n_heads, D)

        bs, k_len, n_heads, D = phi_k.size()
        assert phi_q.size() == (bs, k_len, n_heads, D)
        S = phi_q.new_zeros((bs, self.n_heads, D, self.dim_per_head))
        z = phi_q.new_zeros((bs, self.n_heads, D, 1))
        h = phi_q.new_zeros((bs, k_len, self.n_heads, self.dim_per_head))

        with profiler.record_function("t iteration"):
            # very slow, should probably construct the matrices S_t, z_t
            # for all timesteps, using a cum_sum
            # but memory footprint will grow
            for t in range(k_len):
                phi_k_t = phi_k[:, t]
                phi_q_t = phi_q[:, t].unsqueeze(2)
                v_t = v[:, t]
                with profiler.record_function("S z computation"):
                    S += torch.einsum("bhi,bhj->bhij", phi_k_t, v_t)
                    z += phi_k_t.unsqueeze(-1)
                with profiler.record_function("h computation"):
                    h_t = phi_q_t @ S / torch.clamp(phi_q_t @ z, min=1e-5)
                    h[:, t] = h_t.squeeze(2)
        return h


if __name__ == "__main__":
    import logging
    from evariste.model.transformer import MultiHeadAttention

    logging.basicConfig(level=logging.INFO)
    dim = 512
    target_len = 2048
    src_len = target_len
    bs = 8
    type_ = "rfa"
    # type_ = "softmax"
    print(f"Attention: {type_}")
    if type_ == "rfa":
        att: nn.Module = RFA(
            n_heads=8, dim=dim, phi_size_ratio=2, phi_type="arccos"
        ).cuda()
    elif type_ == "softmax":
        att = MultiHeadAttention(n_heads=8, dim=dim, dropout=0).cuda()
    else:
        raise NotImplementedError

    with profiler.profile(record_shapes=True, use_cuda=True) as prof:
        for i in range(5):
            input = torch.randn(bs, target_len, dim).to("cuda")
            mask = torch.ones((bs, target_len), dtype=torch.bool).to("cuda")
            # kv = torch.randn(bs, src_len, dim).to("cuda")
            # lenghts = torch.ones(bs) * target_len
            # mask = get_causal_mask(lenghts).to("cuda")
            # print(mask[0, :, :])
            start = time.time()
            # input = torch.tensor([[[1, 1.5, 1, 1.5], [2, 2.5, 2, 2.5], [3, 3.5, 3, 3.5]]], dtype=torch.float)
            # mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)
            with autocast(enabled=True):
                with profiler.record_function("att"):
                    out = att(input, mask)
            # out = rfa(input, mask)
            # out = out.cpu()
            torch.cuda.synchronize()
            print(torch.cuda.max_memory_allocated() / (1024 ** 3))
            print("dur", time.time() - start)
    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="cuda_time_total", row_limit=40
        )
    )
    # k = torch.tensor([[1, 1.5], [2, 2.5], [3, 3.5]], dtype=torch.float)
    # v = torch.tensor([[1., 1.], [2., 2.], [3., 3.]], dtype=torch.float)
    #
    # k = k.unsqueeze(0)
    # v = v.unsqueeze(0).to("cuda")
    #
    # print(k.shape, v.shape)
    #
    # def phi(x_):
    #     return torch.cat([x_, -x_], dim=-1)
    #
    # phi_k = phi(k).to("cuda")
    # print(phi_k.shape, v.shape)
    #
    #
    #
    #
    # q = torch.tensor([[1, 1.5], [1, 1.5], [2, 2.5]], dtype=torch.float)
    # q = q.unsqueeze(0).to("cuda")
    # phi_q = phi(q).to("cuda")
    # print(phi_q.shape, S.shape, z.shape)
    #
    # # den =
    # # print("den", den.shape, den)
    # h = phi_q @ S / (phi_q @ z)
    #
    #
    # print("h", h.shape, h)
    #
    # # print(res, z)

    # with profiler.profile(record_shapes=True, use_cuda=True) as prof:
    #     key = torch.randn(16, 2048, 8, 1, 64).cuda()
    #     proj = torch.randn(8, 64, 128).cuda()
    #     _ = key @ proj
    #     with profiler.record_function("matmul_"):
    #         res1 = key @ proj
    #     print("matmul", res1.shape, torch.cuda.max_memory_allocated()/ (1024 ** 3))
    #     torch.cuda.reset_peak_memory_stats()
    #     _ = torch.einsum('blhij,hjk->blhik', key, proj)
    #     with profiler.record_function("einsum_"):
    #         res2 = torch.einsum('blhij,hjk->blhik', key, proj)
    #     print("einsum", res2.shape, torch.cuda.max_memory_allocated() / (1024 **3))
    #     print((res2 - res1).abs().max())
    #
    # print(
    #     prof.key_averages(group_by_input_shape=True).table(
    #         sort_by="cuda_time_total", row_limit=40
    #     )
    # )
