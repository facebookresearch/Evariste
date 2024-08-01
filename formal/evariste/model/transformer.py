# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Optional, Union, Tuple, List, Dict, Any
from dataclasses import dataclass
from logging import getLogger
import math
import time
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evariste.backward.graph import GoalParams
from evariste.model.data.dictionary import (
    Dictionary,
    GEN_REAL,
    GEN_FAKE,
    SOLVABLE_WORD,
    NON_SOLVABLE_WORD,
)
from evariste.model.transformer_args import (
    eval_layers_to_keep,
    DecodingParams,
    ModelArgs,
)
from evariste.model.transformer_utils import update_scores, get_clm_mask_target


N_MAX_POSITIONS = 8192  # maximum input sequence length

DECODER_ONLY_PARAMS = [
    "layer_norm15.%i.weight",
    "layer_norm15.%i.bias",
    "encoder_attn.%i.q_lin.weight",
    "encoder_attn.%i.q_lin.bias",
    "encoder_attn.%i.k_lin.weight",
    "encoder_attn.%i.k_lin.bias",
    "encoder_attn.%i.v_lin.weight",
    "encoder_attn.%i.v_lin.bias",
    "encoder_attn.%i.out_lin.weight",
    "encoder_attn.%i.out_lin.bias",
]

ENCODER_CONVOL_PARAMS = ["conv.weight", "conv.bias"]

TRANSFORMER_TOK_EMB_PARAMS = [
    "embeddings.weight",
    "proj_layer.weight",
    "proj_layer.bias",
]

TRANSFORMER_LAYER_PARAMS = [
    "attentions.%i.q_lin.weight",
    "attentions.%i.q_lin.bias",
    "attentions.%i.k_lin.weight",
    "attentions.%i.k_lin.bias",
    "attentions.%i.v_lin.weight",
    "attentions.%i.v_lin.bias",
    "attentions.%i.out_lin.weight",
    "attentions.%i.out_lin.bias",
    "layer_norm1.%i.weight",
    "layer_norm1.%i.bias",
    "ffns.%i.lin1.weight",
    "ffns.%i.lin1.bias",
    "ffns.%i.lin2.weight",
    "ffns.%i.lin2.bias",
    "layer_norm2.%i.weight",
    "layer_norm2.%i.bias",
]


def prepare_extra_input(
    x: Optional[Union[int, torch.Tensor]], out_like: torch.Tensor
) -> Optional[torch.Tensor]:
    """
    Prepare extra inputs (e.g. languages, types, etc.)
    """
    if x is None:
        return x
    if type(x) is int:
        return torch.full_like(out_like, x)
    assert x.size() == (len(out_like),), (x.size(), out_like.size())
    return x.unsqueeze(1).expand_as(out_like)


def conv_len_out(conv: nn.Conv1d, len_in: torch.Tensor) -> torch.Tensor:
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    assert len(conv.kernel_size) == 1 and isinstance(conv.kernel_size, tuple)
    assert len(conv.stride) == 1 and isinstance(conv.stride, tuple)
    assert len(conv.padding) == 1 and isinstance(conv.padding, tuple)
    assert len(conv.dilation) == 1 and isinstance(conv.dilation, tuple)
    (kernel_size,) = conv.kernel_size
    (stride,) = conv.stride
    (dilation,) = conv.dilation
    (padding,) = conv.padding
    return torch.floor(
        (len_in + 2 * padding - dilation * (kernel_size - 1) - 1).float() / stride + 1
    ).long()


def get_topk1(scores: torch.Tensor, bs: int, beam_size: int, n_words: int):
    """
    Select the most promising beam hypotheses.
    """
    assert scores.shape == (bs * beam_size, n_words)
    scores = scores.view(bs, beam_size * n_words)
    scores, idx = scores.topk(2 * beam_size, dim=1, largest=True, sorted=True)
    return scores, idx


def get_topk2(scores: torch.Tensor, bs: int, beam_size: int, n_words: int):
    """
    Select the most promising beam hypotheses. Faster version.
    """
    assert scores.shape == (bs * beam_size, n_words)
    scores, _idx = scores.topk(2 * beam_size, dim=1, largest=True, sorted=True)
    scores = scores.view(bs, 2 * beam_size ** 2)
    scores, idx2 = scores.topk(2 * beam_size, dim=1, largest=True, sorted=True)
    offset = torch.arange(beam_size, device=_idx.device).repeat(bs) * n_words
    idx = (_idx + offset[:, None]).view(bs, 2 * beam_size ** 2).gather(1, idx2)
    return scores, idx


def get_truncated_gumbel(truncator, samples):
    # To truncate, we substract max(samples) to the samples then add the requested max
    # This is eq 22 which looks like log(exp(-max)-exp(max(samples))+exp(-samples))
    # To avoid numerical instability we use eq 23 / 24 instead
    z = torch.max(samples, dim=1, keepdim=True).values
    v = truncator - samples + torch.log1p(-torch.exp(samples - z))
    truncated = truncator - v.clamp(min=0) - torch.log1p(torch.exp(-torch.abs(v)))

    assert truncated.shape == samples.shape, f"{truncated.shape} // {samples.shape}"
    return truncated


logger = getLogger()


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def create_sinusoidal_embeddings(n_pos: int, dim: int, out):
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
            for pos in range(n_pos)
        ]
    )
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


def train_layers_to_keep(
    n_layers: int, layer_dropout: float, min_layers: int
) -> List[bool]:
    """
    Sample layers to keep at training time (used for layer dropout).
    Optionally set a minimum number of layers to keep.
    """
    assert 0 <= min_layers <= n_layers
    rates = np.random.rand(n_layers)
    to_keep = rates >= layer_dropout
    if to_keep.sum() < min_layers:
        idx = rates.argsort()[::-1][:min_layers]
        to_keep[idx] = True
        assert to_keep.sum() == min_layers
    return to_keep


def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def get_masks(
    slen: int, lengths: torch.Tensor, causal: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.shape[0]
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask


def get_decoder_scores(
    decoder,
    encoded,
    xlen,
    y,
    ylen,
    langs2,
    pred_mask,
    target,
    epsilon: float,
    grad_enabled: bool,
    get_critic: bool,
    output_hidden_states,
    q=None,
):
    with torch.set_grad_enabled(grad_enabled):
        if output_hidden_states:
            decoded, hidden_states = decoder(
                "fwd",
                causal=True,
                tokens=y,
                lengths=ylen,
                src_enc=encoded,
                src_len=xlen,
                langs=langs2,
                output_hidden_states=True,
            )
        else:
            decoded = decoder(
                "fwd",
                causal=True,
                tokens=y,
                lengths=ylen,
                src_enc=encoded,
                src_len=xlen,
                langs=langs2,
            )
        tok_scores, loss = decoder(
            "compute_loss",
            tensor=decoded,
            pred_mask=pred_mask,
            target=target,
            epsilon=epsilon,
        )
        result = (
            {"s2s": tok_scores},
            {"s2s": loss},
        )
        if output_hidden_states:
            result[0]["hidden_states"] = hidden_states
        if get_critic:
            critics, critics_loss = decoder(
                "compute_critic", q=q, src_enc=encoded, src_len=xlen
            )
            result[0]["critic"] = critics
            result[1]["critic"] = critics_loss
        return result


@dataclass
class AllAttentionHiddenStates:
    embeddings: torch.Tensor
    hiddens: List[torch.Tensor]


class MultiHeadAttention(nn.Module):
    NEW_ID = itertools.count()

    def __init__(
        self,
        n_heads: int,
        dim: int,
        dropout: float,
        src_dim: Optional[int] = None,
        learn_scaling: bool = False,
        init_scaling: Optional[float] = None,
        xav_init: bool = False,
    ):
        super().__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.src_dim = dim if src_dim is None else src_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.dim_per_head = self.dim // self.n_heads
        self.learn_scaling = learn_scaling
        assert self.dim % self.n_heads == 0
        assert (init_scaling is None) == (not learn_scaling)

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(self.src_dim, dim)
        self.v_lin = nn.Linear(self.src_dim, dim)
        self.out_lin = nn.Linear(dim, dim)

        # learnable of fixed scaling factor
        if self.learn_scaling:
            assert init_scaling is not None and init_scaling > 0
            self.scaling_factor = nn.Parameter(torch.tensor(init_scaling))

        if xav_init:
            gain = (1 / math.sqrt(2)) if self.src_dim == self.dim else 1.0
            nn.init.xavier_uniform_(self.q_lin.weight, gain=gain)
            nn.init.xavier_uniform_(self.k_lin.weight, gain=gain)
            nn.init.xavier_uniform_(self.v_lin.weight, gain=gain)
            nn.init.xavier_uniform_(self.out_lin.weight)
            nn.init.constant_(self.out_lin.bias, 0.0)

    def forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
            - input (bs, qlen, dim)
            - mask (bs, klen) (non-causal) or (bs, klen, klen)
        """
        assert not (use_cache and self.cache is None)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if not use_cache else self.cache["slen"] + qlen
        else:
            klen = kv.shape[1]
        assert dim == self.dim, "Dimensions do not match: %s input vs %s configured" % (
            dim,
            self.dim,
        )
        mask_reshape = (bs, 1, qlen, klen) if mask.dim() == 3 else (bs, 1, 1, klen)

        def shape(x) -> torch.Tensor:
            """projection"""
            return x.view(bs, -1, self.n_heads, self.dim_per_head).transpose(1, 2)

        def unshape(x) -> torch.Tensor:
            """compute context"""
            return (
                x.transpose(1, 2)
                .contiguous()
                .view(bs, -1, self.n_heads * self.dim_per_head)
            )

        q = shape(self.q_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k_lin(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        elif not use_cache or self.layer_id not in self.cache:
            k = v = kv
            k = shape(self.k_lin(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))  # (bs, n_heads, qlen, dim_per_head)

        if use_cache:
            if self.layer_id in self.cache:
                if kv is None:
                    k_, v_ = self.cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                    v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = self.cache[self.layer_id]
            self.cache[self.layer_id] = (k, v)

        # rescale with a fixed or a learnable factor. if the factor is learnable,
        # normalize keys and queries along their last dimension (after head split)
        # (bs, n_heads, qlen, dim_per_head)
        # https://www.aclweb.org/anthology/2020.findings-emnlp.379.pdf
        # https://github.com/CyndxAI/QKNorm/blob/main/QKNorm/layers.py
        if self.learn_scaling:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            q = q * self.scaling_factor
        else:
            q = q / math.sqrt(self.dim_per_head)

        # (bs, n_heads, qlen, klen)
        scores = torch.matmul(q, k.transpose(2, 3))
        mask = (mask == 0).view(mask_reshape).expand_as(scores)
        scores.masked_fill_(mask, -math.inf)

        # (bs, n_heads, qlen, klen)
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        weights = F.dropout(weights, p=self.dropout, training=self.training)

        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        return self.out_lin(context)


class TransformerFFN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        dim_hidden: int,
        out_dim: int,
        dropout: float,
        gelu_activation: bool,
        xav_init: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.lin1 = nn.Linear(in_dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, out_dim)
        self.act = gelu if gelu_activation else F.relu
        if xav_init:
            nn.init.xavier_uniform_(self.lin1.weight)
            nn.init.xavier_uniform_(self.lin2.weight)
            nn.init.constant_(self.lin1.bias, 0.0)
            nn.init.constant_(self.lin2.bias, 0.0)

    def forward(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        params: ModelArgs,
        dico: Dictionary,
        is_encoder: bool,
        with_output: bool,
        n_layers: Optional[int] = None,
        layer_dropout: float = 0,
        min_layers: int = 0,
        n_words_output: Optional[int] = None,
    ):
        """
        Transformer model (encoder or decoder).

        n_layers: (Optional[int])
            if not None, this parameter will be used to set the number of layers,
            else params.n_layers will be used.
        n_words_output: Optional[int]: if not None, replaces n_word for proj_layer

        """
        assert type(params) is ModelArgs
        super().__init__()

        # encoder / decoder, output layer
        self.fp16 = params.fp16
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output

        # dictionary
        self.n_words = len(dico)
        self.eos_index = dico.eos_index
        self.pad_index = dico.pad_index
        self.emb_index = dico.emb_index
        self.dico = dico

        # model parameters
        if self.is_encoder:
            self.dim = params.enc_emb_dim
        else:
            self.dim = params.dec_emb_dim
            self.src_dim = params.enc_emb_dim
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_heads = params.n_heads  # 8 by default
        self.n_layers = params.n_layers if n_layers is None else n_layers
        self.xav_init = params.xav_init
        self.rescale_embeddings = params.rescale_embeddings
        self.mha_learn_scaling = params.mha_learn_scaling
        self.mha_init_scaling = params.mha_init_scaling

        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        assert (
            self.dim % self.n_heads == 0
        ), f"transformer dim {self.dim} must be a multiple of n_heads {self.n_heads}"

        # layer drop
        self.layer_dropout = layer_dropout
        self.min_layers = min_layers
        assert 0 <= self.min_layers <= self.n_layers

        self.conv = None
        if is_encoder and params.enc_conv_kernel > 0:
            assert params.enc_conv_stride > 0
            assert is_encoder
            self.conv = nn.Conv1d(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=(params.enc_conv_kernel,),
                stride=(params.enc_conv_stride,),
                padding=(params.enc_conv_kernel - 1,),  # needed to be sure that we
                # are not missing some tokens
            )

        # embeddings
        self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
        if params.sinusoidal_embeddings:
            create_sinusoidal_embeddings(
                N_MAX_POSITIONS, self.dim, out=self.position_embeddings.weight
            )
        self.types_embeddings = Embedding(16, self.dim)
        self.langs_embeddings = Embedding(16, self.dim)
        self.embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.embed_scale = math.sqrt(self.dim)

        self.layer_norm_emb = nn.LayerNorm(self.dim, eps=1e-5)

        # transformer layers
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()
        if self.is_decoder:
            self.layer_norm15 = nn.ModuleList()
            self.encoder_attn = nn.ModuleList()

        for layer_id in range(self.n_layers):
            self.attentions.append(
                MultiHeadAttention(
                    self.n_heads,
                    self.dim,
                    dropout=self.attention_dropout,
                    learn_scaling=self.mha_learn_scaling,
                    init_scaling=self.mha_init_scaling,
                    xav_init=self.xav_init,
                )
            )
            self.layer_norm1.append(nn.LayerNorm(self.dim, eps=1e-5))
            if self.is_decoder:
                self.layer_norm15.append(nn.LayerNorm(self.dim, eps=1e-5))
                self.encoder_attn.append(
                    MultiHeadAttention(
                        self.n_heads,
                        self.dim,
                        dropout=self.attention_dropout,
                        src_dim=self.src_dim,
                        learn_scaling=self.mha_learn_scaling,
                        init_scaling=self.mha_init_scaling,
                        xav_init=self.xav_init,
                    )
                )
            self.ffns.append(
                TransformerFFN(
                    self.dim,
                    self.hidden_dim,
                    self.dim,
                    dropout=params.activation_dropout,
                    gelu_activation=params.gelu_activation,
                    xav_init=self.xav_init,
                )
            )
            self.layer_norm2.append(nn.LayerNorm(self.dim, eps=1e-5))

        # output layer
        if self.with_output:
            self.proj_layer = nn.Linear(self.dim, n_words_output or self.n_words)
            if self.xav_init:
                nn.init.xavier_uniform_(self.proj_layer.weight)
                nn.init.constant_(self.proj_layer.bias, 0.0)
            if params.share_inout_emb and self.embeddings:
                assert (
                    n_words_output is None
                ), f"can't share embeddings with {n_words_output} output words"
                self.proj_layer.weight = self.embeddings.weight
            assert (
                next(self.proj_layer.parameters()).type()
                == next(self.embeddings.parameters()).type()
            )

        self.cache: Optional[Dict[str, Any]] = None

    def empty_cache(self):
        self.cache.clear()

    @property
    def dtype(self):
        return torch.half if self.fp16 else torch.float

    def is_changing_input_lengths(self) -> bool:
        return self.conv is not None

    def new_input_lengths(self, lengths):
        assert self.is_changing_input_lengths()
        return conv_len_out(self.conv, lengths)

    def get_discr_emb(self, idx0: int, idx1: int, discr: torch.Tensor) -> torch.Tensor:
        """
        Convert float discriminator d in [0,1] into vectors.
        Compute from 2 embeddings V_0 and V_1 : (1-d) * V_0 + d * V_1
        """
        bs = len(discr)
        emb0 = self.embeddings(torch.LongTensor([idx0]).to(discr.device))
        emb1 = self.embeddings(torch.LongTensor([idx1]).to(discr.device))
        if self.rescale_embeddings:
            emb0 = emb0 * self.embed_scale
            emb1 = emb1 * self.embed_scale
        discr_emb = emb0 + (emb1 - emb0) * discr.view(bs, 1)
        assert discr_emb.size() == (bs, self.dim)
        return discr_emb

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        with torch.cuda.amp.autocast(enabled=self.fp16):
            if mode == "fwd":
                return self.fwd(**kwargs)
            elif mode == "compute_loss":
                return self.compute_loss(**kwargs)
            elif mode == "compute_critic":
                return self.compute_critic(**kwargs)
            elif mode == "best_classes_slow":
                return self.best_classes_slow(**kwargs)
            elif mode == "best_classes_fast":
                return self.best_classes_fast(**kwargs)
            elif mode == "best_classes_fast_split":
                return self.best_classes_fast_split(**kwargs)
            elif mode == "best_classes_tim":
                return self.best_classes_tim(**kwargs)
            elif mode == "fwd_tgt":
                return self.fwd_tgt(**kwargs)
            else:
                raise Exception("Unknown mode: %s" % mode)

    def fwd(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        causal: bool,
        positions: Union[bool, torch.Tensor] = True,
        types=None,
        langs=None,
        attn_mask=None,
        src_enc=None,
        src_len=None,
        use_cache=False,
        input_embs=None,
        enc_mask=None,
        discr=None,
        discr_mode: str = "",
        eval_to_keep_str: Optional[str] = None,
        output_hidden_states=False,
    ):
        """
        Inputs:
            `x` LongTensor(bs, slen), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `masks` BoolTensor(bs, slen, slen)
            `positions` LongTensor(bs, slen), containing word positions
            `langs` LongTensor(bs, slen), containing language IDs
            `discr`: discriminator score (float tensor). 1 = real, 0 = fake
        """
        # input checks
        if tokens.dim() == 2:
            bs, slen = tokens.size()
        else:
            bs, slen, emb_size = tokens.size()
            assert emb_size == self.dim
        assert discr_mode in ["", "sum", "prefix"]
        assert (discr is None) == (discr_mode == "")
        assert input_embs is None or input_embs.size() == (bs, slen, self.dim)
        assert lengths.size() == (bs,)
        assert type(causal) is bool
        assert (src_enc is None) == (src_len is None)
        assert (src_enc is None) or (self.is_decoder and len(src_enc) == bs)
        assert not (use_cache and self.cache is None)
        eval_to_keep: Optional[List[bool]] = None
        if eval_to_keep_str is not None:
            eval_to_keep = eval_layers_to_keep(eval_to_keep_str, self.n_layers)

        if self.conv:
            lengths = self.new_input_lengths(lengths)
            slen = conv_len_out(self.conv, torch.tensor(slen)).item()

        # self-attention masks
        if attn_mask is None:
            mask, attn_mask = get_masks(slen, lengths, causal)
        else:
            mask, _ = get_masks(slen, lengths, causal)
            attn_mask = attn_mask.type_as(mask)
            assert attn_mask.shape == (bs, slen, slen)

        # source attention mask
        if self.is_decoder and src_enc is not None:
            src_mask = (
                torch.arange(src_enc.size(1), dtype=torch.long, device=lengths.device)
                < src_len[:, None]
            )
            if enc_mask is not None:
                assert enc_mask.shape == src_mask.shape
                src_mask &= enc_mask

        # positions
        if positions is False:
            pass
        elif positions is True:
            positions = tokens.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (bs, slen)

        # extra inputs
        langs = prepare_extra_input(langs, out_like=tokens)
        types = prepare_extra_input(types, out_like=tokens)

        if discr is not None:
            if discr.shape == (bs,):
                discr_emb = self.get_discr_emb(
                    idx0=self.dico.index(GEN_FAKE),
                    idx1=self.dico.index(GEN_REAL),
                    discr=discr,
                )
            elif discr.shape == (bs, self.dim):
                discr_emb = discr
            else:
                raise RuntimeError(
                    f"Unexpected discr shape {discr.shape}, (bs, dim) = ({bs}, {self.dim})"
                )
            assert discr_emb.size() == (bs, self.dim), (discr_emb.size(), discr.size())
            if discr_mode == "sum":
                discr_emb = discr_emb.unsqueeze(1).expand(bs, slen, self.dim)
        else:
            discr_emb = None

        # do not recompute cached elements
        if use_cache:
            assert self.cache is not None
            _slen = slen - self.cache["slen"]
            tokens = tokens[:, -_slen:]
            if positions is not False:
                positions = positions[:, -_slen:]
            types = types[:, -_slen:] if types is not None else None
            langs = langs[:, -_slen:] if langs is not None else None

            if discr_mode == "sum" and discr is not None:
                discr_emb = discr_emb[:, -_slen:]

            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]
            if input_embs is not None:
                input_embs = input_embs[:, -_slen:]

        # embeddings
        if tokens.dim() == 2:
            tensor = self.embeddings(tokens)
            if self.rescale_embeddings:
                tensor = tensor * self.embed_scale
        else:
            tensor = tokens

        # if discr mode is prefix, discr replaces the </s> embedding,
        # so it should be applied before convs and all position/langs etc. sum
        if discr_mode == "prefix" and (
            not use_cache or (self.cache is not None and self.cache["slen"] == 0)
        ):
            assert discr_emb is not None
            assert tensor[:, 0].shape == discr_emb.shape
            tensor[:, 0] = tensor[:, 0] + discr_emb

        if self.conv:
            tensor = tensor.transpose(1, 2)
            tensor = self.conv(tensor)
            tensor = tensor.transpose(1, 2)

        if positions is not False:
            assert (
                positions.max().item() < N_MAX_POSITIONS
            ), f"{positions.max().item()} >= {N_MAX_POSITIONS}"
            tensor = tensor + self.position_embeddings(positions).expand_as(tensor)

        # is discr mode is sum then discr can be applied after convs
        if discr_mode == "sum":
            tensor = tensor + discr_emb

        # types and langs are supposed to be applied before the convs,
        # so we should have different slen for types, langs vs positions, mask
        if self.conv and (types is not None or langs is not None):
            raise NotImplementedError

        tensor = tensor if types is None else (tensor + self.types_embeddings(types))
        tensor = tensor if langs is None else (tensor + self.langs_embeddings(langs))
        if input_embs is not None:
            if tokens.dim() == 2:
                tensor[tokens == self.emb_index] = 0
                # TODO: check assert below. could be slower, but prevents silent bugs
                assert (
                    (input_embs.abs().sum(-1) == 0).long()
                    + (tensor.abs().sum(-1) == 0).long()
                    == 1
                ).all()
            tensor = tensor + input_embs
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # First hiddens states is the embeddings
        if output_hidden_states:
            all_hidden_states = AllAttentionHiddenStates(embeddings=tensor, hiddens=[])

        # layer drop
        if self.training and self.layer_dropout > 0:
            to_keep = train_layers_to_keep(
                self.n_layers, self.layer_dropout, self.min_layers
            )

        # transformer layers
        for i in range(self.n_layers):

            # layer drop
            if self.training and self.layer_dropout > 0 and not to_keep[i]:
                continue
            elif not self.training and eval_to_keep is not None and not eval_to_keep[i]:
                continue

            # self attention
            self.attentions[i].cache = self.cache
            attn = self.attentions[i](tensor, attn_mask, use_cache=use_cache)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # encoder attention (for decoder only)
            if self.is_decoder and src_enc is not None:
                self.encoder_attn[i].cache = self.cache
                attn = self.encoder_attn[i](
                    tensor, src_mask, kv=src_enc, use_cache=use_cache
                )
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.layer_norm15[i](tensor)

            # FFN
            ffn = self.ffns[i](tensor)
            ffn = F.dropout(ffn, p=self.dropout, training=self.training)
            tensor = tensor + ffn
            tensor = self.layer_norm2[i](tensor)

            tensor *= mask.unsqueeze(-1).to(tensor.dtype)
            if output_hidden_states:
                all_hidden_states.hiddens.append(tensor)

        # update cache length
        if use_cache:
            self.cache["slen"] += tensor.shape[1]
        if output_hidden_states:
            assert all_hidden_states.embeddings is not None
            assert len(all_hidden_states.hiddens) == self.n_layers
            return tensor, all_hidden_states
        else:
            return tensor

    def compute_loss(
        self,
        tensor: torch.Tensor,
        pred_mask: torch.Tensor,
        target: torch.Tensor,
        epsilon: float = 0,
        reduction="mean",
    ):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` torch.ByteTensor of shape (bs, slen), filled with 1 when we need to predict a word
            `target`    torch.LongTensor of shape (pred_mask.sum(),)
            `
        """
        # assert not (target == self.pad_index).any()
        # assert not (target == self.emb_index).any()
        assert pred_mask.any()
        scores = self.proj_layer(tensor[pred_mask]).view(-1, self.n_words)
        loss = smoothed_labels_xe_loss(
            scores=scores.float(), target=target, epsilon=epsilon, reduction=reduction
        )
        return scores, loss

    def fwd_tgt(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        n_classes: int,
        hard: bool,
        T: float,
    ):
        bs = len(tokens)
        encoded = self.fwd(tokens=tokens, lengths=lengths, causal=False)[:, 0]
        scores = self.proj_layer(encoded).float()  # batch_size x vocab_size
        best_classes = scores.argmax(dim=-1)
        assert best_classes.size() == (bs,)

        if not hard:
            assert T > 0, f"{T} <= 0, but hard"
            reg_loss = entropy_loss(scores)
            weights = torch.softmax(scores / T, dim=-1) / math.log(n_classes)
        else:
            weights = F.one_hot(best_classes, num_classes=n_classes).type(scores.dtype)
            reg_loss = 0

        assert weights.size() == (bs, n_classes)
        return weights, best_classes, reg_loss

    @torch.no_grad()
    def best_classes_slow(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        src_enc: torch.Tensor,
        src_len: torch.Tensor,
        n_classes: int,
        class_embs: torch.Tensor,
    ):
        assert class_embs.size() == class_embs.size(), (self.dim, n_classes)
        bs, slen = tokens.shape
        # TODO: N at a time!

        pred_mask, target = get_clm_mask_target(tokens, lengths)
        assert pred_mask.shape == (bs, slen)

        all_losses = []

        for cond_id in range(n_classes):

            discr = class_embs[:, cond_id].view(1, self.dim).expand(bs, self.dim)

            decoded = self.fwd(
                causal=True,
                tokens=tokens,
                lengths=lengths,
                src_enc=src_enc,
                src_len=src_len,
                discr=discr,
                discr_mode="sum",  # TODO: tune
            )

            # compute loss / optimize
            scores = self.proj_layer(decoded[pred_mask])
            loss = F.cross_entropy(scores, target, reduction="none")
            assert loss.shape == target.shape, (loss.shape, target.shape)
            seq_scores = target.new_zeros(pred_mask.shape, dtype=torch.float)
            seq_scores[pred_mask] += loss
            all_losses.append(seq_scores.sum(-1, keepdim=True))

        all_cond_losses = torch.cat(all_losses, dim=-1)
        assert all_cond_losses.size() == (bs, n_classes)
        return all_cond_losses.argmin(dim=-1)

    @torch.no_grad()
    def best_classes_fast(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        src_enc: torch.Tensor,
        src_len: torch.Tensor,
        n_classes: int,
        class_embs: torch.Tensor,
    ):
        assert class_embs.size() == class_embs.size(), (self.dim, n_classes)
        bs, slen = tokens.shape

        src_enc = (
            src_enc[:, None]
            .repeat(1, n_classes, 1, 1)
            .view(bs * n_classes, src_len.max().item(), self.src_dim)
        )
        src_len = src_len[:, None].repeat(1, n_classes).view(bs * n_classes)
        tokens = tokens[:, None].repeat(1, n_classes, 1).view(bs * n_classes, slen)
        lengths = lengths[:, None].repeat(1, n_classes).view(bs * n_classes)
        pred_mask, target = get_clm_mask_target(tokens, lengths)
        assert pred_mask.shape == (bs * n_classes, slen)

        discr = class_embs.T[None].repeat(bs, 1, 1).view(bs * n_classes, self.dim)

        decoded = self.fwd(
            causal=True,
            tokens=tokens,
            lengths=lengths,
            src_enc=src_enc,
            src_len=src_len,
            discr=discr,
            discr_mode="sum",  # TODO: tune
        )

        # compute loss / optimize
        scores = self.proj_layer(decoded[pred_mask])
        loss = F.cross_entropy(scores, target, reduction="none")
        assert loss.shape == target.shape, (loss.shape, target.shape)
        seq_scores = target.new_zeros(pred_mask.shape, dtype=torch.float)
        seq_scores[pred_mask] += loss
        assert seq_scores.shape == (bs * n_classes, slen)
        all_cond_losses = seq_scores.view(bs, n_classes, slen).sum(dim=-1)

        assert all_cond_losses.size() == (bs, n_classes)
        return all_cond_losses.argmin(dim=-1)

    @torch.no_grad()
    def best_classes_fast_split(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        src_enc: torch.Tensor,
        src_len: torch.Tensor,
        n_classes: int,
        class_embs: torch.Tensor,
        max_classes: int,
    ):
        assert class_embs.size() == class_embs.size(), (self.dim, n_classes)
        bs, slen = tokens.shape
        assert n_classes % max_classes == 0

        src_enc = (
            src_enc[None]
            .repeat(n_classes, 1, 1, 1)
            .view(n_classes * bs, src_len.max().item(), self.src_dim)
        )
        src_len = src_len[None].repeat(n_classes, 1).view(n_classes * bs)
        tokens = tokens[None].repeat(n_classes, 1, 1).view(n_classes * bs, slen)
        lengths = lengths[None].repeat(n_classes, 1).view(n_classes * bs)
        discr = class_embs.T[:, None].repeat(1, bs, 1).view(n_classes * bs, self.dim)

        _all_cond_losses: List[torch.Tensor] = []
        step = max_classes * bs
        assert (n_classes * bs) % step == 0

        for i in range(0, n_classes * bs, step):
            j = i + step

            # mini-batch
            _src_len = src_len[i:j]
            _lengths = lengths[i:j]
            _xlen = _src_len.max().item()
            _ylen = _lengths.max().item()
            _tokens = tokens[i:j, :_ylen]
            _src_enc = src_enc[i:j]  # , :_xlen]
            _discr = discr[i:j]

            _pred_mask, _target = get_clm_mask_target(_tokens, _lengths)
            assert _pred_mask.shape == (step, _ylen)

            decoded = self.fwd(
                causal=True,
                tokens=_tokens,
                lengths=_lengths,
                src_enc=_src_enc,
                src_len=_src_len,
                discr=_discr,
                discr_mode="sum",
            )

            # compute loss / optimize
            scores = self.proj_layer(decoded[_pred_mask])
            loss = F.cross_entropy(scores, _target, reduction="none")
            assert loss.shape == _target.shape, (loss.shape, _target.shape)
            seq_scores = _target.new_zeros(_pred_mask.shape, dtype=torch.float)
            seq_scores[_pred_mask] += loss
            assert seq_scores.shape == (step, _ylen)
            _all_cond_losses.append(seq_scores.view(max_classes, bs, _ylen).sum(dim=-1))

        all_cond_losses = torch.cat(_all_cond_losses, dim=0).T
        assert all_cond_losses.size() == (bs, n_classes)
        return all_cond_losses.argmin(dim=-1)

    @torch.no_grad()
    def best_classes_tim(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        src_enc: torch.Tensor,
        src_len: torch.Tensor,
        n_classes: int,
        class_embs: torch.Tensor,
    ):
        assert class_embs.size() == class_embs.size(), (self.dim, n_classes)
        bs, slen = tokens.shape

        pred_mask, target = get_clm_mask_target(tokens, lengths)
        assert pred_mask.shape == (bs, slen)

        all_losses = []
        for cond_id in range(n_classes):

            discr = class_embs[:, cond_id].view(1, self.dim).expand(bs, self.dim)

            decoded = self.fwd(
                causal=True,
                tokens=tokens,
                lengths=lengths,
                src_enc=src_enc,
                src_len=src_len,
                discr=discr,
                discr_mode="sum",  # TODO: tune
            )

            # compute loss / optimize
            _scores, loss = self.compute_loss(
                tensor=decoded,
                pred_mask=pred_mask,
                target=target,
                epsilon=0.0,
                reduction="none",
            )
            all_losses.append(loss[None, :])

        all_cond_losses = torch.cat(all_losses)  # shape = n_classes x sum y_len
        assert all_cond_losses.shape == (n_classes, len(target))
        start = 0
        best_tokens = []
        for i in (lengths - 1).tolist():
            best_tokens.append(
                torch.argmin(all_cond_losses[:, start : start + i].sum(dim=-1)).item()
            )
            start = start + i
        assert start == all_cond_losses.size(1)

        return target.new(best_tokens)

    @torch.no_grad()
    def generate(
        self,
        src_enc: Optional[torch.Tensor],
        src_len: Optional[torch.Tensor],
        decoding_params: DecodingParams,
        types=None,
        langs=None,
        forbidden_tokens=None,
        discr: Optional[torch.Tensor] = None,
        discr_mode: str = "",
        goal_params: Optional[List[GoalParams]] = None,  # already duplicated
    ):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - None / int / LongTensor(bs)
        `forbidden_tokens`:
            - either a BoolTensor mask of shape (n_words,), or (bs, n_words), or
              a list of `bs` LongTensor of forbidden token IDs which we convert to a
              BoolTensor mask of shape (bs, n_words)
        `discr`:
            - optional float tensor for discriminator experiments. 1 = real, 0 = fake
        """
        assert (src_enc is None) == (src_len is None)

        # sanity check (useless to modify scores if we do not sample)
        modify_scores = any(
            x is not None
            for x in [
                decoding_params.fixed_temperature,
                decoding_params.target_entropy_temperature,
                decoding_params.top_k,
                decoding_params.top_p,
                # goal_params,  # we can have goal_params not None (conditioning label) even without sampling
            ]
        )
        assert not modify_scores or decoding_params.use_sampling

        # stop index (default is the index of "</s>") / max_len
        stop_index = self.dico.word2id[decoding_params.stop_symbol]
        max_len = decoding_params.max_gen_len

        # batch size / device
        if src_len is None:
            bs = decoding_params.n_samples
            device = self.embeddings.weight.device
        else:
            assert src_enc is not None
            bs = len(src_len)
            device = src_enc.device
            assert len(src_enc) == bs
            assert goal_params is None or len(goal_params) == bs

        # temperature
        extra_temperatures: Optional[torch.Tensor] = None
        if goal_params is not None and goal_params[0].temperature is not None:
            extra_temperatures = src_enc.new([p.temperature for p in goal_params])
            assert len(extra_temperatures) == bs

        # length penalty
        length_penalty = decoding_params.length_penalty
        if goal_params is not None and goal_params[0].length_penalty is not None:
            length_penalty = src_enc.new([p.length_penalty for p in goal_params])
            assert len(length_penalty) == bs

        # forbidden tokens
        if forbidden_tokens is None:
            forbidden = None
        elif type(forbidden_tokens) is list:
            assert len(forbidden_tokens) == bs
            forbidden = torch.full(
                size=(bs, self.n_words),
                fill_value=False,
                dtype=torch.bool,
                device=device,
            )
            for i, l in enumerate(forbidden_tokens):
                if len(l) == 0:
                    continue
                forbidden[i, l] = True
        else:
            assert forbidden_tokens.size() in [(self.n_words,), (bs, self.n_words)]
            forbidden = forbidden_tokens.to(device=device)

        # generated sentences
        generated = torch.full(
            size=(bs, max_len),
            fill_value=self.pad_index,
            dtype=torch.long,
            device=device,
        )
        generated[:, 0].fill_(self.eos_index)  # we use <EOS> for <BOS> everywhere

        # scores for each sentence in the beam
        sentence_scores = torch.zeros(bs, dtype=torch.float64, device=device)

        # positions
        positions = (
            torch.arange(max_len, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand_as(generated)
        )

        # current position / max lengths / length of generated sentences / unfinished sentences
        cur_bs = bs
        cur_len = 1
        gen_len = generated.new_full((bs,), fill_value=1)
        unfinished = gen_len.new_full((bs,), fill_value=True, dtype=torch.bool)
        arange = torch.arange(bs, device=device)

        # cache computed states
        self.cache = {"slen": 0}

        # optional sequence prefix
        prefix = decoding_params.prefix
        if prefix is not None:
            assert type(prefix) is list and len(prefix) > 0
            assert all(type(w) is str for w in prefix)
            assert len(prefix) < max_len
            prefix_toks = [self.dico.word2id[w] for w in prefix]
            cur_len += len(prefix_toks)
            gen_len += len(prefix_toks)
            generated[:, 1 : 1 + len(prefix_toks)] = torch.LongTensor(prefix_toks)

        first_step = True

        def sub(x: Optional[torch.tensor]) -> Optional[torch.Tensor]:
            return x if x is None or cur_bs == bs else x[unfinished]

        while cur_len < max_len:

            # compute word scores
            tensor = self(
                "fwd",
                tokens=generated[unfinished, :cur_len],
                lengths=gen_len[unfinished],
                causal=True,
                positions=positions[unfinished, :cur_len],
                types=sub(types),
                langs=sub(langs),
                src_enc=sub(src_enc),
                src_len=sub(src_len),
                use_cache=decoding_params.use_cache,
                eval_to_keep_str=decoding_params.dec_gen_to_keep,
                discr=sub(discr),
                discr_mode=discr_mode,
            )
            dec_len = (1 + len(prefix)) if (prefix is not None and first_step) else 1
            assert tensor.size() == (cur_bs, dec_len, self.dim)
            first_step = False
            tensor = tensor[:, -1, :]  # .to(self.dtype)  # (cur_bs, dim)
            scores = self.proj_layer(tensor).float()  # (cur_bs, n_words)
            assert scores.size() == (cur_bs, self.n_words)

            # forbidden words
            if forbidden is not None:
                if forbidden.size() == (self.n_words,):
                    scores[:, forbidden] = -math.inf
                else:
                    assert forbidden.size() == (bs, self.n_words)
                    scores[sub(forbidden)] = -math.inf

            # update scores with temperature / top-k / top-p
            scores = update_scores(
                scores,
                dec_params=decoding_params,
                extra_temperatures=(
                    None
                    if extra_temperatures is None
                    else extra_temperatures[unfinished]
                ),
            )

            # compute log-probs
            logits = F.log_softmax(scores, dim=-1)

            # select next words: sampling or greedy
            if decoding_params.use_sampling:
                next_words = torch.multinomial(logits.exp(), num_samples=1).squeeze(1)
            else:
                next_words = torch.argmax(scores, dim=-1)

            # selected next word scores
            next_scores = logits[arange[:cur_bs], next_words]
            assert next_words.size() == (cur_bs,)

            # if we reach max_len, force `stop_index`
            if cur_len == max_len - 1:
                next_words.fill_(stop_index)

            # update generations / lengths / finished sentences / current length
            generated[unfinished, cur_len] = next_words
            sentence_scores[unfinished] += next_scores
            gen_len[unfinished] += 1
            still_unfinished = next_words != stop_index
            cur_len = cur_len + 1

            # update cache and unfinished
            if not still_unfinished.all():
                unfinished[unfinished.clone()] = still_unfinished
                for k in self.cache.keys():
                    if k != "slen":
                        self.cache[k] = (
                            self.cache[k][0][still_unfinished],
                            self.cache[k][1][still_unfinished],
                        )

            # stop when there is a </s> in each sentence, or if we exceed the maximum length
            cur_bs = unfinished.sum().item()
            if cur_bs == 0:
                break

        assert cur_bs == 0, cur_bs

        # normalize sentence scores
        # gen_len includes the 2 </s> delimitors, so we divide by gen_len - 1
        sentence_scores /= (gen_len - 1) ** length_penalty

        # sanity check
        assert (generated[:, 1:] == stop_index).sum() == bs

        # empty cache (saves a lot of GPU memory)
        self.empty_cache()

        return generated[:, :cur_len], gen_len, sentence_scores

    @torch.no_grad()
    def generate_beam(
        self,
        src_enc: Optional[torch.Tensor],
        src_len: Optional[torch.Tensor],
        decoding_params: DecodingParams,
        types=None,
        langs=None,
        forbidden_tokens: Optional[List[torch.Tensor]] = None,
        discr: Optional[torch.Tensor] = None,
        discr_mode: str = "",
        goal_params: Optional[List[GoalParams]] = None,  # not duplicated
    ):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - None / int / LongTensor(bs)
        `forbidden_tokens`:
            - Optional[List[LongTensor]]. If it's a list, it's a list of length B and shape (n_tokens,)
            listing ids of tokens that should not be output for each element in the batch.
        """
        # stop index (default is the index of "</s>")
        stop_index = self.dico.word2id[decoding_params.stop_symbol]

        # input batch
        assert src_len is not None and src_enc is not None
        bs = len(src_len)
        assert src_enc.shape[0] == bs

        # beam size (can be larger than n_samples)
        stochastic_beam = decoding_params.use_sampling
        beam_size = (
            decoding_params.n_samples
            if decoding_params.beam_size is None
            else decoding_params.beam_size
        )
        assert beam_size >= decoding_params.n_samples >= 1

        # temperature
        extra_temperatures: Optional[torch.Tensor] = None
        if goal_params is not None and goal_params[0].temperature is not None:
            temp = [[p.temperature] * beam_size for p in goal_params]
            extra_temperatures = src_enc.new(sum(temp, []))
            assert len(extra_temperatures) == bs * beam_size

        # length penalty
        if goal_params is not None and goal_params[0].length_penalty is not None:
            length_penalty = [p.length_penalty for p in goal_params]
        else:
            length_penalty = [decoding_params.length_penalty] * bs
        assert len(length_penalty) == bs

        # forbidden tokens
        if forbidden_tokens is not None:
            forbidden_tokens = [x.to(device=src_enc.device) for x in forbidden_tokens]

        # expand to beam size the source latent representations / source lengths
        src_enc = src_enc.repeat(1, beam_size, 1).view(
            (bs * beam_size,) + src_enc.shape[1:]
        )
        src_len = src_len.unsqueeze(1).repeat(1, beam_size).view(-1)

        if discr is not None:
            discr = discr.unsqueeze(1).repeat(1, beam_size).view(-1)

        # generated sentences (batch with beam current hypotheses)
        generated = src_len.new(bs * beam_size, decoding_params.max_gen_len)
        generated.fill_(self.pad_index)  # fill upcoming ouput with <PAD>
        generated[:, 0].fill_(self.eos_index)  # we use <EOS> for <BOS> everywhere

        # positions
        positions = src_len.new(decoding_params.max_gen_len).long()
        positions = (
            torch.arange(decoding_params.max_gen_len, out=positions)
            .unsqueeze(0)
            .expand_as(generated)
        )

        # generated hypotheses
        generated_hyps: List[BeamHypotheses] = [
            BeamHypotheses(
                beam_size,
                decoding_params.max_gen_len,
                length_penalty[i],
                decoding_params.early_stopping,
            )
            for i in range(bs)
        ]

        # scores for each sentence in the beam
        beam_scores = src_enc.new(bs, beam_size).float().fill_(0)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        # mean_perturbation = phi in https://arxiv.org/pdf/1903.06059.pdf
        if stochastic_beam:
            mean_perturbation = beam_scores.clone()

        # current position
        cur_len = 1

        # cache computed states
        self.cache = {"slen": 0}

        # optional sequence prefix
        prefix = decoding_params.prefix

        if prefix is not None:
            assert type(prefix) is list and len(prefix) > 0
            assert all(type(w) is str for w in prefix)
            assert len(prefix) < decoding_params.max_gen_len
            prefix_toks = [self.dico.word2id[w] for w in prefix]
            cur_len += len(prefix_toks)
            generated[:, 1 : 1 + len(prefix_toks)] = torch.LongTensor(prefix_toks)

        # done sentences
        done: List[bool] = [False for _ in range(bs)]

        cur_bs = bs
        sent_ids: List[int] = list(range(bs))
        first_step = True

        while cur_len < decoding_params.max_gen_len:
            # compute word scores
            tensor = self(
                "fwd",
                tokens=generated[:, :cur_len],
                lengths=src_len.new(cur_bs * beam_size).fill_(cur_len),
                causal=True,
                positions=positions[:, :cur_len],
                types=types,
                langs=langs,
                src_enc=src_enc,
                src_len=src_len,
                use_cache=decoding_params.use_cache,
                eval_to_keep_str=decoding_params.dec_gen_to_keep,
                discr=discr,
                discr_mode=discr_mode,
            )
            _slen = (1 + len(prefix)) if (prefix is not None and first_step) else 1
            assert tensor.size() == (cur_bs * beam_size, _slen, self.dim,)
            first_step = False
            tensor = tensor[:, -1, :]  # (cur_bs * beam_size, dim)
            scores = self.proj_layer(tensor).float()  # (cur_bs * beam_size, n_words)

            # forbidden words
            if forbidden_tokens is not None:
                for i, forbidden in enumerate(forbidden_tokens):
                    if len(forbidden) == 0:
                        continue
                    scores[i * beam_size : (i + 1) * beam_size, forbidden] = -math.inf

            # update scores with temperature / top-k / top-p
            scores = update_scores(
                scores,
                dec_params=decoding_params,
                extra_temperatures=extra_temperatures,
            )

            # compute log-probs
            scores = F.log_softmax(scores, dim=-1)  # (cur_bs * beam_size, n_words)
            assert scores.size() == (cur_bs * beam_size, self.n_words)

            # select next words with scores
            if not stochastic_beam:
                scores = beam_scores[:, None] + scores  # (cur_bs * beam_size, n_words)
            else:
                next_mean_perturbation = mean_perturbation[:, None] + scores
                untruncated_perturbation = next_mean_perturbation - torch.log(
                    -torch.rand_like(next_mean_perturbation).log()
                )
                scores = get_truncated_gumbel(
                    beam_scores[:, None], untruncated_perturbation
                )

            next_scores, next_words = get_topk2(scores, cur_bs, beam_size, self.n_words)

            if stochastic_beam:
                next_phi = next_mean_perturbation.view(cur_bs, -1).gather(1, next_words)
                assert next_phi.size() == (cur_bs, 2 * beam_size)
                next_phi = next_phi.tolist()

            assert next_scores.size() == next_words.size() == (cur_bs, 2 * beam_size)
            next_scores, next_words = next_scores.tolist(), next_words.tolist()
            generated_cpu = None

            # next batch beam content
            # list of (bs * beam_size) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []

            batch_indexes: List[int] = []

            # for each sentence
            for row_id in range(cur_bs):
                sent_id = sent_ids[row_id]

                # if we are done with this sentence
                best_possible = (
                    max(next_scores[row_id])
                    if not stochastic_beam
                    else max(next_phi[row_id])
                )
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(
                    best_possible
                )
                if done[sent_id]:
                    continue

                batch_indexes.append(row_id)
                # next sentence beam content
                next_sent_beam = []

                # next words for this sentence
                for i, (idx, value) in enumerate(
                    zip(next_words[row_id], next_scores[row_id])
                ):

                    # get beam and word IDs
                    beam_id = idx // self.n_words
                    word_id = idx % self.n_words

                    # end of sentence, or next word
                    if (
                        word_id == stop_index
                        or cur_len + 1 == decoding_params.max_gen_len
                    ):
                        if generated_cpu is None:
                            generated_cpu = generated.cpu()
                        # for stochastic, next_scores is actually an upper-bound on the sent_log_prob. It's phi we want.
                        sent_log_prob = (
                            value if not stochastic_beam else next_phi[row_id][i]
                        )
                        generated_hyps[sent_id].add(
                            generated_cpu[
                                row_id * beam_size + beam_id, :cur_len
                            ].clone(),
                            sent_log_prob,
                        )
                    elif not stochastic_beam:
                        next_sent_beam.append(
                            (value, word_id, row_id * beam_size + beam_id, 0)
                        )
                    else:
                        next_sent_beam.append(
                            (
                                value,
                                word_id,
                                row_id * beam_size + beam_id,
                                next_phi[row_id][i],
                            )
                        )

                    # the beam for next step is full
                    if len(next_sent_beam) == beam_size:
                        break

                # update next beam content
                assert (
                    len(next_sent_beam) == 0
                    if cur_len + 1 == decoding_params.max_gen_len
                    else beam_size
                )
                if len(next_sent_beam) == 0:
                    next_sent_beam = [
                        (0, self.pad_index, 0, 0)
                    ] * beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == beam_size * len(batch_indexes)

            # stop when we are done with each sentence
            if all(done):
                break

            # sanity check / prepare next batch
            assert len(next_batch_beam) == len(batch_indexes) * beam_size
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_len.new([x[2] for x in next_batch_beam])
            if stochastic_beam:
                mean_perturbation = mean_perturbation.new(
                    [x[3] for x in next_batch_beam]
                )

            batch_idx = src_len.new(batch_indexes)

            # re-order batch and internal states
            cur_bs = len(batch_idx)
            assert cur_bs * beam_size == len(beam_idx)
            sent_ids = [sent_ids[i] for i in batch_idx]
            generated = generated[beam_idx]
            positions = positions[beam_idx]
            generated[:, cur_len] = beam_words

            src_len = src_len[beam_idx]
            src_enc = src_enc[beam_idx]
            if discr is not None:
                discr = discr[beam_idx]

            assert self.cache is not None
            for k in self.cache.keys():
                if k != "slen":
                    self.cache[k] = (
                        self.cache[k][0][beam_idx],
                        self.cache[k][1][beam_idx],
                    )
            if forbidden_tokens is not None:
                forbidden_tokens = [forbidden_tokens[i] for i in batch_idx]
                assert len(forbidden_tokens) == cur_bs
            if extra_temperatures is not None:
                extra_temperatures = extra_temperatures[beam_idx]

            # update current length
            cur_len = cur_len + 1

        # sort sequences by scores / only take n_samples
        for gen_hyps in generated_hyps:
            assert len(gen_hyps.hyp) == beam_size
            gen_hyps.hyp = sorted(gen_hyps.hyp, key=lambda x: x[0], reverse=True)
            if decoding_params.n_samples < beam_size:
                gen_hyps.hyp = gen_hyps.hyp[: decoding_params.n_samples]

        # # beam should contain no duplicates
        # for beam_hyps in generated_hyps:
        #     hyp_toks = [tuple(hyp.tolist()) for _, hyp in beam_hyps.hyp]
        #     assert len(hyp_toks) == len(set(hyp_toks)), hyp_toks

        # empty cache (saves a lot of GPU memory)
        self.empty_cache()

        return generated_hyps

        # visualize hypotheses
        # print([len(x) for x in generated_hyps], cur_len)
        # globals().update( locals() );
        # !import code; code.interact(local=vars())
        # for ii in range(bs):
        #     for ss, ww in sorted(generated_hyps[ii].hyp, key=lambda x: x[0], reverse=True):
        #         print("%.3f " % ss + " ".join(self.dico[x] for x in ww.tolist()))
        #     print("")

        # # select the best hypotheses
        # tgt_len = src_len.new(bs)
        # best = []
        #
        # for i, hypotheses in enumerate(generated_hyps):
        #     best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
        #     tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
        #     best.append(best_hyp)
        #
        # # generate target batch
        # decoded = src_len.new(bs, tgt_len.max().item()).fill_(self.pad_index)
        # for i, hypo in enumerate(best):
        #     decoded[i, : tgt_len[i] - 1] = hypo
        #     decoded[i, tgt_len[i] - 1] = stop_index
        #
        # # sanity check
        # assert (decoded[:, 1:] == stop_index).sum() == bs
        #
        # return decoded, tgt_len, generated_hyps

    def compute_critic(
        self,
        src_enc: torch.Tensor,
        src_len: torch.Tensor,
        freeze_params: str = "",
        q: Optional[torch.Tensor] = None,
        target_toks: Tuple[str, ...] = (SOLVABLE_WORD, NON_SOLVABLE_WORD),
    ):
        """
        return log p(idx) given the src_enc and a first token critic_index
        @param src_enc: the encoding of the src sentence
        @param src_len: the lengths of the src sentences
        @param q
        @param target_toks: the list of tokens for which we'll extract the scores
        """
        bs = src_enc.size(0)
        assert len(src_len) == bs
        assert freeze_params in ["", "enc", "encdec"]
        idx = [self.dico.word2id[x] for x in target_toks]

        # optionally freeze encoder
        src_enc_ = src_enc if freeze_params == "" else src_enc.detach()

        # decode
        decoded = self(
            "fwd",
            tokens=torch.full_like(src_len, self.dico.critic_index)[:, None],
            lengths=torch.ones_like(src_len),
            causal=True,
            positions=torch.zeros_like(src_len)[:, None],
            types=None,
            langs=None,
            src_enc=src_enc_,
            src_len=src_len,
            use_cache=False,
        )
        assert decoded.size() == (bs, 1, self.dim)
        decoded = decoded[:, -1, :]  # (bs, dim)

        # optionally freeze decoder
        decoded_ = decoded.detach() if freeze_params == "encdec" else decoded

        # compute critic
        scores = self.proj_layer(decoded_)  # (bs, n_words)
        to_grab = torch.tensor(idx, device=scores.device)
        solvable = scores[:, to_grab]
        lprobs = F.log_softmax(solvable, dim=-1)  # (bs, n_idx)
        assert lprobs.size() == (bs, len(idx))
        if q is None:
            return lprobs, None

        # compute loss
        target = torch.cat([q[:, None], 1 - q[:, None]], 1)
        loss = -torch.sum(target * lprobs, 1).mean()  # y.log x + (1-y) * log (1-x)

        # pytorch hack to ensure all parameters are updated
        if freeze_params == "enc":
            loss = loss + 0 * src_enc.sum()
        elif freeze_params == "encdec":
            loss = loss + 0 * src_enc.sum() + 0 * decoded.sum()

        return lprobs, loss


class BeamHypotheses(object):
    def __init__(
        self, n_hyp: int, max_len: int, length_penalty: float, early_stopping: bool
    ):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_len = max_len - 1  # ignoring <BOS>
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.n_hyp = n_hyp
        self.hyp: List[Tuple[float, torch.Tensor]] = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp: torch.Tensor, sum_logprobs: float) -> None:
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted(
                    [(s, idx) for idx, (s, _) in enumerate(self.hyp)]
                )
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            return (
                self.worst_score
                >= best_sum_logprobs / self.max_len ** self.length_penalty
            )


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    loss = (policy * log_policy).sum(-1).mean()
    return loss


def smoothed_labels_xe_loss(scores, target, epsilon, reduction):
    """
    Cross-entropy loss, with smoothed labels.

    https://fairseq.readthedocs.io/en/latest/_modules/fairseq/criterions/label_smoothed_cross_entropy.html
    https://github.com/tascj/kaggle-kuzushiji-recognition/blob/master/mmdetection/mmdet/models/losses/cross_entropy_loss.py#L11
    """
    assert scores.dim() == 2
    assert target.dim() == 1
    assert scores.size(0) == target.size(0)
    assert 0 <= epsilon < 1
    assert reduction in ["mean", "sum", "none"]
    assert scores.dtype == torch.float32

    # regular cross-entropy loss
    if epsilon == 0:
        return F.cross_entropy(scores, target, reduction=reduction)

    bs, n_classes = scores.size()
    lprobs = F.log_softmax(scores, dim=-1)
    target = target.unsqueeze(-1)
    eps_i = epsilon / (n_classes - 1)

    nll_loss = -lprobs.gather(dim=-1, index=target).squeeze(-1)
    smooth_loss = -lprobs.sum(dim=-1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss

    assert loss.size() == (bs,)
    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()

    return loss


def kl_div_loss(scores, tgt_scores, kl_temp):
    """
    KL div loss of for scores tensors.
    Note that F.kl_div(x.log, y) = KL_DIV(y,x)
    """
    assert kl_temp > 0
    assert scores.dim() == 2 and tgt_scores.dim() == 2
    assert scores.shape == tgt_scores.shape
    tgt_probs = F.softmax(tgt_scores.float() / kl_temp, dim=-1)
    lprobs = F.log_softmax(scores.float() / kl_temp, dim=-1)
    loss = F.kl_div(lprobs, tgt_probs, reduction="batchmean")
    return loss


def distillation_hidden_states_loss(
    tensors: List[torch.Tensor],
    tgt_tensors: List[torch.Tensor],
    linear_mapping: nn.Linear,
):
    assert isinstance(tensors, list) and isinstance(tgt_tensors, list)
    # n_layers
    assert len(tgt_tensors) in [2 * len(tensors), len(tensors)]
    if 2 * len(tensors) == len(tgt_tensors):
        tgt_tensors = tgt_tensors[::2]
    assert all(tensors[0].shape == t.shape for t in tensors)
    assert all(tgt_tensors[0].shape == t.shape for t in tgt_tensors)
    assert tensors[0].shape[0] == tgt_tensors[0].shape[0]  # bs
    assert tensors[0].shape[1] == tgt_tensors[0].shape[1]  # slen

    mse = nn.MSELoss()
    loss = 0
    for i in range(len(tensors)):
        loss += mse(linear_mapping(tensors[i]), tgt_tensors[i].detach())
    return loss


if __name__ == "__main__":

    def test():

        bs = 64
        beam_size = 17
        n_words = 65432
        n_iter = 10

        with torch.no_grad():

            for device in ["cpu", "cuda"]:
                print(device)

                scores = torch.randn(bs * beam_size, n_words)
                scores = scores.to(device=torch.device(device))

                start = time.time()
                for _ in range(n_iter):
                    ns1, ni1 = get_topk1(scores, bs, beam_size, n_words)
                    torch.cuda.synchronize()
                print(f"topk-1: {time.time() - start:.04}s")

                start = time.time()
                for _ in range(n_iter):
                    ns2, ni2 = get_topk2(scores, bs, beam_size, n_words)
                    torch.cuda.synchronize()
                print(f"topk-2: {time.time() - start:.04}s")

                print((ns1 - ns2).abs().sum().item())
                print((ni1 - ni2).abs().sum().item())

    test()
