# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from logging import getLogger
import math

from evariste.model.data.dictionary import EOS_WORD, EOU_WORD
from params import Params, ConfStore
from functools import lru_cache


logger = getLogger()


@lru_cache()
def eval_layers_to_keep(s: str, n_layers: int) -> List[bool]:
    """
    Parse layers to keep with layer drop at inference.
    ''       -> keep all layers
    '0,3,7'  -> keep layers 0, 3, and 7
    '0::3'   -> keep every 3 layers
    '0::0.2' -> remove every 5 layers
    """
    assert n_layers >= 1
    if s == "":
        return [True for _ in range(n_layers)]
    if "::" in s:
        start_str, step_str = s.split("::")
        start = int(start_str)
        assert 0 <= start < n_layers
        if step_str.isdigit():
            step_int = int(step_str)
            to_keep = list(range(n_layers))[start::step_int]
        else:
            step_float = float(step_str)
            assert 0 < step_float < 1
            step_int = math.floor(1 / step_float)
            to_keep = [i for i in range(n_layers) if (i + start) % step_int != 0]
    else:
        to_keep = [int(x) for x in s.split(",")]
        assert len(to_keep) == len(set(to_keep))
        assert all(0 <= x < n_layers for x in to_keep)
    assert len(to_keep) == len(set(to_keep)) > 0
    return [i in to_keep for i in range(n_layers)]


@lru_cache()
def get_n_samples_per_depth(s: str) -> Optional[Tuple[str, float]]:
    """
    Parse input format of `f` where `n_samples = f(depth)`.
    Each function has a single float parameter.
    """
    if s == "":
        return None
    method, alpha_str = s.split(",")
    alpha = float(alpha_str)
    assert method in ["linear", "exponential", "cosine"] and alpha > 0
    return method, alpha


@dataclass  # modified in build_modules
class ModelArgs(Params):
    fp16: bool = field(default=False, metadata={"help": "Run model with float16"})
    emb_dim: int = field(default=512, metadata={"help": "Embedding layer size"})
    enc_emb_dim: int = field(default=-1, metadata={"help": "Encoder dimension"})
    dec_emb_dim: int = field(default=-1, metadata={"help": "Decoder dimension"})
    n_layers: int = field(default=4, metadata={"help": "Number of Transformer layers"})
    n_heads: int = field(default=8, metadata={"help": "Number of Transformer heads"})
    xav_init: bool = field(
        default=False,
        metadata={"help": "Xavier type initialization for Multi-head attention layers"},
    )
    rescale_embeddings: bool = field(
        default=False, metadata={"help": "Rescale embeddings"}
    )
    dropout: float = field(default=0, metadata={"help": "Dropout"})
    attention_dropout: float = field(
        default=0, metadata={"help": "Dropout in the attention layer"}
    )
    activation_dropout: float = field(
        default=0, metadata={"help": "Dropout in the FFN layer"}
    )
    gelu_activation: bool = field(
        default=False, metadata={"help": "Use a GELU activation instead of ReLU"}
    )
    share_inout_emb: bool = field(
        default=True, metadata={"help": "Share input and output embeddings"}
    )
    share_all_emb: bool = field(
        default=False, metadata={"help": "Share embeddings across modules"}
    )
    sinusoidal_embeddings: bool = field(
        default=False, metadata={"help": "Use sinusoidal embeddings"}
    )
    dec_n_layers: int = field(
        default=-1,
        metadata={
            "help": "Number of Transformer layers to use for the decoder, n_layers is used "
            "if dec_n_layers == -1"
        },
    )
    enc_n_layers: int = field(
        default=-1,
        metadata={
            "help": "Number of Transformer layers to use for the encoder, n_layers is used "
            "if dec_n_layers == -1"
        },
    )
    emb_n_layers: int = field(
        default=-1,
        metadata={
            "help": "Number of Transformer layers to use for the embedder, n_layers is used "
            "if dec_n_layers == -1"
        },
    )
    # choices=["first", "mean", "max"],
    tf_build_emb: str = field(
        default="first",
        metadata={
            "help": "How to build sentence embedding from transformer output : {first|mean|max}"
        },
    )

    layer_dropout: float = field(
        default=0, metadata={"help": "Layer dropout rate"},
    )
    enc_layer_dropout: float = field(
        default=-1,
        metadata={
            "help": "Layer dropout rate in the encoder, layer_dropout is used if set to -1"
        },
    )
    dec_layer_dropout: float = field(
        default=-1,
        metadata={
            "help": "Layer dropout rate in the decoder, layer_dropout is used if set to -1"
        },
    )
    min_layers: int = field(
        default=0, metadata={"help": "Minimum number of layers to not drop"},
    )
    enc_min_layers: int = field(
        default=-1,
        metadata={
            "help": "Minimum number of layers to not drop in the encoder, min_layers is used if set to -1"
        },
    )
    dec_min_layers: int = field(
        default=-1,
        metadata={
            "help": "Minimum number of layers to not drop in the decoder, min_layers is used if set to -1"
        },
    )

    # multihead attention scaling factor / logits regularization
    mha_learn_scaling: bool = field(
        default=False,
        metadata={"help": "Learn the scaling factor in multihead attention layers"},
    )
    mha_init_scaling: Optional[float] = field(
        default=None,
        metadata={
            "help": "Initial value for the scaling parameter in multihead attention layers"
        },
    )

    # apply a first 1d convolution on encoder inputs to reduce dimension
    enc_conv_kernel: int = -1
    enc_conv_stride: int = -1

    def _check_and_mutate_args(self):
        # embedding dimension
        self.enc_emb_dim = self.enc_emb_dim if self.enc_emb_dim > -1 else self.emb_dim
        self.dec_emb_dim = self.dec_emb_dim if self.dec_emb_dim > -1 else self.emb_dim
        assert self.emb_dim > 0
        assert self.enc_emb_dim > 0
        assert self.dec_emb_dim > 0

        # number of layers
        self.emb_n_layers = (
            self.emb_n_layers if self.emb_n_layers > -1 else self.n_layers
        )
        self.enc_n_layers = (
            self.enc_n_layers if self.enc_n_layers > -1 else self.n_layers
        )
        self.dec_n_layers = (
            self.dec_n_layers if self.dec_n_layers > -1 else self.n_layers
        )
        assert self.n_layers > 0
        assert self.emb_n_layers > 0
        assert self.enc_n_layers > 0
        assert self.dec_n_layers > 0

        # layer dropout
        self.enc_layer_dropout = (
            self.enc_layer_dropout
            if self.enc_layer_dropout > -1
            else self.layer_dropout
        )
        self.dec_layer_dropout = (
            self.dec_layer_dropout
            if self.dec_layer_dropout > -1
            else self.layer_dropout
        )
        assert 0 <= self.layer_dropout < 1
        assert 0 <= self.enc_layer_dropout < 1
        assert 0 <= self.dec_layer_dropout < 1

        # minimum number of layers to not drop
        self.enc_min_layers = (
            self.enc_min_layers if self.enc_min_layers > -1 else self.min_layers
        )
        self.dec_min_layers = (
            self.dec_min_layers if self.dec_min_layers > -1 else self.min_layers
        )
        # assert 0 <= self.min_layers <= self.n_layers
        assert 0 <= self.enc_min_layers <= self.enc_n_layers
        assert 0 <= self.dec_min_layers <= self.dec_n_layers

        assert self.emb_dim % self.n_heads == 0
        assert self.enc_emb_dim == -1 or self.enc_emb_dim % self.n_heads == 0
        assert self.dec_emb_dim == -1 or self.dec_emb_dim % self.n_heads == 0

        # parameters sharing
        assert not self.share_all_emb or self.share_inout_emb
        assert not self.share_all_emb or self.enc_emb_dim == self.dec_emb_dim

        assert (self.enc_conv_kernel > 0) == (self.enc_conv_stride >= 0)
        assert self.enc_conv_kernel >= self.enc_conv_stride

        # multihead attention scaling factor / logits regularization
        assert (self.mha_init_scaling is None) == (not self.mha_learn_scaling)
        assert (self.mha_init_scaling is None) or (self.mha_init_scaling > 0)

    def update_nlayers_dim(self, n_layers, emb_dim):
        assert n_layers > 0
        assert emb_dim > 0
        self.dec_n_layers = n_layers
        self.dec_emb_dim = emb_dim


ConfStore["model_default"] = ModelArgs(
    emb_dim=512,
    emb_n_layers=6,
    enc_n_layers=6,
    dec_n_layers=6,
    n_heads=8,
    gelu_activation=False,
    share_inout_emb=True,
    sinusoidal_embeddings=False,
    tf_build_emb="first",
    fp16=True,
)


@dataclass
class DecodingParams(Params):
    """Stores parameters for generate / generate beam"""

    # number of samples / sequence lengths
    max_gen_len: int = field(
        default=1024, metadata={"help": "Max generated tactic length"}
    )
    n_samples: int = field(
        default=10, metadata={"help": "Number of samples (beam size when using a beam)"}
    )
    beam_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Beam size. If None, is set to n_samples. "
                "Allows to have large beams and select a smaller number of sequences."
            )
        },
    )

    # greedy / sampling / beam decoding
    use_beam: bool = field(default=True, metadata={"help": "Use beam"})
    use_sampling: bool = field(
        default=False,
        metadata={
            "help": "If beam, stochastic beam vs regular beam, else, sampling vs greedy decoding."
        },
    )
    length_penalty: float = field(
        default=1.0, metadata={"help": "Generation length penalty (beam only)"}
    )
    early_stopping: bool = field(
        default=True, metadata={"help": "Early stopping (beam only)"}
    )

    # modify scores with temperature
    fixed_temperature: Optional[float] = field(
        default=None, metadata={"help": "Divide scores by a fixed temperature"},
    )
    target_entropy_temperature: Optional[float] = field(
        default=None,
        metadata={"help": "Adapt the temperature to set a target entropy"},
    )
    ignore_higher_entropy: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Do not modify the distribution when the entropy is already above the target one"
        },
    )

    # filtering (ignore tokens with low scores)
    top_k: Optional[int] = field(
        default=None, metadata={"help": "Only consider words in the top-k"}
    )
    top_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "Only consider words that contribute to a fraction p of the probability mass."
        },
    )
    top_p_min_tokens: Optional[int] = field(
        default=None,
        metadata={
            "help": "When top_p is set, ensure that a minimum number of words are considered."
        },
    )

    # prefixes / sequence delimiters
    stop_symbol: str = field(
        default=EOS_WORD, metadata={"help": "Early stopping"}
    )  # if None, use the index of EOS_WORD = "</s>"
    has_success_token: bool = field(
        default=False,
        metadata={
            "help": "Ignore success token in prover decoding to produce correct tactics"
        },
    )
    prefix_success: bool = field(
        default=False, metadata={"help": "Set first generated token to success"}
    )
    prefix: Optional[List[str]] = field(
        default=None, metadata={"help": "Generated sequence prefixes"},
    )

    # layer drop
    enc_gen_to_keep: Optional[str] = field(
        default=None,
        metadata={
            "help": "Layers to keep in the encoder with layer drop during generation"
        },
    )
    dec_gen_to_keep: Optional[str] = field(
        default=None,
        metadata={
            "help": "Layers to keep in the decoder with layer drop during generation"
        },
    )

    # various parameters
    use_cache: bool = field(
        default=True, metadata={"help": "Use cache in transformer when decoding"}
    )
    cpu: bool = False
    precision: str = field(
        default="half",
        metadata={
            "help": "half, float, or double. double is useful to reproduce MCTS evals"
        },
    )

    # q conditioning
    q_conditioning: str = ""
    q_conditioning_inference_ts: float = field(
        default=1.0,
        metadata={
            "help": (
                "Q conditioning mode: At inference time, we generate tactics conditioned "
                "on Q=q avec q sampled between [q_conditioning_inference_ts,1]"
            )
        },
    )

    # decoding parameters that vary based on input sequences (e.g. input node depths)
    n_samples_per_depth_str: str = ""

    # if > 0, reloads a cond_embeddings module for conditioning
    n_conditioning_classes: int = 0
    conditioning_prop: float = 0
    conditioning_input_mode: str = "sum"
    conditioning_strategy: str = "random"

    @property
    def n_samples_per_depth(self) -> Optional[Tuple[str, float]]:
        return get_n_samples_per_depth(self.n_samples_per_depth_str)

    @property
    def uuid(self):
        return "_".join(
            str(x)
            for x in [
                self.n_samples,
                self.use_beam,
                self.max_gen_len,
                self.length_penalty,
                self.early_stopping,
                self.fixed_temperature,
                self.top_k,
            ]
        )

    def __post_init__(self):

        # conditioning
        assert self.n_conditioning_classes >= 0
        assert 0 <= self.conditioning_prop <= 1
        if self.n_conditioning_classes > 0:
            assert self.conditioning_prop > 0
        if self.conditioning_prop < 1:
            assert self.conditioning_input_mode == "sum"
        assert self.conditioning_input_mode in ["prefix", "sum"]
        assert self.conditioning_strategy in ["fixed", "random", "split"]

        # layers to keep (10 is just to verify the strings' correctness)
        if self.enc_gen_to_keep is not None:
            _ = eval_layers_to_keep(self.enc_gen_to_keep, 10)
        if self.dec_gen_to_keep is not None:
            _ = eval_layers_to_keep(self.dec_gen_to_keep, 10)

        # number of samples
        assert type(self.n_samples) is int and self.n_samples >= 1
        if self.beam_size is not None:
            assert self.use_beam and self.beam_size >= self.n_samples

        # beam with 1 sample does not really make sense
        # (but can be useful for sanity checks)
        if self.use_beam and self.n_samples == 1:
            logger.warning("Using beam with n_samples == 1")

        # if n_samples > 1, we must be using beam, or sampling, or both
        if self.n_samples > 1 and not self.use_beam and not self.use_sampling:
            raise RuntimeError(
                f"Decoding with n_samples={self.n_samples} is not compatible "
                f"with use_beam=False and use_sampling=False."
            )

        # if we use a greedy decoding (no beam + no sampling, or beam with 1 sample),
        # modifying token scores will not have any impact on the results
        if (
            self.use_beam is False
            and self.use_sampling is False
            or self.use_beam
            and ((self.n_samples if self.beam_size is None else self.beam_size) == 1)
        ):
            assert self.fixed_temperature is None
            assert self.target_entropy_temperature is None
            assert self.ignore_higher_entropy is None
            assert self.top_k is None
            assert self.top_p is None
            assert self.top_p_min_tokens is None

        # temperature (cannot be defined in two different ways)
        assert self.fixed_temperature is None or self.target_entropy_temperature is None
        assert (
            self.target_entropy_temperature is None
            or self.target_entropy_temperature > 0
        )
        assert (
            self.ignore_higher_entropy is None
            or self.target_entropy_temperature is not None
            and type(self.ignore_higher_entropy) is bool
        )

        # filtering (top-k / top-p)
        assert self.top_k is None or self.top_k >= 2
        assert self.top_p is None or 0 < self.top_p < 1
        assert (
            self.top_p_min_tokens is None
            or self.top_p is not None
            and self.top_p_min_tokens >= 2
        )
        assert not (
            self.top_k is not None
            and self.top_p_min_tokens is not None
            and self.top_k < self.top_p_min_tokens
        )

        # prefix
        if self.prefix is not None:
            assert type(self.prefix) is list and len(self.prefix) > 0
            assert all(type(x) is str for x in self.prefix)

        assert self.q_conditioning in ["", "sum", "prefix"]
        assert 0 <= self.q_conditioning_inference_ts <= 1

        # precision. float is not enough to reproduce large MCTS evals
        assert self.precision in ["half", "float", "double"]

        # for now, variable n_samples is only implemented for generate (without beam)
        assert self.n_samples_per_depth_str == "" or not self.use_beam


ConfStore["decoding_greedy"] = DecodingParams(
    max_gen_len=1024,
    n_samples=1,
    use_beam=False,
    use_sampling=False,
    fixed_temperature=None,
    stop_symbol=EOU_WORD,
)
ConfStore["decoding_bwd_eval"] = DecodingParams(
    max_gen_len=1024,
    n_samples=8,
    use_beam=True,
    use_sampling=False,
    fixed_temperature=None,
    stop_symbol=EOU_WORD,
)
ConfStore["decoding_fast"] = DecodingParams(
    max_gen_len=1024,
    n_samples=8,
    use_beam=False,
    use_sampling=True,
    fixed_temperature=1.0,
    stop_symbol=EOU_WORD,
)
ConfStore["decoding_slow"] = DecodingParams(
    max_gen_len=1024,
    n_samples=32,
    use_beam=False,
    use_sampling=True,
    fixed_temperature=1.0,
    stop_symbol=EOU_WORD,
)
