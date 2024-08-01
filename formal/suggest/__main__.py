# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Union, Tuple, List, Dict, Any
from evariste import json as json
from abc import abstractmethod
from pathlib import Path
import sys
import fire
import math
import traceback
import numpy as np

import tornado.ioloop
import tornado.web
from tornado.web import url, RequestHandler

import torch
from torch.nn import functional as F
from evariste.backward.graph import GoalParams

from params import ConfStore
from evariste.logger import create_logger
from evariste.model_zoo import ZOO, ZOOModel
from evariste.datasets import LeanDatasetConf
from evariste.model.utils import to_cuda
from evariste.model.data.dictionary import EOU_WORD, EOS_WORD, Dictionary
from evariste.model.transformer_utils import update_scores, get_clm_mask_target
from evariste.backward.model.beam_search import (
    load_beam_search_from_checkpoint,
    BeamSearchModel,
    DecodingParams,
)
from evariste.backward.env.lean.tokenizer import LeanTokenizer
from evariste.backward.env.lean.graph import LeanTheorem, LeanTactic, LeanContext


logger = create_logger(None)


AVAILABLE_MODELS: Dict[str, ZOOModel] = {
    "DEFAULT_SUGGEST_MODEL": ZOOModel("YOUR_PATH", "lean_v1.1", "lean")
}
TOKENIZERS: Dict[str, LeanTokenizer] = {}
for name, model in AVAILABLE_MODELS.items():
    data_conf: LeanDatasetConf = ConfStore[model.dataset]
    tokenizer = LeanTokenizer.build(data_conf.tokenizer)
    TOKENIZERS[name] = tokenizer

DECODING_PARAMS: DecodingParams = ConfStore["decoding_fast"]
DECODING_PARAMS.n_samples = 8
DECODING_PARAMS.use_sampling = True
DECODING_PARAMS.use_beam = True
DECODING_PARAMS.precision = "half"
DECODING_PARAMS.n_conditioning_classes = 0  # TODO: FIX


TUNABLE_DECODING_PARAMS = {
    "n_samples",
    "use_beam",
    "use_sampling",
    "fixed_temperature",
    "length_penalty",
    "top_k",
    "prefix",
}

DECODING_PARAMS_NOT_WORKING_WITH_CLI = {"top_k", "prefix"}

DEVICE = "cpu"
TEMPLATE_MODELS: Dict[str, BeamSearchModel] = {}


def create_dec_params(
    client_dec_params: Dict, tokenizer: LeanTokenizer
) -> DecodingParams:
    local_dec_params = {**client_dec_params}

    if "prefix" in local_dec_params:
        local_dec_params["prefix"] = tokenizer.encode(local_dec_params["prefix"])
    assert (
        set(local_dec_params) <= TUNABLE_DECODING_PARAMS
    ), f"extra non tunable params: {set(local_dec_params) - TUNABLE_DECODING_PARAMS}"

    new_dec_params = DecodingParams.from_cli(
        param_dict={
            k: v
            for k, v in local_dec_params.items()
            if k not in DECODING_PARAMS_NOT_WORKING_WITH_CLI
        },
        default_instance=DECODING_PARAMS,
        allow_incomplete=True,
    )

    # HACK: some decoding params do not work with from_cli
    for k in DECODING_PARAMS_NOT_WORKING_WITH_CLI:
        # empty list and strings must be set to None
        setattr(new_dec_params, k, local_dec_params[k] or None)
    new_dec_params.check_and_mutate_args()

    return new_dec_params


def instanciate_local_models(client_dec_params: Dict) -> Dict[str, BeamSearchModel]:
    assert TEMPLATE_MODELS, "no model loaded"
    models: Dict[str, BeamSearchModel] = {}
    for i, (name, template_model) in enumerate(TEMPLATE_MODELS.items()):
        decoding_params = create_dec_params(client_dec_params, TOKENIZERS[name])
        if i == 0:
            print(decoding_params)
        # no reload: cannot change decoding_params.precision
        models[name] = BeamSearchModel(
            decoding_params=decoding_params,
            device=DEVICE,
            decoder_type=template_model.decoder_type,
        )
        models[name].encoder = template_model.encoder
        models[name].decoder = template_model.decoder
        models[name].cond_embeddings = template_model.cond_embeddings
        models[name].dico = template_model.dico
    return models


def create_input_tensors(
    tokens: List[str], dico: Dictionary
) -> Tuple[torch.Tensor, torch.Tensor]:

    # prepare inputs
    unk_tokens = [t for t in tokens if t not in dico]
    src_tokens = torch.LongTensor([[dico.index(t) for t in tokens if t in dico]])
    src_len = torch.LongTensor([src_tokens.shape[1]])

    # log UNKs
    if len(unk_tokens) > 0:
        print(f"Found {len(unk_tokens)} unknown tokens: {unk_tokens}")

    return src_tokens.to(DEVICE), src_len.to(DEVICE)


def parse_tactic_state(
    tactic_state: str, dico: Dictionary
) -> Tuple[torch.Tensor, torch.Tensor]:
    goal = LeanTheorem(
        conclusion=tactic_state, state=None, context=LeanContext(namespaces=set())
    )
    tokens: List[str] = [EOS_WORD, *goal.tokenize(), EOS_WORD]
    return create_input_tensors(tokens, dico)


def do_suggest(model: BeamSearchModel, tactic_state: str):
    assert model is not None and model.dico is not None
    dico = model.dico

    print(f"[do_suggest] tactic_state={repr(tactic_state)}")
    print(
        "MODEL ",
        model.cond_embeddings is not None,
        model.decoding_params.n_conditioning_classes,
    )

    # tactic state
    src_tokens, src_len = parse_tactic_state(tactic_state, dico)
    params: Optional[List[GoalParams]] = None
    if model.decoding_params.n_conditioning_classes > 0:
        params = [
            GoalParams(cond_id=int(x))
            for x in np.random.randint(
                0,
                model.decoding_params.n_conditioning_classes,
                size=model.decoding_params.n_samples,
            )
        ]

    # generate tactics
    tac_toks, log_priors, critics = model.do_batch(
        src_len=src_len,
        src_tokens=src_tokens,
        forbiddens=None,
        infos=None,
        params=params,
    )
    assert len(tac_toks) == len(log_priors) == len(critics) == 1
    tactics = [LeanTactic.from_tokens(toks) for toks in tac_toks[0]]
    assert len(tactics) == model.decoding_params.n_samples
    return {
        "critic": math.exp(critics[0]),
        "tactic_infos": [
            {
                "tactic": repr(tactic),
                "prior": math.exp(lp),
                "logPrior": lp,
                "cond_id": None if params is None else params[i].cond_id,
            }
            for i, (lp, tactic) in enumerate(zip(log_priors[0], tactics))
            if tactic.is_valid
        ],
    }


def do_taclogprob(
    model: BeamSearchModel, tactic_state: str, tactic_str: str
) -> Dict[str, Any]:
    assert model is not None and model.dico is not None
    dico = model.dico

    print(
        f"[do_taclogprob] tactic_state={repr(tactic_state)} tactic_str={repr(tactic_str)}"
    )

    # tactic state
    src_tokens, src_len = parse_tactic_state(tactic_state, dico)

    # tactic
    tactic = LeanTactic(tactic=tactic_str)
    tactic_toks = [EOS_WORD, *tactic.tokenize(), EOU_WORD]
    out_tokens, out_len = create_input_tensors(tactic_toks, dico)

    # tokens to predict
    pred_mask, target = get_clm_mask_target(out_tokens, out_len)
    out_tokens, out_len = to_cuda(out_tokens, out_len)
    pred_mask, target = to_cuda(pred_mask, target)

    encoder = model.encoder
    decoder = model.decoder
    assert encoder is not None
    assert decoder is not None

    encoded = encoder("fwd", causal=False, tokens=src_tokens, lengths=src_len)
    decoded = decoder(
        "fwd",
        causal=True,
        tokens=out_tokens,
        lengths=out_len,
        src_enc=encoded,
        src_len=src_len,
    )
    scores = decoder.proj_layer(decoded[pred_mask]).view(-1, decoder.n_words)
    # update scores with temperature / top-k / top-p
    scores = update_scores(
        scores, dec_params=model.decoding_params, extra_temperatures=None
    )
    token_log_probs = -F.cross_entropy(scores, target, reduction="none")
    log_prob = token_log_probs.sum().item()
    return {
        "log_prob": log_prob / (out_len.item() - 1),
        "token_log_probs": token_log_probs.cpu().tolist(),
        "tokens": tactic.tokenize(),
    }


class LocalModelsRequestHandler(RequestHandler):
    @classmethod
    @abstractmethod
    def single_model_process(cls, local_model: BeamSearchModel, data: Dict):
        ...

    def process(self, local_models: Dict[str, BeamSearchModel], data: Dict):
        return {
            "perModel": {
                model_name: type(self).single_model_process(local_model, data)
                for model_name, local_model in local_models.items()
            }
        }

    def post(self):
        try:
            data = json.loads(self.request.body.decode("utf8"))
            client_decoding_params = data["decoding_params"]
            try:
                local_models = (
                    instanciate_local_models(client_decoding_params)
                    if "decoding_params" in data
                    else TEMPLATE_MODELS
                )
                processed_data = self.process(local_models, data)
            except Exception as e:
                processed_data = {"error": f"{type(e).__name__}: {e}"}
                print(
                    f"Exception of type {type(e).__name__} "
                    f"when parsing decoding params: {e}",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exc()
            print(processed_data)
            self.write(processed_data)
        except Exception as e:
            traceback.print_exc()
            print(e)


class SuggestHandler(LocalModelsRequestHandler):
    @classmethod
    def single_model_process(cls, local_model: BeamSearchModel, data: Dict):
        return do_suggest(model=local_model, tactic_state=data["tactic_state"])


class TacLogProbHandler(LocalModelsRequestHandler):
    @classmethod
    def single_model_process(cls, local_model: BeamSearchModel, data: Dict):
        return do_taclogprob(
            model=local_model,
            tactic_state=data["tactic_state"],
            tactic_str=data["tactic"],
        )


def make_app():
    settings = {
        "debug": True,
        "gzip": True,
        "autoreload": True,
    }
    return tornado.web.Application(
        handlers=[
            url(r"/suggest", SuggestHandler),
            url(r"/tac_log_prob", TacLogProbHandler),
        ],
        **settings,
    )


def get_device(device: Optional[str] = None) -> str:
    cuda_is_available = torch.cuda.is_available()
    if device is None:
        print(f"CUDA available: {cuda_is_available}")
        device = "cuda:0" if cuda_is_available else "cpu"
    print(f"Using device {device}")
    assert device == "cpu" or cuda_is_available
    torch.randn(1).to(device=device)  # sanity check
    return device


def main(models: Optional[Union[str, Tuple[str]]] = None, device: Optional[str] = None):
    global DEVICE, TEMPLATE_MODELS

    # device (CPU / GPU)
    DEVICE = get_device(device)

    # parse models to reload
    if models is None:
        model_names = sorted(AVAILABLE_MODELS.keys())
    elif isinstance(models, str):
        model_names = [models]
    else:
        assert isinstance(models, tuple)
        model_names = list(models)
    print(f"Selected models: {model_names}")
    assert len(model_names) == len(set(model_names)) > 0
    assert set(model_names) <= AVAILABLE_MODELS.keys()

    # reload models
    for name in model_names:
        print(f"Reloading model {name} ...")
        TEMPLATE_MODELS[name] = load_beam_search_from_checkpoint(
            path=Path(AVAILABLE_MODELS[name].path),
            decoding_params=DECODING_PARAMS,
            device=DEVICE,
        )

    # start server
    app = make_app()
    app.listen(8000)
    tornado.ioloop.IOLoop.current().start()
    logger.info("Server started.")


if __name__ == "__main__":
    # python -m suggest --device cpu
    # python -m suggest --models NAME1,NAME2 --device cuda
    fire.Fire(main)
