# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from typing import Dict, Optional
import os
import copy
import torch
from copy import deepcopy

from evariste.model.data.dictionary import Dictionary
from evariste.model.pointer_network import PointerNetwork
from evariste.model.transformer import (
    DECODER_ONLY_PARAMS,
    ENCODER_CONVOL_PARAMS,
    TRANSFORMER_TOK_EMB_PARAMS,
    TransformerModel,
)
from evariste.model.utils import remove_module_prefix
from evariste.refac.utils import safe_load
from evariste.trainer.args import TrainerArgs


logger = getLogger()


def get_dico_and_pretrained(params: TrainerArgs):
    """
    Reload pretrained models, and their associated dictionaries.
    """
    pretrained = {}
    ignored = 0
    for name, path in params.parsed_reload_model.items():
        # reload model
        assert os.path.isfile(path)
        assert name in params.module_names
        logger.info(f"Reloading {name} model from {path} ...")
        reloaded = safe_load(path, map_location="cpu")

        # reload dictionary
        dico = Dictionary(
            reloaded["dico_id2word"],
            reloaded["dico_word2id"],
            reloaded["dico_counts"],
            frozen=True,
        )
        logger.info(f"Found {len(dico)} pretrained words in reloaded {name}.")

        k: Optional[str] = name
        if name not in reloaded:
            if name in ["decoder", "discriminator"]:
                k = "encoder"
            elif name == "big_decoder":
                k = "decoder"
            elif name == "target_encoder":
                k = None
                ignored += 1
            elif name == "cond_embeddings":
                k = None
                ignored += 1
        if k is not None:
            pretrained[name] = {
                "model": remove_module_prefix(reloaded[k]),
                "dico": dico,
            }

    assert len(pretrained) + ignored == len(params.parsed_reload_model)
    return pretrained


def reload_embeddings(
    tgt_state_dict: Dict,
    tgt_dico: Dictionary,
    src_state_dict: Dict,
    src_dico: Dictionary,
    no_proj_layer: bool = False,
):
    """
    Reload weights of token embeddings of source model into target model.
    """
    assert isinstance(src_dico, Dictionary)
    assert isinstance(tgt_dico, Dictionary)
    logger.info("Reloading embeddings ...")

    assert len(tgt_dico) == tgt_state_dict["embeddings.weight"].shape[0]

    src_words = set(src_dico.word2id.keys())
    tgt_words = set(tgt_dico.word2id.keys())
    if src_words != tgt_words and "proj_layer.weight" in tgt_state_dict:
        src_only = ", ".join(sorted(src_words - tgt_words)[:20])
        tgt_only = ", ".join(sorted(tgt_words - src_words)[:20])
        logger.warning(
            f"There is a vocabulary mismatch and proj_layer in current model. "
            f"This will cause a logits prediction mismatch.\n"
            f"Source words: {len(src_dico)}\n"
            f"Target words: {len(tgt_dico)}\n"
            f"Source & Target words: {len(src_words & tgt_words)}\n"
            f"Source only: {src_only}\n"
            f"Target only: {tgt_only}"
        )

    tensors = []
    for t in TRANSFORMER_TOK_EMB_PARAMS:
        if no_proj_layer and t.startswith("proj_layer"):
            continue
        try:
            tgt_weights = tgt_state_dict[t]
            tensors.append((tgt_weights, src_state_dict[t]))
        except KeyError:
            logger.warning(f"Parameters {t} not found in model.")
            continue

    n_found = 0
    n_found_yttm = 0
    not_found = []
    for tgt_id, word in tgt_dico.id2word.items():
        # look for source word in the pretrained model. if we cannot find it, look
        # whether we can find it by adding or removing the YouTokenToMe prefix ▁
        src_id = None
        if word in src_dico.word2id:
            src_id = src_dico.word2id[word]
        elif "▁" + word in src_dico.word2id:
            src_id = src_dico.word2id["▁" + word]
            n_found_yttm += 1
        elif word[0] == "▁" and word[1:] in src_dico.word2id:
            src_id = src_dico.word2id[word[1:]]
            n_found_yttm += 1
        if src_id is not None:
            n_found += 1
            for tgt_tensor, src_tensor in tensors:
                tgt_tensor[tgt_id] = src_tensor[src_id]
        else:
            not_found.append(word)

    weighted_total = sum(tgt_dico.counts.values())
    weighted_not_found = sum(tgt_dico.counts[w] for w in not_found)
    weighted_found = weighted_total - weighted_not_found
    logger.info(
        f"Embeddings for {n_found}/{len(tgt_dico)} words "
        f"({100 * n_found / len(tgt_dico):.2f}%) were properly reloaded "
        f"({100 * weighted_found / weighted_total:.2f}% including occurrences counts), "
        f"{n_found_yttm}/{n_found} using YTTM prefix."
    )
    if len(not_found) > 0:
        logger.warning(
            f"Dico mismatch: Could not reload {len(not_found)} words: "
            f"{not_found[0:5]} ... {not_found[-5:]}. "
            f"({100 * weighted_not_found / weighted_total:.2f}% including "
            f"occurrences counts). Init Random."
        )


def reload_model(tgt_model, tgt_dico: Dictionary, src: Dict, model_type: str) -> None:
    """
    Reload weights of source model into target model.
    """
    logger.info(f"======= Reloading {model_type} ...")
    tgt_state_dict = tgt_model.state_dict()

    # reload src token embeddings into tgt token embeddings
    if model_type not in ["classifier", "cond_embeddings"]:
        reload_embeddings(
            tgt_state_dict,
            tgt_dico,
            src["model"],
            src["dico"],
            no_proj_layer=(model_type == "target_encoder"),
        )

    # reload other parameters
    if model_type == "encoder":
        allowed_missing_params = ENCODER_CONVOL_PARAMS
    elif model_type == "decoder":
        allowed_missing_params = [
            name % i for name in DECODER_ONLY_PARAMS for i in range(tgt_model.n_layers)
        ]
    else:
        allowed_missing_params = []

    # check that all source parameters exist in the target
    not_in_tgt = [k for k in src["model"].keys() if k not in tgt_state_dict]
    if len(not_in_tgt) > 0:
        raise Exception(
            f"Unexpected parameters found in the model to reload: {not_in_tgt}"
        )

    for k, tgt_weights in tgt_state_dict.items():
        if k in TRANSFORMER_TOK_EMB_PARAMS:
            continue
        if k not in src["model"]:
            if k in allowed_missing_params:
                logger.warning(
                    f"{model_type} param {k} not found in pretrained model. Init Random."
                )
            else:
                raise RuntimeError(
                    f"{model_type} param {k} not found in pretrained model."
                )
        tgt_weights.copy_(src["model"][k])
    logger.info(f"Successfully reloaded {model_type}.")


def share_embeddings(modules: Dict[str, torch.nn.Module]) -> None:
    """
    Share all embeddings across modules.
    """
    ref_mods = [k for k in ["encoder", "decoder", "embedder"] if k in modules]
    logger.info(f"Sharing all embeddings across {', '.join(ref_mods)} ...")
    if len(ref_mods) == 0:
        logger.warning("No module with embeddings to share!")
    else:
        ref_mod = ref_mods[0]
        m1 = getattr(modules[ref_mod], "embeddings")
        m2 = getattr(modules[ref_mod], "proj_layer")
        assert m1.weight.eq(m2.weight).all()
        if len(ref_mods) == 1:
            logger.warning(f"Only one module has embeddings to share: {ref_mod}")
        for k in ref_mods[1:]:
            logger.info(f"Setting {k} embeddings with {ref_mod}")
            getattr(modules[k], "embeddings").weight = m1.weight
            getattr(modules[k], "proj_layer").weight = m1.weight


@torch.no_grad()
def build_modules(dico: Dictionary, params: TrainerArgs) -> Dict[str, torch.nn.Module]:
    """
    Build modules and optionally reload pretrained models.
    """
    logger.info("\n=========== Building models ===========")
    modules: Dict[str, torch.nn.Module] = {}

    # build encoder
    if "encoder" in params.module_names:
        encoder = TransformerModel(
            params.model,
            dico,
            is_encoder=True,
            with_output=True,
            n_layers=params.model.enc_n_layers,
            layer_dropout=params.model.enc_layer_dropout,
            min_layers=params.model.enc_min_layers,
        )
        modules["encoder"] = encoder

    if "target_encoder" in params.module_names:
        te_params = deepcopy(params.model)
        te_params.share_inout_emb = False
        te_params.dropout = 0
        te_params.attention_dropout = 0
        te_params.activation_dropout = 0
        te_params.layer_dropout = 0
        te_params.enc_layer_dropout = 0
        te_params.dec_layer_dropout = 0
        if params.cond.small_tgt_encoder:
            te_params.n_heads = 8
            te_params.emb_dim = 512
            te_params.enc_emb_dim = 512
            te_params.dec_emb_dim = 512
            te_params.n_layers = 6

        target_encoder = TransformerModel(
            te_params,
            dico,
            is_encoder=True,
            with_output=True,
            n_words_output=params.cond.n_classes,
            n_layers=te_params.enc_n_layers,
        )
        modules["target_encoder"] = target_encoder

    if "cond_embeddings" in params.module_names:
        modules["cond_embeddings"] = torch.nn.Linear(
            params.cond.n_classes, params.model.dec_emb_dim, bias=False
        )

    # build decoder
    if "decoder" in params.module_names:
        decoder = TransformerModel(
            params.model,
            dico,
            is_encoder=False,
            with_output=True,
            n_layers=params.model.dec_n_layers,
            layer_dropout=params.model.dec_layer_dropout,
            min_layers=params.model.dec_min_layers,
        )
        modules["decoder"] = decoder

    # build classifier
    if "classifier" in params.module_names:
        classifier = torch.nn.Linear(params.model.enc_emb_dim, 1)
        modules["classifier"] = classifier

    # build critic
    if "critic" in params.module_names:
        critic = torch.nn.Linear(params.model.enc_emb_dim, 1)
        modules["critic"] = critic

    # build embedder
    if "embedder" in params.module_names:
        embedder = TransformerModel(
            params.model,
            dico,
            is_encoder=True,
            with_output=False,
            n_layers=params.model.emb_n_layers,
            layer_dropout=params.model.layer_dropout,
            min_layers=params.model.min_layers,
        )
        modules["embedder"] = embedder

    # build discriminator
    if "discriminator" in params.module_names:
        discriminator = TransformerModel(
            params.model,
            dico,
            is_encoder=True,
            with_output=False,
            n_layers=params.model.enc_n_layers,
            layer_dropout=params.model.enc_layer_dropout,
            min_layers=params.model.enc_min_layers,
        )
        modules["discriminator"] = discriminator

    # build pointer
    if "pointer" in params.module_names:
        pointer = PointerNetwork(
            params.model.dec_emb_dim,
            params.pn.proj_x,
            params.pn.proj_y,
            params.pn.proj_s,
        )
        modules["pointer"] = pointer

    # build decoder that we want to distill
    if "big_decoder" in params.module_names:
        assert "decoder" in params.module_names
        dist_params = copy.deepcopy(params.model)
        dist_params.update_nlayers_dim(
            params.distillation.n_layers, params.distillation.emb_dim
        )
        big_decoder = TransformerModel(
            dist_params,
            dico,
            is_encoder=False,
            with_output=True,
            n_layers=dist_params.dec_n_layers,
            layer_dropout=dist_params.dec_layer_dropout,
            min_layers=dist_params.dec_min_layers,
        )
        modules["big_decoder"] = big_decoder

    if "hidden_states_linear" in params.module_names:
        assert "big_decoder" and "decoder" in params.module_names
        modules["hidden_states_linear"] = torch.nn.Linear(decoder.dim, big_decoder.dim)
    if "embedding_linear" in params.module_names:
        assert "big_decoder" and "decoder" in params.module_names
        modules["embedding_linear"] = torch.nn.Linear(decoder.dim, big_decoder.dim)

    # model summary
    for k, v in modules.items():
        n_params = sum(p.numel() for p in v.parameters() if p.requires_grad)
        logger.info(f"Number of parameters ({k}): {n_params}")

    # reload pretrained modules
    logger.info("=========== Reloading pretrained models ===========")
    pretrained = get_dico_and_pretrained(params)
    for name, model in modules.items():
        if name in pretrained:
            reload_model(model, dico, pretrained[name], name)

    # share embeddings accross modules
    if params.model.share_all_emb:
        share_embeddings(modules)

    logger.info("")

    return {k: v.cuda() for k, v in modules.items()}
