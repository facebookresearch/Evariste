# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, Any, List, Dict, OrderedDict, Union
from logging import getLogger
from pathlib import Path
import os
import copy
import math
import numpy as np
import torch
import collections
from torch import nn

from evariste.model.transformer import TransformerModel
from evariste.model.data.dictionary import Dictionary
from evariste.trainer.args import TrainerArgs
from evariste.trainer.migrations import migrate_train_args
from evariste.refac.utils import safe_load
from params import ConfStore

logger = getLogger()


def show_batch(logger, to_print, dico, example_type):

    """
    log first element of batch.
    to print = {label:batch}
    """
    logger.info("")
    logger.info(f"========== {example_type} example ==========")
    for label, x in to_print:
        source_sentence = " ".join(
            [dico.id2word[int(w)] for w in x[0] if w != dico.pad_index]
        )
        logger.info(f"{label} sent: {source_sentence}")
    logger.info("")


def assert_equal_state_dict(state_dict1, state_dict2):
    assert state_dict1.keys() == state_dict2.keys()
    for k, v in state_dict1.items():
        assert torch.equal(state_dict2[k], v)


def reload_ckpt(
    ckpt_path: Path, only_model_params: bool = True
) -> Tuple[TrainerArgs, Dictionary, Dict[str, Any]]:

    reloaded = safe_load(ckpt_path, map_location="cpu")
    params = migrate_train_args(reloaded["params"])
    assert isinstance(params, TrainerArgs)

    # only reload model parameters
    if only_model_params:
        cfg: TrainerArgs = ConfStore["default_cfg"]
        cfg.model = params.model
        cfg.dico = params.dico
        params = cfg

    # build dictionary
    dico = Dictionary(
        id2word=reloaded["dico_id2word"],
        word2id=reloaded["dico_word2id"],
        counts=reloaded["dico_counts"],
    )
    assert params.dico == dico.conf, f"{params.dico} // {dico.conf}"

    return params, dico, reloaded


def load_from_checkpoint(
    path: Path, module_names: List[str], device: str, fp16: bool = False
) -> Tuple[Dict[str, Union[TransformerModel, nn.Linear]], Dictionary, TrainerArgs]:
    assert path is not None
    params, dico, reloaded = reload_ckpt(path)
    loaded_modules: Dict[str, OrderedDict[str, torch.Tensor]] = {}
    for module in module_names:
        loaded_modules[module] = collections.OrderedDict(
            {
                (k[len("module.") :] if k.startswith("module.") else k): v
                for k, v in reloaded[module].items()
            }
        )
    # build dictionary / update parameters
    assert params.dico == dico.conf, f"{params.dico, dico.conf}"

    # build model / reload weights
    params.model.fp16 = fp16
    modules: Dict[str, Union[TransformerModel, nn.Linear]] = {}
    for module, loaded_module in loaded_modules.items():
        if module == "encoder":
            modules[module] = TransformerModel(
                params.model,
                dico,
                is_encoder=True,
                with_output=True,
                n_layers=params.model.enc_n_layers,
            )
        elif module == "decoder":
            modules[module] = TransformerModel(
                params.model,
                dico,
                is_encoder=False,
                with_output=True,
                n_layers=params.model.dec_n_layers,
            )
        elif module == "big_decoder":
            dist_params = copy.deepcopy(params.model)
            dist_params.update_nlayers_dim(
                params.distillation.n_layers, params.distillation.emb_dim
            )
            modules[module] = TransformerModel(
                dist_params,
                dico,
                is_encoder=False,
                with_output=True,
                n_layers=dist_params.dec_n_layers,
            )
        elif module == "cond_embeddings":
            # n_conditioning_classes isn't reloaded because of reload_ckpt only_model args = True by default
            n_classes = loaded_module["weight"].shape[1]
            params.cond.n_classes = n_classes
            modules[module] = nn.Linear(n_classes, params.model.dec_emb_dim, bias=False)
        else:
            raise RuntimeError(f"Unknown module {module}")
        modules[module].load_state_dict(loaded_module)

    dtype = torch.float16 if fp16 else torch.float32
    for name in module_names:
        assert name == "cond_embeddings" or modules[name].dtype == dtype
        modules[name].to(device=torch.device(device))
        modules[name].eval()

    return modules, dico, params


def check_dicts(tgt_dict, src_dict):
    """
    Will check dicts have same keys.
    And that objects they contain are of same types
    and same shape (if appliable).
    """
    # check keys
    assert tgt_dict.keys() == src_dict.keys()
    # check types
    assert all(isinstance(tgt_dict[k], type(src_dict[k]),) for k in tgt_dict.keys())
    # if object are tensors check shapes
    for k in tgt_dict.keys():
        if isinstance(tgt_dict[k], torch.Tensor):
            assert tgt_dict[k].shape == src_dict[k].shape


def reload_optimizer_state(
    tgt_state, src_state, n_params: int, tgt_ind: int = 0, src_ind: int = 0
):
    """
    Reload optimizer state from src state to tgt state.
    Optimizers state may not have the same numbers of parameters.
    as they may come from trainings with different modules.
    In that case, we give the starting index in both the tgt and src state dicts.
    """
    for i in range(n_params):
        # checks state dictn
        assert tgt_ind + i in tgt_state
        assert src_ind + i in src_state
        check_dicts(tgt_state[tgt_ind + i], src_state[src_ind + i])
        # assign
        for k, v in src_state[src_ind + i].items():
            if isinstance(v, torch.Tensor):
                tgt_state[tgt_ind + i][k].copy_(v)
            else:
                tgt_state[tgt_ind + i][k] = v


def load_module_state_dict(
    module: torch.nn.Module,
    module_name: str,
    data: Dict[str, OrderedDict[str, torch.Tensor]],
    params: TrainerArgs,
):
    """
    Reload module state dict safely.
    """
    checkpoint_multi_gpu = all(
        [x.startswith("module.") for x in data[module_name].keys()]
    )
    if checkpoint_multi_gpu and not params.slurm_conf.multi_gpu:
        logger.info("Reloading muli-gpu on single-gpu. stripping .module.")
        data[module_name] = remove_module_prefix(data[module_name])
    if not checkpoint_multi_gpu and params.slurm_conf.multi_gpu:
        logger.info("Reloading single-gpu on multi-gpu. adding .module.")
        data[module_name] = add_module_prefix(data[module_name])
    module.load_state_dict(data[module_name])


def to_cuda(*args, device: Optional[str] = None):
    """
    Move tensors to CUDA.
    """
    if device is None:
        y = [None if x is None else x.cuda() for x in args]
    else:
        y = [None if x is None else x.to(device) for x in args]
    return y[0] if len(y) == 1 else y


def batch_sequences(sequences, pad_index):
    """
    Batch a list of sequences.
    """
    lengths = torch.LongTensor([len(seq) for seq in sequences])
    batch = torch.full(
        size=(len(lengths), lengths.max().item()),
        fill_value=pad_index,
        dtype=torch.long,
    )
    for i, (seq, slen) in enumerate(zip(sequences, lengths)):
        assert len(seq) == slen
        batch[i, :slen] = seq
    return batch, lengths


@torch.no_grad()
def get_knn_pytorch(embeddings, indices, k, ignore_self=True):
    """
    Input:
        - embeddings matrix of size (m, d)
        - N indices of vectors for which we want to know the nearest neighbors
        - number of nearest neighbors
        - do not return the same indices
    Output:
        - `scores`  matrix of size (N, k) with nearest neighors scores
        - `indices` matrix of size (N, k) with nearest neighors indices
    """
    scores = embeddings.mm(embeddings[indices].t())
    if ignore_self:
        scores[indices] = -math.inf
    scores, indices = scores.topk(k=k, dim=0, largest=True, sorted=True)
    return scores.t(), indices.t()


def get_embs(x, xlen, method: str):
    """
    Return an embedding from a transformer output.
    """
    assert x.dim() == 3
    assert xlen.dim() == 1 and len(xlen) == len(x)
    assert method in ["first", "mean", "max"]
    if method == "first":
        return x[:, 0]
    alen = torch.arange(xlen.max(), dtype=torch.long, device=xlen.device)
    mask = alen[None] >= xlen[:, None]
    if method == "mean":
        x[mask] = 0
        return x.sum(1) / xlen[:, None]
    if method == "max":
        x[mask] = -math.inf
        return x.max(1).values


def create_subseq_pos(seqs, labels, offset: int = 0):
    """
    Given a list of sequences with their labels, return
    a dictionary with the sub-sequences positions.
    """
    assert len(labels) == len(seqs)
    lengths = [offset] + [len(s) for s in seqs]
    idx = list(np.cumsum(lengths))
    pos = list(zip(idx[:-1], idx[1:]))
    assert len(pos) == len(labels)
    return dict(zip(labels, pos))


def create_subseq_mask(pos: List[Tuple[int, int]], slen: int) -> torch.Tensor:
    """
    Given a list of tuples of positions, create a mask for these tokens.
    Input:
        [[0, 3], [2, 4]], 6
    Output:
        torch.BoolTensor([
            [1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0]
        ])
    """
    assert type(pos) is list
    assert all(0 <= a <= b <= slen for a, b in pos)
    alen = torch.arange(slen, dtype=torch.long)
    a = torch.LongTensor([a for a, _ in pos])
    b = torch.LongTensor([b for _, b in pos])
    l_mask = alen[None] >= a[:, None]
    r_mask = alen[None] < b[:, None]
    mask = l_mask & r_mask
    assert mask.long().sum(1).eq(b - a).all()
    return mask


def create_subseq_masks(subseq_pos, ylen):
    """
    Create sub-sequence masks. `full` will contain the full sequence.
    The optional other masks should contain everything, be non-overlapping,
    and will have a sum equal to `full` (except for the last <EOS> token).
    """
    # bs = len(ylen)
    ylen_max = ylen.max().item()
    for pos, l in zip(subseq_pos, ylen.tolist()):
        for x, y in pos.items():
            pos[x] = (y[0], y[1])
        pos["full"] = (0, l - 1)

    names = list(subseq_pos[0].keys())
    masks = {
        name: create_subseq_mask([v[name] for v in subseq_pos], ylen_max)
        for name in names
    }
    if len(masks) > 1:
        non_full_sum = sum(v.long() for k, v in masks.items() if k != "full")
        assert isinstance(non_full_sum, torch.Tensor)
        assert non_full_sum.max().item() == 1
        assert non_full_sum.sum(1).eq(masks["full"].sum(1) - 1).all()
    return names, masks


def remove_module_prefix(x, prefix: str = "module."):
    """
    Remove the potential 'module.' prefix of multi-GPU state dictionaries.
    """
    if all([k.startswith(prefix) for k in x.keys()]):
        x = {k[len(prefix) :]: v for k, v in x.items()}
    return x


def add_module_prefix(x, prefix: str = "module."):
    """
    Add a 'module.' prefix for multi-GPU state dictionaries.
    """
    return {(prefix + k): v for k, v in x.items()}


def get_path_bwd_proving_eval(
    root,
    lang: str,
    split: str,
    epoch: Optional[int] = None,
    decoder_type: str = "decoder",
) -> str:
    """
    Return the folder where the bwd proving eval will be stored for a given epoch.
    If epoch is none, return the folder containing all epochs.
    During evalution res and model will be stored there.
    During MCTS model will be loaded from there.
    """
    assert decoder_type in ["decoder", "big_decoder"]
    name = "bwd_prover_eval" if decoder_type == "decoder" else "bwd_prover_eval_big"
    if epoch is None:
        return os.path.join(root, name, lang, split)
    else:
        return os.path.join(root, name, lang, split, str(epoch))
