# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, List
from pathlib import Path
import torch
import logging

from evariste.datasets import DatasetConf
from evariste.forward.forward_prover import ProverConfig, ForwardProver
from evariste.forward.env_specifics.fwd_env_helper import FwdEnvHelper, get_env_name
from evariste.model.data.dictionary import Dictionary
from evariste.model.transformer import TransformerModel
from evariste.model.utils import reload_ckpt
from evariste.trainer.args import TrainerArgs


logger = logging.getLogger()


def from_checkpoint(
    ckpt_path: str,
    device_str: str,
    cfg: ProverConfig,
    overwrite_dataset: Optional[DatasetConf] = None,
    overwrite_max_inp_len: Optional[int] = None,
    prover_dir: Optional[str] = None,
    overwrite_tasks: Optional[str] = None,
    overwrite_prefix: Optional[List[str]] = None,
    overwrite_gen_type: Optional[str] = None,
) -> Tuple[ForwardProver, Dictionary, TrainerArgs, FwdEnvHelper]:
    """
    Helper to create a ForwardProver from a checkpoint.
    In separate file than ForwardProver to remove cyclic dependencies
    """
    params, dico, reloaded = reload_ckpt(Path(ckpt_path), only_model_params=False)
    logging.info(f"Model trained for {reloaded['epoch']} epochs")
    params.model.fp16 = cfg.fp16
    logging.info(f"Fp16: {params.model.fp16}")

    if overwrite_gen_type is not None:
        params.eq.dataset.gen_type = overwrite_gen_type

    use_critic = "critic" in reloaded

    modules = ["encoder", "decoder"]
    if use_critic:
        modules += ["critic"]
    assert "embedder" not in modules
    assert "pointer" not in modules

    for name in modules:
        reloaded[name] = {
            (k[len("module.") :] if k.startswith("module.") else k): v
            for k, v in reloaded[name].items()
        }

    # build model
    encoder = TransformerModel(
        params.model,
        dico,
        is_encoder=True,
        with_output=True,
        n_layers=params.model.enc_n_layers,
    )
    decoder = TransformerModel(
        params.model,
        dico,
        is_encoder=False,
        with_output=True,
        n_layers=params.model.dec_n_layers,
    )

    # reload weights
    encoder.load_state_dict(reloaded["encoder"])
    decoder.load_state_dict(reloaded["decoder"])

    # cuda
    device = torch.device(device_str)
    encoder.eval().to(device=device)
    decoder.eval().to(device=device)

    critic: Optional[torch.nn.Module]
    if use_critic:
        logger.info("Using a critic for forward prover")
        critic = torch.nn.Linear(params.model.enc_emb_dim, 1)
        critic.load_state_dict(reloaded["critic"])
        critic.eval().to(device=device)
    else:
        critic = None

    if overwrite_tasks is not None:
        params.tasks = overwrite_tasks

    env_helper = FwdEnvHelper.from_trainer_args(
        dico, params, overwrite_dataset=overwrite_dataset, prover_dir=prover_dir
    )
    env_specifics = env_helper.get_prover_env_specifics()
    env_specifics.fwd_params.command_prefix = overwrite_prefix

    prover = ForwardProver.from_trainer_args(
        params=params,
        prover_env_specifics=env_specifics,
        cfg=cfg,
        dico=dico,
        # env specifics
        # models
        encoder=encoder,
        decoder=decoder,
        critic=critic,
        overwrite_max_inp_len=overwrite_max_inp_len,
    )

    return prover, dico, params, env_helper


def make_prover_cfg(
    args: TrainerArgs,
    prover_type: str = "greedy",  # greedy, sampling, stochastic_beam
    is_generator: bool = False,
    async_model: bool = False,
) -> ProverConfig:
    from evariste.forward.forward_model_configs import (
        STOCHASTIC_BEAM_PROVER_CFG,
        SAMPLING_PROVER_CFG,
        GREEDY_PROVER_CFG,
    )

    type2config = {
        "greedy": GREEDY_PROVER_CFG,
        "sampling": SAMPLING_PROVER_CFG,
        "stochastic_beam": STOCHASTIC_BEAM_PROVER_CFG,
    }

    assert (
        prover_type in type2config.keys()
    ), f"{prover_type} not in {type2config.keys()}"

    prover_cfg: ProverConfig = type2config[prover_type]
    prover_cfg.is_generator = is_generator
    prover_cfg.fp16 = args.model.fp16
    prover_cfg.async_model = async_model

    if get_env_name(args) == "eq":
        prover_cfg.batch_config.max_batch_size = 256
    elif get_env_name(args) == "mm":
        prover_cfg.batch_config.max_batch_mem = 200_000_000
        prover_cfg.batch_config.max_batch_size = 512
        prover_cfg.search_cfg.n_simultaneous_proofs = 2048
    elif get_env_name(args) == "lean":
        prover_cfg.search_cfg.n_simultaneous_proofs = min(
            prover_cfg.search_cfg.n_simultaneous_proofs, 64
        )
        prover_cfg.batch_config.max_batch_size = min(
            prover_cfg.batch_config.max_batch_size, 64
        )

    if args.debug.debug:
        prover_cfg.batch_config.max_batch_mem = 100_000_000
        prover_cfg.batch_config.max_batch_size = 16

    if is_generator and args.debug.debug:
        logging.warning(
            "Changing generator config for debug:\n"
            "\tsetting max_nodes to 15\n"
            "\tsetting n_simultaneous_proofs to 32\n"
            # f"\tsetting max_cons_inv_allowed to 1\n"
        )
        prover_cfg.search_cfg.max_nodes = 15
        prover_cfg.search_cfg.n_simultaneous_proofs = 32
        # prover_cfg.search_cfg.max_cons_inv_allowed = 1

    if (not is_generator) and args.debug.debug:
        logging.warning(
            "Changing prover config for debug:\n"
            "\tsetting max_nodes to 15\n"
            "\tsetting n_simultaneous_proofs to 32\n"
            "\tsetting max_cons_inv_allowed to 2\n"
        )
        prover_cfg.search_cfg.max_nodes = 15
        prover_cfg.search_cfg.n_simultaneous_proofs = 32
        prover_cfg.search_cfg.max_cons_inv_allowed = 2

    return prover_cfg
