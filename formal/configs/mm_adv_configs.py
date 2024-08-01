# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Dict, Optional

from params import ConfStore
from configs.adv_configs import AdversarialConfig
from configs.bwd_configs import _bwd_config
from evariste.adversarial.prover import AdvProverKind
from evariste.trainer.args import TrainerArgs
from configs.fwd_configs import fwd_mm_config


_models: Dict[str, str] = {}  # add your models here


def mm_adversarial_backward_prover_config(
    debug: bool = False, from_scratch: str = "", multi_task: bool = False
) -> TrainerArgs:
    mm_fmt = "goal--label-mandsubst-EOU-theorem-predsubst"
    task = f"mm_bwd_{mm_fmt}_rl"
    sl_task = f"mm_x2y_{mm_fmt}_seq2seq"
    if multi_task:
        task += f",{sl_task}"
    cfg = _bwd_config(task=task, debug=debug)

    if from_scratch:
        cfg.reload_model = ""
    if debug:
        cfg.bwd_proving_eval_str = ""
    else:
        cfg.bwd_proving_eval_str = "mm:valid"

    cfg.stopping_criterion = f""
    cfg.validation_metrics = f""
    return cfg


def bigger_bwd_model(cfg: TrainerArgs, reload_arxiv=True):
    cfg.accumulate_gradients = 1
    cfg.model.fp16 = True
    cfg.batch.size = 16
    cfg.model.emb_n_layers = 6
    cfg.model.enc_n_layers = 12
    cfg.model.dec_n_layers = 6
    cfg.batch.tokens = 3000
    cfg.model.enc_emb_dim = 1600
    cfg.model.dec_emb_dim = 1024
    cfg.model.n_heads = 8
    cfg.model.dropout = 0.3
    cfg.model.layer_dropout = 0
    cfg.model.enc_layer_dropout = -1
    cfg.model.dec_layer_dropout = -1
    cfg.model.min_layers = 0
    cfg.model.enc_min_layers = -1
    cfg.model.dec_min_layers = -1
    cfg.model.attention_dropout = 0
    cfg.model.activation_dropout = 0.1
    cfg.model.gelu_activation = False
    cfg.model.share_inout_emb = True
    cfg.model.share_all_emb = False
    cfg.model.sinusoidal_embeddings = False
    cfg.model.tf_build_emb = "first"
    cfg.n_th_to_prove = 500
    cfg.stopping_criterion = ""
    cfg.batch.max_len = 1024
    cfg.env_base_seed = -1
    cfg.num_workers = 1
    cfg.optimizer = "adam_inverse_sqrt,warmup_updates=10000,lr=0.0001,weight_decay=0.01"
    cfg.clip_grad_norm = 5
    cfg.epoch_size = 500000
    cfg.max_epoch = 10000

    return cfg


def _adversarial_cfg(
    debug: bool = False,
    local: bool = False,
    prover_pretrained: Optional[str] = None,
    gen_pretrained: Optional[str] = None,
    data: str = "mm_new3",
    train_prover_with_generated_only: bool = False,
    condition_generator: bool = False,
    fixed_prover: bool = False,
    fixed_generator: bool = False,
    restore_optimizer_for_gen: bool = False,
    restore_optimizer_for_prover: bool = False,
    bwd_prover: bool = False,
    bwd_big_model: bool = False,
    bwd_mcts_data: bool = False,
) -> AdversarialConfig:
    assert not (fixed_generator and fixed_prover)
    use_zip = not bwd_prover

    def _common_updates(cfg_: TrainerArgs):
        if data == "mm_inequal2":
            cfg_.mm.dataset = ConfStore["inequal2"]
            cfg_.mm.graph.axioms_only = True
        else:
            cfg_.mm.dataset = ConfStore["new3"]
            if debug:
                cfg_.mm.dataset = ConfStore["new3_100"]
            assert data == "mm_new3", data
        cfg_.rl_params.replay_buffer.min_len = 1000 if not debug else 10
        if use_zip:
            cfg_.rl_params.replay_buffer.refresh_every = 10_000 if not debug else 1000
        cfg_.stopping_criterion = ""
        cfg_.mm.stop_action = True

    def _restore_opti(cfg_: TrainerArgs):
        ckpt = cfg_.reload_model
        cfg_.reload_checkpoint = ckpt
        cfg_.reload_model = ""

    def _make_gen_cfg() -> TrainerArgs:
        gen_cfg_ = fwd_mm_config(
            debug=debug,
            task="mm_gen_seq2seq",
            is_generation=True,
            reload_model=_models[gen_pretrained] if gen_pretrained else "",
        )
        if restore_optimizer_for_gen:
            _restore_opti(gen_cfg_)
        _common_updates(gen_cfg_)
        gen_cfg_.no_eval_on_train = True
        gen_cfg_.mm.graph.generated_prob = 0.5
        if condition_generator:
            gen_cfg_.mm.cond_gen_with_proved = True
            gen_cfg_.rl_params.replay_buffer.filter_if_rewards_zero = False
        else:
            gen_cfg_.rl_params.replay_buffer.filter_if_rewards_zero = True
        return gen_cfg_

    def _make_fwd_cfg() -> TrainerArgs:
        assert prover_pretrained is not None
        fwd_cfg_ = fwd_mm_config(
            debug=debug,
            task="mm_fwd_seq2seq",
            is_generation=False,
            reload_model=_models[prover_pretrained] if prover_pretrained else "",
        )
        if restore_optimizer_for_prover:
            _restore_opti(fwd_cfg_)
        _common_updates(fwd_cfg_)
        fwd_cfg_.no_eval_on_train = True
        fwd_cfg_.rl_params.replay_buffer.filter_if_rewards_zero = True
        if train_prover_with_generated_only:
            fwd_cfg_.mm.graph.generated_prob = 1.0
        else:
            fwd_cfg_.mm.graph.generated_prob = 0.5
        return fwd_cfg_

    def _make_bwd_cfg() -> TrainerArgs:
        assert prover_pretrained is not None
        bwd_cfg_ = mm_adversarial_backward_prover_config(
            debug=debug,
            from_scratch=_models[prover_pretrained],
            multi_task=not train_prover_with_generated_only,
        )
        bwd_cfg_.validation_metrics = "valid-mm-proven-greedy"
        if bwd_big_model:
            bigger_bwd_model(bwd_cfg_, reload_arxiv=False)
        if bwd_mcts_data:
            bwd_cfg_.tasks += ",mm_mcts_goal--label-mandsubst-EOU-theorem-predsubst"
            bwd_cfg_.mcts_train.jsonl_data_dir = "MCTS_DATA_PATH"
        if restore_optimizer_for_prover:
            _restore_opti(bwd_cfg_)
        _common_updates(bwd_cfg_)
        bwd_cfg_.rl_params.replay_buffer.filter_if_rewards_zero = True
        return bwd_cfg_

    num_prover_trainers = 1
    fixed_prover_path = ""
    if fixed_prover:
        assert not bwd_prover
        assert prover_pretrained
        num_prover_trainers = 0
        fixed_prover_path = _models[prover_pretrained]

    num_gen_trainers = 1
    fixed_gen_path = ""
    if fixed_generator:
        assert gen_pretrained
        num_gen_trainers = 0
        fixed_gen_path = _models[gen_pretrained]

    overwrite_slurm_world_size = 4 if (debug and local) else -1
    if debug and local and (fixed_prover or fixed_generator):
        overwrite_slurm_world_size = 3

    return AdversarialConfig(
        env_name="mm",
        prover_trainer_args=_make_bwd_cfg() if bwd_prover else _make_fwd_cfg(),
        generator_trainer_args=_make_gen_cfg(),
        num_prover_trainers=num_prover_trainers,
        num_generator_trainers=num_gen_trainers,
        num_prover_actors=1,
        num_generator_actors=1,
        prover_kind=(
            AdvProverKind.BackwardGreedy if bwd_prover else AdvProverKind.ForwardGreedy
        ),
        debug=debug,
        overwrite_slurm_world_size=overwrite_slurm_world_size,
        exp_name="debug_adversarial" if (debug and local) else "",
        fixed_prover=fixed_prover_path,
        fixed_generator=fixed_gen_path,
    )
