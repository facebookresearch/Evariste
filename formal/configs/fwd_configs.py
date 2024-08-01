# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path
import os
import copy

from params import ConfStore
from evariste.model.utils import reload_ckpt
from evariste.forward.fwd_lean.training.lean_graph_sampler import LEAN_FWD_TASK
from evariste.forward.fwd_hl.hl_graph_sampler import HL_FWD_TASK
from evariste.trainer.args import TrainerArgs


def _base_fwd_config(
    debug: bool, reload_model: str, task: str, other_tasks: str = ""
) -> TrainerArgs:
    if other_tasks:
        tasks = ",".join([task, other_tasks])
    else:
        tasks = task

    # default training args
    cfg: TrainerArgs = ConfStore["default_cfg"]
    cfg.tasks = tasks
    cfg.num_workers = 1
    cfg.epoch_size = 40000
    cfg.max_epoch = 100000
    cfg.batch.max_len = 2048
    cfg.batch.tokens = 15000
    cfg.batch.size = 8
    cfg.optimizer = "adam_inverse_sqrt,warmup_updates=10000,lr=0.0002,weight_decay=0.01"
    cfg.gpu_oom_retry = -1  # doesn't seems to work anymore with pytorch v1.8.1
    cfg.no_eval_on_test = True

    # reload model args
    if reload_model != "":
        assert os.path.isfile(reload_model)
        ckpt_train_params, _, _ = reload_ckpt(Path(reload_model))
        cfg.model = ckpt_train_params.model
        cfg.reload_model = reload_model
    cfg.model.fp16 = True

    if debug:
        cfg.epoch_size = 20000
        cfg.max_epoch = 100
        cfg.debug.debug = debug

    return cfg


def fwd_mm_config(
    debug: bool = False,
    task: str = "mm_fwd_seq2seq",
    legacy: bool = False,
    is_generation: bool = False,
    reload_model: str = "",
    stop_action: bool = False,
    other_tasks: str = "",
) -> TrainerArgs:
    cfg = _base_fwd_config(
        debug=debug, task=task, reload_model=reload_model, other_tasks=other_tasks
    )
    cfg.mm.dataset = ConfStore["new3"]
    cfg.mm.graph.remap_label = False
    cfg.mm.stop_action = stop_action
    cfg.mm.graph.topo_sort_version = 2

    if legacy:
        # legacy, for keeping bwd compatibility with previous sweep files
        # were model is a 6 layers without convolutions
        cfg.model.enc_n_layers = 6
        cfg.model.enc_conv_kernel = -1
        cfg.model.enc_conv_stride = -1
        if not debug:
            cfg.epoch_size = 40000
            cfg.batch.tokens = 17500

    if not is_generation:
        cfg.stopping_criterion = ""
        cfg.validation_metrics = (
            f"mm_valid_fwd_proving-prop_proved,valid-{task}-full-seq-acc"
        )
        cfg.fwd_proving_eval_str = "mm:valid"
    else:
        cfg.mm.graph.sample_goal = False
        cfg.stopping_criterion = ""
        cfg.validation_metrics = f"_valid-{task}-tok-ppl,valid-{task}-full-seq-acc"
        cfg.fwd_proving_eval_str = ""

    if debug:
        cfg.mm.dataset = ConfStore["new3_100"]

    return cfg


def fwd_hl_config(debug: bool = False, task: str = HL_FWD_TASK) -> TrainerArgs:
    # Mutating a config, it's bad but convenient
    # Can probably be buggy if a parameter name change (this might be silent)
    cfg = _base_fwd_config(debug=debug, task=task, reload_model="")
    cfg.hl = ConfStore["hl_plus_default_args"]
    cfg.stopping_criterion = f"valid-{task}-full-seq-acc,100"
    cfg.validation_metrics = f"valid-{task}-full-seq-acc"
    cfg.fwd_proving_eval_str = ""
    return cfg


def fwd_lean_config(debug: bool = False, task: str = LEAN_FWD_TASK) -> TrainerArgs:
    from evariste.datasets.lean import LEAN_DATASET_LATEST_FWD

    cfg = _base_fwd_config(debug=debug, task=task, reload_model="")
    cfg.lean.dataset = copy.deepcopy(LEAN_DATASET_LATEST_FWD)

    cfg.lean.graph.insert_noise_prob = 0.2

    cfg.stopping_criterion = ""
    cfg.validation_metrics = f"_valid-{task}-tok-ppl"
    cfg.fwd_proving_eval_str = "lean:valid"
    if debug:
        cfg.model.enc_n_layers = 2
        cfg.model.dec_n_layers = 2
    return cfg


def fwd_eq_config(
    debug: bool = False,
    task: str = "eq_fwd_graph_seq2seq",
    is_generation: bool = False,
    reload_model: str = "",
    stop_action: bool = False,
    other_tasks: str = "",
    fwd_eval: bool = True,
) -> TrainerArgs:
    cfg = _base_fwd_config(
        debug=debug, task=task, reload_model=reload_model, other_tasks=other_tasks
    )
    cfg.eq.dataset = ConfStore["eq_dataset_exp_trigo_hyper"]
    if fwd_eval:
        if not is_generation:
            cfg.fwd_proving_eval_str = "eq:identities"
            cfg.stopping_criterion = f"fwd-identities-eq-proven-greedy,100"
            cfg.validation_metrics = f"fwd-identities-eq-proven-greedy"
        else:
            cfg.stopping_criterion = f"_valid-{task}-tok-ppl,100"
            cfg.validation_metrics = f"_valid-{task}-tok-ppl"
    cfg.eq.stop_action = stop_action
    return cfg


def register_fwd_cfgs():
    # TODO move this in a register function in another file?
    # MM - FWD - LEGACY
    ConfStore["fwd"] = fwd_mm_config(task="mm_fwd_graph2subst_seq2seq", legacy=True)
    ConfStore["fwd_debug"] = fwd_mm_config(
        task="mm_fwd_graph2subst_seq2seq", debug=True, legacy=True
    )
    # MM - FWD
    ConfStore["fwd_mm"] = fwd_mm_config()
    ConfStore["fwd_mm_debug"] = fwd_mm_config(debug=True)

    # MM - GEN
    ConfStore["gen_mm"] = fwd_mm_config(task="mm_gen_seq2seq", is_generation=True)
    ConfStore["gen_mm_debug"] = fwd_mm_config(
        task="mm_gen_seq2seq", debug=True, is_generation=True
    )

    # HL - FWD
    ConfStore["fwd_hl"] = fwd_hl_config()
    ConfStore["fwd_hl_debug"] = fwd_hl_config(debug=True)

    # LEAN - FWD
    ConfStore["fwd_lean_debug"] = fwd_lean_config(debug=True)
    ConfStore["fwd_lean"] = fwd_lean_config()

    # EQ - FWD
    ConfStore["fwd_eq"] = fwd_eq_config(task="eq_fwd_graph_seq2seq")
    ConfStore["fwd_eq_debug"] = fwd_eq_config(debug=True, task="eq_fwd_graph_seq2seq")
    # # EQ - GEN
    # ConfStore["gen_eq_debug"] = fwd_eq_config(
    #     debug=True, task="eq_gen_equality_equiv_seq2seq", is_generation=True
    # )
    # ConfStore["gen_eq"] = fwd_eq_config(
    #     task="eq_gen_equality_equiv_seq2seq", is_generation=True
    # )
    # ConfStore["gen_eq_with_stop"] = fwd_eq_config(
    #     task="eq_gen_equality_equiv_seq2seq", stop_action=True, is_generation=True
    # )
