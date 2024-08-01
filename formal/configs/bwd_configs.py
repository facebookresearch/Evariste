# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path
import os

from params import ConfStore
from evariste.utils import COND_TOK
from evariste.model.utils import reload_ckpt
from evariste.trainer.args import TrainerArgs
from evariste.model.data.dictionary import DicoConf


def _bwd_config(
    task: str, debug: bool = False, other_tasks: str = "", reload_model: str = ""
) -> TrainerArgs:

    if other_tasks:
        tasks = ",".join([task, other_tasks])
    else:
        tasks = task

    lang = task.split("_")[0]
    eval_split = "identities" if lang == "eq" else "valid"
    assert lang in ["eq", "mm", "hl", "lean", "isabelle"]

    cfg: TrainerArgs = ConfStore["default_cfg"]
    cfg.exp_name = "debug"
    cfg.tasks = tasks
    cfg.stopping_criterion = f"_valid-{task}-tok-ppl,50"
    cfg.validation_metrics = f"_valid-{task}-tok-ppl"
    cfg.bwd_proving_eval_str = f"{lang}:{eval_split}"
    cfg.optimizer = "adam_inverse_sqrt,warmup_updates=10000,lr=0.0001,weight_decay=0.01"

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


def register_bwd_cfgs():

    # Fabian debug
    cfg = _bwd_config("eq_gen_graph_offline_seq2seq", True, reload_model="",)
    cfg.eq.dataset = ConfStore["eq_dataset_lean"]
    cfg.eq.dataset.offline_dataset_path = ""
    cfg.eq.dataset.offline_dataset_splits = "0.99,0.01,0.0"
    cfg.stopping_criterion = "_valid-eq_gen_graph_offline_seq2seq-tok-ppl,10"
    cfg.validation_metrics = "_valid-eq_gen_graph_offline_seq2seq-tok-ppl"
    cfg.fwd_proving_eval_str = ""
    cfg.bwd_proving_eval_str = ""
    cfg.generation_eval_str = "eq:train"
    cfg.generation_eval_gen_args.prover_params.beam_path = ""
    cfg.generation_eval_gen_args.n_gens = 10
    cfg.generation_eval_gen_args.sequential = True
    cfg.generation_eval_gen_args.dataset_conf = ConfStore["eq_dataset_lean"]
    cfg.generation_eval_gen_args.forward_cfg.max_nodes = 20
    cfg.generation_eval_gen_args.forward_cfg.max_generations = 60
    cfg.generation_eval_gen_args.forward_cfg.max_cons_inv_allowed = 60
    cfg.generation_eval_gen_args.save_chunk_length = 10
    cfg.epoch_size = 1
    cfg.no_eval_on_test = True
    ConfStore["bwd_fabian"] = cfg

    cfg = _bwd_config("eq_gen_graph_offline_seq2seq", True, reload_model="",)
    cfg.eq.dataset = ConfStore["eq_dataset_lean"]
    cfg.eq.dataset.offline_dataset_path = "PATH/generations/"
    cfg.eq.dataset.offline_dataset_splits = "0.7,0.3,0.0"
    cfg.stopping_criterion = "_valid-eq_gen_graph_offline_seq2seq-tok-ppl,10"
    cfg.validation_metrics = "_valid-eq_gen_graph_offline_seq2seq-tok-ppl"
    cfg.fwd_proving_eval_str = ""
    cfg.bwd_proving_eval_str = ""
    cfg.generation_eval_str = ""
    cfg.epoch_size = 1
    cfg.no_eval_on_test = True
    cfg.eq.proved_conditioning = True
    from evariste.model.data.envs.node_sampling_strategy import NodeSamplingStrategy

    cfg.eq.dataset.fwd_offline_sampling_strategy = NodeSamplingStrategy.SamplingMinimal
    ConfStore["bwd_offline_gen"] = cfg

    # Lean
    cfg = _bwd_config("lean_x2y_statement--tactic-EOU_seq2seq", False)
    cfg.lean = ConfStore["default_lean"]
    cfg.bwd_proving_eval_str = "lean:valid"
    ConfStore["bwd_lean"] = cfg

    # Lean (debug)
    cfg = _bwd_config("lean_x2y_statement--tactic-EOU_seq2seq", True)
    cfg.lean = ConfStore["default_lean"]
    cfg.bwd_proving_eval_str = "lean:valid"
    ConfStore["bwd_lean_debug"] = cfg

    # Lean PACT (debug)
    cfg = _bwd_config("lean_pact_seq2seq", True)
    cfg.lean = ConfStore["default_lean"]
    cfg.bwd_proving_eval_str = "lean:valid"
    ConfStore["bwd_pact_lean_debug"] = cfg

    # Lean PACT (no debug)
    cfg = _bwd_config("lean_pact_seq2seq", False)
    cfg.lean = ConfStore["default_lean"]
    cfg.bwd_proving_eval_str = "lean:valid"
    ConfStore["bwd_pact_lean"] = cfg

    # Isabelle (debug)
    cfg = _bwd_config("isabelle_x2y_statement--tactic-EOU_seq2seq", True)
    cfg.isabelle = ConfStore["default_isabelle"]
    cfg.bwd_proving_eval_str = ""
    ConfStore["bwd_isabelle_debug"] = cfg

    # eq (debug)
    ConfStore["bwd_eq_debug"] = _bwd_config(task="eq_bwd_rwalk_seq2seq", debug=True)

    # eq (no debug)
    ConfStore["bwd_eq_test"] = _bwd_config(task="eq_bwd_rwalk_seq2seq", debug=False)

    # HOL-Light
    ConfStore["bwd_hl_debug"] = _bwd_config(
        task="hl_x2y_goal--tactic-EOU-subgoals_seq2seq", debug=True
    )

    # Metamath
    ConfStore["bwd_mm_debug"] = _bwd_config(
        task="mm_x2y_goal--label-mandsubst-EOU-theorem-predsubst_seq2seq", debug=True
    )
