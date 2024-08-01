# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum, unique
from datetime import datetime
from pathlib import Path
import os
import numpy as np

from params import ConfStore, Params, MISSING
from evariste.model_zoo import ZOO, ZOOModel
from evariste.refac.utils import safe_load
from evariste.clusters.utils import clusterify_path, clusterify_partitions
from evariste.backward.env.lean.filter_tactics import TacticFilter
from evariste.backward.remote.search_params_sampler import HyperOptKind
from evariste.datasets.lean import LeanDatasetConf, LEAN_SPLITS
from evariste.trainer.args import TrainerArgs
from evariste.trainer.utils import module_reload_paths
from evariste.model.utils import reload_ckpt, assert_equal_state_dict
from evariste.model.data.dictionary import DicoConf
from evariste.model.data.subproof_args import MCTSSubProofArgs
from evariste.backward.prover.prover_args import BeamSearchKind
from evariste.backward.prover.zmq_prover import ZMQProverParams
from evariste.backward.prover.mcts_samples import ALL_MCTS_SUBTASKS
from evariste.backward.remote.prioritized_label_sampler import LabelSamplerCfg


@dataclass
class MCTSOnlineParams(Params):
    exp_name: str
    train_params: TrainerArgs = field(default_factory=lambda: ConfStore["default_cfg"])
    zmq_prover: ZMQProverParams = field(default_factory=lambda: ConfStore["zmq_fast"])
    train_partition: str = clusterify_partitions("Theorem_Proving")
    no_trainer: bool = False
    n_gpu_trainer: int = 1
    lang: Optional[str] = None
    pre_existing_run: Optional[str] = None
    distillation: bool = False
    debug: bool = False
    local: bool = False
    data_src_props: str = "train:1,valid:1"
    hyperopt: HyperOptKind = HyperOptKind.Fixed
    hyperopt_param_str: str = ""  # fmt: {'temperature':[0.3,1.5],'n_samples':[8,16,32]}
    label_sampler: LabelSamplerCfg = field(default_factory=lambda: LabelSamplerCfg())
    launcher_seed: int = -1  # used by label sampling, None if -1

    @property
    def splits_props(self) -> Tuple[List[str], List[float]]:
        splits: List[str] = []
        props: List[float] = []
        for x in self.data_src_props.split(","):
            sp = x.split(":")
            assert len(sp) == 2
            splits.append(sp[0])
            props.append(float(sp[1]))
        lean_allowed_splits = set()
        if self.zmq_prover.lean_dataset is not None:
            if self.zmq_prover.lean_dataset:
                lean_allowed_splits = set(LEAN_SPLITS.keys())
            else:
                lean_allowed_splits = set(self.zmq_prover.lean_dataset.splits)
        allowed_splits = {
            "eq": {"eq_bwd_rwalk_seq2seq", "eq_bwd_graph_seq2seq", "identities"},
            "mm": {"train", "valid", "test"},
            "hl": {"train", "valid", "test"},
            "lean": lean_allowed_splits,
        }
        assert self.lang is not None
        assert all(s in allowed_splits[self.lang] for s in splits), (
            self.lang,
            splits,
            allowed_splits[self.lang],
        )
        assert len(splits) == len(set(splits)) == len(props)
        return splits, props

    def __post_init__(self):

        if self.debug:
            self.zmq_prover.n_machines = 2
            self.n_gpu_trainer = 2
            self.train_params.debug.train = True
            self.train_params.mcts_train.replay_buffer.min_len = 10

        if self.local:
            self.n_gpu_trainer = 1
            self.train_params.master_port = -1
            self.train_params.exp_id = datetime.now().strftime("%d_%m__%H_%M")
            self.zmq_prover.n_machines = 1
            self.zmq_prover.n_trainers = 1
            self.zmq_prover.max_restarts = 0
            self.zmq_prover.local = self.local
            self.zmq_prover.prover.mcts = ConfStore["mcts_very_fast"]
            self.zmq_prover.prover.mcts.early_stop = False
            self.zmq_prover.prover.mcts.n_expansions = 50
            self.zmq_prover.prover.mcts.count_threshold = 0
            self.zmq_prover.prover.mcts.expander.tokens_per_batch = 1000
            self.zmq_prover.prover.mcts.only_learn_tactics_from = ""
            self.train_params.batch.tokens = 500
            self.train_params.batch.max_len = 256
            self.zmq_prover.prover.mcts.expander.max_input_len = 256
            self.train_params.mcts_train.replay_buffer.min_len = 3
            self.train_params.mcts_train.minproof_rb.min_recv_proofs = 1
            self.train_params.mcts_train.minproof_rb.dump_interval = 10.0

        if not self.no_trainer:
            assert self.n_gpu_trainer > 0
            self.zmq_prover.n_trainers = self.n_gpu_trainer
            assert (
                self.n_gpu_trainer <= 8 or self.n_gpu_trainer % 8 == 0
            ), self.n_gpu_trainer
        else:
            self.zmq_prover.n_trainers = 0

        # update partitions
        partition = clusterify_partitions(self.train_partition)
        if self.train_partition != partition:
            print(f"Updating cluter partition: {self.train_partition} -> {partition}")
            self.train_partition = partition

        # check that we do not process theorems / tactics on which we will not be able to train
        max_len = self.train_params.batch.max_len
        max_inp_len = self.zmq_prover.prover.mcts.expander.max_input_len
        max_gen_len = self.zmq_prover.decoding.max_gen_len
        assert max_len >= max_inp_len, (max_len, max_inp_len)
        assert max_len >= max_gen_len, (max_len, max_gen_len)

    def _check_and_mutate_args(self):
        print("Check and mutate Online MCTS")
        assert self.train_params.reload_model == "", self.train_params.reload_model

        if self.lang == "mm":
            self.zmq_prover.mm_dataset = self.train_params.mm.dataset
        elif self.lang == "eq":
            self.zmq_prover.eq_dataset = self.train_params.eq.dataset
        elif self.lang == "hl":
            self.zmq_prover.hl_dataset = self.train_params.hl.dataset
        elif self.lang == "lean":
            self.zmq_prover.lean_dataset = self.train_params.lean.dataset

        # If pre_existing_run  TODO: fix
        if self.pre_existing_run is not None:
            pass
        #     assert os.path.isdir(self.pre_existing_run)
        #     params_dict = json.load(
        #         open(os.path.join(self.pre_existing_run, "params.json"))
        #     )
        #     train_params = migrate_train_args(params_dict)
        #     self.train_params.dump_path = train_params.dump_path
        #     self.train_params.reload_checkpoint = ""
        #     self.train_params.exp_id = train_params.exp_id
        else:
            assert len(
                self.train_params.reload_partial_checkpoint
            ) > 0 or os.path.exists(self.train_params.reload_checkpoint)

        # Check MCTS Tasks
        has_minproof_task: bool = False
        subtasks = self.zmq_prover.prover.mcts_subtasks_online_training
        for task in self.train_params.parsed_tasks():
            for subtask in ALL_MCTS_SUBTASKS:
                if f"mcts_{subtask}" in task and subtask not in subtasks:
                    subtasks.append(subtask)
                    if subtask == "minproof":
                        has_minproof_task = True
        assert len(subtasks) == len(set(subtasks))

        # ZMQ prover check
        self.zmq_prover.prover.beam_path = Path(
            os.path.dirname(self.train_params.reload_checkpoint)
        )
        assert os.path.isdir(self.zmq_prover.prover.beam_path) or os.path.exists(
            self.zmq_prover.prover.beam_path
        )  # TODO: should be a file or a folder ...
        if (
            self.zmq_prover.prover.beam_kind == BeamSearchKind.IncreasingQuality
            and self.pre_existing_run is None
        ):
            # if params.increasing_quality_only, we expect `zmq_prover.beam_path`
            # to exist and contain `checkpoint.-1.pth`, `results.json`, and `done`
            root = Path(self.zmq_prover.prover.beam_path)
            assert root.is_dir(), f"{root} does not exist"
            assert (root / "checkpoint.-1.pth").is_file(), f"checkpoint not in {root}"
            assert (root / "results.json").is_file(), f"results.json not in {root}"
            assert (root / "done").is_file(), f"done not in {root}"

        assert (
            len(self.train_params.stats_socket_addr) > 0
        ), self.train_params.stats_socket_addr

        # TODO: move this? this looks like a dataset specific check
        if self.lang != "eq":
            src_splits, _ = self.splits_props
            for src_split in src_splits:
                if self.lang == "lean" and src_split == "imo":
                    continue
                if (
                    self.lang == "lean"
                    and src_split != "train"
                    and not self.train_params.lean.dataset.is_old
                ):
                    continue
                split_path = (
                    Path(self.zmq_prover.dataset.data_dir) / f"split.{src_split}"
                )
                assert split_path.is_file(), split_path
        # TODO add assert on the trainer model and beam model ?
        # TODO add more checks to free the checks in launch_mcts

        # check that the partial checkpoint that we reload are valid.
        # We should have same encoder and big decoder in the checkpoint of decoder
        # as the one of encoder and big decoder
        if self.distillation and len(self.train_params.reload_partial_checkpoint):
            module_paths = module_reload_paths(
                self.train_params.reload_partial_checkpoint,
                self.train_params.module_names,
            )
            assert module_paths["encoder"] == module_paths["big_decoder"]
            ckp_original = safe_load(module_paths["encoder"], map_location="cpu")
            ckp_distillation = safe_load(module_paths["decoder"], map_location="cpu")
            assert_equal_state_dict(
                ckp_original["encoder"], ckp_distillation["encoder"]
            )
            assert_equal_state_dict(
                ckp_original["decoder"], ckp_distillation["big_decoder"]
            )

        eval_split = self.zmq_prover.prover.eval_split
        bwd_proving_eval = self.train_params.bwd_proving_eval
        assert isinstance(bwd_proving_eval, list)
        to_check = (self.lang, eval_split)
        assert to_check in bwd_proving_eval, f"{to_check} not in {bwd_proving_eval}"

        # WTF - Do not understand this
        # if self.train_params.mcts_train.only_learn_tactics_from == "solving":
        #     assert (
        #         self.zmq_prover.prover.mcts.backup_one_for_solved
        #     ), "backup_one_for_solved must be true for only_learn_from_solving"

        # q conditioning
        assert (
            self.zmq_prover.decoding.q_conditioning
            == self.train_params.mcts_train.q_conditioning
        )
        if len(self.train_params.mcts_train.q_conditioning) > 0:
            self.zmq_prover.prover.mcts.train_sample_for_q_conditioning = True
            if self.train_params.mcts_train.only_learn_tactics_from in [
                "minproof",
                "proof",
                "minproof-solving",
            ]:
                raise RuntimeError(
                    "Q conditioning makes no sense if with learn_tactic_from proof/minproof "
                    "as we send only the solving tactics for proof and minproof nodes."
                )
        self.zmq_prover.prover.mcts.only_learn_tactics_from = (
            self.train_params.mcts_train.only_learn_tactics_from
        )

        if self.hyperopt != HyperOptKind.Fixed:
            assert (
                self.zmq_prover.decoding.n_samples_per_depth_str == ""
            ), "Cannot use variable n_samples with nevergrad"

        if has_minproof_task and (
            self.train_params.mcts_train.minproof_rb.split_props is not None
        ):
            prover_splits, _ = self.splits_props
            rb_splits, _ = self.train_params.mcts_train.minproof_rb.split_props
            if set(prover_splits) != set(rb_splits):
                raise RuntimeError(
                    "prover splits ('splits_props') and rb splits "
                    "('train_params.mcts_train.minproof_rb.split_props')"
                    f"seems different {set(prover_splits)} != {set(rb_splits)}"
                )


def _online_mcts_config(model: ZOOModel) -> MCTSOnlineParams:
    """
    Build an online mcts config that is language agnostic.
    Will have to be completed for a specific language.
    """
    # Initialize train params with the one of the reloaded checkpoint
    ckpt_train_params, _, _ = reload_ckpt(Path(clusterify_path(model.path)))

    # 2048 here because we may want to train on theorems of this size
    ckpt_train_params.batch.max_len = 2048

    cfg = MCTSOnlineParams(
        exp_name="mcts_training",
        train_params=ckpt_train_params,
        zmq_prover=ConfStore["zmq_slow"],
        train_partition=clusterify_partitions("Theorem_Proving"),
        no_trainer=False,
        debug=False,
        n_gpu_trainer=8,
        lang=MISSING,
        pre_existing_run=None,
        local=False,
    )

    cfg.train_params.lean.dataset = ConfStore["lean_latest"]

    # Trainer params
    cfg.train_params.reload_checkpoint = clusterify_path(model.path)
    cfg.train_params.reload_checkpoint_optimizer_state_only = True
    cfg.train_params.batch.tokens = 8000
    cfg.train_params.optimizer = "adam_warmup,lr=0.0001,warmup_updates=10000"
    cfg.train_params.dico = DicoConf()
    cfg.train_params.use_checkpoint_dico = True
    cfg.train_params.epoch_size = 250_000
    cfg.train_params.async_bwd_eval_freq = 1
    cfg.train_params.async_bwd_eval_timeout = 480
    cfg.train_params.tasks = MISSING
    cfg.train_params.exp_name = "mcts_training"
    cfg.train_params.master_port = np.random.randint(9_999) + 10000 + 1

    cfg.train_params.__post_init__()  # fixes root_dump_path

    # Trainer params for proving eval
    cfg.train_params.bwd_proving_eval_str = MISSING

    # ZMQ prover params
    cfg.zmq_prover.decoding.fixed_temperature = 1.0
    cfg.zmq_prover.decoding.use_beam = True
    cfg.zmq_prover.decoding.use_sampling = True
    cfg.zmq_prover.prover.beam_kind = BeamSearchKind.IncreasingQuality
    cfg.zmq_prover.prover.n_simultaneous_proofs = 40
    cfg.zmq_prover.prover.mcts.early_stop = False
    cfg.zmq_prover.prover.heartbeats_freq = 60
    cfg.zmq_prover.n_machines = 128
    cfg.zmq_prover.max_restarts = 2 * cfg.zmq_prover.n_machines

    # sequence lengths
    cfg.zmq_prover.prover.mcts.expander.tokens_per_batch = 10_000
    cfg.zmq_prover.prover.mcts.expander.max_input_len = 2048
    cfg.zmq_prover.decoding.max_gen_len = 512

    # These paths can only be set once we know in what folder the mcts is running
    cfg.zmq_prover.prover.beam_path = Path("later")
    cfg.zmq_prover.prover.dump_path = Path("later")
    cfg.zmq_prover.root_dir = Path("later")

    return cfg


def online_mcts_config_eq(
    model: ZOOModel, mode: str = "graph", eval_critic: bool = True
) -> MCTSOnlineParams:
    assert mode in {"rwalk", "graph"}
    cfg = _online_mcts_config(model=model)
    cfg.lang = "eq"
    cfg.data_src_props = f"identities:1"

    cfg.train_params.bwd_proving_eval_str = "eq:identities"
    cfg.zmq_prover.prover.eval_split = "identities"

    cfg.zmq_prover.eq_dataset = cfg.train_params.eq.dataset
    cfg.zmq_prover.eq_dataset.n_async_envs = 4

    tasks = [f"eq_bwd_{mode}_seq2seq", "eq_mcts_tactic_fmt", "eq_mcts_critic"]
    if eval_critic:
        critic_task = f"eq_critic_{mode}_seq2seqtok"
        tasks.append(critic_task)
        cfg.train_params.no_train = critic_task
    cfg.train_params.tasks = ",".join(tasks)

    cfg.zmq_prover.n_machines = 8
    cfg.zmq_prover.prover.mcts.expander.tokens_per_batch = 15_000
    cfg.train_params.mcts_train.replay_buffer.min_len = 10_000
    cfg.train_params.mcts_train.replay_buffer.max_len = 100_000
    cfg.train_params.epoch_size = 200_000  # ~ 40 min for eq
    cfg.train_params.no_eval_on_train = True  # should be identical to valid eval
    cfg.train_params.no_eval_on_test = True  # should be identical to valid eval

    cfg.train_params.batch.max_len = 1024
    cfg.zmq_prover.prover.mcts.expander.max_input_len = 1024
    cfg.zmq_prover.decoding.max_gen_len = 64

    return cfg


def online_mcts_config_mm(model: ZOOModel) -> MCTSOnlineParams:
    cfg = _online_mcts_config(model)
    cfg.lang = "mm"
    cfg.data_src_props = "train:1,valid:1"

    cfg.train_params.bwd_proving_eval_str = "mm:valid,mm:test"
    cfg.zmq_prover.prover.eval_split = "valid"

    cfg.zmq_prover.mm_dataset = cfg.train_params.mm.dataset
    # cfg.zmq_prover.dataset.n_async_envs = 4  # TODO: implement for MM?

    cfg.train_params.tasks = (
        "mm_x2y_goal--label-mandsubst-EOU-theorem-predsubst_seq2seq,"
        "mm_mcts_tactic_goal--label-mandsubst-EOU-theorem-predsubst,"
        "mm_mcts_critic"
    )

    cfg.zmq_prover.n_machines = 8
    cfg.zmq_prover.prover.mcts.expander.tokens_per_batch = 15_000
    cfg.train_params.mcts_train.replay_buffer.min_len = 10_000
    cfg.train_params.mcts_train.replay_buffer.max_len = 100_000
    cfg.train_params.epoch_size = 200_000

    cfg.train_params.batch.max_len = 1024
    cfg.zmq_prover.prover.mcts.expander.max_input_len = 1024
    cfg.zmq_prover.decoding.max_gen_len = 256

    return cfg


def online_mcts_distillation_config_mm(model: ZOOModel) -> MCTSOnlineParams:
    cfg = online_mcts_config_mm(model)
    model_path = clusterify_path(model.path)
    cfg.distillation = True
    cfg.train_params.distillation.online = True
    cfg.train_params.distillation.critic = True
    cfg.train_params.tasks = (
        "mm_x2y_goal--label-mandsubst-EOU-theorem-predsubst_distillation,"
        "mm_mcts_goal--label-mandsubst-EOU-theorem-predsubst_distillation"
    )
    cfg.train_params.reload_checkpoint = ""
    cfg.train_params.reload_partial_checkpoint = f"encoder:"
    cfg.train_params.reload_checkpoint_optimizer_state_only = False
    # cfg.train_params.per_module_optimizer_params_str = "decoder:lr=0.00003"
    cfg.zmq_prover.prover.mcts.expander.tokens_per_batch = 28000
    cfg.zmq_prover.prover.n_simultaneous_proofs = 30
    return cfg


@unique
class SyntheticKind(str, Enum):
    No = "no"
    V1 = "v1"
    V2 = "v2"


def new_online_mcts_config_lean(
    model: ZOOModel,
    pact: bool = False,
    effect: bool = False,
    prove_self: bool = False,
    synthetic: SyntheticKind = SyntheticKind.No,
) -> MCTSOnlineParams:
    cfg = _online_mcts_config(model)
    cfg.lang = "lean"
    cfg.train_params.bwd_proving_eval_str = "lean:minif2f_valid"
    cfg.zmq_prover.prover.eval_split = "minif2f_valid"
    task_weights = [
        ("lean_x2y_statement--tactic-EOU_seq2seq", 10),
        ("lean_mcts_tactic_statement--tactic-EOU", 20),
        ("lean_mcts_critic", 20),
    ]
    if pact:
        task_weights.append(("lean_pact_seq2seq", 20))
    if effect:
        task_weights.append(("lean_mcts_effect", 20))
    if prove_self:
        task_weights.append(("lean_prove_self_seq2seq", 1))
    if synthetic == SyntheticKind.V1:
        task_weights.append(("lean_syntheticrwalk_statement--tactic-EOU_seq2seq", 5))
        task_weights.append(("lean_syntheticgraph_statement--tactic-EOU_seq2seq", 5))
    if synthetic == SyntheticKind.V2:
        task_weights.append(("lean_synthetic2_statement--tactic-EOU_seq2seq", 10))
    tasks, weights = zip(*task_weights)
    cfg.train_params.tasks = ",".join(tasks)
    cfg.train_params.tasks_weight = ",".join([str(x) for x in weights])
    cfg.zmq_prover.prover.mcts.expander.tokens_per_batch = 5000

    cfg.zmq_prover.max_restarts = 1_000_000

    cfg.zmq_prover.decoding.fixed_temperature = 1.3
    cfg.zmq_prover.prover.mcts.depth_penalty = 0.95

    cfg.zmq_prover.prover.mcts.n_expansions = 2000
    cfg.zmq_prover.decoding.n_samples = 16

    cfg.train_params.optimizer = "adam_warmup,lr=0.00001,warmup_updates=10000"

    return cfg


def online_mcts_config_lean(
    model: ZOOModel,
    minif2f: bool = False,
    n_workers: int = 496,
    timeout: int = 2000,
    lean_cluster: bool = False,
    num_instances: int = 1,
    num_threads: int = 10,
    pact: bool = False,
    effect: bool = False,
    prove_self: bool = False,
    synthetic: SyntheticKind = SyntheticKind.No,
    lean_dataset: Optional[str] = None,
    filter_tactics: Optional[TacticFilter] = None,
    fingerprint: Optional[str] = None,
    lean_cluster_partition: Optional[str] = None,
) -> MCTSOnlineParams:
    cfg = new_online_mcts_config_lean(model, pact, effect, prove_self, synthetic)

    assert isinstance(filter_tactics, TacticFilter)

    print(f"Using model Lean dataset {model.dataset}")
    cfg.zmq_prover.lean_dataset = ConfStore[model.dataset]
    if lean_dataset is not None:
        print(f"Overwriting for Lean dataset {lean_dataset}")
        cfg.zmq_prover.lean_dataset = ConfStore[lean_dataset]

    assert isinstance(cfg.zmq_prover.lean_dataset, LeanDatasetConf)

    if lean_cluster:
        cfg.zmq_prover.lean_dataset.lean_cluster = lean_cluster
        cfg.zmq_prover.lean_dataset.partition = clusterify_partitions(
            "Theorem_Proving"
            if lean_cluster_partition is None
            else lean_cluster_partition
        )
        cfg.zmq_prover.lean_dataset.num_instances = num_instances
        cfg.zmq_prover.lean_dataset.num_threads = num_threads
    cfg.zmq_prover.lean_dataset.timeout = timeout

    cfg.zmq_prover.lean_dataset.filter_tactics = filter_tactics
    if fingerprint is not None:
        cfg.zmq_prover.lean_dataset.fingerprint = fingerprint

    cfg.train_params.lean.dataset = cfg.zmq_prover.lean_dataset

    # enough to fill GPU, not too much to avoid blowing up the ram
    cfg.zmq_prover.prover.n_simultaneous_proofs = 5
    if lean_cluster:
        cfg.zmq_prover.prover.n_simultaneous_proofs = 20

    if minif2f:
        cfg.data_src_props = (
            "train:1,minif2f_valid:1,minif2f_test:1"  # without test for 119
        )
    else:
        cfg.data_src_props = "train:1,valid:1"

    cfg.zmq_prover.n_machines = n_workers

    return cfg


def online_subproof_mcts_config_mm(model: ZOOModel):
    cfg = online_mcts_config_mm(model)

    cfg.train_params.tasks = "mm_subproof_online_mcts_bwd_goal--label-mandsubst-EOU-theorem-predsubst_seq2seq"

    subproofs_params = MCTSSubProofArgs()
    subproofs_params.internal_nodes = True

    cfg.zmq_prover.prover.mcts_subproof_params = subproofs_params
    cfg.zmq_prover.prover.mcts_subproofs_online_training = True
    cfg.train_params.online_bwd_gen = True
    cfg.train_params.online_bwd_gen_params = subproofs_params
    cfg.train_params.epoch_size = 5000

    assert cfg.train_params.reload_checkpoint != "", cfg.train_params.reload_checkpoint

    return cfg


# def online_subproof_mcts_config_lean(model: ZOOModel):
#     cfg = lambda: online_mcts_config_lean(model_path)
#     cfg.train_params.tasks = ",".join(
#         [
#             "lean_x2y_statement--tactic-EOU-subgoals_seq2seq",
#             "lean_subproof_online_mcts_bwd_statement--tactic-EOU-subgoals_seq2seq",
#         ]
#     )
#     subproofs_params = MCTSSubProofArgs()
#     subproofs_params.internal_nodes = True
#
#     cfg.zmq_prover.prover.mcts_subproof_params = subproofs_params
#     cfg.zmq_prover.prover.mcts_subproofs_online_training = True
#     # cfg.train_params.epoch_size = 5000
#     cfg.train_params.online_bwd_gen = True
#     cfg.train_params.online_bwd_gen_params = subproofs_params
#     assert cfg.train_params.reload_checkpoint != "", cfg.train_params.reload_checkpoint
#
#     return cfg


def register_online_mcts_configs() -> None:

    models: Dict[str, ZOOModel] = {}  # add your models here

    # base cfg for eq, mm, and lean
    ConfStore["graph_online_mcts_eq__eq_graph"] = lambda: online_mcts_config_eq(
        models["eq_graph"], mode="graph"
    )
    ConfStore["rwalk_online_mcts_eq__eq_rwalk"] = lambda: online_mcts_config_eq(
        models["eq_rwalk"], mode="rwalk"
    )
    ConfStore["base_online_mcts_mm"] = lambda: online_mcts_config_mm(models["mm"])
    ConfStore["base_subproof_online_mcts_mm"] = lambda: online_subproof_mcts_config_mm(
        models["mm"]
    )

    ConfStore["distillation_online_mcts_mm"] = (
        lambda: online_mcts_distillation_config_mm(models["mm_distil"])
    )

    # Keeping v28, paper configs. -- set ProofCleaningParams.level to TacRemoval
    ConfStore["minif2f_v28_with_newgen_no_merge"] = lambda: online_mcts_config_lean(
        models["lean_full_names_102"],
        minif2f=True,
        timeout=2000,
        n_workers=128,  # 128 to get to 119
        lean_cluster=True,
        num_instances=4,
        num_threads=12,  # 10 to get to 119
        pact=True,
        lean_dataset="lean_v28_no_merge",
        filter_tactics=ConfStore["tac_filter_no_split"],
        fingerprint="node_id",
        synthetic=SyntheticKind.V2,
    )

    ConfStore["minif2f_v31_with_newgen"] = lambda: online_mcts_config_lean(
        models["lean_full_names_102"],
        minif2f=True,
        timeout=2000,
        n_workers=128,  # 128 to get to 119
        lean_cluster=True,
        num_instances=4,
        num_threads=12,  # 10 to get to 119
        pact=True,
        lean_dataset="lean_v31",
        filter_tactics=ConfStore["tac_filter_no_split"],
        fingerprint="node_id",
        synthetic=SyntheticKind.V2,
    )

    # TODO: is it problematic to use v1.1 if the checkpoint used v31 ?
    ConfStore["400provers_model_v1.1"] = lambda: online_mcts_config_lean(
        models["lean__22_06_10_lean_v31_merge_400provers_v0__1482540__250"],
        minif2f=True,
        timeout=2000,
        n_workers=128,  # 128 to get to 119
        lean_cluster=True,
        num_instances=4,
        num_threads=12,  # 10 to get to 119
        pact=True,
        lean_dataset="lean_v1.1",
        filter_tactics=ConfStore["tac_filter_no_split"],
        fingerprint="node_id",
        synthetic=SyntheticKind.V2,
    )

    ConfStore["base_model_v1.2"] = lambda: online_mcts_config_lean(
        models["lean_full_names_102"],
        minif2f=True,
        timeout=2000,
        n_workers=128,  # 128 to get to 119
        lean_cluster=True,
        num_instances=4,
        num_threads=12,  # 10 to get to 119
        pact=True,
        lean_dataset="lean_v1.2",
        filter_tactics=ConfStore["tac_filter_no_split"],
        fingerprint="node_id",
        synthetic=SyntheticKind.V2,
    )

    ConfStore["minif2f_v31_with_newgen_effect"] = lambda: online_mcts_config_lean(
        models["lean_full_names_102"],
        effect=True,
        minif2f=True,
        timeout=2000,
        n_workers=128,  # 128 to get to 119
        lean_cluster=True,
        num_instances=4,
        num_threads=12,  # 10 to get to 119
        pact=True,
        lean_dataset="lean_v31",
        filter_tactics=ConfStore["tac_filter_no_split"],
        fingerprint="node_id",
        synthetic=SyntheticKind.V2,
    )

    ConfStore["minif2f_v31_with_newgen_cond_moe_32"] = lambda: online_mcts_config_lean(
        models["lean_cond_moe_32"],
        minif2f=True,
        timeout=2000,
        n_workers=128,  # 128 to get to 119
        lean_cluster=True,
        num_instances=4,
        num_threads=12,  # 10 to get to 119
        pact=True,
        lean_dataset="lean_v31",
        filter_tactics=ConfStore["tac_filter_no_split"],
        fingerprint="node_id",
        synthetic=SyntheticKind.V2,
    )

    for version in ["v33"]:
        for merge in [False, True]:
            merge_s = "" if merge else "_no_merge"
            for synthetic in [SyntheticKind.No, SyntheticKind.V2]:
                synthetic_s = "" if synthetic == SyntheticKind.No else "_synthetic"
                ConfStore[f"minif2f_{version}{merge_s}{synthetic_s}"] = (
                    lambda synthetic=synthetic: online_mcts_config_lean(
                        models["lean_full_names_102"],
                        minif2f=True,
                        n_workers=32,
                        timeout=2000,
                        lean_cluster=False,
                        num_instances=1,
                        num_threads=10,
                        pact=True,
                        effect=False,
                        prove_self=False,
                        synthetic=synthetic,
                        lean_dataset=f"lean_{version}{merge_s}",
                        filter_tactics=ConfStore["tac_filter_no_split"],
                        fingerprint="node_id",
                        lean_cluster_partition=None,
                    )
                )
                ConfStore[f"minif2f_{version}{merge_s}{synthetic_s}_big"] = (
                    lambda synthetic=synthetic: online_mcts_config_lean(
                        models["lean_full_names_102"],
                        minif2f=True,
                        n_workers=200,
                        timeout=2000,
                        lean_cluster=True,
                        num_instances=4,
                        num_threads=12,
                        pact=True,
                        effect=False,
                        prove_self=False,
                        synthetic=synthetic,
                        lean_dataset=f"lean_{version}{merge_s}",
                        filter_tactics=ConfStore["tac_filter_no_split"],
                        fingerprint="node_id",
                        lean_cluster_partition=None,
                    )
                )

    ConfStore["minif2f_v31_with_newgen_cond_moe_32_half"] = (
        lambda: online_mcts_config_lean(
            models["lean_cond_moe_32_half"],
            minif2f=True,
            timeout=2000,
            n_workers=128,  # 128 to get to 119
            lean_cluster=True,
            num_instances=4,
            num_threads=12,  # 10 to get to 119
            pact=True,
            lean_dataset="lean_v31",
            filter_tactics=ConfStore["tac_filter_no_split"],
            fingerprint="node_id",
            synthetic=SyntheticKind.V2,
        )
    )

    # DEFAULT FOR NEW STYLE CONFIGS
    ConfStore["lean_mcts_pact_synthv2"] = lambda: new_online_mcts_config_lean(
        models["lean_full_names_102"],
        pact=True,
        synthetic=SyntheticKind.V2,
    )
