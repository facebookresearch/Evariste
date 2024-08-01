# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, List, Set, Dict, Tuple, Any
from dataclasses import dataclass, field
from functools import cached_property
from logging import getLogger
from pathlib import Path
import os
import getpass
import re

from evariste.adversarial_offline.args import GeneratorArgs
from params import Params, ConfStore
from evariste.slurm import SlurmConf
from evariste.clusters.utils import clusterify_path
from evariste.comms.rl_distributed_config import RLDistributedConfig
from evariste.model.data.subproof_args import MCTSSubProofArgs
from evariste.model.kl_teacher_model import KLCfg
from evariste.model.transformer_args import ModelArgs, DecodingParams
from evariste.model.pointer_network import PointerNetworkArgs
from evariste.model.data.envs.metamath_args import OnlineGenerationArgs
from evariste.model.data.dictionary import DicoConf
from evariste.model.data.envs.latex_args import LatexArgs
from evariste.model.data.envs.metamath_args import MetamathArgs
from evariste.model.data.envs.hol_light_args import HOLLightArgs
from evariste.model.data.envs.equations_args import EquationArgs
from evariste.model.data.envs.multi_args import GANTrainingArgs
from evariste.model.data.envs.lean_args import LeanArgs
from evariste.model.data.envs.isabelle_args import IsabelleArgs
from evariste.model.data.envs.sr_args import SRArgs
from evariste.model.data.envs.rl_train_args import RLTrainArgs
from evariste.model.data.envs.mcts_trainer_args import MCTSTrainArgs, ReplayBufferArgs
from evariste.trainer.utils import module_reload_paths


logger = getLogger()


class CheckpointMissing(Exception):
    pass


@dataclass
class DistillationArgs(Params):
    n_layers: int = 6
    emb_dim: int = 512
    kl_temperature: float = 1
    critic: bool = True
    online: bool = False
    task_loss: bool = False
    cross_entropy_loss: bool = True
    hidden_states_loss: bool = False
    embedding_loss: bool = False

    def __post_init__(self):
        assert self.kl_temperature > 0
        assert self.n_layers > 0
        assert self.emb_dim > 0
        assert (
            self.task_loss
            or self.cross_entropy_loss
            or self.hidden_states_loss
            or self.embedding_loss
        )


@dataclass
class MLMArgs(Params):
    word_pred: float = field(
        default=0.15,
        metadata={"help": "Fraction of words for which we need to make a prediction"},
    )
    sample_alpha: float = field(
        default=0,
        metadata={
            "help": "Sample masked words with a frequency proportional to counts ** -sample_alpha"
        },
    )
    word_mask_keep_rand_str: str = field(
        default="0.8,0.1,0.1",
        metadata={
            "help": "Fraction of words to mask out / keep / randomize, among the words to predict"
        },
    )

    @cached_property
    def word_mask_keep_rand(self) -> Tuple[float, float, float]:
        probs = [float(p) for p in self.word_mask_keep_rand_str.split(",")]
        assert len(probs) == 3
        assert all([0 <= p <= 1 for p in probs]) and sum(probs) == 1
        return probs[0], probs[1], probs[2]

    @property
    def word_mask(self) -> float:
        return self.word_mask_keep_rand[0]

    @property
    def word_keep(self) -> float:
        return self.word_mask_keep_rand[1]

    @property
    def word_rand(self) -> float:
        return self.word_mask_keep_rand[2]

    def __post_init__(self):
        assert 0 <= self.word_pred < 1
        assert 0 <= self.sample_alpha
        _, _, _ = self.word_mask_keep_rand


@dataclass
class InputNoiseArgs(Params):
    shuffle: float = field(
        default=0, metadata={"help": "Randomly shuffle input words (0 to disable)"}
    )
    dropout: float = field(
        default=0, metadata={"help": "Randomly dropout input words (0 to disable)"}
    )
    blank: float = field(
        default=0, metadata={"help": "Randomly blank input words (0 to disable)"}
    )

    def __post_init__(self):
        assert self.shuffle == 0 or self.shuffle > 1
        assert self.dropout == 0 or 0 < self.dropout < 1
        assert self.blank == 0 or 0 < self.blank < 1


@dataclass
class BatchArgs(Params):
    bptt: int = field(default=512, metadata={"help": "Sequence length"})
    max_len: int = field(default=1024, metadata={"help": "Maximum sequence length"})
    group_by_size: bool = field(
        default=True, metadata={"help": "Sort sentences by size during the training"}
    )
    size: int = field(default=32, metadata={"help": "Number of sequences per batch"})
    tokens: int = field(
        default=5000,
        metadata={"help": "Number of tokens per batch (-1 uses batch.size instead)"},
    )
    quadratic_max_cost: int = field(
        default=-1,
        metadata={
            "help": "Max quadratic cost per batch (-1 uses batch.tokens instead)"
        },
    )
    queue_strategy: str = field(
        default="uniform_sampling", metadata={"help": "Caching strategy for training."}
    )

    # sample with replacement in collate queue to oversample big batches
    # of small sequences
    collate_queue_with_replacement: bool = False

    # how often (every N batches) we update the queue.
    # when `queue_strategy == "uniform_sampling_replacement"` only
    collate_queue_update_freq: int = 100

    collate_queue_size: int = 10_000

    def __post_init__(self):
        assert self.bptt >= 1
        assert self.queue_strategy in [
            "uniform_sampling",
            "uniform_sampling_replacement",
        ]


@dataclass  # modified in train.py
class DebugArgs(Params):
    train: bool = field(
        default=False,
        metadata={"help": "Use valid sets for train sets (faster loading)"},
    )
    slurm: bool = field(
        default=False,
        metadata={"help": "Debug multi-GPU / multi-node within a SLURM job"},
    )
    dataloader: bool = field(
        default=False,
        metadata={
            "help": "Set num_workers to 0 so the data are loaded in the main process"
        },
    )
    debug: bool = field(
        default=False, metadata={"help": "Enable all debug flags"}
    )  # warning, now --debug 1
    size: int = field(
        default=3000,
        metadata={
            "help": "Number of examples to use in debug mode (for train / valid / test). "
            "Can be used for overfitting tests."
        },
    )

    # to debug distributed jobs locally, if using debug.debug, you can set
    # the rank and world_size of your worker here
    rank: int = -1
    world_size: int = -1


@dataclass
class ConditioningArgs(Params):
    # how many classes to learn
    n_classes: int = 0
    max_classes: int = (
        4  # maximum number of classes in parallel (for best_classes_fast_split)
    )
    prob_cond: float = (
        1.0  # probability that we use conditioning (normal training otherwise)
    )
    enc_tgt: bool = True
    input_mode: str = "sum"
    reg: float = 0.1
    # number of epochs for T to linearly go to 0
    n_epoch_decrease: int = 5
    proba_hard: float = 0.25
    small_tgt_encoder: bool = True

    def __post_init__(self):
        assert self.n_classes == 0 or self.n_classes > 0 and 0 < self.prob_cond <= 1
        assert self.max_classes > 0
        assert self.n_classes % self.max_classes == 0
        assert self.input_mode in ["sum", "prefix"]
        if self.prob_cond < 1:
            assert self.input_mode == "sum"
        assert self.n_classes == 0 or self.n_classes >= 2


def parse_proving_eval(s: str) -> List[Tuple[str, str]]:
    """
    Parse evaluation proving parameters.
    """
    assert type(s) is str
    splits = [x for x in s.split(",") if len(x) > 0]
    assert len(splits) == len(set(splits))
    res = []
    for x in splits:
        env, split = x.split(":")
        assert (env in {"mm", "hl", "eq", "lean", "isabelle"}) or (env in ConfStore)
        assert re.match(
            r"train|valid|test|identities|minif2f_valid"
            r"|minif2f_valid_false|minif2f_test|minif2f_test_false"
            r"|oai_curriculum|oai_curriculum_false|synthetic_rwalk_20_10"
            r"|annotations_v\d+|annotations_v\d+_false",
            split,
        ), split

        res.append((env, split))
    return res


@dataclass
class TrainerArgs(Params):
    distillation: DistillationArgs
    mlm: MLMArgs
    pn: PointerNetworkArgs
    model: ModelArgs
    input_noise: InputNoiseArgs
    batch: BatchArgs
    gan: GANTrainingArgs
    cond: ConditioningArgs
    debug: DebugArgs
    slurm_conf: SlurmConf
    dico: DicoConf
    mcts_train: MCTSTrainArgs
    mm: MetamathArgs
    hl: HOLLightArgs
    eq: EquationArgs
    sr: SRArgs
    lean: LeanArgs
    isabelle: IsabelleArgs
    latex: LatexArgs

    rl_params: RLTrainArgs
    online_bwd_gen_params: MCTSSubProofArgs = field(
        default_factory=lambda: MCTSSubProofArgs()
    )
    online_bwd_gen: bool = False
    online_fwd_generation: bool = False
    online_gen_cfg: OnlineGenerationArgs = field(
        default_factory=lambda: OnlineGenerationArgs()
    )
    # when doing online generation, no eval on train since it messes
    # with data refreshing
    no_eval_on_train: bool = False
    no_eval_on_test: bool = False
    rl_distributed: RLDistributedConfig = field(
        default_factory=lambda: RLDistributedConfig()
    )
    tasks: str = ""
    tasks_weight: str = ""
    command: str = ""  # set in initialize_exp

    n_provers: int = 10

    # duplicate with SlurmConf for compatibility reasons with torch.distributed launch and our launch tool
    master_port: int = -1
    local_rank: int = -1

    root_dump_path: str = field(
        default="dumped",
        metadata={"help": "Experiment root dump path"},
    )
    dump_path: str = field(default="", metadata={"help": "Experiment dump path"})

    # if dump_path is set we still mutate it by adding sweep_id, exp_id
    # if override_dump_path is set, we use it without modification as dump_path
    # TODO: maybe change behaviour of dump_path field instead?
    override_dump_path: str = ""
    exp_name: str = field(default="", metadata={"help": "Experiment name"})
    exp_id: str = field(default="", metadata={"help": "Experiment ID"})

    n_kept_checkpoints: int = field(
        default=2, metadata={"help": "Number of checkpoints to keep around"}
    )

    aug_prob: float = field(
        default=0,
        metadata={
            "help": "When doing augmented goal2tactic, probability of augmenting a sample"
        },
    )
    env_base_seed: int = field(
        default=-1,
        metadata={"help": "Base seed for environments (-1 to use timestamp seed)"},
    )
    num_workers: int = field(
        default=1, metadata={"help": "Number of CPU workers for DataLoader"}
    )
    split_data: bool = field(
        default=False, metadata={"help": "Split data across workers of a same node"}
    )

    check_memory: bool = field(
        default=False,
        metadata={
            "help": "Tries to deepcopy the env as many times as there are workers. slow."
        },
    )

    optimizer: str = field(
        default="adam,lr=0.0001",
        metadata={"help": "Optimizer (SGD / RMSprop / Adam, etc.)"},
    )
    per_module_optimizer_params_str: str = field(
        default="",
        metadata={
            "help": (
                "format 'module_name:lr=0.0002', if we want to set"
                "a special optimizer parameter for one module."
            )
        },
    )
    clip_grad_norm: float = field(
        default=1.0, metadata={"help": "Clip gradients norm (0 to disable)"}
    )
    epoch_size: int = field(
        default=100_000,
        metadata={
            "help": "Epoch size / evaluation frequency (-1 for parallel data size)"
        },
    )

    max_epoch: int = field(default=100_000, metadata={"help": "Maximum epoch size"})
    stopping_criterion: str = field(
        default="",
        metadata={
            "help": "Stopping criterion, and number of non-increase before stopping the experiment"
        },
    )
    validation_metrics: str = field(default="", metadata={"help": "Validation metrics"})
    accumulate_gradients: int = field(
        default=1,
        metadata={
            "help": "Accumulate model gradients over N iterations (N times larger batch sizes)"
        },
    )

    reload_model: str = field(
        default="", metadata={"help": "Reload a pretrained model"}
    )
    reload_checkpoint: str = field(default="", metadata={"help": "Reload a checkpoint"})

    use_checkpoint_dico: bool = field(
        default=False, metadata={"help": "Force to use checkpoint dico."}
    )
    use_reloaded_dico: bool = field(
        default=False, metadata={"help": "Included reloaded model words in dico"}
    )
    reload_checkpoint_optimizer_state_only: bool = field(
        default=False,
        metadata={
            "help": "When reloading optimizer from checkpoint, reload only the optimizer state."
        },
    )
    reload_partial_checkpoint: str = field(
        default="", metadata={"help": "Reload a pretrained optimizer"}
    )
    beam_eval: bool = field(
        default=False, metadata={"help": "Perform beam search decoding"}
    )
    eval_bleu: bool = field(
        default=False, metadata={"help": "Evaluate BLEU score for seq2seq tasks"}
    )
    fwd_proving_eval_str: str = field(
        default="",
        metadata={
            "help": (
                "Perform forward proving evaluation with greedy forward prover. "
                'Empty for nothing, or: "mm:valid,eq:test,eq:identities"'
            )
        },
    )
    bwd_proving_eval_str: str = field(
        default="",
        metadata={
            "help": (
                "Perform backward proving evaluation. "
                'Empty for nothing, or: "mm:valid,eq:test,eq:identities"'
            )
        },
    )
    generation_eval_str: str = field(
        default="",
        metadata={
            "help": (
                "Perform generation evaluation. "
                'Empty for nothing, or: "eq:train" for generated.'
            )
        },
    )

    generation_eval_gen_args: GeneratorArgs = field(
        default_factory=lambda: ConfStore["gen_args_512_512_greedy"]
    )

    generation_eval_max_nodes: str = field(
        default="20",
        metadata={
            "help": (
                "Maximum number of nodes and generator steps used for generator evaluation. "
                "Comma-separated list of integers."
            )
        },
    )

    eval_freq: int = 1  # eval every K epoch. must be > 0
    eval_greedy: bool = True
    async_fwd_eval_freq: int = (
        -1
    )  # never if -1, otherwise every K epoch. eval_freq must divide K
    async_bwd_eval_freq: int = -1  # same
    async_bwd_eval_timeout: int = 240
    async_bwd_eval_max_attempts: int = 1
    async_bwd_eval_skip_zero: bool = False
    async_bwd_eval_dec_params: DecodingParams = field(
        default_factory=lambda: ConfStore["decoding_bwd_eval"]
    )

    n_th_to_prove: int = field(
        default=500,
        metadata={
            "help": "Total number of valid theorem on which to call the eval prover"
        },
    )

    no_train: str = field(default="", metadata={"help": "Tasks to not train on"})
    eval_only: bool = field(default=False, metadata={"help": "Only run evaluations"})

    stats_socket_addr: str = field(
        default="",
        metadata={
            "help": "The address of a zmq socket where we can send training stats."
        },
    )

    max_vocab: int = field(
        default=-1, metadata={"help": "Maximum vocabulary size (-1 to disable)"}
    )
    min_count: int = field(default=0, metadata={"help": "Minimum vocabulary count"})

    n_pretrained_words: int = field(
        default=0, metadata={"help": "Holds number of pretrained words"}
    )

    # TODO: remove or fix (if possible)?
    gpu_oom_retry: int = field(
        default=-1,
        metadata={
            "help": "If >0, when an OOM is detected during training, "
            "the OOM is catched and the trainer will try to do a new step "
            "(with a max retry of 'gpu_oom_retry'). WARNING: this doesn't "
            "seems to work anymore with pytorch v1.8.1"
        },
    )

    log_network_stats_freq: int = field(
        default=300,
        metadata={
            "help": (
                "Periodically log statistics about trained "
                "modules. (every n_total_iter, 0 to disable)"
            )
        },
    )

    sync_worker_tasks: bool = field(
        default=True,
        metadata={
            "help": (
                "If True, all workers will process the same tasks at the same time. "
                "Adds bias in the data iterator, but necessary when the tasks use different modules. "
                "Can potentially accelerate training if the tasks require different processing time."
            )
        },
    )

    # entropy_cost for entropy loss (as in RL)
    entropy_cost: Optional[float] = None

    # label smoothing epsilon (0 to disable)
    label_smoothing_eps: float = 0
    kl: KLCfg = field(default_factory=lambda: KLCfg())

    module_names: List[str] = field(default_factory=lambda: [])

    log_freq: int = 60

    # if label is known, proba that we select a random conditioning anyway
    proba_random_conditioning: float = 0.5

    # what to freeze when training the critic (nothing, encoder, encoder + decoder)
    critic_freeze: str = ""

    @property
    def parsed_reload_model(self) -> Dict[str, Path]:
        # reload pretrained models
        if self.reload_model != "":
            logger.info(f"Parsing Reload Model {self.reload_model}")
            # use the same checkpoint for all modules, or provide different module checkpoints
            self.reload_model = clusterify_path(self.reload_model)
            if os.path.isfile(self.reload_model):
                return {name: Path(self.reload_model) for name in self.module_names}
            else:
                return module_reload_paths(self.reload_model, self.module_names)
        else:
            return {}

    @property
    def parsed_reload_partial_checkpoint(self) -> Dict[str, Path]:
        # reload checkpoints for specific modules
        # TODO refacto this
        if self.reload_partial_checkpoint != "":
            return module_reload_paths(
                self.reload_partial_checkpoint, self.module_names
            )
        else:
            return {}

    def parsed_tasks(self, prefix: str = "") -> List[str]:
        res = [x for x in self.tasks.split(",") if x.startswith(prefix)]
        assert all(len(x) > 0 for x in res)
        return res

    def parsed_tasks_weight(self) -> List[float]:
        """
        --tasks mm_a,mm_b,hl_a,hl_b --tasks_weight 1,2,2,1  => 1/6, 2/6 and 2/6, 1/6 for mm / hl
        --tasks mm_a,mm_b,hl_a,hl_b --tasks_weight ''  => 0.25, 0.25 and 0.25, 0.25 for mm / hl
        """
        n_tasks = len(self.parsed_tasks())
        if self.tasks_weight == "":
            return [1 / n_tasks] * n_tasks
        weights = [float(x) for x in self.tasks_weight.split(",")]
        total = sum(weights)
        assert len(weights) == n_tasks
        return [w / total for w in weights]

    @cached_property
    def no_train_tasks(self):
        no_train = self.no_train.split(",")
        no_train = set(x for x in no_train if len(x) > 0)
        assert all(t in self.parsed_tasks() for t in no_train)
        assert len(set(self.parsed_tasks()) - no_train) > 0
        return no_train

    @cached_property
    def per_module_optimizer_params(self) -> Dict[str, Dict[str, float]]:
        """
        Parse per module optimizer parameters.
        Format is a string, e.g. 'decoder:lr=0.0003-beta1=0.0098,big_decoder:lr=0.0001'
        Converted to a dictionary:
            {
                'decoder': {'lr': 0.0003, 'beta1':0.0098},
                'big_decoder': {'lr': 0.0001},
            }
        """
        s = self.per_module_optimizer_params_str
        assert type(s) is str
        module_params = [x for x in s.split(",") if len(x) > 0]
        res: Dict[str, Dict[str, float]] = {}
        for x in module_params:  # for each module
            module_name: str
            params_str: str
            module_name, params_str = x.split(":")
            assert (
                module_name not in res and len(module_name) > 0 and len(params_str) > 0
            )
            res[module_name] = {}
            for param_with_value in params_str.split("-"):  # for each parameter
                param_name: str
                value: str
                param_name, value = param_with_value.split("=")
                assert param_name not in res[module_name]
                res[module_name][param_name] = float(value)
        return res

    @cached_property
    def fwd_proving_eval(self) -> List[Tuple[str, str]]:
        return parse_proving_eval(self.fwd_proving_eval_str)

    @cached_property
    def bwd_proving_eval(self) -> List[Tuple[str, str]]:
        return parse_proving_eval(self.bwd_proving_eval_str)

    @cached_property
    def generation_eval(self) -> List[Tuple[str, str]]:
        return parse_proving_eval(self.generation_eval_str)

    def __post_init__(self):
        if "USERNAME" in self.root_dump_path:
            self.root_dump_path = self.root_dump_path.replace(
                "USERNAME", getpass.getuser()
            )
        self.root_dump_path = clusterify_path(self.root_dump_path)
        # assert os.path.isdir(self.root_dump_path), self.root_dump_path

        assert self.critic_freeze in ["", "enc", "encdec"]

    def _check_and_mutate_args(self):
        """
        Check models parameters.
        """
        modules: Set[str] = set()
        tasks = self.parsed_tasks()
        if any(task.split("_")[-1] == "clm" for task in tasks):
            modules.add("encoder")
        if any(task.split("_")[-1] == "mlm" for task in tasks):
            modules.add("encoder")
        if any(task.split("_")[-1] == "cclm" for task in tasks):
            modules.add("encoder")
            modules.add("discriminator")
            modules.add("classifier")
        if any(task.split("_")[-1] == "disc" for task in tasks):
            modules.add("encoder")
            modules.add("decoder")
            modules.add("discriminator")
            modules.add("classifier")
        if any(task.split("_")[-1] == "dae" for task in tasks):
            modules.add("encoder")
            modules.add("decoder")
        if any(task.split("_")[-1] == "mass" for task in tasks):
            modules.add("encoder")
            modules.add("decoder")
        if any(task.split("_")[-1] == "bt" for task in tasks):
            modules.add("encoder")
            modules.add("decoder")
        if any(task.split("_")[-1] == "seq2seq" for task in tasks):
            modules.add("encoder")
            modules.add("decoder")
        if any(task.split("_")[-1] == "seq2seqtok" for task in tasks):
            modules.add("encoder")
            modules.add("decoder")
        if any(task.split("_")[-1] == "seq2tok" for task in tasks):
            modules.add("encoder")
            modules.add("classifier")
        if any(task.split("_")[-1] == "seq2emb" for task in tasks):
            modules.add("embedder")
            modules.add("encoder")
        if any(task.split("_")[-1] == "embseq2seq" for task in tasks):
            modules.add("embedder")
            modules.add("encoder")
            modules.add("decoder")
        if any(task.split("_")[-1] == "embseq2ptrseq" for task in tasks):
            modules.add("embedder")
            modules.add("encoder")
            modules.add("decoder")
            modules.add("pointer")
        if any(task.split("_")[-1] == "seq2ptrseq" for task in tasks):
            modules.add("encoder")
            modules.add("decoder")
            modules.add("pointer")
        if any(task.split("_")[-1] == "embseq2subst" for task in tasks):
            modules.add("embedder")
            modules.add("encoder")
            modules.add("decoder")
        if any("mcts" in task for task in tasks):
            modules.add("encoder")
            modules.add("decoder")
        if any("distillation" in task for task in tasks):
            modules.add("encoder")
            modules.add("decoder")
            modules.add("big_decoder")
            if self.distillation.hidden_states_loss:
                modules.add("hidden_states_linear")
            if self.distillation.embedding_loss:
                modules.add("embedding_linear")
        if any("rl" in task for task in tasks):
            modules.add("encoder")
            modules.add("decoder")
        # TODO: change this condition
        if any(("critic" in t and t.startswith(("mm_fwd", "mm_gen"))) for t in tasks):
            logger.info("Adding a critic layer for fwd")
            modules.add("critic")

        if self.cond.n_classes > 0:
            if self.cond.enc_tgt:
                modules.add("target_encoder")
            modules.add("cond_embeddings")

        self.module_names = sorted(modules)

        if len(self.per_module_optimizer_params_str) > 0:
            if "AdamCosineWithWarmup" in self.optimizer:
                raise Exception(
                    "Per module optimizer parameter option is not available "
                    "for AdamCosineWithWarmup, please implement it."
                )
            assert all(k in modules for k in self.per_module_optimizer_params.keys())

        # checkpoint reloading
        assert (
            len(self.reload_checkpoint) == 0 or len(self.reload_partial_checkpoint) == 0
        )

        # tasks weights
        assert len(self.parsed_tasks_weight()) == len(self.parsed_tasks())

        # the check below is to prevent setting the rng in the main process
        assert self.num_workers > 0 or self.debug.dataloader

        if self.online_fwd_generation:
            assert isinstance(self.online_gen_cfg, OnlineGenerationArgs)
            if not self.debug.debug:
                assert self.slurm_conf.is_slurm_job
            assert (
                self.slurm_conf.torch_world_size > 0
            ), self.slurm_conf.torch_world_size
            assert self.no_eval_on_train
            assert self.mm.graph.generated_proof_path == ""
            assert (
                self.mm.graph.generated_prob > 0 or self.lean.graph.generation_prob > 0
            )

        if self.mm.graph.generated_prob > 0:
            if (
                self.rl_distributed.is_adversarial_training
                + self.online_fwd_generation
                + (self.mm.graph.generated_proof_path != "")
                != 1
            ):
                raise ValueError(
                    "To sample generated proofs you need to be in adversarial training "
                    "setup, or (exclusive) with fwd online generation or (exclusive) a "
                    "generated_proof_path"
                )

            if (
                self.rl_distributed.is_adversarial_training
                or self.online_fwd_generation
            ):
                if not self.no_eval_on_train:
                    raise ValueError(
                        "Impossible for the moment to evaluate "
                        "on train with online generation (because of "
                        "distributed communications between workers)"
                    )

        if not self.online_fwd_generation:
            assert self.slurm_conf.torch_world_size == -1

        if self.mm.graph.reward_quantile_conditioning:
            assert self.rl_params.replay_buffer.n_reward_quantiles > 0

        if self.debug.dataloader:
            self.num_workers = 0

        assert not (self.use_checkpoint_dico and self.use_reloaded_dico)
        assert self.use_reloaded_dico is False or self.reload_model

        # reload_checkpoint
        if self.reload_checkpoint_optimizer_state_only:
            assert len(self.reload_checkpoint) > 0

        # check proving eval format
        _ = self.fwd_proving_eval
        _ = self.bwd_proving_eval
        _ = self.generation_eval

        # check that for Equations, the provided backward evaluation tasks are
        # possible (i.e. that the splits are available in the Equation environment)
        eq_bwd_proving_tasks = {"eq_bwd_rwalk_seq2seq", "eq_bwd_graph_seq2seq"}
        for env_name, split in self.bwd_proving_eval:
            if env_name != "eq":
                continue
            p_tasks = eq_bwd_proving_tasks & set(self.parsed_tasks("eq"))
            if len(p_tasks) > 1:
                raise RuntimeError("There can only be one proving task.")
            if split not in ["valid", "identities"]:
                raise RuntimeError(
                    f"No backward evaluation on {split} is possible for Equations."
                    f"Please provide `valid` or `identities`"
                )
            if split == "valid" and len(p_tasks) == 0:
                raise RuntimeError(
                    f"Did not find a proving task for backward evaluation on Equations. "
                    f"You must provide a proving task."
                )
            assert self.eval_freq > 0, "must eval sometimes"
            freqs = [self.async_bwd_eval_freq, self.async_fwd_eval_freq]
            for freq, name in zip(freqs, ["bwd", "fwd"]):
                assert freq == -1 or (
                    freq >= self.eval_freq and freq % self.eval_freq == 0
                ), f"{name} : {freq} % {self.eval_freq} != 0"

    def has_mcts(self):
        return any(["mcts" in x for x in self.tasks.split(",")])


ConfStore["default_cfg"] = TrainerArgs(
    mlm=MLMArgs(),
    distillation=DistillationArgs(
        n_layers=ModelArgs().n_layers, emb_dim=ModelArgs().emb_dim
    ),
    pn=PointerNetworkArgs(),
    model=ModelArgs(),
    input_noise=InputNoiseArgs(),
    batch=BatchArgs(),
    gan=GANTrainingArgs(),
    cond=ConditioningArgs(),
    debug=DebugArgs(),
    slurm_conf=SlurmConf(),
    dico=DicoConf(),
    mcts_train=MCTSTrainArgs(replay_buffer=ReplayBufferArgs()),
    mm=ConfStore["default_mm"],
    hl=ConfStore["hl_plus_default_args"],
    eq=ConfStore["default_eq"],
    sr=ConfStore["default_sr"],
    lean=ConfStore["default_lean"],
    isabelle=ConfStore["default_isabelle"],
    latex=ConfStore["default_latex"],
    rl_params=ConfStore["default_rl"],
)
