# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from glob import glob
from math import floor
from typing import Optional, Union, Tuple, List, Set, Dict, Any
from collections import OrderedDict
from logging import getLogger
from evariste.backward.graph import Theorem
from evariste.model.data.envs.metamath import N_MAX_NODES
from evariste.model.data.envs.node_sampling_strategy import NodeSamplingStrategy
from pympler import asizeof
from pathlib import Path
import numpy as np
import random

from evariste import json as json
from evariste.comms.store import AnnotatedGeneration
from evariste.datasets.equations import EquationsDatasetConf
from evariste.forward.fwd_eq.gen.proof_search import (
    EqGenProofSearch,
    EqGenForwardGraph,
    EqGenForwardTactic,
    CycleInGeneration,
)
from evariste.model.data.envs.replay_buffer_factory import (
    requires_adv_rb,
    build_adv_replay_buffer,
)
from evariste.model.data.envs.env import DataEnvironment
from evariste.model.data.envs.replay_buffer_loader import ReplayBuffer

from evariste.trainer.args import TrainerArgs
from evariste.envs.eq.env import EquationEnv
from evariste.envs.eq.graph import (
    VARIABLES,
    CONSTANTS,
    U_OPS,
    B_OPS,
    C_OPS,
    NodeSet,
    Node,
    eq_nodes_are_equal,
)
from evariste.envs.eq.identities import (
    IDENTITIES_EXP,
    IDENTITIES_TRIGO,
    IDENTITIES_HYPER,
)
from evariste.envs.eq.graph import RULE_VARS
from evariste.envs.eq.rules import ARule, TRule
from evariste.envs.eq.rules_lib import ALL_RULES, NAME_TO_RULE
from evariste.envs.eq.generation import (
    EquationGraphStats,
    GraphNode,
    EquationGraphGenerator,
    EquationGraphSampler,
    NothingToSample,
    extract_walk_steps,
    extract_graph_steps,
)

from evariste.model.data.dictionary import (
    B_NODE_WORD,
    E_NODE_WORD,
    EOS_WORD,
    PROVED_WORD,
    UNPROVED_WORD,
    Dictionary,
)
from evariste.model.data.envs.mcts_loader import MCTSSubProofDataLoader
from evariste.model.data.mcts_subproof import load_mcts_dumps
from evariste.backward.env.equations.env import EQEnvWorker
from evariste.backward.env.equations.graph import (
    T_RULE_TOK,
    A_RULE_TOK,
    SIMP_TAC_TOK,
    NN_TAC_TOK,
    FWD_TOK,
    BWD_TOK,
    N_PREFIX_TOKS,
    B_EQ_NODE_WORD,
    E_EQ_NODE_WORD,
    prefix_pos_tok,
    prefix_var_name,
    EQTactic,
    EQTheorem,
)
from evariste.forward.online_generation.worker_type import WorkerType
from evariste.forward.fwd_eq.eq_fwd_tokenizer import tokenize_graph, EqFwdTokenizer
from evariste.backward.prover.mcts import MCTSSampleTactics


# MIN_VALID_TEST_SAMPLES = 1000
# MIN_VALID_LABELS = 100
from evariste.utils import load_stream, print_memory

MIN_VALID_TEST_SAMPLES = 5
MIN_VALID_LABELS = 5


logger = getLogger()


class EquationsEnvironment(DataEnvironment):

    TRAINING_TASKS = [
        "eq_bwd_rwalk_seq2seq",
        "eq_bwd_graph_seq2seq",
        "eq_bwd_graph_offline_seq2seq",
        "eq_fwd_graph_seq2seq",
        # these use old generation code
        "eq_gen_graph_seq2seq",
        "eq_gen_graph_offline_seq2seq",
        # these use forward prover generation
        "eq_newgen_graph_seq2seq",
        "eq_bwd_newgen_graph_seq2seq",
        "eq_newgen_graph_offline_seq2seq",
        "eq_bwd_newgen_graph_offline_seq2seq",
        "eq_critic_walk_seq2seqtok",
        "eq_critic_graph_seq2seqtok",
        "eq_gen_rl",  # TODO: implement
        "eq_fwd_rl",  # TODO: implement
        "eq_bwd_rl",  # TODO: implement
        "eq_mcts_tactic_fmt",
        "eq_mcts_minproof_fmt",
        "eq_mcts_critic",
        "eq_subproof_mcts_bwd_seq2seq",
        # task to initialize an equations environment without a training task:
        "eq_notask",
        # task to initialize an equations environment without a training task that will be treated like a generation environment
        "eq_gen_notask",
    ]

    PROVING_TASKS = {
        "eq_bwd_rwalk_seq2seq",
        "eq_bwd_graph_seq2seq",
        "eq_fwd_graph_seq2seq",
    }

    REQUIRES_NEWGEN = {
        "eq_newgen_graph_seq2seq",
        "eq_newgen_graph_offline_seq2seq",
        "eq_bwd_newgen_graph_seq2seq",
        # "eq_bwd_newgen_graph_offline_seq2seq"
    }

    def __init__(self, dico: Dictionary, params: TrainerArgs, fast: bool = False):
        """
        Initialize environment.
        fast: fast flag used for testing
        """

        super().__init__(
            dico=dico,
            params=params,
            env_name="eq",
            tactic_cls=EQTactic,
            theorem_cls=EQTheorem,
            dataset=params.eq.dataset,
        )

        self.fast = fast

        # valid / test samples
        if self.params.debug.train:
            global MIN_VALID_TEST_SAMPLES
            MIN_VALID_TEST_SAMPLES = 500

        eq_tasks = params.parsed_tasks("eq")
        # skip if no Equations task
        # if len(eq_tasks) == 0:
        #     return

        # there can only be one proving task
        if len(self.PROVING_TASKS & set(eq_tasks)) > 1:
            raise RuntimeError("There can only be one proving task.")

        logger.info("\n=========== Initializing Equations Environment ===========")

        # create Equations environment instance
        logger.info("Creating Equations environment ...")
        self.eq_env = EquationEnv.build(params.eq.dataset.env)

        # for each split, the list of labels in the split
        self.labels: Dict[Union[str, Tuple[str, str]], List[str]] = {}
        self.label_to_eq: Dict[str, EQTheorem] = {}

        # build rules
        self.build_rules()

        requires_egg = {
            "eq_bwd_rwalk_seq2seq",
            "eq_bwd_graph_seq2seq",
            "eq_bwd_graph_offline_seq2seq",
            "eq_fwd_graph_seq2seq",
            "eq_gen_graph_seq2seq",
            "eq_gen_graph_offline_seq2seq",
        }
        if len(requires_egg & set(eq_tasks)) > 0:
            # initialize generator
            self.egg = EquationGraphGenerator(
                env=self.eq_env,
                rules=self.rules,
                hyp_max_ops=params.eq.dataset.hyp_max_ops,
                tf_prob=params.eq.dataset.tf_prob,
                bias_nodes=params.eq.dataset.bias_nodes,
                bias_rules=params.eq.dataset.bias_rules,
                max_true_nodes=params.eq.dataset.max_true_nodes,
            )
        self.graph_sampler = EquationGraphSampler(
            self.eq_env.rng, params.eq.dataset.sampler_params
        )
        self.forward_prover, self.gen_stream = None, None
        if len(self.REQUIRES_NEWGEN & set(eq_tasks)) > 0:
            from evariste.forward.fwd_eq.gen.for_trainer import get_gen

            self.forward_prover, self.gen_stream = get_gen(
                self.params.eq.dataset, self.eq_env
            )

        # skip if no Equations task
        if len(eq_tasks) == 0:
            return

        # build vocabulary
        self.build_vocab()

        # create datasets
        self.create_datasets()
        for split, labels in self.labels.items():
            logger.info(f"Loaded {len(labels)} {split} labels.")
        assert len(self.label_to_eq) == sum(len(x) for x in self.labels.values()), (
            len(self.label_to_eq),
            sum(len(x) for x in self.labels.values()),
            list(self.labels.keys()),
            [len(x) for x in self.labels.values()],
        )
        # assert set(self.label_to_eq.keys()) == set(sum(self.labels.values(), []))

        # # replay buffer for adversarial training
        self.adv_rb: Optional[ReplayBuffer[AnnotatedGeneration]] = None
        if requires_adv_rb(env_name="eq", args=params):
            self.adv_rb = build_adv_replay_buffer(
                env_name="eq", worker_type=self.get_worker_type(), args=params
            )

        # load MCTS subproof data
        self.mcts_subproof: Optional[MCTSSubProofDataLoader] = None
        if self.params.online_bwd_gen:
            self.mcts_subproof = MCTSSubProofDataLoader(
                tactic_cls=EQTactic, theorem_cls=EQTheorem, params=params
            )

        # preview data
        self.preview_data()

    @staticmethod
    def get_identities(dataset: EquationsDatasetConf) -> Dict[str, EQTheorem]:
        """
        Reload set of equation identities. Since proofs are not available,
        identities are only used in proving evaluations.
        """
        logger.info(f"Loading equation identities ...")

        unary_ops = dataset.env.unary_ops
        binary_ops = dataset.env.binary_ops
        identities = (
            IDENTITIES_EXP + IDENTITIES_TRIGO + IDENTITIES_HYPER
        )  # TODO: add calculus

        skip_ops = 0
        label_to_eq = {}

        for identity in identities:
            assert type(identity) is list and 1 <= len(identity) <= 2
            eq, hyps = identity if len(identity) == 2 else (identity[0], [])
            assert eq.is_comp() and eq.value == "=="

            # check operators
            if not (
                eq.get_unary_ops().issubset(unary_ops)
                and eq.get_binary_ops().issubset(binary_ops)
            ):
                skip_ops += 1
                continue

            # add identity
            name = f'identities_{str(eq).replace(" ", "_").replace("/", "|")}'
            assert name not in label_to_eq
            label_to_eq[name] = EQTheorem(node=eq, hyps=hyps)

        logger.info(
            f"Loaded {len(label_to_eq)}/{len(identities)} equation identities. "
            f"{skip_ops} were ignored because they contain unauthorized operators."
        )
        return label_to_eq

    def set_rng(self, rng: np.random.RandomState):
        """
        Set random generator.
        """
        assert not hasattr(self, "rng")
        self.rng = rng
        self.eq_env.set_rng(rng)

    def build_rules(self):
        """
        Build rules. Only select equivalent rules that have authorized operators.
        """
        self.rules = ALL_RULES[self.params.eq.dataset.rule_env]
        logger.info(f"Found {len(self.rules)} {self.params.eq.dataset.env} rules.")
        logger.info(self.params.eq.dataset.rule_types)
        # filter rules by type
        rtypes = self.params.eq.dataset.rule_types
        assert all(any(rule.rule_type == t for rule in self.rules) for t in rtypes)
        self.rules = [rule for rule in self.rules if rule.rule_type in rtypes]
        assert len(self.rules) > 0
        logger.info(f"Filtered {len(self.rules)} rules with valid types.")

        # filter rules with valid operators
        self.rules = [
            rule
            for rule in self.rules
            if rule.get_unary_ops().issubset(self.params.eq.dataset.env.unary_ops)
            and rule.get_binary_ops().issubset(self.params.eq.dataset.env.binary_ops)
        ]
        assert len(self.rules) > 0
        logger.info(f"Filtered {len(self.rules)} rules with valid operators.")

        # split rules by category
        self.rules_t = [rule for rule in self.rules if isinstance(rule, TRule)]
        self.rules_a = [rule for rule in self.rules if isinstance(rule, ARule)]
        rules_t_e = [rule for rule in self.rules_t if not rule.left.is_comp()]
        rules_t_c = [rule for rule in self.rules_t if rule.left.is_comp()]
        assert len(self.rules_t) + len(self.rules_a) == len(self.rules)
        logger.info(
            f"Found {len(self.rules)} rules in total, including {len(self.rules_t)} "
            f"transformation rules ({len(rules_t_e)} for expressions, "
            f"{len(rules_t_c)} for comparisons), and {len(self.rules_a)} assertion rules."
        )

    def build_vocab(self):
        """
        Build vocabulary.
        """
        logger.info("\n==================== Building vocabulary ====================")

        vocab = OrderedDict()

        # add environment variables / constants
        for x in [*VARIABLES, *CONSTANTS]:
            vocab[x] = 1

        # add environment operators
        for x in [*U_OPS, *B_OPS, *C_OPS]:
            vocab[x] = 1

        # add delimiter / forward / backward tokens
        vocab[T_RULE_TOK] = 1
        vocab[A_RULE_TOK] = 1
        vocab[SIMP_TAC_TOK] = 1
        vocab[NN_TAC_TOK] = 1

        vocab[FWD_TOK] = 1
        vocab[BWD_TOK] = 1

        # integer representation tokens
        for digit in range(10):
            vocab[str(digit)] = 1
        vocab["+"] = 1
        vocab["-"] = 1

        # prefix position tokens
        for prefix_pos in range(N_PREFIX_TOKS):
            tok = prefix_pos_tok(prefix_pos)
            vocab[tok] = 1

        # add rule names
        for rule in self.rules:
            name = rule.name
            assert name not in vocab
            vocab[name] = 1

        # add variable names
        for v in RULE_VARS:
            assert v.is_var(), v
            name = prefix_var_name(v.value)
            assert name not in vocab
            vocab[name] = 1

        # prefix position tokens
        if len(self.REQUIRES_NEWGEN & set(self.params.parsed_tasks("eq"))) > 0:
            from evariste.envs.mm.utils import node_tok
            from evariste.forward.fwd_eq.gen.tactics import (
                HYP_HOLE_TOK,
                EQ_FWD_TAC_TOKS,
            )

            for v in VARIABLES:
                name = prefix_var_name(v)
                assert name not in vocab
                vocab[name] = 1

            vocab[HYP_HOLE_TOK] = 1
            vocab[B_EQ_NODE_WORD] = 1
            vocab[E_EQ_NODE_WORD] = 1
            for word in EQ_FWD_TAC_TOKS:
                vocab[word] = 1
            for word in EQ_FWD_TAC_TOKS:
                vocab[word] = 1

            for node_id in range(N_MAX_NODES):
                tok = node_tok(node_id)
                vocab[tok] = 1
                # vocab[f"_PRED_{tok}"] = 1

        assert not any(" " in k for k in vocab)

        # update global dictionary
        self.dico.add_vocab(vocab)

    def create_datasets(self):
        """
        Pre-process datasets for fast data iteration.
        """

        logger.info(f"\n==================== Creating datasets ====================")
        params = self.params
        tasks = set(params.parsed_tasks("eq"))

        for task in tasks:
            if task not in self.TRAINING_TASKS:
                raise Exception(f"Unknown task: {task}")

        # create datasets

        if "eq_gen_graph_offline_seq2seq" in tasks:
            self.load_offline_dataset(direction="gen")
        elif "eq_bwd_graph_offline_seq2seq" in tasks:
            self.load_offline_dataset(direction="bwd")
        elif "eq_newgen_graph_offline_seq2seq" in tasks:
            self.load_new_offline_dataset(direction="newgen")
        elif "eq_bwd_newgen_graph_offline_seq2seq" in tasks:
            self.load_new_offline_dataset(direction="bwd_newgen")

        if (
            set(
                [
                    "eq_newgen_graph_seq2seq",
                    "eq_bwd_newgen_graph_seq2seq",
                    "eq_newgen_graph_offline_seq2seq",
                    "eq_bwd_newgen_graph_offline_seq2seq",
                    "eq_gen_graph_offline_seq2seq",
                    "eq_bwd_graph_offline_seq2seq",
                    "eq_notask",
                    "eq_gen_notask",
                ]
            )
            & tasks
        ):
            # do not create datasets for eq_bwd_rwalk_seq2seq and eq_bwd_graph_seq2seq
            create_tasks = tasks
        else:
            create_tasks = tasks | {"eq_bwd_rwalk_seq2seq", "eq_bwd_graph_seq2seq"}
        for task in {
            "eq_bwd_rwalk_seq2seq",
            "eq_bwd_graph_seq2seq",
            "eq_fwd_graph_seq2seq",
            "eq_gen_graph_seq2seq",
            "eq_gen_graph_offline_seq2seq",
            "eq_newgen_graph_seq2seq",
            "eq_bwd_newgen_graph_seq2seq",
            "eq_newgen_graph_offline_seq2seq",
            "eq_bwd_newgen_graph_offline_seq2seq",
            "eq_bwd_graph_offline_seq2seq",
            "eq_critic_rwalk_seq2seqtok",
            "eq_critic_graph_seq2seqtok",
        } & create_tasks:
            logger.info(f"{create_tasks}")
            if self.fast:
                logger.warning("Not creating datasets for fast testing!")
                continue
            logger.info(f"========== Creating {task} dataset ...")
            if not task in self.data.keys():
                self.data[task] = {}
            for split in ["valid", "test"]:
                data = self.create_dataset(task, split)
                self.data[task][split] = data
                size = asizeof.asizeof(self.data[task][split]) / 1024 ** 2
                logger.info(f"====> Size of data for {task} ({split}): {size:.3f}MB")

        # identities dataset
        self.create_identities_dataset()

        # MCTS subproof tasks
        mcts_subproof_tasks = [
            task
            for task in params.parsed_tasks("eq")
            if task.startswith("eq_subproof_mcts")
        ]
        if len(mcts_subproof_tasks) > 0:
            assert len(mcts_subproof_tasks) == 1
            task = mcts_subproof_tasks[0]
            assert task == "eq_subproof_mcts_bwd_seq2seq"

            # build env worker
            env_worker = EQEnvWorker(self.params.eq.dataset, eq_env=self.eq_env)
            env_worker.init()

            # load MCTS dumps
            (
                self.mcts_subproof_samplers,
                self.cumulative_mcts_subproofs,
            ) = load_mcts_dumps(
                env_name="eq",
                params=params,
                subproof_params=params.eq.mcts_subproof,
                env_worker=env_worker,
            )

            self.data[task] = {}
            for split in ["valid", "test"]:  # train is sampled on the fly
                data = self.create_mcts_subproof_bwd_dataset(split)
                self.data[task][split] = data

    def create_mcts_subproof_bwd_dataset(self, split: str):
        assert split in ["valid", "test"]
        rng = np.random.RandomState(0 if split == "valid" else 1)

        # generate data
        data = []

        for _ in range(MIN_VALID_TEST_SAMPLES):
            data.append(self.get_mcts_subproof_bwd_sample(split, rng))

        # skip too long sequences (or failed samples)
        too_long = len([True for item in data if item is None])
        data = [item for item in data if item is not None]

        # log data statistics
        logger.info(
            f"Created {len(data)} MCTS subproof step {split} sequences. "
            f"Skipped {too_long} too long sequences."
        )

        return data

    def get_mcts_subproof_bwd_sample(self, split: str, rng: np.random.RandomState):
        """
        Get a subproof from MCTS nodes.
        # TODO: set rng in samplers
        """
        samplers = self.mcts_subproof_samplers[split]
        # select a random sampler
        if self.params.eq.mcts_subproof.weight_samplers_alpha == 0:
            index = rng.randint(len(samplers))
        else:
            cumulative_scores = self.cumulative_mcts_subproofs[split]
            index = np.searchsorted(
                cumulative_scores,  # a
                rng.random() * cumulative_scores[-1],  # v
                side="right",  # a[i-1] <= v < a[i]
            )
        sampler = samplers[index]
        goal, tactic, children = sampler.sample_proof_step()
        name = f"mcts_subproof__{sampler.name}__{sampler.n_gen_proofs}"

        # sanity check
        assert isinstance(goal, EQTheorem)
        assert isinstance(tactic, EQTactic)
        assert type(children) is list
        assert all(isinstance(sg, EQTheorem) for sg in children)

        eos = self.dico.eos_word

        # add sequence delimiters
        x = [eos, *goal.tokenize(), eos]
        y = [eos, *tactic.tokenize(), eos]

        # max length
        if max(len(x), len(y)) > self.params.batch.max_len:
            return None

        # index sequences
        x = [self.dico.index(t) for t in x]
        y = [self.dico.index(t) for t in y]

        return {"name": name, "x": x, "y": y}

    def create_identities_dataset(self):
        label_to_eq = self.get_identities(self.params.eq.dataset)
        self.labels["identities"] = list(label_to_eq.keys())
        assert len(set(label_to_eq.keys()).intersection(self.label_to_eq.keys())) == 0
        self.label_to_eq.update(label_to_eq)

    def preview_data(self):
        """
        Preview small snapshots of created datasets.
        """
        N_PREVIEW = 1000
        logger.info(f"\n==================== Dataset preview ====================")
        rng = np.random.RandomState(seed=0)
        for task, content in self.data.items():
            if "train" in content:
                split = "train"
            elif "valid" in content:
                split = "valid"
            else:
                continue
            logger.info(f"========== {task} ({split})")
            data = content[split]
            if not data:
                logger.warn(f"No data for {task} {split} provided!")
                continue
            for k in ["x", "y"]:
                if k not in data[0]:
                    continue
                lengths = [len(x[k]) for x in data]
                logger.info(
                    f"Lengths on {k} -- "
                    f"avg: {np.mean(lengths):.2f} (+/- {np.std(lengths):.2f}) -- "
                    f"min: {np.min(lengths)} -- "
                    f"max: {np.max(lengths)}"
                )
            idx = rng.randint(0, len(data), size=(N_PREVIEW,))
            for i in idx:
                for k in ["x", "y"]:
                    if k not in data[i]:
                        continue
                    v = data[i][k]
                    if type(v) is list:
                        try:
                            v = " ".join(self.dico[j] for j in v)
                        except TypeError as e:
                            logger.warning(e)
                            v = str(v)
                    else:
                        v = str(v)
                    logger.info(f"{k} {v}")

    def dump_data(self, dump_folder: str):
        dump_folder = Path(dump_folder)
        assert dump_folder.is_dir()
        for task, content in self.data.items():
            for split, data in content.items():
                path = dump_folder.joinpath(f"{task}_{split}")
                data_words = []
                for i in range(len(data)):
                    sample = {}
                    for k in ["x", "y"]:
                        if k not in data[i]:
                            continue
                        v = data[i][k]
                        if type(v) is list:
                            try:
                                v = " ".join(self.dico[j] for j in v)
                            except TypeError as e:
                                logger.warning(e)
                                v = str(v)
                        else:
                            v = str(v)
                        sample[k] = v
                    data_words.append(sample)
                json.dump(data_words, open(path, "w"))

    def generate_random_walk(self, rng: np.random.RandomState) -> Tuple[Dict, str]:
        """
        Generate a random walk.
        """
        params = self.params

        n_init_ops = rng.randint(params.eq.dataset.max_init_ops + 1)
        max_created_hyps = rng.randint(params.eq.dataset.max_created_hyps + 1)
        prob_add_hyp = rng.random()

        # perform a random walk
        walk = self.egg.random_walk(
            bidirectional=params.eq.dataset.bidirectional_walk,
            n_steps=params.eq.dataset.n_walk_steps,
            n_init_ops=n_init_ops,
            max_created_hyps=max_created_hyps,
            prob_add_hyp=prob_add_hyp,
        )

        walk = {
            "steps": walk["steps"],
            "start": walk["start"],
            "end": walk["end"],
            "hyps": walk["hyps"],
        }
        name = f"rwalk_{rng.randint(1 << 60)}"

        return walk, name

    def generate_random_graph(
        self, rng: np.random.RandomState
    ) -> Tuple[List[GraphNode], NodeSet, str]:
        """
        Generate a random graph.
        """
        params = self.params

        n_init_hyps = rng.randint(params.eq.dataset.max_init_hyps + 1)
        # TODO: sample tf_prob ?

        # generate a random graph
        nodes, hyps = self.egg.generate_graph(
            n_nodes=params.eq.dataset.n_nodes,
            max_trials=params.eq.dataset.max_trials,
            n_init_hyps=n_init_hyps,
        )

        name = f"graph_{rng.randint(1 << 60)}"
        assert len(hyps) == n_init_hyps

        return nodes, hyps, name

    def get_bwd_rwalk_samples(
        self, store_eq: bool, skip_too_long: bool, rng: np.random.RandomState
    ) -> List[Dict[str, Union[str, List[int]]]]:
        """
        Create a dataset of equations generated with a random walk.
        """
        data = []
        eos = self.dico.eos_word

        # perform a random walk
        walk, name = self.generate_random_walk(rng)
        goals_with_tactics, _ = extract_walk_steps(walk)

        # store valid / test equations
        if store_eq:
            self.label_to_eq[name] = goals_with_tactics[-1][0]

        for goal, tactic, _ in goals_with_tactics:

            # add sequence delimiters
            x = [eos, *goal.tokenize(), eos]
            y = [eos, *tactic.tokenize(), eos]

            # max length
            if skip_too_long and max(len(x), len(y)) > self.params.batch.max_len:
                continue

            # index sequences
            x = [self.dico.index(t) for t in x]
            y = [self.dico.index(t) for t in y]

            data.append({"name": name, "x": x, "y": y})

        return data

    def get_bwd_graph_samples(
        self, store_eq: bool, skip_too_long: bool, rng: np.random.RandomState
    ) -> List[Dict[str, Union[str, List[int]]]]:
        """
        Create a dataset of equations generated with a random graph.
        Used for backward proving.
            Input: goal
            Output: tactic
        """
        data = []
        eos = self.dico.eos_word

        # generate a random graph
        nodes, _, graph_name = self.generate_random_graph(rng)
        # sample theorems to prove from the graph
        n_samples = self.params.eq.dataset.n_graph_samples
        sampled_ids, _ = self.graph_sampler.sample(graph=self.egg, n_samples=n_samples)
        seen_goal: Set[Tuple[EQTheorem, EQTactic]] = set()
        for node_id in sampled_ids:

            node: GraphNode = nodes[node_id]
            name = f"{graph_name}_node{node_id}"

            # extract goals and tactics
            goals_with_tactics, _, _ = extract_graph_steps(node)
            # store valid / test equations
            new_data = []
            for goal, tactic, _ in goals_with_tactics:
                if (goal, tactic) in seen_goal:
                    continue
                seen_goal.add((goal, tactic))
                # add sequence delimiters
                x = [eos, *goal.tokenize(), eos]
                y = [eos, *tactic.tokenize(), eos]

                # max length
                if skip_too_long and max(len(x), len(y)) > self.params.batch.max_len:
                    continue

                # index sequences
                x = [self.dico.index(t) for t in x]
                y = [self.dico.index(t) for t in y]

                new_data.append({"name": name, "x": x, "y": y})

            if store_eq and new_data:
                self.label_to_eq[name] = goals_with_tactics[-1][0]
            data += new_data
        return data

    def get_bwd_newgen_graph_samples(self):
        assert self.gen_stream is not None
        while True:
            proof_search = next(self.gen_stream)
            final_graph = proof_search.next_graph
            n_samples = self.params.eq.dataset.n_graph_samples
            try:
                sampled_ids, _ = self.graph_sampler.sample(
                    graph=EquationGraphStats(final_graph.nodes, rules=self.rules),
                    n_samples=n_samples,
                )
                break
            except NothingToSample:
                logger.warning("Found nothing to sample")
        data = []
        already_added = set()
        for node_id in sampled_ids:
            name = f"{proof_search.goal.name}_{node_id}"
            goals_with_tactics, hyps, node_ids = extract_graph_steps(
                final_graph.nodes[node_id]
            )

            for goal, tactic, _ in goals_with_tactics:
                if (goal, tactic) in already_added:
                    continue
                already_added.add((goal, tactic))
                x = [EOS_WORD, *goal.tokenize(), EOS_WORD]
                y = [EOS_WORD, *tactic.tokenize(), EOS_WORD]

                if max(len(x), len(y)) > self.params.batch.max_len:
                    continue
                # index sequences
                x_id: List[int] = [self.dico.index(t) for t in x]
                y_id: List[int] = [self.dico.index(t) for t in y]

                data.append({"name": name, "x": x_id, "y": y_id})

        return data

    def get_newgen_graph_samples(self):
        assert self.gen_stream is not None
        while True:
            proof_search = next(self.gen_stream)
            final_graph = proof_search.next_graph
            n_samples = self.params.eq.dataset.n_graph_samples
            try:
                sampled_ids, _ = self.graph_sampler.sample(
                    graph=EquationGraphStats(final_graph.nodes, rules=self.rules),
                    n_samples=n_samples,
                )
                break
            except NothingToSample:
                logger.warning("Found nothing to sample")
        data = []
        already_added = set()
        for node_id in sampled_ids:
            name = f"{proof_search.goal.name}_{node_id}"
            try:
                for graph, tactic in proof_search.get_forward_steps(node_id):
                    x = graph.tokenize()
                    y = [
                        EOS_WORD,
                        *tactic.tokenize(graph),
                        EOS_WORD,
                    ]
                    # max length
                    if max(len(x), len(y)) > self.params.batch.max_len:
                        continue
                    hsh = " ".join(x) + " ".join(y)
                    if hsh in already_added:
                        continue
                    already_added.add(hsh)
                    # index sequences
                    try:
                        data.append(
                            {
                                "name": name,
                                "x": [self.dico.index(t) for t in x],
                                "y": [self.dico.index(t) for t in y],
                            }
                        )
                    except KeyError:
                        print(x)
                        print(y)
                        raise
            except CycleInGeneration:
                logger.warning("Cycle in generation")
            except:
                import pickle
                from evariste.utils import rstr

                pickle.dump(
                    (proof_search, node_id), open(f"repros/{rstr()}.pkl", "wb"),
                )
                raise
        return data

    def get_fwd_graph_samples(
        self,
        store_eq: bool,
        skip_too_long: bool,
        include_goal: bool,
        rng: np.random.RandomState,
    ) -> List[Dict[str, Union[str, List[int]]]]:
        """
        Create a dataset of equations generated with a random graph.
        Used for forward proving.
            Input: goal + graph_of_nodes
            Output: new_node + tactic
        """
        from evariste.forward.common import ForwardGoal
        from evariste.forward.common import ForwardGraph
        from evariste.forward.fwd_eq.eq_fwd_env import EqForwardTactic

        data = []

        # generate a random graph
        nodes, init_hyps, graph_name = self.generate_random_graph(rng)

        # sample theorems to prove from the graph
        n_samples = self.params.eq.dataset.n_graph_samples
        sampled_ids, _ = self.graph_sampler.sample(graph=self.egg, n_samples=n_samples)

        already_seen = set()
        for node_id in sampled_ids:

            node: GraphNode = nodes[node_id]
            name = f"fwd_{graph_name}_node{node_id}"

            # extract goals and tactics
            goals_with_tactics, hyps, node_ids = extract_graph_steps(node, init_hyps)
            final_goal, tactic, _ = goals_with_tactics[-1]
            # store valid / test equations
            if store_eq:
                self.label_to_eq[name] = final_goal

            previous_nodes: List[Node] = []
            for goal, tactic, _ in goals_with_tactics:

                # add random nodes
                if self.params.eq.dataset.insert_noise_prob > 0:
                    temp = []
                    others = [
                        j for j, n in enumerate(nodes) if n.node_id not in node_ids
                    ]
                    while len(previous_nodes) > 0 and len(others) > 0:
                        if rng.random() < self.params.eq.dataset.insert_noise_prob:
                            next_id = rng.randint(len(others))
                            temp.append(nodes[others[next_id]].node)
                            others.pop(next_id)
                        else:
                            temp.append(previous_nodes.pop(0))
                    previous_nodes = temp + previous_nodes

                fwd_goal = ForwardGoal(
                    thm=final_goal if include_goal else None,
                    label=name,
                    global_hyps=[EQTheorem(node=hyp, hyps=[]) for hyp in hyps],
                )
                generated = [
                    EQTheorem(node=prev_node, hyps=final_goal.eq_hyps)
                    for prev_node in previous_nodes
                ]
                fwd_graph = ForwardGraph(fwd_goal=fwd_goal, generated_thms=generated)

                tokenizer = EqFwdTokenizer(is_generation=not include_goal)

                x = tokenizer.tokenize_graph(graph=fwd_graph)
                y = tokenizer.tokenize_command(
                    fwd_tactic=EqForwardTactic(next_node=final_goal, bwd_tactic=tactic)
                )

                previous_nodes.append(goal.eq_node)
                if (goal, tactic) in already_seen:
                    continue
                already_seen.add((goal, tactic))

                # max length
                if skip_too_long and max(len(x), len(y)) > self.params.batch.max_len:
                    continue

                # index sequences
                x = [self.dico.index(t) for t in x]
                y = [self.dico.index(t) for t in y]

                data.append({"name": name, "x": x, "y": y})
        return data

    def load_offline_theorems(
        self,
        proofs: List[Tuple[str, Any, EqGenProofSearch, bool]],
        skip_too_long: bool = False,
        direction: str = "newgen",
    ) -> List[Dict[str, Union[str, List[int]]]]:
        """
        Extract proof steps from proofs depending on fwd_offline_sampling_strategy parameter.
        For generation (direction == "gen"):
            Input: graph_of_nodes
            Output: new_node + tactic
        For backward proving (direction == "bwd"):
            Input: goal
            Output: tactic
        """
        data: List[Dict[str, Union[str, List[int]]]] = []
        eos = self.dico.eos_word
        sampling_strategy = self.params.eq.dataset.fwd_offline_sampling_strategy

        already_added = set()
        skipped = {"duplicate": 0, "too_long": 0}

        def append_step(name: str, x: List[str], y: List[str]):
            # max length
            if skip_too_long and max(len(x), len(y)) > self.params.batch.max_len:
                skipped["too_long"] += 1
                return
            # index sequences
            x_id: List[int] = [self.dico.index(t) for t in x]
            y_id: List[int] = [self.dico.index(t) for t in y]

            data.append({"name": name, "x": x_id, "y": y_id})

        def extract_and_append_bwd(node: GraphNode, bwd_proved: bool):
            goals_with_tactics, hyps, node_ids = extract_graph_steps(node)
            for goal, tactic, _ in goals_with_tactics:
                if (goal, tactic) not in already_added:
                    already_added.add((goal, tactic))
                    x_bwd = [eos, *goal.tokenize(), eos]
                    y_bwd = [eos, *tactic.tokenize(), eos]
                    append_step(name, x_bwd, y_bwd)

        def extract_and_append_fwd(
            goals_and_tactics: List[Tuple[EqGenForwardGraph, EqGenForwardTactic]],
            bwd_proved: bool,
        ):
            prefix = []
            if self.params.eq.proved_conditioning:
                prefix = [PROVED_WORD if bwd_proved else UNPROVED_WORD]
            for graph, tactic in goals_and_tactics:
                x = graph.tokenize()
                y = [
                    EOS_WORD,
                    *prefix,
                    *tactic.tokenize(graph),
                    EOS_WORD,
                ]
                # max length
                if max(len(x), len(y)) > self.params.batch.max_len:
                    continue
                hsh = " ".join(x) + " ".join(y)
                if hsh in already_added:
                    continue
                already_added.add(hsh)
                # index sequences
                data.append(
                    {
                        "name": name,
                        "x": [self.dico.index(t) for t in x],
                        "y": [self.dico.index(t) for t in y],
                    }
                )

        easy, hard = [], []
        for item in proofs:
            (easy if item[3] else hard).append(item)

        if self.params.eq.proved_conditioning:
            random.seed(42)
            # if we have more hard than easy, pick all easy
            if len(hard) > len(easy):
                chosen_easy = easy
            # otherwise, get a balanced mix
            else:
                to_choose = min(len(easy), max(len(hard), 30000))  # always at least
                chosen_easy = random.sample(easy, k=to_choose)
            to_train = chosen_easy + hard
            logger.info(
                f"EQ OFFLINE Proportions : {100*len(chosen_easy)/max(1,len(to_train))}% easy  ({len(hard)})"
            )
        else:
            to_train = hard

        for item in to_train:
            name, proof, bwd_proved = item[0], item[2], item[3]
            assert isinstance(proof, EqGenProofSearch)
            nodes = proof.next_graph.nodes
            if sampling_strategy == NodeSamplingStrategy.AllMinimal:
                for node_id, node in enumerate(nodes):
                    if not node.ntype in ["transform", "assert"]:
                        continue
                    if direction == "bwd_newgen":
                        extract_and_append_bwd(node, bwd_proved)
                    else:
                        try:
                            extract_and_append_fwd(
                                proof.get_forward_steps(node_id), bwd_proved
                            )
                        except CycleInGeneration:
                            logger.warning("Cycle in generation")
                        except:
                            import pickle
                            from evariste.utils import rstr

                            pickle.dump(
                                (proof, node_id), open(f"repros/{rstr()}.pkl", "wb"),
                            )
                            raise
            elif sampling_strategy == NodeSamplingStrategy.SamplingMinimal:
                n_samples = self.params.eq.dataset.n_graph_samples
                graph = EquationGraphStats(nodes, self.rules)
                sampled_ids, _ = self.graph_sampler.sample(
                    graph=graph, n_samples=n_samples
                )
                for node_id in sampled_ids:
                    if direction == "bwd_newgen":
                        extract_and_append_bwd(nodes[node_id], bwd_proved)
                    else:
                        try:
                            extract_and_append_fwd(
                                proof.get_forward_steps(node_id), bwd_proved
                            )
                        except CycleInGeneration:
                            logger.warning("Cycle in generation")
                        except:
                            import pickle
                            from evariste.utils import rstr

                            pickle.dump(
                                (proof, node_id), open(f"repros/{rstr()}.pkl", "wb"),
                            )
                            raise
        logger.info(
            f"Extracted {len(data)} data from {len(to_train)} offline proofs. Avoided {skipped} samples"
        )
        return data

    def load_offline_dataset(self, skip_too_long: bool = False, direction: str = "gen"):
        """
        Load a dataset of equations from params.eq.dataset.offline_dataset_path.
        For generation (direction == "gen"):
            Input: graph_of_nodes
            Output: new_node + tactic
        For backward proving (direction == "bwd"):
            Input: goal
            Output: tactic
        """
        task = f"eq_{direction}_graph_offline_seq2seq"

        seed = 2357
        rank = self.params.slurm_conf.global_rank
        world_size = self.params.slurm_conf.world_size
        logger.info(
            f"EquationsEnvironment loading offline dataset with split RNG(seed={seed}), rank: {rank}/{world_size}"
        )
        print_memory(logger, "load_offline_dataset before load")
        rng = np.random.RandomState(seed)

        path_globs = self.params.eq.dataset.offline_dataset_path.split(",")
        dirs = (Path(d) for glb in path_globs for d in glob(glb))

        # load data in per worker shards
        proofs = list(
            p
            for i, p in enumerate(
                load_stream(
                    dirs,
                    verbose_interval=10000,
                    max=self.params.debug.size if self.params.debug.debug else None,
                )
            )
            if i % world_size == rank
        )
        logger.info(f"Loaded {len(proofs)} proofs for worker {rank}/{world_size}.")
        print_memory(logger, "load_offline_dataset after load")

        # split proofs as indicated by params.eq.dataset.offline_dataset_splits
        self.data[task] = {}

        rng.shuffle(proofs)
        start = 0
        cum_prop = 0.0
        for split, prop in zip(
            ["train", "valid", "test"],
            self.params.eq.dataset.offline_dataset_splits.split(","),
        ):
            cum_prop += float(prop)
            end = floor(cum_prop * len(proofs))
            # extract steps from proofs
            data = self.load_offline_theorems(
                proofs=proofs[start:end],
                skip_too_long=skip_too_long,
                direction=direction,
            )
            if not data:
                logger.warn(f"{task}: no data assigned for split {split}!")
            else:
                logger.info(
                    f"{task}: Using [{start}:{end}]={end - start} proofs for split {split}."
                )
                logger.info(f"{task}: Using {len(data)} steps for split {split}.")
            self.data[task][split] = data
            start = end

    def load_new_offline_dataset(self, direction: str, skip_too_long: bool = False):
        """
        Load a dataset of equations from params.eq.dataset.offline_dataset_path.
        For generation (direction == "newgen"):
            Input: graph_of_nodes
            Output: new_node + tactic
        For backward proving (direction == "newgen_bwd"):
            Input: goal
            Output: tactic
        """
        task = f"eq_{direction}_graph_offline_seq2seq"

        seed = 2357
        rank = self.params.slurm_conf.global_rank
        world_size = self.params.slurm_conf.world_size
        logger.info(
            f"EquationsEnvironment loading offline dataset with split RNG(seed={seed}), rank: {rank}/{world_size}"
        )
        print_memory(logger, "load_offline_dataset before load")
        rng = np.random.RandomState(seed)

        path_globs = self.params.eq.dataset.offline_dataset_path.split(",")
        dirs = (Path(d) for glb in path_globs for d in glob(glb))

        # load data in per worker shards
        proofs = list(
            p
            for i, p in enumerate(
                load_stream(
                    dirs,
                    verbose_interval=10000,
                    max=self.params.debug.size if self.params.debug.debug else None,
                )
            )
            if i % world_size == rank
        )
        logger.info(f"Loaded {len(proofs)} proofs for worker {rank}/{world_size}.")
        print_memory(logger, "load_offline_dataset after load")

        # split proofs as indicated by params.eq.dataset.offline_dataset_splits
        self.data[task] = {}

        rng.shuffle(proofs)
        start = 0
        cum_prop = 0.0
        for split, prop in zip(
            ["train", "valid", "test"],
            self.params.eq.dataset.offline_dataset_splits.split(","),
        ):
            cum_prop += float(prop)
            end = floor(cum_prop * len(proofs))
            # extract steps from proofs
            data = self.load_offline_theorems(
                proofs=proofs[start:end],
                skip_too_long=skip_too_long,
                direction=direction,
            )
            if not data:
                logger.warn(f"{task}: no data assigned for split {split}!")
            else:
                logger.info(
                    f"{task}: Using [{start}:{end}]={end - start} proofs for split {split}."
                )
                logger.info(f"{task}: Using {len(data)} steps for split {split}.")
            self.data[task][split] = data
            start = end

    def add_critic_samples(
        self,
        goals_with_tactics: List[Tuple[EQTheorem, EQTactic, List[Node]]],
        hyps: List[Node],
        data: List[Dict[str, Union[str, List[int]]]],
        skip_too_long: bool,
        return_one: bool,
        name: str,
        rng: np.random.RandomState,
    ) -> None:
        """
        Given a proof, create critic samples.
        """
        eos = self.dico.eos_word
        n_nodes = len(goals_with_tactics)

        for node_id, (goal, _, _) in enumerate(goals_with_tactics):

            # for valid and test, return only one sample (the final goal)
            if return_one and node_id < n_nodes - 1:
                continue

            assert goal.eq_hyps == hyps
            is_true = rng.randint(2) == 0
            if is_true:
                theorem = goal
            else:
                # to create false examples, we negate true ones, or randomly remove
                # some hypotheses. the last node is the only one for which we know
                # that all hypotheses are necessary.
                if node_id < n_nodes - 1 or len(hyps) == 0 or rng.randint(2) == 0:
                    theorem = EQTheorem(node=goal.eq_node.negation(), hyps=hyps)
                else:
                    n_old_hyps = len(hyps)
                    n_new_hyps = rng.randint(n_old_hyps)
                    new_hyps = [
                        hyps[hyp_id]
                        for hyp_id in rng.permutation(n_old_hyps)[:n_new_hyps]
                    ]
                    theorem = EQTheorem(node=goal.eq_node, hyps=new_hyps)

            # tokenize / add sequence delimiters
            x = [eos, *theorem.tokenize(), eos]
            y = [1 if is_true else 0]

            # max length
            if skip_too_long and max(len(x), len(y)) > self.params.batch.max_len:
                continue

            # index sequences
            x = [self.dico.index(t) for t in x]

            data.append({"name": name, "x": x, "y": y})

    def get_critic_rwalk_samples(
        self, valid_test: bool, rng: np.random.RandomState
    ) -> List[Dict[str, Union[str, List[int]]]]:
        """
        Create a dataset of true / false equations for the critic.
        Generate samples with a random walk.
            Input: goal
            Output: critic (true or false)
        """
        data: List[Dict[str, Union[str, List[int]]]] = []

        # perform a random walk
        walk, name = self.generate_random_walk(rng)
        goals_with_tactics, hyps = extract_walk_steps(walk)

        # for each node in the random walk
        self.add_critic_samples(
            goals_with_tactics=goals_with_tactics,
            hyps=hyps,
            data=data,
            skip_too_long=not valid_test,
            return_one=valid_test,
            name=f"critic_{name}",
            rng=rng,
        )

        return data

    def get_critic_graph_samples(
        self, valid_test: bool, rng: np.random.RandomState
    ) -> List[Dict[str, Union[str, List[int]]]]:
        """
        Create a dataset of true / false equations for the critic.
        Generate samples with a random graph.
            Input: goal
            Output: critic (true or false)
        """
        data: List[Dict[str, Union[str, List[int]]]] = []

        # generate a random graph
        nodes, init_hyps, graph_name = self.generate_random_graph(rng)

        # sample theorems to prove from the graph
        n_samples = self.params.eq.dataset.n_graph_samples
        sampled_ids, _ = self.graph_sampler.sample(graph=self.egg, n_samples=n_samples)

        # for each node sampled from the graph
        for node_id in sampled_ids:

            # extract goals and tactics
            node: GraphNode = nodes[node_id]
            goals_with_tactics, hyps, _ = extract_graph_steps(node, init_hyps)

            # for each proof node
            self.add_critic_samples(
                goals_with_tactics=goals_with_tactics,
                hyps=hyps,
                data=data,
                skip_too_long=not valid_test,
                return_one=valid_test,
                name=f"critic_{graph_name}_node{node_id}",
                rng=rng,
            )

        return data

    def create_dataset(self, task: str, split: str):
        """
        Create valid / test datasets.
        """
        assert split in ["valid", "test"]

        # set a random generator to have a consistent valid / test set
        rng = np.random.RandomState(0 if split == "valid" else 1)
        old_rng = self.eq_env.set_rng(rng)

        data: List[Dict] = []
        too_long = 0
        labels: Set[str] = set()

        while len(data) < MIN_VALID_TEST_SAMPLES or (
            split == "valid" and len(labels) < MIN_VALID_LABELS
        ):
            if task == "eq_bwd_rwalk_seq2seq":
                samples = self.get_bwd_rwalk_samples(
                    store_eq=True, skip_too_long=False, rng=rng
                )
            elif task == "eq_bwd_graph_seq2seq":
                samples = self.get_bwd_graph_samples(
                    store_eq=True, skip_too_long=False, rng=rng
                )
            elif task == "eq_fwd_graph_seq2seq":
                samples = self.get_fwd_graph_samples(
                    store_eq=True, skip_too_long=False, include_goal=True, rng=rng
                )
            elif task == "eq_gen_graph_seq2seq":
                samples = self.get_fwd_graph_samples(
                    store_eq=False, skip_too_long=False, include_goal=False, rng=rng
                )
            elif task == "eq_newgen_graph_seq2seq":
                samples = self.get_newgen_graph_samples()
            elif task == "eq_bwd_newgen_graph_seq2seq":
                samples = self.get_bwd_newgen_graph_samples()
            elif task == "eq_newgen_graph_offline_seq2seq":
                assert task in self.data.keys()
                return self.data[task][split]
            elif task == "eq_bwd_newgen_graph_offline_seq2seq":
                assert task in self.data.keys()
                return self.data[task][split]
            elif task == "eq_gen_graph_offline_seq2seq":
                assert task in self.data.keys()
                return self.data[task][split]
            elif task == "eq_bwd_graph_offline_seq2seq":
                assert task in self.data.keys()
                return self.data[task][split]
            elif task == "eq_critic_rwalk_seq2seqtok":
                samples = self.get_critic_rwalk_samples(valid_test=True, rng=rng)
            elif task == "eq_critic_graph_seq2seqtok":
                samples = self.get_critic_graph_samples(valid_test=True, rng=rng)
            else:
                raise RuntimeError(f"Unknown task {task}")

            for sample in samples:
                labels.add(sample["name"])
                if max(len(sample["x"]), len(sample["y"])) > self.params.batch.max_len:
                    too_long += 1
                    print("TOO LONG")
                    continue
                data.append(sample)

        logger.info(
            f"Created {len(data)} {split} {task} sequences. "
            f"Skipped {too_long} too long sequences."
        )

        if task in self.PROVING_TASKS:
            assert (task, split) not in self.labels, (task, split)
            self.labels[(task, split)] = sorted(list(labels))

        # restore previous random generators
        self.eq_env.set_rng(old_rng)

        return data

    def get_train_sample(self, task: str):
        """
        Get a train sample.
        """
        if task not in self._cache:
            self._cache[task] = []

        # populate cache if empty
        while len(self._cache[task]) == 0:
            if task == "eq_bwd_rwalk_seq2seq":
                samples = self.get_bwd_rwalk_samples(
                    store_eq=False, skip_too_long=True, rng=self.rng
                )
                self._cache[task] = samples
            elif task == "eq_bwd_graph_seq2seq":
                samples = self.get_bwd_graph_samples(
                    store_eq=False, skip_too_long=True, rng=self.rng
                )
                self._cache[task] = samples
            elif task == "eq_fwd_graph_seq2seq":
                samples = self.get_fwd_graph_samples(
                    store_eq=False, skip_too_long=True, include_goal=True, rng=self.rng
                )
                self._cache[task] = samples
            elif task == "eq_newgen_graph_seq2seq":
                samples = self.get_newgen_graph_samples()
                self._cache[task] = samples
            elif task == "eq_bwd_newgen_graph_seq2seq":
                samples = self.get_bwd_newgen_graph_samples()
                self._cache[task] = samples
            elif task == "eq_newgen_graph_offline_seq2seq":
                assert task in self.data.keys()
                data = self.data[task]["train"]
                return data[self.rng.randint(len(data))]
            elif task == "eq_bwd_newgen_graph_offline_seq2seq":
                assert task in self.data.keys()
                data = self.data[task]["train"]
                return data[self.rng.randint(len(data))]
            elif task == "eq_gen_graph_seq2seq":
                samples = self.get_fwd_graph_samples(
                    store_eq=False, skip_too_long=True, include_goal=False, rng=self.rng
                )
                self._cache[task] = samples
            elif task == "eq_gen_graph_offline_seq2seq":
                assert task in self.data.keys()
                data = self.data[task]["train"]
                return data[self.rng.randint(len(data))]
            elif task == "eq_bwd_graph_offline_seq2seq":
                assert task in self.data.keys()
                data = self.data[task]["train"]
                return data[self.rng.randint(len(data))]
            elif task == "eq_critic_rwalk_seq2seqtok":
                samples = self.get_critic_rwalk_samples(valid_test=False, rng=self.rng)
                self._cache[task] = samples
            elif task == "eq_critic_graph_seq2seqtok":
                samples = self.get_critic_graph_samples(valid_test=False, rng=self.rng)
                self._cache[task] = samples
            elif task == "eq_subproof_mcts_bwd_seq2seq":
                return self.get_mcts_subproof_bwd_sample(split="train", rng=self.rng)
            elif task.endswith("_rl"):
                return self.get_rl_sample(task, split="train", rng=self.rng)
            else:
                raise Exception(f'Unknown task: "{task}"')

        # return element
        return self._cache[task].pop()

    def get_theorem(
        self, task: str, seed: Optional[int] = None
    ) -> Tuple[EQTheorem, str]:
        if not hasattr(self, "rng"):
            print(f"EquationsEnvironment setting RNG with seed {seed}", flush=True)
            self.set_rng(np.random.RandomState(seed))
        if task == "eq_bwd_rwalk_seq2seq":
            walk, name = self.generate_random_walk(self.rng)
            goals_with_tactics, _ = extract_walk_steps(walk)
            return goals_with_tactics[-1][0], name
        elif task == "eq_bwd_graph_seq2seq":
            nodes, init_hyps, graph_name = self.generate_random_graph(self.rng)
            # TODO: check that it is not too slow
            n_samples = 1  # self.params.eq.dataset.n_graph_samples
            sampled_ids, _ = self.graph_sampler.sample(self.egg, n_samples=n_samples)
            node_id = sampled_ids[0]
            node = nodes[node_id]
            name = f"{graph_name}_node{node_id}"
            goals_with_tactics, _, _ = extract_graph_steps(node, init_hyps)
            return goals_with_tactics[-1][0], name
        else:
            raise RuntimeError(f"Wrong task: {task}")

    def get_mcts_y_fmt(self, sample: MCTSSampleTactics):
        assert isinstance(sample.goal, EQTheorem)
        assert all(isinstance(tactic, EQTactic) for tactic in sample.tactics)
        eos = self.dico.eos_word
        y = []
        for tactic in sample.tactics:
            this_y = [eos, *tactic.tokenize(), eos]
            this_y = [self.dico.index(tok) for tok in this_y]
            y.append(this_y)
        return y

    def get_sample(self, task: str, split: str, index: Optional[int]):
        """
        Get a data sample.
        """
        assert (split == "train") == (index is None)
        if task.startswith("eq_mcts"):
            return self.get_mcts_sample(task, split, index)
        # elif task.startswith("eq_subproof_online_mcts_bwd_"):
        #     return self.get_mcts_subproof_online_bwd_sample(task, split, self.rng)
        else:
            if split == "train":
                return self.get_train_sample(task)
            assert index is not None
            data = self.data[task][split]
            return data[index]

    def get_rl_sample(self, task: str, split: str, rng: np.random.RandomState):
        """
        block: if True, block in replay_buffer.get_sample until replay buffer
        return a sample. if False, it raises EmptyStore if Empty.
        """
        assert task in ["eq_gen_rl", "eq_fwd_rl", "eq_bwd_rl"], task
        eos = self.dico.eos_word

        # retrieve graph from replay buffer
        final_goal, nodes, annotated_gen = self.sample_node_sequence_from_rb(split, rng)

        # extract graph steps / hypotheses
        init_hyps = NodeSet([node.node for node in nodes if node.ntype == "hyp"])
        goals_with_tactics, used_hyps, _ = extract_graph_steps(final_goal, init_hyps)
        goals_with_tactics: List[Tuple[EQTheorem, EQTactic, List[Node]]]
        used_hyps: List[Node]

        # select next node / graph
        tgt_idx = rng.randint(len(goals_with_tactics))
        previous_nodes = [g.eq_node for g, _, _ in goals_with_tactics[:tgt_idx]]
        next_node, tactic, _ = goals_with_tactics[tgt_idx]

        if task in ["eq_gen_rl", "eq_fwd_rl"]:

            # build graph
            graph = used_hyps + previous_nodes
            is_generation = task == "eq_gen_rl"
            graph_tokens = tokenize_graph(
                goal="" if is_generation else final_goal.node.prefix(),
                graph=[node.prefix() for node in graph],
                is_generation=is_generation,
            )

            # tokenize input / output
            x = [eos, *graph_tokens, eos]
            y = [
                eos,
                B_NODE_WORD,
                *next_node.eq_node.prefix_tokens(),
                E_NODE_WORD,
                *tactic.tokenize(),
                eos,
            ]
        elif task == "eq_bwd_rl":
            x = [eos, *next_node.tokenize(), eos]
            y = [eos, *tactic.tokenize(), eos]
        else:
            raise RuntimeError(f"Unexpected task: {task}")

        # skip too long sequences
        if max(len(x), len(y)) > self.params.batch.max_len:
            return None

        # index sequences
        x = [self.dico.index(t) for t in x]
        y = [self.dico.index(t) for t in y]

        return {"name": "undefined", "x": x, "y": y, "return": 1}

    # static to be borrowed by EqGraphSampler
    def sample_node_sequence_from_rb(
        self, split: str, rng: np.random.RandomState,
    ) -> Tuple[GraphNode, List[GraphNode], AnnotatedGeneration]:
        """
        Sample from replay buffer.
        """
        assert split == "train"

        # TODO: add
        # if params.eq.cond_gen_with_proved:
        #     assert not replay_buffer.filter_if_rewards_zero

        annotated_generation, _ = self.adv_rb.get_sample(
            split=split, index=None, rng=rng, block=True
        )
        if len(annotated_generation.generation.forward_steps()) == 0:
            raise RuntimeError("This should have been filtered!")
        from evariste.forward.fwd_eq.eq_helpers import history_to_eq_nodes

        nodes, hyps = history_to_eq_nodes(annotated_generation.generation)
        n_gen = len(nodes) - len(hyps)
        annotated_generation.grabbed()
        assert n_gen > 0

        # # compute discounted returns
        # power = torch.FloatTensor(range(n_gen)[::-1])
        # discounts = params.rl_params.replay_buffer.discount ** power
        # returns = discounts * annotated_generation.reward

        # sanity check
        if self.adv_rb.filter_if_rewards_zero:
            # assert any(r != 0 for r in returns)
            assert annotated_generation.reward != 0

        # for the moment, we suppose that the generated
        # node is the last node in the sequence
        # TODO: be a bit smarter here (i.e. add heuristic)
        goal = nodes[-1]
        if goal.ntype not in ["transform", "assert"]:
            raise RuntimeError(f"Unexpected node type: {goal.ntype}")

        return goal, nodes, annotated_generation

    def get_worker_type(self) -> WorkerType:
        assert self.params.rl_distributed.is_adversarial_training
        if self.params.parsed_tasks("eq_gen_rl"):
            worker_type = WorkerType.GENERATOR_TRAINER
        elif self.params.parsed_tasks("eq_fwd_rl"):
            worker_type = WorkerType.PROVER_TRAINER
        elif self.params.parsed_tasks("eq_bwd_rl"):
            worker_type = WorkerType.PROVER_TRAINER
        else:
            raise RuntimeError(f"Unexpected eq tasks {self.params.parsed_tasks('eq')}")
        return worker_type

    def get_stats(self) -> Dict[str, float]:
        stats = super().get_stats()
        # replay buffer stats
        if self.adv_rb is not None:
            stats.update(self.adv_rb.store_stats_and_reset())
        return stats

    def close(self):
        logger.info("Closing EquationsEnvironment ...")
        super().close()
        if self.forward_prover is not None:
            self.forward_prover.close()
        # if self.replay_buffer:
        #     self.replay_buffer.close()
        # if self.mcts_subproof is not None:
        #     self.mcts_subproof.replay_buffer.close()
        logger.info("Closed EquationsEnvironment")
