# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Union, Tuple, List, Set, Dict
from collections import OrderedDict
from logging import getLogger
from pympler import asizeof
import numpy as np
from evariste.envs.sr.tokenizer import FloatTokenizer

from evariste.model.data.envs.env import DataEnvironment
from evariste.trainer.args import TrainerArgs
from evariste.envs.eq.graph import (
    U_OPS,
    B_OPS,
    C_OPS,
    RULE_VARS,
    VARIABLES,
    CONSTANTS,
    Node,
)
from evariste.envs.sr.env import (
    SREnv,
    XYValues,
    Transition,
)
from evariste.envs.sr.rules import RULES
from evariste.envs.sr.generation import traj_to_goal_with_tactics

from evariste.model.data.dictionary import Dictionary

from evariste.backward.env.equations.graph import (
    N_PREFIX_TOKS,
    prefix_pos_tok,
    prefix_var_name,
)
from evariste.backward.env.sr.graph import (
    SRTactic,
    SRTheorem,
)
from evariste.backward.prover.mcts import MCTSSampleTactics
from evariste.envs.eq.sympy_utils import SympyException, simplify_sp
from evariste.envs.sr.simplification import NotValidExpression, simplify_via_rules


MIN_VALID_TEST_SAMPLES = 100
MIN_VALID_LABELS = 10

# MIN_VALID_TEST_SAMPLES = 5
# MIN_VALID_LABELS = 5


logger = getLogger()


class SRDataEnvironment(DataEnvironment):

    TRAINING_TASKS = ["sr_bwd_backtrack_walk_seq2seq"]

    PROVING_TASKS = {"sr_bwd_backtrack_walk_seq2seq"}

    def __init__(self, dico: Dictionary, params: TrainerArgs):
        """
        Initialize environment.
        """

        super().__init__(
            dico=dico,
            params=params,
            env_name="sr",
            tactic_cls=SRTactic,
            theorem_cls=SRTheorem,
            dataset=params.sr.dataset,
        )

        # valid / test samples
        if self.params.debug.train:
            global MIN_VALID_TEST_SAMPLES
            MIN_VALID_TEST_SAMPLES = 500

        # skip if no Equations task
        sr_tasks = params.parsed_tasks("sr")
        if len(sr_tasks) == 0:
            return

        # there can only be one proving task
        if len(self.PROVING_TASKS & set(sr_tasks)) > 1:
            raise RuntimeError("There can only be one proving task.")

        # create SymbolicRegression environment instance
        logger.info("\n========= Initializing SymbolicRegression Environment =========")
        self.sr_env = SREnv.build(params.sr.dataset.sr_env)
        self.eq_env = self.sr_env.eq_env

        # for each split, the list of labels in the split
        self.labels: Dict[Union[str, Tuple[str, str]], List[str]] = {}
        self.label_to_eq: Dict[str, SRTheorem] = {}

        # build rules
        self.build_rules()

        # skip if no Equations task
        if len(sr_tasks) == 0:
            return

        # build vocabulary
        self.build_vocab()

        # create datasets
        self.create_datasets()
        for split, labels in self.labels.items():
            logger.info(f"Loaded {len(labels)} {split} labels.")
        assert len(self.label_to_eq) == sum(len(x) for x in self.labels.values())
        assert set(self.label_to_eq.keys()) == set(sum(self.labels.values(), []))

        # preview data
        # self.preview_data()

    def set_rng(self, rng: np.random.RandomState):
        """
        Set random generator.
        """
        assert not hasattr(self, "rng")
        self.rng = rng
        self.sr_env.set_rng(rng)

    def build_rules(self):
        """
        Build rules. Only select equivalent rules that have authorized operators.
        """
        self.rules = RULES  # TODO: add different sets of rules
        logger.info(f"Found {len(self.rules)} {self.params.sr.dataset.sr_env} rules.")

        # # filter rules with valid operators  # TODO: implement
        # self.rules = [
        #     rule
        #     for rule in self.rules
        #     if rule.get_unary_ops().issubset(self.params.eq.dataset.env.unary_ops)
        #     and rule.get_binary_ops().issubset(self.params.eq.dataset.env.binary_ops)
        # ]
        # assert len(self.rules) > 0
        # logger.info(f"Filtered {len(self.rules)} rules with valid operators.")

        # split rules by category
        logger.info(f"Found {len(self.rules)} rules in total.")

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

        for symbol in FloatTokenizer().symbols:
            vocab[symbol] = 1

        assert not any(" " in k for k in vocab)

        # update global dictionary
        self.dico.add_vocab(vocab)

    def create_datasets(self):
        """
        Pre-process datasets for fast data iteration.
        """
        logger.info(f"\n==================== Creating datasets ====================")
        params = self.params
        tasks = set(params.parsed_tasks("sr"))

        for task in tasks:
            if task not in self.TRAINING_TASKS:
                raise Exception(f"Unknown task: {task}")

        for task in {"sr_bwd_backtrack_walk_seq2seq"} & (
            tasks | {"sr_bwd_backtrack_walk_seq2seq"}
        ):
            logger.info(f"========== Creating {task} dataset ...")
            self.data[task] = {}
            for split in ["valid", "test"]:
                data = self.create_dataset(task, split)
                self.data[task][split] = data
                size = asizeof.asizeof(self.data[task][split]) / 1024 ** 2
                logger.info(f"====> Size of data for {task} ({split}): {size:.3f}MB")

    def preview_data(self):
        """
        Preview small snapshots of created datasets.
        """
        N_PREVIEW = 10
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

    def generate_backtrack_walk(
        self, rng: np.random.RandomState
    ) -> Tuple[List[Transition], XYValues, str]:

        params = self.params
        sr_env = self.sr_env
        eq_env = self.eq_env
        min_init_ops = params.sr.dataset.min_init_ops
        max_init_ops = params.sr.dataset.max_init_ops
        n_init_ops = rng.randint(min_init_ops, max_init_ops + 1)
        tgt_eq: Node = eq_env.generate_expr(n_init_ops)

        if params.sr.dataset.simplify_equations:
            tgt_eq = simplify_via_rules(tgt_eq)
            try:
                tgt_eq = simplify_sp(tgt_eq)
            except (SympyException) as e:
                return None, None, None
        true_xy: XYValues = sr_env.sample_dataset(tgt_eq)
        if np.isnan(true_xy.y).mean() > params.sr.dataset.nan_rejection_proportion:
            return None, None, None

        trajectory: List[Transition] = sr_env.sample_trajectory(
            tgt_eq
        )  ##TODO: sample trajectory with better strategy (taking into account mse for instance?)
        sr_env.check_trajectory(tgt_eq, trajectory)
        name = f"backtrack_walk_{rng.randint(1 << 60)}"

        return trajectory, true_xy, name

    def get_bwd_backtrack_walk_samples(
        self, store_eq: bool, skip_too_long: bool, rng: np.random.RandomState
    ) -> List[Dict[str, Union[str, List[int]]]]:
        """
        Create a dataset of trajectories for symbolic regression.
        """
        env = self.sr_env
        data = []
        eos = self.dico.eos_word

        # perform a random walk
        trajectory, true_xy, name = self.generate_backtrack_walk(rng)
        if trajectory is None:
            return []
        goals_with_tactics = traj_to_goal_with_tactics(env, trajectory, true_xy)

        # happens when outputs are identical (especially when function is null or non defined)
        # TODO: fix
        if len(goals_with_tactics) == 0:
            return []

        # store valid / test equations
        if store_eq:
            self.label_to_eq[name] = goals_with_tactics[0][0]

        for goal, tactic, _ in goals_with_tactics:

            # tokenize / add sequence delimiters
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

    def create_dataset(self, task: str, split: str):
        """
        Create valid / test datasets.
        """
        assert split in ["valid", "test"]

        # set a random generator to have a consistent valid / test set
        rng = np.random.RandomState(0 if split == "valid" else 1)
        old_rng = self.sr_env.set_rng(rng)

        data: List[Dict] = []
        too_long = 0
        labels: Set[str] = set()

        while len(data) < MIN_VALID_TEST_SAMPLES or (
            split == "valid" and len(labels) < MIN_VALID_LABELS
        ):
            if task == "sr_bwd_backtrack_walk_seq2seq":
                samples = self.get_bwd_backtrack_walk_samples(
                    store_eq=True, skip_too_long=False, rng=rng
                )
            else:
                raise RuntimeError(f"Unknown task {task}")

            for sample in samples:
                labels.add(sample["name"])
                if max(len(sample["x"]), len(sample["y"])) > self.params.batch.max_len:
                    too_long += 1
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
        self.sr_env.set_rng(old_rng)

        return data

    def get_train_sample(self, task: str):
        """
        Get a train sample.
        """
        if task not in self._cache:
            self._cache[task] = []

        # populate cache if empty
        while len(self._cache[task]) == 0:
            if task == "sr_bwd_backtrack_walk_seq2seq":
                samples = self.get_bwd_backtrack_walk_samples(
                    store_eq=False, skip_too_long=True, rng=self.rng
                )
                self._cache[task] = samples
            else:
                raise Exception(f'Unknown task: "{task}"')

        # return element
        return self._cache[task].pop()

    def get_theorem(
        self, task: str, seed: Optional[int] = None
    ) -> Tuple[SRTheorem, str]:
        if not hasattr(self, "rng"):  # TODO: check this
            print(f"SRDataEnvironment setting RNG with seed {seed}", flush=True)
            self.set_rng(np.random.RandomState(seed))
        if task == "sr_bwd_backtrack_walk_seq2seq":
            while True:
                trajectory, true_xy, name = self.generate_backtrack_walk(self.rng)
                goals_with_tactics = traj_to_goal_with_tactics(
                    self.sr_env, trajectory, true_xy
                )
                if len(goals_with_tactics) > 0:  # TODO: clean
                    break
            return goals_with_tactics[0][0], name
        else:
            raise RuntimeError(f"Wrong task: {task}")

    def get_mcts_y_fmt(self, sample: MCTSSampleTactics):
        assert isinstance(sample.goal, SRTheorem)
        assert all(isinstance(tactic, SRTactic) for tactic in sample.tactics)
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
        if task.startswith("sr_mcts"):
            return self.get_mcts_sample(task, split, index)
        else:
            if split == "train":
                # training data is generated on the fly
                return self.get_train_sample(task)
            assert index is not None
            data = self.data[task][split]
            return data[index]

    def close(self):
        logger.info("Closing SRDataEnvironment ...")
        super().close()
        logger.info("Closed SRDataEnvironment")
