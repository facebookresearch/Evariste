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
from pathlib import Path
from tqdm import tqdm

import numpy as np

from evariste import json as json
from evariste.model.data.envs.env import DataEnvironment
from evariste.model.data.dictionary import Dictionary
from evariste.trainer.args import TrainerArgs
from evariste.backward.prover.mcts import MCTSSampleTactics
from evariste.backward.env.isabelle.graph import IsabelleTactic, IsabelleTheorem
from evariste.backward.env.isabelle.tokenizer import IsabelleTokenizer
from evariste.utils import print_memory

# TODO: Add this import when ready
# from evariste.envs.isabelle.env import IsabelleEnv

logger = getLogger()

MIN_VALID_TEST_SAMPLES = 500
MIN_VALID_LABELS = 10

isabelle_task_name = "isabelle_x2y_statement--tactic-EOU_seq2seq"
with_state_only_isabelle_data_path = ()


class IsabelleDataEnvironment(DataEnvironment):
    TRAINING_TASKS = [isabelle_task_name]
    PROVING_TASKS = {isabelle_task_name}

    def __init__(self, dico: Dictionary, params: TrainerArgs):
        """
        Initialise the environment.
        """
        super().__init__(
            dico=dico,
            params=params,
            env_name="isabelle",
            tactic_cls=IsabelleTactic,
            theorem_cls=IsabelleTheorem,
            dataset=params.isabelle.dataset,
        )

        self.data_paths = {"with_state_only": Path(with_state_only_isabelle_data_path)}

        # valid / test samples
        if self.params.debug.train:
            global MIN_VALID_TEST_SAMPLES
            MIN_VALID_TEST_SAMPLES = 500

        # skip if no Equations task
        isabelle_tasks = params.parsed_tasks("isabelle")
        if len(isabelle_tasks) == 0:
            return

        # create Isabelle environment instance
        # TODO: Add the isabelle environment
        # logger.info("\n========= Initialising Isabelle environment =========")
        # self.isabelle_env = IsabelleEnv.build(params.isabelle.dataset.isabelle_env)

        # Build tokenizer
        self.tokenizer: IsabelleTokenizer = IsabelleTokenizer.build(
            "bpe_arxiv_lean_utf8_20k"
        )
        self.tokenizer.init_bpe()
        self.build_vocab()

        # for each split, the list of labels in the split
        self.labels: Dict[Union[str, Tuple[str, str]], List[str]] = {}
        self.labels_to_theorems: Dict[str, IsabelleTheorem] = {}

        # create datasets
        self.create_datasets()
        # for split, labels in self.labels.items():
        #     logger.info(f"Loaded {len(labels)} {split} labels.")
        # assert len(self.labels_to_theorems) == sum(len(x) for x in self.labels.values())
        # assert set(self.labels_to_theorems.keys()) == set(sum(self.labels.values(), []))

        # preview data
        self.preview_data()

    def set_rng(self, rng: np.random.RandomState):
        """
        Set the random number generator.
        """
        assert not hasattr(self, "rng")
        self.rng = rng
        # TODO: Add back in when isabelle env is ready
        # self.isabelle_env.set_rng(rng)

    def build_vocab(self):
        """
        Build the vocabulary.
        TODO: Add the Isabelle specific vocabulary and separators.
        """
        logger.info("\n==================== Building vocabulary ====================")

        vocab = OrderedDict()

        # TODO: Add the actual vocabulary
        bpe_vocab = self.tokenizer.bpe.vocab()
        bpe_vocab = set(bpe_vocab[4:])
        extra_voc = bpe_vocab - vocab.keys()
        vocab.update({tok: 0 for tok in extra_voc})
        logger.info(
            f"Added {len(extra_voc)} extra words from YTTM tokenizer to vocabulary."
        )

        # Update global dictionary
        self.dico.add_vocab(vocab)

    def create_datasets(self):
        """
        Pre-process datasets for fast data iteration.
        """
        logger.info("\n==================== Creating datasets ====================")
        params = self.params
        tasks = set(params.parsed_tasks("isabelle"))

        for task in tasks:
            if task not in self.TRAINING_TASKS:
                raise Exception(f"Unknown task {task}.")

        for task in {isabelle_task_name} & (tasks | {isabelle_task_name}):
            logger.info(f"Creating {task} dataset ...")
            self.data[task] = {}
            for split in ["train", "valid", "test"]:
                data = self.create_dataset(task, split)
                self.data[task][split] = data
                size = asizeof.asizeof(self.data[task][split]) / (1024 ** 2)
                logger.info(f"====> Size of data for {task} ({split}): {size:.3f}MB")

        del self.rng

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
                            print(v)
                            logger.warning(e)
                            v = str(v)
                    else:
                        v = str(v)
                    logger.info(f"{k} {v}")

    def load_bwd_data(self, source: str, split: str):
        """
        Load Isabelle data
        """
        assert split in ["train", "valid", "test"]

        logger.info(
            f"\n================== Load Isabelle dataset {source}, paritition {split} ==================="
        )
        data_path = self.data_paths[source]
        print_memory(logger, f"Loading backward data from {data_path}")

        data_path = Path(data_path)
        assert data_path.is_dir(), data_path

        eos = self.dico.eos_word

        # load data
        data = []
        with open(Path(data_path) / f"{split}.jsonl") as fin:
            for datapoint in tqdm(fin.readlines(), desc=f"Loading {split} dataset"):
                datapoint = json.loads(datapoint.strip())
                x = self.tokenizer.encode(datapoint["x"])
                y = self.tokenizer.encode(datapoint["y"])
                name = datapoint["problem_name"]

                x = [eos, *x, eos]
                y = [eos, *y, eos]
                x = [self.dico.index(t) for t in x]
                y = [self.dico.index(t) for t in y]
                data.append(
                    {"name": name, "x": x, "y": y,}
                )
        return data

    def create_dataset(self, task: str, split: str):
        """
        Create valid / test datasets
        """
        assert split in ["train", "valid", "test"]

        # set a random generator to have a consistent valid / test set
        if not hasattr(self, "rng"):
            rng = np.random.RandomState(0 if split == "valid" else 1)
            self.set_rng(rng)
        # TODO: Add back in when isabelle env is ready
        # old_rng = self.isabelle_env.set_rng(rng)

        data: List[Dict] = []
        too_long = 0
        labels: Set[str] = set()

        while len(data) < MIN_VALID_TEST_SAMPLES or (
            split == "valid" and len(labels) < MIN_VALID_LABELS
        ):
            if task == isabelle_task_name:
                samples = self.load_bwd_data("with_state_only", split)
            else:
                raise RuntimeError(f"Unknown task {task}.")

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
        # TODO: Add back in when isabelle env is ready
        # self.isabelle_env.set_rng(old_rng)

        return data

    def get_train_sample(self, task: str):
        """
        Get a train sample.
        """
        if task not in self._cache:
            self._cache[task] = []

        # populate cache if empty
        while len(self._cache[task]) == 0:
            if task == isabelle_task_name:
                samples = self.load_bwd_data("with_state_only", "train")
                self._cache[task] = samples
            else:
                raise Exception(f"Unknown task {task}.")

        # return element
        return self._cache[task].pop()

    def get_mcts_y_fmt(self, sample: MCTSSampleTactics):
        assert isinstance(sample.goal, IsabelleTheorem)
        assert all(isinstance(tactic, IsabelleTactic) for tactic in sample.tactics)
        eos = self.dico.eos_word
        y = []
        for tactic in sample.tactics:
            this_y = [eos, *tactic.tokenize(), eos]
            this_y = [self.dico.index(tok) for tok in this_y]
            y.append(this_y)
        return y

    def get_sample(self, task: str, split: str, index: Optional[int]):
        """
        Get a data sample
        """
        assert (split == "train") == (index is None)
        if task.startswith("isabelle_mcts"):
            return self.get_mcts_sample(task, split, index)
        else:
            data = self.data[task][split]
            index = self.rng.randint(len(data)) if index is None else index
            return data[index]

    def close(self):
        logger.info("Closing IsabelleDataEnvironment ...")
        super().close()
        logger.info("Closed IsabelleDataEnvironment.")
