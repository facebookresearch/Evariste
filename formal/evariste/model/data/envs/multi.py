# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from torch.utils.data import DataLoader
from logging import getLogger
import os
import numpy as np

from evariste.model.data.dictionary import Dictionary
from evariste.model.data.envs.env import DataEnvironment
from evariste.model.data.loader import load_binarized
from evariste.model.data.envs.batch_iterator import BatchIterator
from evariste.model.trainer_utils import add_noise
from evariste.trainer.args import TrainerArgs


INFORMAL_DATASETS = {
    "python": "",
    "papers": "",
    "arxiv": "",
    "arxivutf8": "",
    "ncen": "",
    "nden": "",
}


logger = getLogger()


class MultiEnvironment(DataEnvironment):

    # TRAINING_TASKS = []
    # multi_LANG_clm
    # multi_LANG_dae_seq2seq
    # multi_LANG1-LANG2-LANG1_bt
    # multi_LANG1-LANG2_seq2seq

    @staticmethod
    def requires_env(s, params):
        """
        Determines whether an environment is needed because
        we invoke it in the MultiEnvironment.
        """
        multi_tasks = set(params.parsed_tasks("multi"))
        for task in multi_tasks:
            if task == f"multi_{s}_clm":
                return True
            if task == f"multi_{s}_cclm":
                return True
            if task == f"multi_{s}_dae_seq2seq":
                return True
            if task.endswith("_bt"):
                langs = task[6:-3].split("-")
                assert len(langs) == 3
                assert langs[0] == langs[2]
                if s in langs:
                    return True
        return False

    def __init__(self, dico: Dictionary, params: TrainerArgs, all_envs):
        """
        Initialize environment.
        """
        super().__init__(
            dico=dico,
            params=params,
            env_name="lean",
            tactic_cls=None,
            theorem_cls=None,
            dataset=params.lean.dataset,
        )

        self.dico = dico
        self.params = params
        self.all_envs = all_envs
        self.formal_envs = set(self.all_envs.keys()) - {"latex"}

        # skip if no multi task
        if len(params.parsed_tasks("multi")) == 0:
            return

        logger.info("\n=========== Initializing Multi Environment ===========")

        # look for required languages
        self.mono_langs = set()
        for task in params.parsed_tasks("multi"):
            # CLM
            if task.endswith("_clm"):
                lang = task[6:-4]
                self.mono_langs.add(lang)
            # CCLM
            elif task.endswith("_cclm"):
                lang = task[6:-5]
                self.mono_langs.add(lang)
            # DAE
            elif task.endswith("_dae_seq2seq"):
                lang = task[6:-12]
                self.mono_langs.add(lang)
            # BT
            elif task.endswith("_bt"):  # BT
                lang1, lang2, lang3 = task[6:-3].split("-")
                assert lang1 == lang3 and lang1 != lang2
                self.mono_langs.add(lang1)
                self.mono_langs.add(lang2)
            else:
                raise Exception(f'Unknown Multi task: "{task}"')

        assert len(self.mono_langs) > 0
        assert self.mono_langs.issubset(self.formal_envs | INFORMAL_DATASETS.keys())

        # language dictionary / update parameters
        self.id2lang = dict(enumerate(sorted(list(self.mono_langs))))
        self.lang2id = {lang: i for i, lang in self.id2lang.items()}
        logger.info(f"Multi Environment languages: {self.id2lang}")

        # load data
        self.load_data()

        # create datasets
        self.create_datasets()

        # preview data
        self.preview_data()

    def set_rng(self, rng: np.random.RandomState):
        """
        Set random generator.
        """
        assert not hasattr(self, "rng")
        self.rng = rng

    def index_sequences(self, sequences):
        """
        Index token sequences.
        """
        eos = self.dico.eos_word
        data = []
        too_long = 0

        for x in sequences:

            # index sequences / add sequence delimiters
            x = [eos, *x, eos]
            x = [self.dico.index(t) for t in x]

            # skip too long sequences
            if len(x) > self.params.batch.max_len:
                too_long += 1
                continue

            data.append(x)

        logger.info(f"Removed {too_long}/{len(sequences)} too long sequences.")

        return data

    def load_informal_sequences(self, data_dir):
        """
        Load informal dataset sequences.
        """
        assert os.path.isdir(data_dir)

        max_len = self.params.batch.max_len
        eos_idx = self.dico.eos_index

        data_dico = None
        remapping = None  # update reloaded data word IDs to match the new dictionary
        sequences = {}

        # for each split
        for split in ["train", "valid", "test"]:

            # reload binarized data
            path = os.path.join(data_dir, f"{split}.pth")
            data = load_binarized(path, self.params)

            # update global dictionary / check dictionaries are identical
            if split == "train":
                data_dico = data["dico"]
                self.dico.add_vocab(data_dico.counts)
                remapping = np.array(
                    [self.dico.index(data_dico[i]) for i in range(len(data_dico))]
                )
            assert data["dico"] == data_dico

            # remap word IDs / sanity check
            sentences = remapping[data["sentences"]]
            positions = data["positions"]
            assert (positions[:, 1] - positions[:, 0]).min() >= 0
            assert len(positions) == (sentences[positions[:, 1]] == eos_idx).sum()

            # remove empty sentences
            non_empty = positions[:, 1] - positions[:, 0] >= 1
            if non_empty.sum() < len(positions):
                n_empty = len(positions) - non_empty.sum()
                positions = positions[non_empty]
                logger.info(f"Ignoring {n_empty} empty sentences.")
            assert (positions[:, 1] - positions[:, 0]).min() >= 1

            # extract sequences
            sequences[split] = [
                [eos_idx, *sentences[a:b].tolist(), eos_idx]
                for a, b in positions
                if b - a + 2 <= max_len
            ]

            # remove too long sequences
            too_long = (positions[:, 1] - positions[:, 0] + 2 > max_len).sum()
            assert len(sequences[split]) + too_long == len(positions)
            logger.info(
                f"Created {len(sequences[split])} informal sequences. "
                f"Skipped {too_long} too long sequences."
            )

        return sequences

    def load_data(self):
        """
        Load datasets.
        """
        logger.info(f"\n==================== Loading datasets ====================")
        params = self.params

        # load monolingual sequences
        self.sequences = {}
        for lang in self.mono_langs:
            logger.info(f"========== Loading {lang} sequences ...")

            # formal sequences
            if lang in self.formal_envs:
                self.sequences[lang] = {}
                for split in ["train", "valid", "test"]:
                    sequences = self.all_envs[lang].get_sequences(split)
                    sequences = self.index_sequences(sequences)
                    self.sequences[lang][split] = sequences

            # informal sequences
            else:
                assert lang in INFORMAL_DATASETS
                data_dir = INFORMAL_DATASETS[lang]
                self.sequences[lang] = self.load_informal_sequences(data_dir)

    def create_dataset(self, task, split):
        """
        Create valid / test datasets.
        """
        assert split in ["valid", "test"]
        data = []
        rng = np.random.RandomState(0 if split == "valid" else 1)

        # CLM
        if task.endswith("_clm"):
            lang = task[6:-4]
            for x in self.sequences[lang][split]:
                data.append({"name": "", "x": x, "lang": self.lang2id[lang]})

        # CCLM
        elif task.endswith("_cclm"):
            lang = task[6:-5]
            for x in self.sequences[lang][split]:
                data.append({"name": "", "x": x, "lang": self.lang2id[lang]})

        # DAE
        elif task.endswith("_dae_seq2seq"):
            lang = task[6:-12]
            for y in self.sequences[lang][split]:
                x = add_noise(y, self.params, self.dico, rng=rng)
                data.append(
                    {
                        "name": "",
                        "x": x,
                        "y": y,
                        "lang1": self.lang2id[lang],
                        "lang2": self.lang2id[lang],
                    }
                )

        # BT
        elif task.endswith("_bt"):
            lang1, lang2, lang3 = task[6:-3].split("-")
            assert lang1 == lang3 and lang1 != lang2
            for x in self.sequences[lang1][split]:
                data.append(
                    {
                        "name": "",
                        "x": x,
                        "lang1": self.lang2id[lang1],
                        "lang2": self.lang2id[lang2],
                    }
                )

        else:
            raise Exception(f'Unknown task: "{task}"')

        logger.info(f"Created {len(data)} {task} {split} sequences.")
        return data

    def create_datasets(self):
        """
        Pre-process datasets for fast data iteration.
        """
        logger.info(f"\n==================== Creating datasets ====================")
        params = self.params

        # create datasets
        for task in params.parsed_tasks("multi"):
            logger.info(f"========== Creating {task} dataset ...")
            self.data[task] = {}
            for split in ["valid", "test"]:
                data = self.create_dataset(task, split)
                self.data[task][split] = data

    def preview_data(self):
        """
        Preview small snapshots of created datasets.
        """
        N_PREVIEW = 10
        logger.info(f"\n==================== Dataset preview ====================")
        for task, content in self.data.items():
            logger.info(f"========== {task}")
            if "train" in content:
                split = "train"
            elif "valid" in content:
                split = "valid"
            else:
                continue
            data = content[split]
            idx = np.random.randint(0, len(data), size=(N_PREVIEW,))
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

    def get_train_sample(self, task):
        """
        Get a train sample.
        """
        # CLM
        if task.endswith("_clm"):
            lang = task[6:-4]
            data = self.sequences[lang]["train"]
            index = self.rng.randint(len(data))
            return {"name": "", "x": data[index], "lang": self.lang2id[lang]}

        # conditional CLM
        elif task.endswith("_cclm"):
            lang = task[6:-5]
            data = self.sequences[lang]["train"]
            index = self.rng.randint(len(data))
            return {"name": "", "x": data[index], "lang": self.lang2id[lang]}

        # DAE
        elif task.endswith("_dae_seq2seq"):
            lang = task[6:-12]
            data = self.sequences[lang]["train"]
            index = self.rng.randint(len(data))
            y = data[index]
            x = add_noise(y, self.params, self.dico, rng=self.rng)
            return {
                "name": "",
                "x": x,
                "y": y,
                "lang1": self.lang2id[lang],
                "lang2": self.lang2id[lang],
            }

        # BT
        elif task.endswith("_bt"):
            lang1, lang2, lang3 = task[6:-3].split("-")
            assert lang1 == lang3 and lang1 != lang2
            data = self.sequences[lang1]["train"]
            index = self.rng.randint(len(data))
            return {
                "name": "",
                "x": data[index],
                "lang1": self.lang2id[lang1],
                "lang2": self.lang2id[lang2],
            }

        else:
            raise Exception(f'Unknown task: "{task}"')

    def get_sample(self, task, split, index):
        """
        Get a data sample.
        """
        assert (split == "train") == (index is None)
        if split == "train":
            return self.get_train_sample(task)
        data = self.data[task][split]
        index = self.rng.randint(len(data)) if index is None else index
        return data[index]

    def create_data_loader(self, split: str, task: str):
        """
        Create a data loader for this environment.
        """
        assert split in ["train", "valid", "test"]
        logger.info(f"Creating {split} iterator for {task} ...")

        batch_iterator = BatchIterator(
            self, split, task, params=self.params, pad_index=self.dico.pad_index
        )
        num_workers = self.params.num_workers if split == "train" else 1
        data_loader = DataLoader(
            batch_iterator, batch_size=None, num_workers=num_workers, shuffle=False,
        )
        return data_loader
