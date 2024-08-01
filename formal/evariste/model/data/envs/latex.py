# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os

from evariste.model.data.loader import load_binarized
from evariste.model.data.dataset import StreamDataset


logger = getLogger()


class LatexDataEnvironment:

    TRAINING_TASKS = ["latex_clm", "latex_mlm", "latex_mass"]

    def __init__(self, dico, params):
        """
        Initialize environment.
        """
        self.dico = dico
        self.params = params

        # skip if no LaTeX task
        if len(params.parsed_tasks("latex")) == 0:
            return

        # load data
        self.load_data()

        # create datasets
        self.create_datasets()

    def load_data(self):
        """
        Load binarized dataset.
        """
        latex_data_dir = self.params.latex.data_dir
        logger.info(f"Reloading informal dataset from {latex_data_dir} ...")
        assert os.path.isdir(latex_data_dir)
        params = self.params

        # load data
        self.loaded = {}

        for split in ["train", "valid", "test"]:

            # load data / update dictionary parameters / update data
            path = os.path.join(latex_data_dir, f"{split}.pth")
            data = load_binarized(path, params)

            # check dictionaries are identical
            if split == "train":
                dico_ = data["dico"]
            assert data["dico"] == dico_

            # loaded data
            self.loaded[split] = {
                "sentences": data["sentences"],
                "positions": data["positions"],
            }

        # either we reload a model, and the LaTeX dictionary much match the one of the reloaded model
        # or the LaTeX dictionary is used to initialize the dictionary in this experiment
        if self.dico == dico_:
            logger.info(f"Reloaded dictionary matches data dictionary.")
        else:
            assert sum(self.dico.counts.values()) == 0
            self.dico.id2word = dico_.id2word
            self.dico.word2id = dico_.word2id
            self.dico.counts = dico_.counts
            self.dico.check_valid()

    def create_datasets(self):
        """
        Create datasets.
        """
        params = self.params
        latex_tasks = params.parsed_tasks("latex")
        if not any(task in self.TRAINING_TASKS for task in latex_tasks):
            return

        self.data = {}

        for split in ["train", "valid", "test"]:
            self.data[split] = StreamDataset(
                self.loaded[split]["sentences"],
                self.loaded[split]["positions"],
                bs=params.batch.size,
                bptt=params.batch.bptt,
                eos_index=self.dico.eos_index,
            )

    def create_data_loader(self, split: str, task: str):
        """
        Create a data loader for this environment.
        """
        assert split in ["train", "valid", "test"]
        assert task in LatexDataEnvironment.TRAINING_TASKS
        logger.info(f"Creating {split} iterator for {task} ...")

        # iterator
        is_train = split == "train"
        return self.data[split].get_iterator(shuffle=is_train, infinite=is_train)
