# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, List
from logging import getLogger
import os
import numpy as np
import itertools
from torch.utils.data import DataLoader


from evariste import json as json
from evariste.envs.hl.utils import get_theorem_tokens
from evariste.forward.fwd_hl.hl_graph_sampler import (
    HLGraphSampler,
    HL_FWD_TASKS,
    HL_FWD_TASK,
    HL_GEN_TASK,
)
from evariste.backward.env.hol_light.graph import (
    # HLTactic,
    # HLTheorem,
    HLProofNode,
    build_hl_proof_graphs,
)
from evariste.forward.training.helpers import postorder_traversal
from evariste.model.utils import create_subseq_pos
from evariste.model.data.envs.batch_iterator import BatchIterator
from evariste.model.data.envs.multi import MultiEnvironment
from evariste.model.data.dictionary import EOU_WORD, Dictionary
from evariste.model.data.utils import split_task, SplitException
from evariste.trainer.args import TrainerArgs


# dataset size for tasks with on-the-fly generated data
# and where there is no defined valid / test sets
EVAL_DATA_SIZE = 2000


logger = getLogger()


class HOLLightDataEnvironment:
    def __init__(self, dico: Dictionary, params: TrainerArgs):
        """
        Initialize environment.
        """
        self.params = params
        self.dico = dico

        self.graph_sampler: Optional[HLGraphSampler] = None

        # skip if no HOL-Light task
        required_for_multi = MultiEnvironment.requires_env("hl", params)
        if len(params.parsed_tasks("hl")) == 0 and not required_for_multi:
            return

        logger.info("\n=========== Initializing HOL-Light Data Environment ===========")

        # build vocabulary
        self.build_vocab()

        # load data
        self.load_data()

        # create datasets
        self.data = {}
        self.create_datasets()

        # # load MCTS data  # TODO: update
        # self.mcts = MCTSDataLoader(
        #     env_name="hl",
        #     params=params,
        #     tactic_cls=HLTactic,
        #     theorem_cms=HLTheorem,
        # )

        # preview data
        self.preview_data()

    def set_rng(self, rng: np.random.RandomState):
        """
        Set random generator.
        """
        assert not hasattr(self, "rng")
        self.rng = rng

    def build_vocab(self):
        """
        Build vocabulary.
        """
        logger.info("\n==================== Building vocabulary ====================")

        vocab_path = os.path.join(self.params.hl.dataset.data_dir, "vocab")
        assert os.path.isfile(vocab_path)

        vocab = {}
        with open(vocab_path, "r") as f:
            for line in f:
                word, count = line.rstrip().split()
                assert word not in vocab
                assert count.isdigit()
                vocab[word] = int(count)
        logger.info(
            f"Loaded {sum(vocab.values())} words "
            f"({len(vocab)} unique) from {vocab_path}"
        )

        # update global dictionary
        self.dico.add_vocab(vocab)

    def load_data(self):
        """
        Load datasets.
        """
        params = self.params
        hl_data_dir = params.hl.dataset.data_dir
        logger.info("\n==================== Loading data ====================")
        assert os.path.isdir(hl_data_dir)

        # load splits
        logger.info(f"Reloading HOL-Light dataset from {hl_data_dir} ...")
        self.splits = {}
        self.label2split = {}
        for split in ["train", "valid", "test"]:
            path = os.path.join(hl_data_dir, f"split.{split}")
            assert os.path.isfile(path)
            with open(path, "r") as f:
                labels = [x.rstrip() for x in f]
            assert len(labels) == len(set(labels))
            if params.debug.train:
                labels = labels[: self.params.debug.size]
            self.splits[split] = labels
            for label in labels:
                assert label not in self.label2split
                self.label2split[label] = split
            logger.info(f"Loaded {len(labels)} {split} theorems from {path}")

        # load theorems
        logger.info(f"Loading theorems from {path} ...")
        path = os.path.join(hl_data_dir, "dataset.json")
        assert os.path.isfile(path)
        theorems = {}
        n_duplicates = 0
        with open(path, "r") as f:
            for line in f:
                theorem = json.loads(line.rstrip())
                name = theorem["name"]
                if name in theorems:
                    logger.warning(
                        f"{name} ({theorem['filename']}) already found in "
                        f"{theorems[name]['filename']} -- Taking last one."
                    )
                    n_duplicates += 1
                theorems[name] = theorem
        if params.debug.train:
            theorems = {k: v for k, v in theorems.items() if k in self.label2split}
        self.theorems = theorems
        logger.info(
            f"Loaded {len(self.theorems)} theorems from {path} -- "
            f"Skipped {n_duplicates} duplicates."
        )

        # build proof trees / remove theorems with proof trees that could not be parsed
        logger.info("Building proof trees ...")
        self.proof_trees, _ = build_hl_proof_graphs(self.theorems)
        logger.info(
            f"Built {len(self.proof_trees)} proof trees. "
            f"Failed to parse {len(self.theorems) - len(self.proof_trees)} theorem proof trees."
        )
        self.theorems = {
            k: v for k, v in self.theorems.items() if k in self.proof_trees
        }
        self.label2split = {
            k: v for k, v in self.label2split.items() if k in self.proof_trees
        }
        self.splits = {
            k: [x for x in v if x in self.proof_trees] for k, v in self.splits.items()
        }

        # sanity check
        assert len(self.label2split) == sum(len(v) for v in self.splits.values())
        for name, theorem in self.theorems.items():
            assert theorem.keys() == {"name", "filename", "line", "steps"}
            assert theorem["name"] == name
            assert name in self.label2split or params.debug.train
            assert all(tok in self.dico for tok in get_theorem_tokens(theorem))
            assert len(theorem["steps"][-1]["subgoals"]) == 0

    def create_datasets(self):
        """
        Pre-process datasets for fast data iteration.
        """
        logger.info("\n==================== Creating datasets ====================")
        params = self.params

        # create datasets
        # seq2seq tasks (backward training)
        #     hl_x2y_goal--tactic_seq2seq
        #     hl_x2y_goal--tactic-EOU-subgoals_seq2seq
        for task in params.parsed_tasks("hl"):
            try:
                s_x, s_y, _ = split_task(task)
            except SplitException:
                continue
            logger.info(f"========== Creating seq2seq {task} dataset ...")
            self.data[task] = {}
            for split in ["train", "valid", "test"]:
                self.data[task][split] = self.create_seq2seq_dataset(split, s_x, s_y)

        for task in HL_FWD_TASKS:
            if task not in params.parsed_tasks("hl"):
                continue
            if self.graph_sampler is None:
                self.graph_sampler = HLGraphSampler.from_proofs(
                    proofs=self.proof_trees,
                    splits=self.splits,
                    dico=self.dico,
                    max_len=self.params.batch.max_len,
                )
            include_goal = task == HL_FWD_TASK
            self.data[task] = {}
            for split in ["valid", "test"]:
                self.data[task][split] = self.create_fwd_seq2seq_dataset(
                    split, include_goal=include_goal
                )

    def preview_data(self):
        """
        Preview small snapshots of created datasets.
        """
        N_PREVIEW = 15
        logger.info("\n==================== Dataset preview ====================")
        for task, content in self.data.items():
            logger.info(f"========== {task}")
            if "train" in content:
                split = "train"
            elif "valid" in content:
                split = "valid"
            else:
                continue
            data = content[split]
            seed = self.params.slurm_conf.global_rank
            rng = np.random.RandomState(seed if seed >= 0 else None)
            idx = rng.randint(0, len(data), size=(N_PREVIEW,))
            for i in idx:
                for k in ["name", "x", "y"]:
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

    def get_sequences(self, split):
        """
        Return a list of all available sequences for monolingual or
        cross-lingual training.
        """
        assert split in ["train", "valid", "test"]

        logger.info(f"Creating HOL-Light {split} sequences...")

        sequences = []

        # for each theorem
        for name, theorem in self.theorems.items():

            # only consider theorems in that split
            assert theorem["name"] == name
            if self.label2split[name] != split:
                continue

            # retrieve all goals and subgoals
            outs = [x["out"] for x in theorem["proof_tok"]]
            assert len(outs) >= 2

            # for each subgoal
            for i, out in enumerate(outs):
                if len(out["cur"]) == 0:
                    assert i == len(outs) - 1
                    continue
                goal = self.wrap_goal(out["cur"])
                sequences.append(" ".join(goal))

            # sanity check
            sequences_ = set(sequences)
            for out in outs:
                for sg in out["subgoals"]:
                    assert " ".join(self.wrap_goal(sg)) in sequences_

        # sort sequences by length / remove duplicates
        n_total = len(sequences)
        sequences = sorted(set(sequences), key=lambda s: (len(s), s))
        sequences = [x.split() for x in sequences]

        logger.info(
            f"Created {len(sequences)} unique HOL-Light "
            f"{split} sequences ({n_total} total)."
        )

        return sequences

    def get_x2y_sample(
        self, name: str, node: HLProofNode, s_x: List[str], s_y: List[str]
    ):

        assert isinstance(node, HLProofNode)

        s_xy = set(s_x + s_y)
        eos = self.dico.eos_word

        # create input / output sequences
        item = {
            "label": [name],
            "EOU": [EOU_WORD],
        }

        # goal
        if "goal" in s_xy:
            item["goal"] = node.theorem.tokenize()

        # tactic
        if "tactic" in s_xy:
            item["tactic"] = node.tactic.tokenize()

        # next subgoals
        if "subgoals" in s_xy:
            subgoals = [sg.theorem.tokenize() for sg in node.children]
            item["subgoals"] = list(itertools.chain.from_iterable(subgoals))

        # build x and y sequences
        x = [item[s] for s in s_x]
        y = [item[s] for s in s_y]

        # sub-sequences positions
        x_subseq_pos = create_subseq_pos(x, labels=s_x)
        y_subseq_pos = create_subseq_pos(y, labels=s_y)

        # add sequence delimiters
        x = [eos, *sum(x, []), eos]
        y = [eos, *sum(y, []), eos]

        # skip too long sequences
        if max(len(x), len(y)) > self.params.batch.max_len:
            return None

        # index sequences
        try:
            xx = [self.dico.index(t) for t in x]
            yy = [self.dico.index(t) for t in y]
        except KeyError:
            logger.info(" ".join(x))
            logger.info(" ".join(y))
            raise

        return {
            "name": name,
            "x": xx,
            "y": yy,
            "x_subseq_pos": x_subseq_pos,
            "y_subseq_pos": y_subseq_pos,
        }

    def create_seq2seq_dataset(self, split: str, s_x: List[str], s_y: List[str]):
        """
        Create a sequence-to-sequence dataset.
        """
        AVAILABLE_SUB_SEQUENCES = {
            "goal",
            "tactic",
            "subgoals",
            "EOU",
        }
        assert len(s_x) == len(set(s_x)) >= 1
        assert len(s_y) == len(set(s_y)) >= 1
        assert len(set(s_x) & set(s_y)) == 0
        assert all(x in AVAILABLE_SUB_SEQUENCES for x in s_x + s_y)

        eos = self.dico.eos_word
        max_len = self.params.batch.max_len

        data = []
        too_long = 0

        # for each theorem
        for name in self.splits[split]:

            # retrieve proof tree nodes
            proof_tree = self.proof_trees[name]
            nodes = postorder_traversal(proof_tree, rng=None)

            # for each step
            for node in nodes:
                sample = self.get_x2y_sample(name, node, s_x, s_y)
                if sample is None:
                    too_long += 1
                    continue
                data.append(sample)

        # log data statistics
        logger.info(
            f"Created {len(data)} ({s_x} -> {s_y}) {split} sequences. "
            f"Skipped {too_long} too long sequences."
        )
        x_len = [len(x["x"]) for x in data]
        y_len = [len(x["y"]) for x in data]
        logger.info(f"Avg. input len: {np.mean(x_len):.02f} (±{np.std(x_len):.02f})")
        logger.info(f"Avg. output len: {np.mean(y_len):.02f} (±{np.std(y_len):.02f})")

        return data

    def create_fwd_seq2seq_dataset(self, split: str, include_goal: bool):
        assert split in ["valid", "test"]
        assert isinstance(include_goal, bool)

        rng = np.random.RandomState(0 if split == "valid" else 1)

        # generate data
        data = []
        for _ in range(EVAL_DATA_SIZE):
            data.append(
                self.graph_sampler.get_graph_sample(
                    split=split, include_goal=include_goal, rng=rng
                )
            )

        # skip too long sequences (or failed samples)
        too_long = len([1 for item in data if item is None])
        data = [item for item in data if item is not None]

        # log data statistics
        logger.info(
            f"Created {len(data)} forward graph {split} sequences."
            f"Skipped {too_long} too long sequences."
        )
        return data

    # def expand_mcts_sample(self, sample):
    #     assert self.mcts.mcts_fmt.startswith("goal--tactic-EOU"), self.mcts.mcts_fmt

    #     goal, tactics = sample["goal"], sample["tactics"]
    #     eos = self.dico.eos_word
    #     # build x and y sequences
    #     y, target_pi = [], []
    #     for i, pi in enumerate(sample["target_pi"]):
    #         this_y = [eos, *tactics[i].tokenize(), EOU_WORD, eos]
    #         if all(tok in self.dico.word2id for tok in this_y):
    #             this_y = [self.dico.word2id[tok] for tok in this_y]
    #         else:
    #             continue
    #         if len(this_y) > self.params.batch.max_len:
    #             continue
    #         y.append(this_y)
    #         target_pi.append(pi)

    #     x = [eos, *goal.tokenize(), eos]
    #     # Some tokens in HOL subgoals might be unknown. We skip those.
    #     x_ids = []
    #     for tok in x:
    #         if tok in self.dico.word2id:
    #             x_ids.append(self.dico.word2id[tok])
    #     x = x_ids

    #     if len(x) > self.params.batch.max_len or len(y) == 0:
    #         return None

    #     target_pi = torch.FloatTensor(target_pi)
    #     target_pi /= target_pi.sum()

    #     return {
    #         "x": x,
    #         "y": y,
    #         "q_estimate": sample["q_estimate"],
    #         "target_pi": target_pi,
    #     }

    def get_sample(self, task, split, index):
        """
        Get a data sample.
        """
        assert (split == "train") == (index is None)
        if task.startswith("hl_mcts"):
            raise NotImplementedError  # TODO: update
            # pre_sample = self.mcts.get_mcts_sample(split, index, self.rng)
            # if pre_sample is None:
            #     return None
            # sample = self.expand_mcts_sample(pre_sample)
            # if sample is None:  # thrown away because of max_len
            #     return None
            # try:
            #     chosen_tactic = torch.multinomial(sample["target_pi"], 1).item()
            #     return {
            #         "x": sample["x"],
            #         "y": sample["y"][chosen_tactic],
            #         "q_estimate": sample["q_estimate"],
            #     }
            # except RuntimeError:
            #     print("ERROR", sample["target_pi"])
            #     return None
        elif task == HL_FWD_TASK and split == "train":
            return self.graph_sampler.get_graph_sample(
                split=split, include_goal=True, rng=self.rng
            )
        elif task == HL_GEN_TASK and split == "train":
            return self.graph_sampler.get_graph_sample(
                split=split, include_goal=False, rng=self.rng
            )
        else:
            data = self.data[task][split]
            index = self.rng.randint(len(data)) if index is None else index
            return data[index]

    def create_data_loader(self, split: str, task: str):
        """
        Create a data loader for this environment.
        """
        assert split in ["train", "valid", "test"]
        logger.info(f"Creating {split} iterator for {task} ...")

        # dataset
        batch_iterator = BatchIterator(
            self, split, task, params=self.params, pad_index=self.dico.pad_index
        )

        if task.startswith("hl_mcts"):
            raise NotImplementedError  # TODO: update
            # collate_fn = dataset.collate_reduce_padding(
            #     dataset.collate_fn,
            #     key_fn=lambda x: (len(x["x"]), len(x["y"])),
            #     size_fn=size_with_pad,
            # )

        # data loader
        num_workers = self.params.num_workers if split == "train" else 1
        data_loader = DataLoader(
            batch_iterator, batch_size=None, num_workers=num_workers, shuffle=False,
        )

        return data_loader

    #
    # TODO: update / clean below
    #

    # TRAINING_TASKS = [task + type for task in [] for type in ["clm", "mlm", "seq2seq"]]
    # "hl_pred_prev_tact_",  # given goal(T-1) and subgoals(T), predict tactic(T)
    # "hl_pred_next_tact_",  # given goal(T-1), predict tactic(T)
    # "hl_pred_next_goal_",  # given goal(T-1) and tactic(T), predict subgoals(T)
    # "hl_pred_theorem_name_",  # given theorem statement, predict the theorem to apply (train only)
    # "hl_list_split_pred_prev_tact_",  # list_split corruption mode (predict previous tactic)
    # "hl_list_split_pred_next_goal_",  # list_split corruption mode (predict subgoals)
    # "hl_list_subset_pred_prev_tact_",  # list_subset corruption mode (predict previous tactic)
    # "hl_list_subset_pred_next_goal_",  # list_subset corruption mode (predict subgoals)
    # "hl_remove_cmds_pred_prev_tact_",  # remove_cmds corruption mode (predict previous tactic)
    # "hl_remove_cmds_pred_next_goal_",  # remove_cmds corruption mode (predict subgoals)

    # #
    # # hl_pred_prev_tact
    # # hl_pred_next_tact
    # # hl_pred_next_goal
    # #
    # for task in {
    #     "hl_pred_prev_tact_clm",
    #     "hl_pred_prev_tact_mlm",
    #     "hl_pred_prev_tact_seq2seq",
    #     "hl_pred_next_tact_clm",
    #     "hl_pred_next_tact_mlm",
    #     "hl_pred_next_tact_seq2seq",
    #     "hl_pred_next_goal_clm",
    #     "hl_pred_next_goal_mlm",
    #     "hl_pred_next_goal_seq2seq",
    # } & set(self.params.hl_tasks):
    #     logger.info(f"========== Creating {task} dataset ...")
    #     single_sequence = task.endswith("_clm") or task.endswith("_mlm")
    #     self.data[task] = {}
    #     for split in ["train", "valid", "test"]:
    #         data = self.create_hl_pred_next_tact_goal(
    #             self.theorems[split],
    #             to_predict=task[8:-4],
    #             single_sequence=single_sequence,
    #         )
    #         self.data[task][split] = data
    #         logger.info(f"Created {len(data)} {task} {split} elements.\n")

    # #
    # # hl_pred_theorem_name
    # #
    # for task in {
    #     "hl_pred_theorem_name_clm",
    #     "hl_pred_theorem_name_mlm",
    #     "hl_pred_theorem_name_seq2seq",
    # } & set(self.params.hl_tasks):
    #     logger.info(f"========== Creating {task} dataset ...")
    #     single_sequence = task.endswith("_clm") or task.endswith("_mlm")
    #     data = self.create_theorem_names(
    #         self.theorems["train"], single_sequence=single_sequence
    #     )
    #     random.Random(0).shuffle(data)
    #     self.data[task] = {
    #         "train": data,
    #         "valid": data[:1000],
    #         "test": data[1000:2000],
    #     }
    #     logger.info(f"Created {len(data)} theorem names.\n")

    # #
    # # hl_list_split_pred_prev_tact
    # # hl_list_split_pred_next_goal
    # # hl_list_subset_pred_prev_tact
    # # hl_list_subset_pred_next_goal
    # # hl_remove_cmds_pred_prev_tact
    # # hl_remove_cmds_pred_next_goal
    # #
    # for task in {
    #     "hl_list_split_pred_prev_tact_clm",
    #     "hl_list_split_pred_prev_tact_mlm",
    #     "hl_list_split_pred_prev_tact_seq2seq",
    #     "hl_list_split_pred_next_goal_clm",
    #     "hl_list_split_pred_next_goal_mlm",
    #     "hl_list_split_pred_next_goal_seq2seq",
    #     "hl_list_subset_pred_prev_tact_clm",
    #     "hl_list_subset_pred_prev_tact_mlm",
    #     "hl_list_subset_pred_prev_tact_seq2seq",
    #     "hl_list_subset_pred_next_goal_clm",
    #     "hl_list_subset_pred_next_goal_mlm",
    #     "hl_list_subset_pred_next_goal_seq2seq",
    #     "hl_remove_cmds_pred_prev_tact_clm",
    #     "hl_remove_cmds_pred_prev_tact_mlm",
    #     "hl_remove_cmds_pred_prev_tact_seq2seq",
    #     "hl_remove_cmds_pred_next_goal_clm",
    #     "hl_remove_cmds_pred_next_goal_mlm",
    #     "hl_remove_cmds_pred_next_goal_seq2seq",
    # } & set(self.params.hl_tasks):

    #     # read data
    #     logger.info(f"========== Creating {task} dataset ...")
    #     fp = getattr(self.params, task[: task.index("_pred_")] + "_path")
    #     with open(fp, "r") as f:
    #         lines = [json.loads(line.rstrip()) for line in f]

    #     # update vocabulary
    #     new_vocab = lines[0]
    #     self.dico.add_vocab(new_vocab)

    #     # create dataset
    #     single_sequence = task.endswith("_clm") or task.endswith("_mlm")
    #     data = self.create_corrupt_theorems(
    #         lines[1:],
    #         to_predict=task[task.index("_pred_") + 6 : -4],
    #         single_sequence=single_sequence,
    #     )
    #     random.Random(0).shuffle(data)
    #     self.data[task] = {
    #         "train": data[:-2000],
    #         "valid": data[-2000:-1000],
    #         "test": data[-1000:],
    #     }
    #     logger.info(f"Created {len(data)} {task} elements.\n")

    # def create_hl_pred_next_tact_goal(self, theorems, to_predict, single_sequence):
    #     """
    #     Create proof steps:
    #         - given a goalstack, predict the next tactic
    #             - hl_pred_next_tact
    #         - given a goalstack and the next tactic, predict the next goalstack
    #             - hl_pred_next_goal
    #     """
    #     assert to_predict in ["prev_tact", "next_tact", "next_goal"]
    #     data = []

    #     # sequence delimiters
    #     eos = self.dico.eos_word

    #     # for each theorem
    #     for theorem in theorems:

    #         # retrieve theorem name, tactics (commands) and goals (outputs)
    #         name = theorem["name"]
    #         cmds = [x["cmd"] for x in theorem["proof_tok"]]
    #         outs = [x["out"] for x in theorem["proof_tok"]]
    #         assert len(cmds) >= 2

    #         # for each tactic (the first command being the theorem statement)
    #         for cid in range(1, len(cmds)):

    #             # wrap sequences
    #             prev_goal = wrap_goal(outs[cid - 1]["cur"])
    #             next_tact = wrap_cmd(cmds[cid])
    #             next_goal = wrap_subgoals(outs[cid])
    #             assert next_goal[0] != EXCEPTION_WORD

    #             # predict a tactic given a goal, or subgoals given a goal and a tactic, or a tactic given a goal and subgoals
    #             if to_predict == "next_tact":
    #                 x = prev_goal
    #                 y = next_tact
    #             if to_predict == "next_goal":
    #                 x = prev_goal + next_tact
    #                 y = [UNAFFECTED_GOAL_WORD] if next_goal == prev_goal else next_goal
    #             if to_predict == "prev_tact":
    #                 # skip tactics that do not affect the goal
    #                 if next_goal == prev_goal:
    #                     continue
    #                 x = prev_goal + next_goal
    #                 y = next_tact

    #             # add sequence delimiters
    #             x = [eos, *x, eos]
    #             y = [eos, *y, eos]

    #             # index sequences
    #             x = [self.dico.index(t) for t in x]
    #             y = [self.dico.index(t) for t in y]

    #             # add data
    #             data.append(
    #                 {
    #                     "name": name,
    #                     "cmd_id": cid,
    #                     "x": torch.LongTensor(x),
    #                     "y": torch.LongTensor(y),
    #                 }
    #             )

    #     # merge into a single sequence
    #     if single_sequence:
    #         data = self.merge_xy(data)

    #     # remove too long sequences
    #     init_len = len(data)
    #     data = self.remove_too_long(data)

    #     # log data statistics
    #     logger.info(
    #         f"Skipped {init_len - len(data)}/{init_len} too long sequences. Now {len(data)} sequences."
    #     )
    #     if to_predict == "next_goal":
    #         n_exceptions = sum(
    #             [
    #                 int(d["y"][1].item() == self.dico.word2id[EXCEPTION_WORD])
    #                 for d in data
    #             ]
    #         )
    #         n_unaffected = sum(
    #             [
    #                 int(d["y"][1].item() == self.dico.word2id[UNAFFECTED_GOAL_WORD])
    #                 for d in data
    #             ]
    #         )
    #         logger.info(
    #             f"Found {n_exceptions} exceptions ({100 * n_exceptions / len(data):.3f}%), "
    #             f"and {n_unaffected} unaffected goals ({100 * n_unaffected / len(data):.3f}%)"
    #         )

    #     return data

    # def create_corrupt_theorems(self, lines, to_predict, single_sequence):
    #     """
    #     Create augmented data theorems for:
    #         - hl_pred_list_split
    #         - hl_pred_list_subset
    #         - hl_pred_remove_cmds
    #     """
    #     # assert all(line["name"] in self.train_theorem_names for line in lines)
    #     assert to_predict in ["prev_tact", "next_goal"]
    #     data = []

    #     # sequence delimiters
    #     eos = self.dico.eos_word

    #     # for each theorem variation
    #     for line in lines:

    #         # retrieve theorem name / commands / outputs
    #         name = line["name"]
    #         new_cid = line[
    #             "cid"
    #         ]  # command ID from which the commands have been modified
    #         cmds = [None if x is None else x["cmd"] for x in line["proof_tok"]]
    #         outs = [None if x is None else x["out"] for x in line["proof_tok"]]

    #         # sanity check
    #         assert len(cmds) == len(outs)
    #         assert all(out is None or len(out["cur"]) > 0 for out in outs[:-1])
    #         assert ((x is None) == (cid < new_cid - 1) for cid, x in enumerate(cmds))
    #         assert ((x is None) == (cid < new_cid - 1) for cid, x in enumerate(outs))

    #         for cid in range(new_cid, len(cmds)):

    #             # wrap sequences
    #             prev_goal = wrap_goal(outs[cid - 1]["cur"])
    #             next_tact = wrap_cmd(cmds[cid])
    #             next_goal = wrap_subgoals(outs[cid])

    #             # predict a tactic given a goal, or subgoals given a goal and a tactic, or a tactic given a goal and subgoals
    #             if to_predict == "next_goal":
    #                 x = prev_goal + next_tact
    #                 y = [UNAFFECTED_GOAL_WORD] if next_goal == prev_goal else next_goal
    #             if to_predict == "prev_tact":
    #                 # skip tactics that do not affect the goal or cause exceptions
    #                 if next_goal == prev_goal or next_goal[0] == EXCEPTION_WORD:
    #                     continue
    #                 x = prev_goal + next_goal
    #                 y = next_tact

    #             # add sequence delimiters
    #             x = [eos, *x, eos]
    #             y = [eos, *y, eos]

    #             # index sequences
    #             x = [self.dico.index(t) for t in x]
    #             y = [self.dico.index(t) for t in y]

    #             # add data
    #             data.append(
    #                 {
    #                     "name": name,
    #                     "cmd_id": cid,
    #                     "new_cmd_id": new_cid,
    #                     "x": torch.LongTensor(x),
    #                     "y": torch.LongTensor(y),
    #                 }
    #             )

    #     # merge into a single sequence
    #     if single_sequence:
    #         data = self.merge_xy(data)

    #     # remove too long sequences
    #     init_len = len(data)
    #     data = self.remove_too_long(data)

    #     # log data statistics
    #     logger.info(
    #         f"Skipped {init_len - len(data)}/{init_len} too long sequences. Now {len(data)} sequences."
    #     )
    #     if to_predict == "next_goal":
    #         n_exceptions = sum(
    #             [
    #                 int(d["y"][1].item() == self.dico.word2id[EXCEPTION_WORD])
    #                 for d in data
    #             ]
    #         )
    #         n_unaffected = sum(
    #             [
    #                 int(d["y"][1].item() == self.dico.word2id[UNAFFECTED_GOAL_WORD])
    #                 for d in data
    #             ]
    #         )
    #         logger.info(
    #             f"Found {n_exceptions} exceptions ({100 * n_exceptions / len(data):.3f}%), "
    #             f"and {n_unaffected} unaffected goals ({100 * n_unaffected / len(data):.3f}%)"
    #         )

    #     return data

    # def create_theorem_names(self, theorems, single_sequence):
    #     """
    #     Create theorem names dataset.
    #         - hl_pred_theorem_name
    #     """
    #     data = []

    #     # sequence delimiters
    #     eos = self.dico.eos_word

    #     # for each theorem
    #     for theorem in theorems:

    #         name = theorem["name"]

    #         # applying MATCH_ACCEPT_TAC with the appropriate theorem directly proves the goal
    #         x = wrap_goal(theorem["proof_tok"][0]["out"]["cur"])
    #         y = wrap_cmd(f"e ( MATCH_ACCEPT_TAC {name} ) ;;".split())

    #         # add sequence delimiters
    #         x = [eos, *x, eos]
    #         y = [eos, *y, eos]

    #         # index sequences
    #         x = [self.dico.index(t) for t in x]
    #         y = [self.dico.index(t) for t in y]

    #         # add data
    #         data.append(
    #             {"name": name, "x": torch.LongTensor(x), "y": torch.LongTensor(y)}
    #         )

    #     # merge into a single sequence
    #     if single_sequence:
    #         data = self.merge_xy(data)

    #     # remove too long sequences
    #     init_len = len(data)
    #     data = self.remove_too_long(data)
    #     logger.info(
    #         f"Skipped {init_len - len(data)}/{init_len} too long sequences. Now {len(data)} sequences."
    #     )

    #     return data
