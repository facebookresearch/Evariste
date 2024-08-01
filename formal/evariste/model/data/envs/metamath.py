# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from typing import Optional, Tuple, List, Iterator
import os
import time
import pickle
import psutil
import itertools
import numpy as np

from evariste.forward.online_generation.worker_type import WorkerType
from evariste.forward.fwd_mm import mm_fwd_tasks
from evariste.forward.fwd_mm.mm_fwd_tasks import MMFwdFormat
from evariste.forward.fwd_mm.training.mm_graph_sampler import (
    MMGraphSampler,
    fake_stop_node,
    fwd_x2y_sample,
    MMFwdGraphData,
)
from evariste.forward.fwd_mm.mm_helpers import (
    MMForwardTactic,
    MMEnvInfo,
)

from evariste.comms.store import EmptyStore
from evariste.forward.common import (
    ForwardGoal,
    ForwardStep,
    MaybeForwardStep,
    GenerationHistory,
)
from evariste.forward.fwd_mm.training.common import MMFwdTrainingProof
from evariste.forward.fwd_mm.training.rb_utils import sample_node_sequence_from_rb
from evariste.model.data.envs.replay_buffer_loader import ReplayBuffer
from evariste.model.data.mcts_subproof import (
    parse_mcts_subproof_xy,
    load_mcts_dumps,
    MCTSProofStepSamplerMM,
    ProofStepSample,
)
from evariste.model.utils import create_subseq_pos
from evariste.model.negative_sampler import NegativeSampler
from evariste.model.data.utils import split_task, SplitException
from evariste.model.data.envs.env import DataEnvironment
from evariste.model.data.envs.multi import MultiEnvironment
from evariste.model.data.envs.mcts_loader import MCTSSubProofDataLoader
from evariste.envs.mm.env import MetamathEnv
from evariste.envs.mm.utils import (
    decompress_all_proofs,
    enumerate_nodes,
    get_mm_informal_data,
    get_canonical_order,
    get_canonical_order_with_hyps_first,
    Node,
    get_false_missing_hyps,
    node_tok,
    Node_a_p,
    Node_e,
    reward_quantile_tok,
)
from evariste.model.data.dictionary import (
    Dictionary,
    B_GOAL_WORD,
    E_GOAL_WORD,
    B_HYP_WORD,
    E_HYP_WORD,
    B_THEOREM_WORD,
    E_THEOREM_WORD,
    EOU_WORD,
)

from evariste.backward.env.metamath.env import MMEnvWorker
from evariste.backward.env.metamath.graph import MMTheorem, MMTactic, get_subst
from evariste.backward.prover.mcts import MCTSSampleTactics
from evariste.refac.utils import safe_pkl_load
from evariste.trainer.args import TrainerArgs

logger = getLogger()

N_MAX_NODES = 1024


def label_remap(mm_env: MetamathEnv):
    # map labels to unique labels (to avoid duplicate theorems, e.g. ax-mp / e0a)
    # NOTE: duplicates could potentially be removed from the vocabulary
    # For all th with the same conclusion and hyps, we only keep the theorem with the fewer disjoint
    theorem2label = {}
    for i, label in enumerate(mm_env.labels.keys()):
        assertion = mm_env.labels[label][1]
        theorem = MMTheorem(
            conclusion=" ".join(assertion["tokens"]),
            hyps=[(None, " ".join(h)) for h in assertion["e_hyps"]],
        )
        if theorem not in theorem2label:
            theorem2label[theorem] = {tuple(sorted(assertion["mand_disj"])): label}
        else:
            to_delete = []
            # Check all disjoints for this theorem. If we find any for which this version is a strict subset,
            # remove them from the remapping and replace them with label.
            found = False
            to_iterate = list(theorem2label[theorem].items())
            for mand_disj, remapped in to_iterate:
                remapped_disj = set(mand_disj)
                label_disj = assertion["mand_disj"]
                if remapped_disj.issubset(label_disj):
                    pass  # we're already remapping toward something with less disj
                elif label_disj.issubset(remapped_disj):
                    to_delete.append(mand_disj)
                    theorem2label[theorem][
                        tuple(sorted(assertion["mand_disj"]))
                    ] = label
                    found = True
            # If we've included the remapping already, we're all good.
            # Otherwise that means we have non-comparable disj than all other elements so we add this label.
            if not found:
                theorem2label[theorem][tuple(sorted(assertion["mand_disj"]))] = label

            # clean up keys for which we found a smaller disj set.
            for key in to_delete:
                del theorem2label[theorem][key]

    label_remap = {}
    for i, label in enumerate(mm_env.labels.keys()):
        assertion = mm_env.labels[label][1]
        theorem = MMTheorem(
            conclusion=" ".join(assertion["tokens"]),
            hyps=[(None, " ".join(h)) for h in assertion["e_hyps"]],
        )
        label_disj = assertion["mand_disj"]
        for mand_disj, remapped in theorem2label[theorem].items():
            remapped_disj = set(mand_disj)
            if remapped_disj.issubset(label_disj):
                label_remap[label] = remapped
                break

    assert len(label_remap) == len(mm_env.labels)

    duplicates = {k for k, v in label_remap.items() if k != v}
    logger.info(f"Found {len(duplicates)}/{len(label_remap)} theorem duplicates.")
    return label_remap


# dataset size for tasks with on-the-fly generated data and where there is no
# defined valid / test sets, e.g. mm_fwd_graph_embseq2ptrseq
EVAL_DATA_SIZE = 2000


class MetamathDataEnvironment(DataEnvironment):

    TRAINING_TASKS = [
        # Given a goal and hypotheses, predict the embedding of the theorem to apply
        "mm_goal2rule_seq2emb",
        #
        # Given a goal and hypotheses, predict whether we can solve it
        "mm_critic_seq2tok",
        "mm_critic_seq2seqtok",
        #
        #
        "mm_fwd_graph2subst_seq2seq",  # legacy, please use mm_fwd_seq2seq
        "mm_fwd_seq2seq",
        "mm_gen_seq2seq",
        "mm_fwd_rl",
        # "mm_bwd_[fmt_str]_rl",
        "mm_gen_rl",
        #
        # "mm_seq_[fmt_str]_[clm|mlm|mass]",
        #     Pretraining task (causal LM / masked LM / MASS)
        #     Where fmt_str is a '-' separated string containing that determines the model's output format
        #     Token in that string can be:
        #         goal  : the tokens of the theorem to prove
        #         proof : the forward proof, preceded by the hypotheses
        #     Example: mm_seq_goal-proof_mass
        #
        # "mm_x2y_[fmt_str]_seq2seq"
        #     The above format is accepted,
        #     Where fmt_str is a '-' separated string containing that determines the model's output format
        #     Token in that string can be:
        #         goal      : the tokens of the current goal to solve
        #         statement : statement of the theorem to prove
        #         ehyps     : hypotheses of the theorem to prove, with labels
        #         label     : the label of the theorem to apply
        #         theorem   : all the tokens of the theorem to apply
        #         mandsubst : the substitutions that cannot be deduced with a parser
        #         predsubst : the substitutions that can be deduced with a parser
        #         proof     : forward proof of the theorem
        #         subgoals  : generated subgoals
        #         EOU       : The EOU token that can be used at test_time to early_stop the beam search
        #     The format string is divided into the input and output part with the '--' token
        #     Example: mm_x2y_goal--label-mandsubst-EOU-theorem-predsubst_seq2seq
        #     We are feeding in the goal and outputting everything else, with an EOU token for early stopping
        #     We can also use informal data using these keys (not compatible with the previous ones):
        #         informallatex  : informal LaTeX comment associated with each Metamath statement
        #         formalmm       : formal metamath statement
        #         formallatex    : formal metamath statement converted to LaTeX using the provided LaTeX mapping
        #
        # "mm_mcts_tactic_[fmt_string]" "mm_mcts_critic"
        #     Listen on a socket for training pairs stemming from MCTS proving
        #     Similar to the seq2seq task above "mm_x2y_[fmt_str]_seq2seq", for MCTS
        #
        # "mm_fwd_x2y_[fmt_str]_seq2seq"
        #     The above format is accepted,
        #     Where fmt_str is a '-' separated string containing that determines the model's output format
        #     Token in that string can be:
        #         goal      : the tokens of the current goal to solve
        #         stack     : stack of all nodes statements in the graph
        #         label     : the label of the theorem to apply
        #         theorem   : all the tokens of the theorem to apply
        #         subst     : the substitutions
        #         generated : next node that will be generated
        #         EOU       : The EOU token that can be used at test_time to early_stop the beam search
        #     The format string is divided into the input and output part with the '--' token
        #     Example: mm_fwd_x2y_goal-stack--label-subst-EOU-generated_seq2seq
        #
        # "mm_subproof_mcts_bwd_[fmt_str]_seq2seq",
        #     The above format is accepted.
        #     Format is the same as "mm_x2y_[fmt_str]_seq2seq"
        #     Example: mm_subproof_mcts_bwd_goal--label-mandsubst-EOU-theorem-predsubst_seq2seq
    ]

    def __init__(self, dico: Dictionary, params: TrainerArgs):
        """
        Initialize environment.
        """
        super().__init__(
            dico=dico,
            params=params,
            env_name="mm",
            tactic_cls=MMTactic,
            theorem_cls=MMTheorem,
            dataset=params.mm.dataset,
        )
        # Attributes used in DataEnvironment parent class

        if self.params.batch.quadratic_max_cost != -1:
            # self.collate_size_fn = self.quadratic_size
            # self.max_size = self.params.batch.quadratic_max_cost
            raise NotImplementedError

        # define here to be called in close even if not mm tasks
        self.replay_buffer: Optional[ReplayBuffer] = None

        # skip if no Metamath task
        required_for_multi = MultiEnvironment.requires_env("mm", params)
        if len(params.parsed_tasks("mm")) == 0 and not required_for_multi:
            return

        logger.info("\n=========== Initializing Metamath Data Environment ===========")

        # create Metamath instance and decompress all proofs
        logger.info("Creating Metamath environment ...")
        self.mm_env = self.build_mm_env(self.params.mm.dataset.database_path)
        decompress_all_proofs(self.mm_env)
        self.label_remap = label_remap(self.mm_env)

        # build vocabulary
        self.build_vocab()

        # load data
        self.load_data()

        # # RB buffer connected to actors
        # if replay_buffer_factory.requires_adv_rb(args=params, env_name="mm"):
        #     self.replay_buffer = replay_buffer_factory.from_trainer_args(
        #         params, worker_type=self.get_worker_type(), env_name="mm"
        #     )

        logger.info("Creating a MMGraphSampler")
        self.graph = MMGraphSampler(
            trainer_args=params,
            proof_trees=self.proof_trees,
            dico=self.dico,
            label_remap=self.label_remap,
            replay_buffer=self.replay_buffer,
        )

        # create datasets
        self.create_datasets()

        # load MCTS subproof data
        self.mcts_subproof = None
        if self.params.online_bwd_gen:
            self.mcts_subproof = MCTSSubProofDataLoader(
                tactic_cls=MMTactic, theorem_cls=MMTheorem, params=params
            )

        # preview data
        self.preview_data()

    @staticmethod
    def build_mm_env(database_path: str) -> MetamathEnv:
        mm_env = MetamathEnv(
            filepath=database_path,
            rename_e_hyps=True,
            decompress_proofs=False,
            verify_proofs=False,
            log_level="info",
        )
        mm_env.process()
        return mm_env

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
        assert len(self.mm_env.fs.frames) == 1
        assert len(self.mm_env.fs.frames[0].d) == 0
        assert len(self.mm_env.fs.frames[0].e_labels) == 0

        vocab = {}

        # add constant names
        for x in self.mm_env.fs.frames[0].c:
            assert x not in vocab
            vocab[x] = 1

        # add variable names
        for x in self.mm_env.fs.frames[0].v:
            assert x not in vocab
            vocab[x] = 1

        # add $e hypotheses names
        for i in range(100):
            vocab[f"E_HYP_{i}"] = 0

        # prefix position tokens
        for prefix_pos in range(N_MAX_NODES):
            tok = node_tok(prefix_pos)
            vocab[tok] = 1

        # I am not sure about how to add it in the vocab, if we should always put
        # it or not
        if self.params.rl_params.replay_buffer.n_reward_quantiles > 0:
            n_quantiles = self.params.rl_params.replay_buffer.n_reward_quantiles
            logger.info(f"Adding {n_quantiles} reward quantiles to vocab")
            for quantile in range(n_quantiles):
                tok = reward_quantile_tok(quantile)
                vocab[tok] = 1

        # add $f hypotheses
        for name, tokens in self.mm_env.fs.frames[0].f_labels.items():
            assert name not in vocab
            vocab[name] = 1
            for x in tokens:
                vocab[x] += 1

        # adding label tokens and proof tokens
        for label, (label_type, assertion) in self.mm_env.labels.items():
            assert label not in vocab
            assert label_type in ["$a", "$p"]

            if label_type == "$p":
                if not self.params.mm.graph.axioms_only:
                    vocab[label] = 1
            else:
                vocab[label] = 1

            for x in assertion["tokens"]:
                vocab[x] += 1
            if label_type == "$a":
                continue
            for x in self.mm_env.decompressed_proofs[label]:
                if x in vocab:
                    vocab[x] += 1
                else:
                    logger.warning(f'Found unexpected token in a proof: "{x}"')
                    vocab[x] = 1

        # add LaTeX tokens from informal data
        if hasattr(self, "latex_data"):
            logger.info(
                f"Adding vocabulary from informal data. Currently "
                f"{len(vocab)} tokens ({sum(vocab.values())} total)"
            )
            new_tokens = set()
            for v in self.latex_data.values():
                if v is None:
                    continue
                sequences = [
                    v["informal_latex"],
                    v["formal_latex"]["goal"],
                ]
                sequences += v["formal_latex"]["hyps"]
                for seq in sequences:
                    for token in seq:
                        if token not in vocab:
                            new_tokens.add(token)
                        vocab[token] = vocab.get(token, 0) + 1
            logger.info(
                f"Added vocabulary from informal data. Found {len(new_tokens)} new tokens "
                f"({sum(vocab[x] for x in new_tokens)} total). Now "
                f"{len(vocab)} tokens ({sum(vocab.values())} total)."
            )

        # update global dictionary
        self.dico.add_vocab(vocab)

    def load_data(self):
        """
        Load datasets.
        """
        params = self.params
        mm_data_dir = params.mm.dataset.data_dir
        logger.info("\n==================== Loading data ====================")
        assert os.path.isdir(mm_data_dir)

        # load splits
        logger.info(f"Reloading Metamath dataset from {mm_data_dir} ...")
        self.splits = {}
        self.label2split = {}
        for split in ["train", "valid", "test"]:
            path = os.path.join(mm_data_dir, f"split.{split}")
            assert os.path.isfile(path)
            with open(path, "r", encoding="utf-8") as f:
                labels = [x.rstrip() for x in f]
            assert len(labels) == len(set(labels))
            assert all(self.mm_env.labels[label][0] == "$p" for label in labels)
            if params.debug.train:
                labels = labels[: self.params.debug.size]
            self.splits[split] = labels
            for label in labels:
                assert label not in self.label2split
                self.label2split[label] = split
            logger.info(f"Loaded {len(labels)} {split} theorems from {path}")

        # load proof trees, unless we only have tasks that do not require them
        path = os.path.join(mm_data_dir, "proof_trees.pkl")
        assert os.path.exists(path), path
        logger.info(f"Loading proof trees from {path} ...")
        with open(path, "rb") as f:
            self.all_proof_trees = safe_pkl_load(f)
        if params.debug.train:
            self.all_proof_trees = {
                k: v for k, v in self.all_proof_trees.items() if k in self.label2split
            }
        # remove from the split theorems for which we do not have any proof tree.
        # these are rare and usually correspond to theorems that are not of type
        # Node_a_p, e.g. dummylink, idi, iin1, iin2, iin3
        for k in set(self.label2split.keys()) - set(self.all_proof_trees.keys()):
            logger.warning(f"Theorem {k} has no proof tree.")
            self.splits[self.label2split[k]].remove(k)
            del self.label2split[k]
        assert set(self.all_proof_trees.keys()) == set(self.label2split.keys())
        logger.info(f"Loaded {len(self.all_proof_trees)} proof trees from {path}")

        if params.mm.additional_proofs_path:
            path = params.mm.additional_proofs_path
            assert os.path.isfile(path)
            logger.info(f"Loading additional proof trees from {path} ...")
            with open(path, "rb") as f:
                additional_proofs = pickle.load(f)
            assert set(additional_proofs.keys()).isdisjoint(
                set(self.all_proof_trees.keys())
            )
            additional_proofs = list(additional_proofs.items())
            if params.debug.train:
                additional_proofs = additional_proofs[: params.debug.train]
            logger.info(f"Adding {len(additional_proofs)} proof trees in train")
            for label, proof in additional_proofs:
                self.all_proof_trees[label] = proof
                self.label2split[label] = "train"
                self.splits["train"].append(label)

        # load proof trees
        self.proof_trees = {}
        for split in ["train", "valid", "test"]:
            data = [
                (k, v)
                for k, v in self.all_proof_trees.items()
                if self.label2split[k] == split
            ]
            self.proof_trees[split] = data
            logger.info(f"Processed {len(data)} {split} proof trees.")

        # MCTS subproofs tasks
        mcts_subproof_tasks = params.parsed_tasks("mm_subproof_mcts_bwd_")
        if len(mcts_subproof_tasks) > 0:
            # check task format
            for task in mcts_subproof_tasks:
                allowed = {
                    "goal",
                    "label",
                    "mandsubst",
                    "predsubst",
                    "subgoals",
                    "theorem",
                    "EOU",
                }
                parse_mcts_subproof_xy(task, allowed=",".join(allowed))

            # build env worker
            env_worker = MMEnvWorker(self.params.mm.dataset, mm_env=self.mm_env)
            env_worker.init()

            # load MCTS dumps
            (
                self.mcts_subproof_samplers,
                self.cumulative_mcts_subproofs,
            ) = load_mcts_dumps(
                env_name="mm",
                params=params,
                subproof_params=params.mm.mcts_subproof,
                env_worker=env_worker,
            )

    def create_datasets(self):
        """
        Pre-process datasets for fast data iteration.
        """
        logger.info("\n==================== Creating datasets ====================")
        params = self.params

        # sequence prediction tasks
        #     mm_seq_goal-proof_mass
        for task in params.parsed_tasks("mm"):
            s = task.split("_")
            if s[1] != "seq":
                continue
            assert len(s) == 4 and s[-1] in ["clm", "mlm", "mass"]
            s_x = s[2].split("-")
            logger.info(f"========== Creating seq {task} dataset ...")
            self.data[task] = {}
            for split in ["train", "valid", "test"]:
                data = self.create_seq_dataset(split, s_x)
                self.data[task][split] = data

        # seq2seq tasks
        #     mm_x2y_goal--theorem-mandsubst-predsubst_seq2seq
        #     mm_x2y_goal--label-mandsubst-EOU-theorem-predsubst_seq2seq
        for task in params.parsed_tasks("mm"):
            try:
                s_x, s_y, s_xy = split_task(task)
            except SplitException:
                continue
            logger.info(f"========== Creating seq2seq {task} dataset ...")
            self.data[task] = {}
            for split in ["train", "valid", "test"]:
                # informal / formal dataset
                if {"informallatex", "formalmm", "formallatex"} & s_xy:
                    data = self.create_seq2seq_informal_dataset(split, s_x, s_y)
                else:
                    data = self.create_seq2seq_dataset(split, s_x, s_y, True)
                self.data[task][split] = data

        # critic
        for task in {"mm_critic_seq2tok", "mm_critic_seq2seqtok"} & set(
            params.parsed_tasks("mm")
        ):
            logger.info(f"========== Creating {task} dataset ...")
            self.data[task] = {}
            for split in ["train", "valid", "test"]:
                data = self.create_seq2critic_dataset(
                    split, no_syntactic=params.mm.critic_no_syntactic
                )
                self.data[task][split] = data

        # forward tasks
        fwd_tasks = {
            t for t in params.parsed_tasks("mm") if mm_fwd_tasks.use_graph_sampler(t)
        }
        for task in fwd_tasks:
            logger.info(f"========== Creating {task} dataset ...")
            start = time.time()

            self.data[task] = {}
            for split in ["valid", "test"]:  # train is sampled on the fly
                data = self.create_fwd_graph_dataset(task, split)
                self.data[task][split] = data

            logger.info(f"Created {task} dataset in {time.time() - start:.02f}s")

        # MCTS subproofs tasks
        for task in params.parsed_tasks("mm"):
            if not task.startswith("mm_subproof_mcts_bwd_"):
                continue
            self.data[task] = {}
            for split in ["valid", "test"]:  # train is sampled on the fly
                data = self.create_mcts_subproof_bwd_dataset(task, split)
                self.data[task][split] = data

    def preview_data(self):
        """
        Preview small snapshots of created datasets.
        """
        N_PREVIEW = 15
        logger.info("\n==================== Dataset preview ====================")
        for task, content in self.data.items():
            if "train" in content:
                split = "train"
            elif "valid" in content:
                split = "valid"
            else:
                continue
            logger.info(f"========== {task} ({split})")
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

        logger.info(f"Creating Metamath {split} sequences...")

        sequences = []

        # for each theorem
        for name, root_node in self.proof_trees[split]:

            assert self.mm_env.labels[name][0] == "$p"

            # retrieve all proof tree nodes
            nodes = enumerate_nodes(
                root=root_node, ignore_f_e=True, ignore_empty=False, no_syntactic=True,
            )

            # for each node
            for node in nodes:
                if node.statement[0] != "|-":
                    continue
                goal = self.wrap_goal(node, node)
                sequences.append(" ".join(goal))

        # for the training split, also add axiom statements
        if split == "train":
            for _, (label_type, assertion) in self.mm_env.labels.items():
                if label_type == "$a" and assertion.tokens[0] == "|-":
                    theorem = MMTheorem(
                        conclusion=assertion.tokens_str,
                        hyps=[(None, " ".join(hyp)) for hyp in assertion.e_hyps],
                    ).tokenize()
                    sequences.append(" ".join(theorem))

        # sort sequences by length / remove duplicates
        n_total = len(sequences)
        sequences = sorted(set(sequences), key=lambda s: (len(s), s))
        sequences = [x.split() for x in sequences]

        logger.info(
            f"Created {len(sequences)} unique Metamath "
            f"{split} sequences ({n_total} total)."
        )

        return sequences

    def build_negative_sampler(self):
        """
        Negative sampler to predict theorems without labels.
        """
        logger.info("Building negative sampler...")
        eos = self.dico.eos_word
        max_len = self.params.batch.max_len
        labels = list(self.mm_env.labels.keys())
        sequences = []
        too_long = []
        for label in labels:
            seq = [eos, *self.wrap_theorem(label), eos]
            if len(seq) > max_len:
                too_long.append(label)
                seq = seq[: max_len - 1] + seq[-1:]
            sequences.append(seq)
        logger.warning(
            f"Found sequences for {len(labels)} labels. {len(too_long)} sequences were "
            f"too long and were truncated to {max_len} tokens: {', '.join(too_long)}."
        )
        return NegativeSampler(
            labels=labels,
            sequences_lst=sequences,
            dico=self.dico,
            worst_offenders=self.params.mm.neg.worst_offenders,
            dim=self.params.model.enc_emb_dim,
            fp16=self.params.model.fp16,
            tf_build_emb=self.params.model.tf_build_emb,
        )

    def wrap_goal(self, node: Node_a_p, root_node: Node_a_p) -> List[str]:
        """
        Wrap Metamath goal.
        A goal is composed of:
            - a set of essential hypotheses
            - a statement
        Essential hypotheses are taken from the root node. Either return everything
        in a single sequence, or multiple sequences (e.g. for the embedder.)
        TODO: randomly remove not required $e hypotheses / shuffle them
        """
        assert node.ltype in ["$a", "$p"]
        assert len(node.statement) >= 1
        assert all(name in root_node.e_hyps for name in node.e_hyps.keys())
        assert all(
            type(name) is str and name.startswith("E_HYP_") and type(term) is str
            for name, term in root_node.e_hyps.items()
        )
        return MMTheorem(
            conclusion=node.statement_str, hyps=list(root_node.e_hyps.items())
        ).tokenize()

    def wrap_theorem(self, label: str) -> List[str]:
        """
        Wrap a theorem. Can be composed of hypotheses followed
        by the theorem statement, or just the theorem label.
        """
        assertion = self.mm_env.labels[label][1]
        return [
            B_THEOREM_WORD,
            *MMTheorem(
                conclusion=" ".join(assertion["tokens"]),
                hyps=[(None, " ".join(h)) for h in assertion["e_hyps"]],
            ).tokenize()[1:-1],
            E_THEOREM_WORD,
        ]

    def wrap_subst(self, node: Node_a_p) -> Tuple[List[str], List[str]]:
        """
        Wrap the substitutions to apply to a goal. We make a distinction between
        the substitutions that can be predicted given the goal, and the one that
        need to be predicted.
        """
        assert node.ltype in ["$a", "$p"]
        assert len(node.label) > 0
        assert all(
            type(name) is str and len(name.split()) == 1 and type(term) is str
            for name, term in node.substitutions.items()
        )

        # retrieve predictable substitutions
        assertion = self.mm_env.labels[node.label][1]
        predictable = set(assertion.tokens) & set(node.substitutions.keys())

        # add substitutions
        mand_subst = []
        pred_subst = []
        for name, term in node.substitutions.items():
            seq = MMTactic.wrap_subs(name, term)
            if name in predictable:
                pred_subst.extend(seq)
            else:
                mand_subst.extend(seq)

        return mand_subst, pred_subst

    def check_forward_proof(self, name, root_node):
        """
        Check whether the computed forward proof is the decompressed one.
        This may not be the case if:
            1) the syntactic nodes were removed from proof_trees
            2) the theorem had nodes with the same statement but different proofs,
               before we simplified the proof tree. in this case the resulting
               forward proof should still be valid.
        """
        p0 = self.mm_env.decompressed_proofs[name]
        p1 = root_node.proof.split()
        if p0 != p1:
            logger.warning(
                f"The computed forward proof for {name} differs from the "
                f"decompressed one. {len(p1)} vs {len(p0)} tokens."
            )

    def create_seq_dataset(self, split: str, s_x: List[str]):
        """
        Create a sequence prediction dataset, for CLM / MLM / MASS tasks.
        """
        assert len(s_x) == len(set(s_x)) >= 1
        assert all(x in {"label", "goal", "proof"} for x in s_x)

        eos = self.dico.eos_word
        data = []
        too_long = 0

        # for each theorem
        for name, root_node in self.proof_trees[split]:

            assert self.mm_env.labels[name][0] == "$p"
            if "proof" in s_x:
                self.check_forward_proof(name, root_node)

            # retrieve all proof tree nodes
            nodes = enumerate_nodes(
                root=root_node, ignore_f_e=True, ignore_empty=False, no_syntactic=True,
            )

            # for each node:
            for node in nodes:

                # create input sequences
                item = {"label": [self.label_remap[node.label]]}

                # goal -- we cannot use wrap_goal because we also need hypotheses' labels
                if "goal" in s_x:
                    e_hyps = [
                        [B_HYP_WORD, hyp_name, *hyp_content.split(), E_HYP_WORD]
                        for hyp_name, hyp_content in sorted(
                            node.e_hyps.items(), key=lambda x: int(x[0].split("_")[-1])
                        )
                    ]
                    item["goal"] = [
                        B_THEOREM_WORD,
                        *sum(e_hyps, []),
                        *node.statement,
                        E_THEOREM_WORD,
                    ]

                # forward proof
                if "proof" in s_x:
                    proof = node.proof.split()
                    item["proof"] = proof
                    assert all(hyp_name in proof for hyp_name in node.e_hyps.keys())

                # build x / create sub-sequences positions
                x = [item[s] for s in s_x]
                x_subseq_pos = create_subseq_pos(x, labels=s_x)

                # index sequences / add sequence delimiters
                x = [eos, *sum(x, []), eos]
                x = [self.dico.index(t) for t in x]

                # skip too long sequences
                if len(x) > self.params.batch.max_len:
                    too_long += 1
                    continue

                data.append({"name": name, "x": x, "x_subseq_pos": x_subseq_pos})

        # log data statistics
        logger.info(
            f"Created {len(data)} ({s_x}) {split} sequences. "
            f"Skipped {too_long} too long sequences."
        )
        x_len = [len(x["x"]) for x in data]
        logger.info(f"Avg. sequence len: {np.mean(x_len):.02f} (±{np.std(x_len):.02f})")

        return data

    def get_x2y_core(self, name, s_x, s_y, item):
        # build x and y sequences
        eos = self.dico.eos_word

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
        except TypeError:
            logger.info("ITEM BOOM ERROR")
            logger.info(str(x))
            logger.info(str(y))
            logger.info(str(item))
            raise

        return {
            "name": name,
            "x": xx,
            "y": yy,
            "x_subseq_pos": x_subseq_pos,
            "y_subseq_pos": y_subseq_pos,
        }

    def get_x2y_sample(
        self,
        name: str,
        node: Node_a_p,
        root_node: Node_a_p,
        s_x: List[str],
        s_y: List[str],
    ):
        s_xy = set(s_x + s_y)

        # create input / output sequences
        item = {
            "label": [
                (
                    self.label_remap[node.label]
                    if self.params.mm.graph.remap_label
                    else node.label
                )
            ],
            "EOU": [EOU_WORD],
        }

        # goal / statement / theorem (full token sequence)
        if "goal" in s_xy:
            item["goal"] = self.wrap_goal(node, root_node)
        if "statement" in s_xy:
            item["statement"] = node.statement
        if "theorem" in s_xy:
            item["theorem"] = self.wrap_theorem(node.label)

        # substitutions
        if {"subst", "mandsubst", "predsubst"} & s_xy:
            mand_subst, pred_subst = self.wrap_subst(node)
            item["subst"] = mand_subst + pred_subst
            item["mandsubst"] = mand_subst
            item["predsubst"] = pred_subst

        # forward proof
        if "proof" in s_xy:
            item["proof"] = node.proof.split()

        # next subgoals
        if "subgoals" in s_xy:
            subgoals = [
                " ".join(node.substitutions.get(tok, tok) for tok in sg)
                for sg in self.mm_env.labels[node.label][1].e_hyps
            ]
            item["subgoals"] = sum(
                [[B_GOAL_WORD, *sg.split(), E_GOAL_WORD] for sg in subgoals], []
            )

        # hypotheses required for this subgoal, with their labels
        if "ehyps" in s_xy:
            e_hyps = [
                [B_HYP_WORD, hyp_name] + hyp_content.split() + [E_HYP_WORD]
                for hyp_name, hyp_content in sorted(
                    node.e_hyps.items(), key=lambda x: int(x[0].split("_")[-1])
                )
            ]
            item["ehyps"] = sum(e_hyps, [])

        return self.get_x2y_core(name, s_x, s_y, item)

    def get_generated_bwd(self, task: str, split: str):
        assert split == "train", "No generated data outside train"
        s = task.split("_")
        assert len(s) == 4
        s_x, s_y = [t.split("-") for t in s[2].split("--")]

        proof_trees = self.graph.generated_proofs[split]
        cumulative = self.graph.generated_cumulative[split]
        # sample weighted proof tree
        index = np.searchsorted(
            cumulative,  # a
            self.rng.random() * cumulative[-1],  # v
            side="right",  # a[i-1] <= v < a[i]
        )
        name, root_node = proof_trees[index]
        # sample a node uniformly
        node = self.rng.choice(
            enumerate_nodes(
                root=root_node, ignore_f_e=True, ignore_empty=False, no_syntactic=True
            )
        )
        # generate sample
        return self.get_x2y_sample(name, node, root_node, s_x, s_y)

    def create_seq2seq_dataset(
        self, split: str, s_x: List[str], s_y: List[str], no_syntactic: bool
    ):
        """
        Create a sequence-to-sequence dataset.
        """
        AVAILABLE_SUB_SEQUENCES = {
            "goal",
            "statement",
            "ehyps",
            "label",
            "theorem",
            "subst",
            "mandsubst",
            "predsubst",
            "proof",
            "subgoals",
            "EOU",
        }
        assert len(s_x) == len(set(s_x)) >= 1
        assert len(s_y) == len(set(s_y)) >= 1
        assert len(set(s_x) & set(s_y)) == 0
        assert all(x in AVAILABLE_SUB_SEQUENCES for x in s_x + s_y)

        data = []
        too_long = 0

        # for each theorem
        for name, root_node in self.proof_trees[split]:

            # retrieve all proof tree nodes
            nodes = enumerate_nodes(
                root=root_node, ignore_f_e=True, ignore_empty=False, no_syntactic=True,
            )

            # check forward proof validity
            if "proof" in s_x + s_y:
                self.check_forward_proof(name, root_node)

            # for each node:
            for node in nodes:
                to_append = self.get_x2y_sample(name, node, root_node, s_x, s_y)
                if to_append is None:
                    too_long += 1
                else:
                    data.append(to_append)

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

    def create_seq2seq_informal_dataset(
        self, split: str, s_x: List[str], s_y: List[str]
    ):
        """
        Create the informal to formal (or formal to informal) dataset.
        """
        AVAILABLE_SUB_SEQUENCES = {"informallatex", "formalmm", "formallatex", "EOU"}
        assert len(s_x) == len(set(s_x)) >= 1
        assert len(s_y) == len(set(s_y)) >= 1
        assert len(set(s_x) & set(s_y)) == 0
        assert all(x in AVAILABLE_SUB_SEQUENCES for x in s_x + s_y)

        data = []
        too_long = 0
        eos = self.dico.eos_word
        s_xy = set(s_x + s_y)

        # $a assertions are not in the splits, but still contain
        # relevant informal data. we add them to the training set
        # and keep the same valid / test sets
        names = self.splits[split]
        assert all(self.mm_env.labels[k][0] == "$p" for k in names)
        if split == "train":
            a_names = [k for k, (t, _) in self.mm_env.labels.items() if t == "$a"]
            names = names + a_names

        # for each theorem in the split
        for name in names:

            v = self.latex_data[name]
            if v is None:
                continue

            # create input / output sequences
            item = {"EOU": [EOU_WORD]}

            # informal LaTeX
            if "informallatex" in s_xy:
                item["informallatex"] = v["informal_latex"]

            # formal LaTeX
            if "formallatex" in s_xy:
                seq = []
                for hyp in v["formal_latex"]["hyps"]:
                    seq.extend([B_HYP_WORD, *hyp, E_HYP_WORD])
                seq += v["formal_latex"]["goal"]
                item["formallatex"] = seq

            # formal statement
            if "formalmm" in s_xy:
                item["formalmm"] = self.wrap_theorem(name)

            # build x and y sequences
            x = [item[s] for s in s_x]
            y = [item[s] for s in s_y]

            # sub-sequences positions
            x_subseq_pos = create_subseq_pos(x, labels=s_x)
            y_subseq_pos = create_subseq_pos(y, labels=s_y)

            # add sequence delimiters
            x = [eos, *sum(x, []), eos]
            y = [eos, *sum(y, []), eos]

            # max length
            if max(len(x), len(y)) > self.params.batch.max_len:
                too_long += 1
                continue

            # index sequences
            x = [self.dico.index(t) for t in x]
            y = [self.dico.index(t) for t in y]

            data.append(
                {
                    "name": name,
                    "x": x,
                    "y": y,
                    "x_subseq_pos": x_subseq_pos,
                    "y_subseq_pos": y_subseq_pos,
                }
            )

        logger.info(
            f"Created {len(data)} sequences. Skipped {too_long} too long sequences."
        )
        return data

    def create_seq2critic_dataset(self, split: str, no_syntactic: bool):
        """
        Create data for critic.
        Input: statement goal with hypotheses
        Output: remaining nodes / depth
        Optionally ignore nodes that can be solved with a parser:
            "class", "setvar", "set", "wff"
        """
        eos = self.dico.eos_word
        max_neg = self.params.mm.critic_max_neg

        data_pos = []
        data_neg = []

        # for each theorem
        for name, root_node in self.proof_trees[split]:

            # retrieve all proof tree nodes
            nodes = enumerate_nodes(
                root_node,
                ignore_f_e=True,
                ignore_empty=False,
                no_syntactic=no_syntactic,
            )

            _pos = []
            _neg = []

            # for each node:
            for node in nodes:

                # only have the required hypotheses in the positive examples
                # TODO: add noise with random other hypotheses?
                _pos.append(self.wrap_goal(node, node))
                _neg.extend(get_false_missing_hyps(node, max_neg=max_neg))

            # add sequence delimiters
            _pos = [[eos, *x, eos] for x in _pos]
            _neg = [[eos, *x, eos] for x in _neg]

            # index sequences
            _pos = [[self.dico.index(t) for t in x] for x in _pos]
            _neg = [[self.dico.index(t) for t in x] for x in _neg]

            # add data
            data_pos.append([{"name": name, "x": x, "y": [1]} for x in _pos])
            data_neg.append([{"name": name, "x": x, "y": [0]} for x in _neg])

        # merge data
        data_pos = list(itertools.chain.from_iterable(data_pos))
        data_neg = list(itertools.chain.from_iterable(data_neg))

        # skip too long sequences
        _len_pos = len(data_pos)
        _len_neg = len(data_neg)
        data_pos = [x for x in data_pos if len(x["x"]) <= self.params.batch.max_len]
        data_neg = [x for x in data_neg if len(x["x"]) <= self.params.batch.max_len]

        # log data statistics
        logger.info(
            f"Created {len(data_pos)} critic {split} positive sequences. "
            f"Skipped {_len_pos - len(data_pos)} too long sequences."
        )
        logger.info(
            f"Created {len(data_neg)} critic {split} negative sequences. "
            f"Skipped {_len_neg - len(data_neg)} too long sequences."
        )

        # shuffle data
        rng = np.random.RandomState(42)
        rng.shuffle(data_pos)
        rng.shuffle(data_neg)

        # balanced valid / test
        if split == "valid" or split == "test":
            n_seq = min(len(data_pos), len(data_neg))
            data_pos = data_pos[:n_seq]
            data_neg = data_neg[:n_seq]
            logger.info(f"Now {n_seq} positive and negative {split} sequences.")

        return data_pos + data_neg

    def create_fwd_graph_dataset(self, task: str, split: str):
        assert split in ["valid", "test"]
        rng = np.random.RandomState(0 if split == "valid" else 1)

        # generate data
        data = []
        for _ in range(EVAL_DATA_SIZE):
            data.append(self.graph.get_graph_sample(split, task, rng))

        # skip too long sequences (or failed samples)
        too_long = len([True for item in data if item is None])
        data = [item for item in data if item is not None]

        # log data statistics
        logger.info(
            f"Created {len(data)} forward graph {split} sequences. "
            f"Skipped {too_long} too long sequences."
        )

        return data

    def create_mcts_subproof_bwd_dataset(self, task: str, split: str):
        assert split in ["valid", "test"]
        rng = np.random.RandomState(0 if split == "valid" else 1)

        # generate data
        data = []

        for _ in range(EVAL_DATA_SIZE):
            data.append(self.get_mcts_subproof_bwd_sample(task, split, rng))

        # skip too long sequences (or failed samples)
        too_long = len([True for item in data if item is None])
        data = [item for item in data if item is not None]

        # log data statistics
        logger.info(
            f"Created {len(data)} MCTS subproof step {split} sequences. "
            f"Skipped {too_long} too long sequences."
        )

        return data

    def get_graph_iterator(self, split: str) -> Iterator[Tuple[int, GenerationHistory]]:
        num_gen = 0
        rng = np.random.RandomState()
        assert split == "train", f"dont use this on {split} split"
        while True:
            sampled: MMFwdTrainingProof = self.graph.sample_proof_root(rng, split)
            name, proof_tree = sampled.name, sampled.root
            assert not sampled.generated, "graph iterator sampled something generated ?"
            res = GenerationHistory(
                goal=ForwardGoal(
                    statement="",
                    e_hyps=list(proof_tree.e_hyps.values()),
                    forbidden=None,
                    mand_disj=proof_tree.disjoint,
                    label=name,
                ),
                stack=[],
            )
            ordered = get_canonical_order_with_hyps_first(proof_tree)

            for node in ordered:
                if not isinstance(node, Node_a_p):
                    continue
                children_ids = []
                for c in node.children:
                    for i, candidate in enumerate(ordered):
                        if c.statement_str == candidate.statement_str:
                            children_ids.append(i)
                            break

                assert len(children_ids) == len(
                    node.children
                ), "Didn't find all my children in the graph :("

                tactic = MMForwardTactic(
                    label=node.label,
                    substitutions=node.substitutions,
                    children_ids=children_ids,
                )
                res.append_step(
                    MaybeForwardStep(
                        step=ForwardStep(
                            score=0,
                            normalised_score=0,
                            statement=node.statement_str,
                            children=children_ids,
                            tactic=tactic,
                            env_info=MMEnvInfo(new_disjoints=node.disjoint),
                        ),
                        err=None,
                    )
                )
            yield num_gen, res
            num_gen += 1

    def expand_proofstep_sample(self, s_x, s_y, name, proof_step: ProofStepSample):
        assert isinstance(proof_step.theorem, MMTheorem)
        assert isinstance(proof_step.tactic, MMTactic)
        mand_subst, pred_subst, all_subst = get_subst(
            proof_step.theorem, [proof_step.tactic]
        )
        item = {
            "label": [self.label_remap[proof_step.tactic.label]],
            "theorem": self.wrap_theorem(proof_step.tactic.label),
            "subst": all_subst,
            "mandsubst": mand_subst[0],
            "predsubst": pred_subst[0],
            "EOU": [EOU_WORD],
            "goal": proof_step.theorem.tokenize(),
            "statement": proof_step.theorem.conclusion,
        }

        return self.get_x2y_core(name, s_x, s_y, item)

    def get_mcts_y_fmt(self, sample: MCTSSampleTactics):
        # Ovewrite
        # Separate substitutions into mand and pred
        assert isinstance(sample.goal, MMTheorem)
        assert all(isinstance(tactic, MMTactic) for tactic in sample.tactics)
        mand_subst, pred_subst, all_subst = get_subst(sample.goal, sample.tactics)
        item = {
            "label": [[self.label_remap[tactic.label]] for tactic in sample.tactics],
            "theorem": [self.wrap_theorem(tactic.label) for tactic in sample.tactics],
            "subst": all_subst,
            "mandsubst": mand_subst,
            "predsubst": pred_subst,
            "EOU": [[EOU_WORD]] * len(sample.tactics),
        }
        assert "tactic" in self.mcts or "minproof" in self.mcts
        key = "tactic" if "tactic" in self.mcts else "minproof"
        s_x, s_y = [x.split("-") for x in self.mcts[key].mcts_fmt.split("--")]
        assert len(s_x) == 1 and s_x[0] == "goal", "Unimplemented fmt for mcts"
        y = []
        eos = self.dico.eos_word
        for i, pi in enumerate(sample.tactics):
            this_y = [item[s][i] for s in s_y]
            this_y = [eos, *sum(this_y, []), eos]
            this_y = [self.dico.index(tok) for tok in this_y]
            y.append(this_y)
        return y

    # Overwrite parent class method as for MM the fmt of y is different

    def get_mcts_subproof_bwd_sample(
        self, task: str, split: str, rng: np.random.RandomState
    ):
        """
        Get a subproof from MCTS nodes.
        # TODO: set rng in samplers
        """
        samplers = self.mcts_subproof_samplers[split]
        # select a random sampler
        if self.params.mm.mcts_subproof.weight_samplers_alpha == 0:
            index = rng.randint(len(samplers))
        else:
            cumulative_scores = self.cumulative_mcts_subproofs[split]
            index = np.searchsorted(
                cumulative_scores,  # a
                rng.random() * cumulative_scores[-1],  # v
                side="right",  # a[i-1] <= v < a[i]
            )
        sampler: MCTSProofStepSamplerMM = samplers[index]
        node, root_node = sampler.sample_mm_proof_step()
        name = f"mcts_subproof__{sampler.name}__{sampler.n_gen_proofs}"

        # sanity check
        assert isinstance(node, Node_a_p)
        assert isinstance(root_node, Node_a_p)

        s_x, s_y = parse_mcts_subproof_xy(task)
        return self.get_x2y_sample(name, node, root_node, s_x, s_y)

    def get_mcts_subproof_online_bwd_sample(
        self, task: str, split: str, rng: np.random.RandomState
    ):
        print("SAMPLING PROOF STEP", flush=True)
        sample = self.mcts_subproof.get_mcts_sample(index=None, split=split, rng=rng)
        print("GOT SAMPLE", flush=True)
        if sample is None:
            return None
        print("GOT NON NONE SAMPLE", flush=True)

        # Is this  useful ?
        name = f"mcts_subproof_online_{self.mcts_subproof.n_gen_proofs}"
        s_x, s_y = parse_mcts_subproof_xy(task)
        print("PARSE OK", flush=True)
        return self.expand_proofstep_sample(s_x, s_y, name, sample)

    def get_sample(self, task: str, split: str, index: Optional[int]):
        """
        Get a data sample.
        """
        assert (split == "train") == (index is None), (split, index)
        if split == "train" and mm_fwd_tasks.use_graph_sampler(task):
            return self.graph.get_graph_sample(split, task, rng=self.rng)
        elif task.startswith("mm_mcts"):
            return self.get_mcts_sample(task, split, index)
        elif (
            self.params.mm.graph.generated_proof_path
            and "x2y" in task
            and split == "train"
        ):
            if self.rng.random() < self.params.mm.graph.generated_prob:
                return self.get_generated_bwd(task, split)
        elif task.endswith("rl"):
            return self.get_rl_sample(task, split, self.rng)
        elif task.startswith("mm_subproof_mcts_bwd_") and split == "train":
            return self.get_mcts_subproof_bwd_sample(task, split, self.rng)
        elif task.startswith("mm_subproof_online_mcts_bwd_") and split == "train":
            return self.get_mcts_subproof_online_bwd_sample(task, split, self.rng)
        data = self.data[task][split]
        index = self.rng.randint(len(data)) if index is None else index
        return data[index]

    def get_rl_sample_with_supervised_start(
        self, task: str, split: str, rng: np.random.RandomState
    ):
        """
        Hack to use supervised data while the ReplayBuffer didn't reach the min size
        to begin training. This is useful when training from scratch, not to get
        stuck waiting from rl samples when the model is still random.

        A better solution would be to detect that there is not enough data directly
        in the trainer.rl_step and return.

        Warning: this is only valid since we are using reinforce without baseline
        as RL algorithm.
        """
        try:
            sample = self.get_rl_sample(task, split, rng, block=False)
        except EmptyStore:
            task_type = task.split("_")[1]  # task = mm_{fwd | gen | bwd_fmt}_rl
            if task_type in {"fwd", "gen"}:
                supervised_task = f"mm_{task_type}_seq2seq"
            else:
                fmt = task.split("_")[2]
                supervised_task = f"mm_x2y_{fmt}_seq2seq"
            assert supervised_task in self.params.parsed_tasks("mm"), (
                supervised_task,
                self.params.parsed_tasks("mm"),
            )
            sample = None
            while not sample:
                sample = self.get_sample(supervised_task, split, index=None)

            sample["return"] = 1.0
        return sample

    def get_worker_type(self) -> WorkerType:
        assert self.params.rl_distributed.is_adversarial_training

        is_bwd_prover = bool(
            [
                x
                for x in self.params.parsed_tasks("mm_")
                if x.endswith("_rl")
                and x not in {"mm_fwd_rl", "mm_gen_rl", "mm_gen_critic_rl"}
            ]
        )  # this is a bit silly but whatever.
        is_fwd_prover = bool(self.params.parsed_tasks("mm_fwd_"))
        is_generator = bool(self.params.parsed_tasks("mm_gen_"))
        assert int(is_bwd_prover) + int(is_fwd_prover) + int(is_generator) == 1, (
            (is_bwd_prover, is_fwd_prover, is_generator),
            self.params.parsed_tasks("mm"),
        )
        if is_fwd_prover or is_bwd_prover:
            return WorkerType.PROVER_TRAINER
        else:
            return WorkerType.GENERATOR_TRAINER

    def get_bwd_rl_sample(self, task: str, node: Node, root_node: Node):
        if isinstance(node, Node_e):
            return None
        s_x, s_y, _ = split_task(task)
        x2y_sample = self.get_x2y_sample("", node, root_node, s_x, s_y)
        if x2y_sample is None:
            return x2y_sample
        x2y_sample["return"] = 1
        return x2y_sample

    def get_rl_sample(
        self, task: str, split: str, rng: np.random.RandomState, block: bool = True
    ):
        """
        block: if True, block in replay_buffer.get_sample until replay buffer
        return a sample. if False, it raises EmptyStore if Empty.
        """
        sample, returns, annotated_gen = sample_node_sequence_from_rb(
            split, self.replay_buffer, block, self.params, rng
        )
        goal = sample.root
        nodes = sample.traj
        descendants = set(
            node for node in get_canonical_order(goal) if node.ltype != "$e"
        )
        # credit assignment: we select target only among goal descendants
        candidates = [i for i, n in enumerate(nodes) if n in descendants]
        # for bwd tasks, grab a candidate and gtfo
        if not mm_fwd_tasks.is_fwd_or_gen(task):
            tgt_idx = rng.choice(candidates)
            annotated_gen.grabbed_elem(tgt_idx)
            return self.get_bwd_rl_sample(task, nodes[tgt_idx], goal)

        is_gen_task = mm_fwd_tasks.is_gen_task(task)
        use_stop_action = self.params.mm.stop_action and is_gen_task

        if use_stop_action:
            # adding the possibility to sample the stop action
            candidates.append(len(nodes))

        assert len(candidates) >= 1
        name = "unk"
        tgt_idx = rng.choice(candidates)
        annotated_gen.grabbed_elem(tgt_idx)
        if tgt_idx == len(nodes):
            assert use_stop_action
            tgt = fake_stop_node(goal)
            is_stop = True
        else:
            tgt = nodes[tgt_idx]
            is_stop = False

        graph = nodes[:tgt_idx]

        node2id = {n: i for i, n in enumerate(graph)}
        children_ids = [node2id[c] for c in tgt.children]

        if is_stop:
            assert graph[-1] == goal
            assert children_ids == [len(graph) - 1]

        graph_data = MMFwdGraphData(
            name=name,
            graph=graph,
            goal=goal,
            target=tgt,
            children_ids=children_ids,
            order=nodes,
            target_id=tgt_idx,
            is_stop=is_stop,
            was_proved=sample.proved,
            is_generated=True,
        )

        sample = fwd_x2y_sample(
            fmt=MMFwdFormat.from_task(task, params=self.params),
            max_len=self.params.batch.max_len,
            dico=self.dico,
            label_remap=self.label_remap if self.params.mm.graph.remap_label else None,
            rng=rng,
            graph_data=graph_data,
        )
        if not sample:
            return None
        if is_stop:
            sample["return"] = returns[-1]
        else:
            sample["return"] = returns[tgt_idx]

        if self.params.rl_params.negative_rewards:
            assert (
                self.params.rl_params.replay_buffer.discount == 1
            ), "only discount 1 to ensure all returns are in {0,1}"
            sample["return"] = sample["return"] * 2 - 1  # from {0, 1} to {-1, 1}

        return sample

    def get_stats(self):
        stats = super().get_stats()
        if self.params.online_bwd_gen:
            stats.update(self.mcts_subproof.get_stats())
        return stats

    def close(self):
        logger.info("Closing MetamathDataEnvironment ...")
        super().close()
        try:
            if self.replay_buffer:
                self.replay_buffer.close()
            # Because closing open things is crucial.
            if self.mcts_subproof is not None:
                self.mcts_subproof.replay_buffer.close()
            self.graph.close()
        except AttributeError:
            # MetamathDataEnvironment was not fully initialized
            pass
        logger.info("Closed MetamathDataEnvironment ...")


def _mem() -> str:
    return f"mem_usage: {psutil.virtual_memory().used / 1024 ** 3:.01f}GB"
