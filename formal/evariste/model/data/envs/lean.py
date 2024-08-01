# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, List, Set, Dict
from logging import getLogger
from numbers import Number
from pathlib import Path
import os
import re
import csv
import sys
import time
import pickle
import itertools
import numpy as np
from numpy.random.mtrand import RandomState

from evariste import json as json
from evariste.backward.prover.mcts import MCTSSampleTactics
from evariste.datasets.lean import PACT_VERSION
from evariste.forward.error_conditioning.online_error_sampler import OnlineErrorSampler
from evariste.forward.fwd_lean.lean_fwd_thm_tokenizer import hyp_tok
from evariste.forward.fwd_lean.training.lean_graph_sampler import (
    LeanBaseGraphSampler,
    LEAN_FWD_TASK,
    LEAN_FWD_ERROR_COND_TASK,
    LEAN_FWD_TASKS,
)
from evariste.forward.fwd_lean.training.lean_graph_sampler_advanced import (
    LeanMixedGraphSampler,
)
from evariste.forward.fwd_lean.training.lean_mcts_subproof_dataset import (
    LeanMCTSSubproofDataset,
    LeanMCTSSolvedDataset,
)
from evariste.forward.fwd_lean.training.lean_online_generation_dataset import (
    LeanOnlineGenerationDataset,
)
from evariste.forward.training.graph_sampler import GraphSampler
from evariste.forward.utils.stats_utils import prefix_dict
from evariste.model.data.envs.lean_utils import (
    generate_goal_input,
    proof_to_items,
    load_first_proofs,
)
from evariste.trainer.args import TrainerArgs
from evariste.model.data.envs.multi import MultiEnvironment
from evariste.model.utils import create_subseq_pos
from evariste.model.data.utils import split_task, SplitException
from evariste.backward.env.lean.tokenizer import LeanTokenizer
from evariste.model.data.envs.mcts_loader import MCTSSubProofDataLoader
from evariste.model.data.envs.env import DataEnvironment
from evariste.clusters.utils import clusterify_path
from evariste.utils import load_and_project_conditioning_vectors, COND_TOK, print_memory

from evariste.backward.env.lean.graph import (
    LeanTactic,
    LeanTheorem,
    LeanContext,
    lean_error_str,
)
from evariste.model.data.mcts_subproof import (
    parse_mcts_subproof_xy,
    load_mcts_dumps,
    ProofStepSample,
)
from evariste.model.data.dictionary import (
    EOU_WORD,
    MASK_WORD,
    SUCCESS_WORD,
    NO_EFFECT_WORD,
    UNK_ERROR_WORD,
    B_GOAL_WORD,
    E_GOAL_WORD,
    B_NS_WORD,
    E_NS_WORD,
    BWD_WORD,
    Dictionary,
    SQUID_TOK,
    M_STACK_WORD,
)

from scripts.load_mcts_min_proofs import (
    LeanSupervisedAndBackwardGraphSampler,
    MinProofDataset,
)

from leanml import parse_goal_structured
from leanml.parse_goal import StructuredGoal

EVAL_DATA_SIZE = 2000
N_MAX_HYPS = 100
MAX_SYNTH_EVAL = 1000


logger = getLogger()


def normalize_goal_pp(s: str) -> str:
    assert "\t" not in s, s
    assert " \n" not in s, s
    assert s.strip() == s, s
    s = re.sub(r"\n +", r" ", s)
    s = re.sub(r" +", " ", s)
    assert re.match(r"[0-9]+ goal", s) is None, s
    ng = s.count("⊢")  # number of goals
    if ng > 1:  # Lean does not precise "1 goal"
        s = f"{ng} goals\n{s}"
    return s


class LeanDataEnvironment(DataEnvironment):

    # TRAINING_TASKS = [
    #     "lean_x2y_[statement--tactic-EOU]_seq2seq",
    #     "lean_pact_seq2seq",
    #     "lean_cond_goal_gen_seq2seq",
    #     "lean_prove_self_seq2seq",
    # ]
    MCTS_TRAINING_TASKS = [
        "lean_mcts_critic",
        "lean_mcts_effect",
        "lean_mcts_tactic_statement--tactic-EOU",
    ]
    AVAILABLE_SUB_SEQUENCES = {
        "statement",
        "tactic",
        "EOU",
        "subgoals",
        "bwd",
    }
    PACT_TASKS = [
        "next_lemma_prediction",
        "premise_classification",
        "proof_term_elab",
        "proof_term_prediction",
        "result_elab",
        "skip_proof",
        "theorem_name_prediction",
        "ts_elab",
        "type_prediction",
    ]
    if PACT_VERSION == 1:
        PACT_TASKS.append("proof_step_classification")
    if PACT_VERSION == 3:
        PACT_TASKS.append("subexpr_type_prediction")

    @staticmethod
    def get_pact_xy(pact_task: str) -> Tuple[str, str]:
        return {
            "next_lemma_prediction": ("goal", "next_lemma"),
            "premise_classification": (
                {1: "goal", 3: "goal_and_premise"}[PACT_VERSION],
                "classify_premise",
            ),
            "proof_step_classification": ("goal", "classify_locals"),
            "proof_term_elab": ("proof_term", "elab_proof_term"),
            "proof_term_prediction": ("goal", "proof_term"),
            "result_elab": ("result", "result_elab"),
            "skip_proof": ("skip_proof", "proof_term"),
            "theorem_name_prediction": ("type", "name"),
            "ts_elab": ("goal", "elab_goal"),
            "type_prediction": ("skip_proof", "goal"),
            "subexpr_type_prediction": ("typed_expression", "type"),
        }[pact_task]

    @staticmethod
    def get_pact_task_token(pact_task: str) -> str:
        return f"<PACT_{pact_task.upper()}>"

    def __init__(self, dico: Dictionary, params: TrainerArgs):
        """
        Initialize environment.
        """
        super().__init__(
            dico=dico,
            params=params,
            env_name="lean",
            tactic_cls=LeanTactic,
            theorem_cls=LeanTheorem,
            dataset=params.lean.dataset,
        )

        # init before returning
        self.graph: Optional[GraphSampler] = None
        self.fwd_error_sampler: Optional[OnlineErrorSampler] = None
        self.loaded: Set[Tuple[str, str]] = set()

        # skip if no Lean task
        required_for_multi = MultiEnvironment.requires_env("lean", params)
        if len(params.parsed_tasks("lean")) == 0 and not required_for_multi:
            return

        # This will create the checkpoint if necessary
        self.materialized_dataset = params.lean.dataset.get_materialized()

        logger.info("\n=========== Initializing Lean Data Environment ===========")

        # build tokenizer
        self.tokenizer = LeanTokenizer.build(self.materialized_dataset.tokenizer)
        s_test = (
            "finset α → Prop, h0 : p ∅, step : ∀ (a : α) (s : finset α), (∀ (x : α), "
            "x ∈ s → x < a) → p s → p (insert a s), n : ℕ, ihn : ∀ (s : finset α), "
            "s.card = n → p s, s : finset α, hn : s.card = n.succ, A : s.nonempty, "
            "B : s.max' A ∈ s ⊢ (s.erase (s.max' A)).card = n "
            "⟦⟦⟦⟦⟦ ⟦⟦⟦⟦⟦⟦⟦⟦⟦⟦⟦⟦⟦⟦⟦⟦⟦ ₉₉₉₉₉₉₉₉ ₉₉"
        )
        s_out = self.tokenizer.encode(s_test)
        logger.info(
            f"===== Loaded tokenizer =====\n"
            f"Input: {s_test}\n"
            f"Output ({len(s_out)} tokens): {' '.join(s_out)}"
        )

        self.parsed_rows: Dict[str, List] = {}
        self.row_splits: Dict[str, Dict[str, List[int]]] = {}
        self.splits: Dict[str, Dict[str, Set[str]]] = {}

        # build vocab
        synth_data_dir = Path(self.params.lean.dataset.synthetic_data_dir)
        self.csv_paths = {
            "mathlib": Path(self.params.lean.dataset.data_dir) / "data.csv",
            "synthetic2": synth_data_dir / "v2" / "all.csv",
            "genwithsimp0905": Path(
                clusterify_path("/generated/gen_with_simp_22_09_05/")
            )
            / "train"
            / "gen_with_simp_22_09_05.csv",
        }
        csv.field_size_limit(sys.maxsize)

        # load backward data
        self.parsed_rows: Dict[str, List] = {}
        self.row_splits: Dict[str, Dict[str, List[int]]] = {}
        self.splits: Dict[str, Dict[str, Set[str]]] = {}
        self.load_bwd_data("mathlib", {"train", "test", "valid"})

        if self.params.lean.additional_training_proofs != "":
            self.add_proofs_to_mathlib_train(
                proof_dir=self.params.lean.additional_training_proofs
            )
        print_memory(logger, "Done loading backward data.")
        self.build_vocab()

        # # dump for debug
        # print(f"Dumping {len(self.parsed_rows['mathlib'])} rows.")
        # with open("YOUR_PATH/dump_parsed_rows.pkl", "wb") as f:
        #     pickle.dump(self.parsed_rows['mathlib'], f)

        # load conditioning vectors
        if self.materialized_dataset.conditioning_path is not None:
            self.z_vectors = load_and_project_conditioning_vectors(
                Path(self.materialized_dataset.conditioning_path),
                target_dim=self.params.model.enc_emb_dim,
            )
            self.z_vectors_sorted_names = sorted(self.z_vectors.keys())
            with open(self.csv_paths["mathlib"], "r") as f:
                all_decls = {x["decl_name"] for x in csv.DictReader(f)}
            z_vector_decls = set(self.z_vectors.keys())
            missing = all_decls - z_vector_decls
            logger.info(
                f"=== Loaded {len(self.z_vectors)} z_vectors. "
                f"{len(missing)} missing from data.csv "
                f"{len(all_decls)} decls ({len(missing) * 100 / len(all_decls):.2f}%)"
            )

        # preview data
        # self.preview_data()

        if self.params.online_bwd_gen:
            self.mcts_subproof = MCTSSubProofDataLoader(
                tactic_cls=LeanTactic, theorem_cls=LeanTheorem, params=params
            )

        self.pact_loaded = False

    def maybe_load(self, task: str, split: str):
        if (task, split) in self.loaded:
            return

        # load backward data
        if task.startswith("lean_x2y_"):
            # self.load_bwd_data("mathlib", split)  # already loaded...
            if self.params.lean.additional_training_proofs != "":
                self.add_proofs_to_mathlib_train(
                    proof_dir=self.params.lean.additional_training_proofs
                )

        self.synth_tasks = list(set(self.csv_paths.keys()) - {"mathlib"})
        for k in self.synth_tasks:
            if task.startswith(f"lean_{k}_"):
                self.load_bwd_data(k, {split})

        # load forward data (also used for subgoals prediction)
        self.load_fwd_data(task)

        # load subproof data
        self.load_subproof_data(task)

        # load prove self dataset
        self.goal_pp_data = self.load_prove_self_dataset(task, split)

        # load PACT dataset
        self.pact_json_data: Dict[str, Dict] = {}
        self.load_pact_dataset(task, split)

        # create datasets
        self.create_datasets(task, split)
        if task in self.data and split in self.data[task]:
            self.preview_data(task, split)
        # fwd data (on the fly sampling and online datasets)
        self.build_fwd_data()

        self.loaded.add((task, split))

    def get_sequences(self, split: str) -> List[List[str]]:
        """
        Return a list of all available sequences for monolingual or
        cross-lingual training.
        """
        assert split in ["train", "valid", "test"]

        logger.info(f"Creating Lean {split} sequences...")

        sequences = []

        for item in self.parsed_rows["mathlib"]:
            if item["split"] != split:
                continue
            sequences.append(" ".join(item["statement"]))

        # sort sequences by length / remove duplicates
        n_total = len(sequences)
        sequences = sorted(set(sequences), key=lambda s: (len(s), s))
        sequences = [x.split() for x in sequences]

        logger.info(
            f"Created {len(sequences)} unique Lean "
            f"{split} sequences ({n_total} total)."
        )

        return sequences

    def set_rng(self, rng: np.random.RandomState) -> None:
        """
        Set random generator.
        """
        assert not hasattr(self, "rng")
        self.rng = rng

    def load_bwd_data(self, source: str, splits_to_load: Set[str]):
        """
        Load .csv file and pre-process data.
        """
        logger.info(
            f"\n================== Load dataset ({source} {splits_to_load}) =================="
        )
        print_memory(logger, f"Loading backward data for {source} ...")
        csv_path = self.csv_paths[source]
        if "mathlib" not in source:
            _path = str(csv_path)
            assert _path.endswith(".csv")
            csv_path = Path(f"{_path[:-4]}.{self.params.slurm_conf.local_rank}.csv")
        assert csv_path.is_file(), csv_path
        logger.info(f"Loading data from {csv_path} ...")

        # parse .csv rows
        n_invalid_concl = 0
        start_time = time.time()

        if source not in self.splits:
            self.splits[source] = {}
        if source not in self.parsed_rows:
            self.parsed_rows[source] = []
        if source not in self.row_splits:
            self.row_splits[source] = {}

        for split_to_load in splits_to_load:
            if split_to_load not in self.splits[source]:
                self.row_splits[source][split_to_load] = []
                self.splits[source][split_to_load] = set()
        row_splits = self.row_splits[source]
        parsed_rows = self.parsed_rows[source]
        logger.info(f"{source, splits_to_load, list(row_splits.keys())}")

        # read CSV columns
        with open(csv_path, "r") as f:
            columns = list(next(csv.reader(f)))

        # look for precomputed tokenizations (for fast reloading)
        assert ("tok_statements" in columns) == ("tok_tactics" in columns)
        assert ("tok_statements" in columns) == ("synthetic" in source)
        if "tok_statements" in columns:
            logger.info(f"tok_statements and tok_tactics available!")

        # open CSV file
        f = open(csv_path, "r")
        all_rows = list(csv.DictReader(f))

        key_to_item: Dict[str, int] = {}
        for row_id, row in enumerate(all_rows):

            # invalid conclusions
            if row["goal_pp"].strip() == "":
                n_invalid_concl += 1
                continue

            # invalid namespaces / context
            open_namespaces = (
                ""
                if self.materialized_dataset.pp_full_names
                else row["open_namespaces"]
            )
            context = LeanContext(namespaces=set(open_namespaces.split()))

            # row split
            split = "train"
            eval_splits = {"valid", "test"}.intersection(splits_to_load)
            if source == "mathlib":
                split = row["split"]
            elif eval_splits:
                for split_to_load in eval_splits:
                    mod_100 = {"valid": 0, "test": 1}[split_to_load]
                    if (
                        row_id % 100 == mod_100
                        and len(row_splits[split_to_load]) < MAX_SYNTH_EVAL
                    ):
                        split = split_to_load

            if split not in splits_to_load:
                continue

            goal_pp = normalize_goal_pp(row["goal_pp"])
            item = {
                "name": row["decl_name"],
                "filename": row["filename"],
                "split": split,
                # needed by LeanGraphSampler
                # TODO: fix LeanTheorem.from_tokens()
                "context": context,
                "goal_pp": goal_pp,
                # id for the previous state or nan to match missing values in pandas
                "previous_key": row.get("previous_key", float("nan")),
            }

            # tokenized statement
            if "tok_statements" in row:
                item["statement"] = row["tok_statements"]
            else:
                item["statement"] = " ".join(
                    LeanTheorem(
                        conclusion=goal_pp, context=context, state=None
                    ).tokenize()
                )

            # tokenized tactic
            if "tok_tactics" in row:
                item["tactic"] = row["tok_tactics"]
            else:
                item["tactic"] = " ".join(
                    LeanTactic(row["human_tactic_code"]).tokenize()
                )
            try:
                key = f"{row['filename']}:{row['line']}:{row['column']}"
                key_to_item[key] = len(parsed_rows)
                row_splits[split].append(len(parsed_rows))
            except KeyError:
                raise RuntimeError(f"{source}, {splits_to_load}")

            parsed_rows.append(item)

            # debug -- do not reload everything
            if self.params.debug.train and len(parsed_rows) == 5000:
                break
        found = 0
        for item in parsed_rows:

            item["previous_tactics"] = []

            past_tactics = self.params.lean.dataset.past_tactics
            cur = item
            for p in range(past_tactics):
                next_item = key_to_item.get(cur["previous_key"], None)
                if next_item is None:
                    break
                cur = parsed_rows[next_item]
                if p < past_tactics:
                    item["previous_tactics"].append(cur["tactic"])

            # reverse the two lists to be ordered from old to recent
            item["previous_tactics"] = item["previous_tactics"][::-1]
        if len(item["previous_tactics"]) > 0:
            found += 1

        # close CSV file
        f.close()

        logger.info(
            f"Parsed {len(parsed_rows)} rows in {time.time() - start_time:.3f}s.\n"
            f"Ignored {n_invalid_concl} rows with invalid conclusions.\n"
            f"{found} rows had at least 1 past state.\n"
        )
        for k, v in row_splits.items():
            logger.info(f"{k} rows: {len(v)}")

        # build splits
        for item in parsed_rows:
            name = item["name"]
            split = item["split"]
            if split not in splits_to_load:
                continue
            if source == "mathlib":
                assert all(
                    name not in v or k == split for k, v in self.splits[source].items()
                )
            self.splits[source][split].add(name)
        for split_to_load in splits_to_load:
            logger.info(
                f"Found {len(self.splits[source][split_to_load])} {split_to_load} labels."
            )
        if source == "mathlib":
            assert sum(len(v) for v in self.splits[source].values()) == len(
                set.union(*self.splits[source].values())
            ), "label splits should be disjoint"

    def add_proofs_to_mathlib_train(self, proof_dir: str):
        from evariste.backward.graph import Proof

        logger.info(f"Loading proofs from {proof_dir}")
        pdir = Path(proof_dir)
        assert pdir.exists(), pdir

        proofs: List[Tuple[str, Proof]]
        if pdir.suffix == ".json":
            from evariste.supervised_proofs.common import load_dataset

            metadata, proofs_ = load_dataset(pdir.parent, dataset_name=pdir.stem)
            logger.info(f"Loaded dataset {pdir} with metadata: {metadata}")
            proofs = [(p.label, p.proof) for p in proofs_]
        else:
            proofs = list(load_first_proofs(pdir))

        items = [
            item
            for name, proof in proofs
            for item in proof_to_items(name, proof, split="train")
        ]
        source = "mathlib"
        splits = self.splits[source]
        row_splits = self.row_splits[source]
        parsed_rows = self.parsed_rows[source]
        for item in items:
            split = item["split"]
            name = item["name"]
            assert split == "train"
            row_splits[split].append(len(parsed_rows))
            parsed_rows.append(item)
            assert all(name not in v or k == split for k, v in splits.items())
            splits[split].add(name)
        logger.info(f"Added {len(proofs)} proofs in matlib train ({len(items)} steps)")

    def load_fwd_data(self, task: str):
        if not "subgoals" in task:
            return
        fwd_data_path = Path(self.materialized_dataset.data_dir) / "extracted_fwd.pkl"
        logger.info(f"Loading forward data from {fwd_data_path} ...")
        with open(fwd_data_path, "rb") as fp:
            _fwd_data = pickle.load(fp)
        self.goal_tactic_to_subgoals = {}
        n_duplicates = 0
        for (decl_name, filename), goals_with_tactics in _fwd_data.items():
            for elem in goals_with_tactics:
                subgoals = [node["full_pp"] for node in elem["res"]["nodes"]]
                k = (
                    decl_name,
                    filename,
                    elem["goal_pp"],
                    elem["tactic"],
                )
                if k in self.goal_tactic_to_subgoals:
                    assert self.goal_tactic_to_subgoals[k] == subgoals
                    n_duplicates += 1
                else:
                    self.goal_tactic_to_subgoals[k] = subgoals
        logger.info(
            f"Reloaded {len(self.goal_tactic_to_subgoals)} "
            f"(decl_name, filename, goal, tactic) -> subgoals. "
            f"Found {n_duplicates} duplicates."
        )

    def load_subproof_data(
        self, task: str,
    ):
        if not task.startswith("lean_subproof_mcts_bwd_"):
            return
        allowed = {"statement", "tactic", "EOU", "subgoals"}
        parse_mcts_subproof_xy(task, allowed=",".join(allowed))
        (
            self.mcts_subproof_samplers,
            self.cumulative_mcts_subproofs,
        ) = load_mcts_dumps(
            env_name="lean",
            params=self.params,
            subproof_params=self.params.lean.mcts_subproof,
            env_worker=None,
        )

    def load_prove_self_dataset(self, task: str, split: str):
        if task != "lean_prove_self_seq2seq":
            return
        data_dir = Path(clusterify_path("YOUR_PATH/pact3/"))
        theorems: List[Dict[str, str]] = []  # label to different goal_pp
        path = data_dir / split / "prove_self.json"
        assert path.is_file()
        logger.info(f"Loading data from {path} ...")
        with open(path, "r") as f:
            lines = [json.loads(line.rstrip()) for line in f]
        logger.info(f"Loaded {len(lines)} lines from {path}")
        theorems.extend(lines)
        assert all(x.keys() == {"tactic_state", "name"} for x in theorems)
        return theorems

    def load_jsonl_data(self, path: str) -> Tuple[List[Dict], int]:
        assert os.path.isfile(path), path
        data = []
        n_duplicates = 0
        seen = set()
        with open(path, "r") as f:
            for idx, line in enumerate(f):
                if self.params.debug.train and idx >= 500:  # debugging
                    break
                if line in seen:
                    n_duplicates += 1
                    continue
                seen.add(line)
                sample = json.loads(line)
                assert len(sample) == 2
                data.append(sample)
        return data, n_duplicates

    def load_pact_dataset(self, task: str, split: str) -> None:
        """
        Load .json files and pre-process data.
        We only load valid and test data. Train data is split and will be loaded by different workers.

        - For the train set, they are rebalanced by sampling first a task then a sample (see `get_pact_train_sample`)
        - For the valid and test set, as we go through all samples, the sets are rebalanced by sampling for each
          PACT task a subset of the original set, whose cardinal is set to be the min cardinal of all PACT task sets
          in the considered split.
        """
        if task != "lean_pact_seq2seq":
            return

        logger.info(f"\n==================== Load PACT dataset ====================")

        # load data (except for train where we specify data path)
        print_memory(logger, "Loading PACT JSON data ...")
        self.pact_json_data[split] = {}
        for pact_task in self.PACT_TASKS:
            if split == "train":
                name = f"{pact_task}.json.{self.params.slurm_conf.local_rank}.tok"
            else:
                name = f"{pact_task}.json.tok"
            json_path = Path(self.materialized_dataset.pact_data_dir) / split / name
            if split == "train":  # data will be loaded / processed by workers
                self.pact_json_data[split][pact_task] = json_path
            else:
                data, n_dupl = self.load_jsonl_data(json_path)
                self.pact_json_data[split][pact_task] = data
                logger.info(
                    f"Loaded {len(data):>7} samples ({n_dupl} duplicates) from Lean PACT "
                    f"co-training task {pact_task:<25} ({split:>5} split) from {json_path}"
                )
        print_memory(logger, "Loaded PACT JSON data.")

        # remove valid / test data to have the same number of samples for each PACT task
        if split == "train":
            return
        rng = np.random.RandomState(0)
        data = self.pact_json_data[split]
        min_size = min(len(v) for v in data.values())
        for task in data.keys():
            idx = rng.choice(len(data[task]), size=min_size, replace=False)
            data[task] = [data[task][i] for i in idx]
        logger.info(f"Selecting {min_size} samples for each {split} PACT task.")

    def build_fwd_data(self):
        fwd_tasks = self.params.parsed_tasks("lean_fwd")
        assert all(t in LEAN_FWD_TASKS for t in fwd_tasks), fwd_tasks
        # forward training
        if LEAN_FWD_TASK in self.params.parsed_tasks("lean"):
            task = LEAN_FWD_TASK
            # standard human dataset
            graph_sampler = LeanBaseGraphSampler.from_human_dataset(
                self.params, dico=self.dico, parsed_rows=self.parsed_rows["mathlib"]
            )

            if self.params.lean.graph.generation_prob > 0:
                # mixing with generated data
                if self.params.online_fwd_generation:
                    generated_dataset = LeanOnlineGenerationDataset.from_trainer_args(
                        params=self.params
                    )
                    allow_global_hyps = False
                elif self.params.lean.graph.use_mcts_subproof:
                    generated_dataset = LeanMCTSSubproofDataset.from_trainer_args(
                        params=self.params
                    )
                    allow_global_hyps = (
                        True  # we need global hyps because of uncompleted proofs
                    )
                elif self.params.lean.graph.use_solved_mcts_subproof:
                    generated_dataset = LeanMCTSSolvedDataset.from_trainer_args(
                        params=self.params
                    )
                    allow_global_hyps = False
                else:
                    raise RuntimeError("No generated data specified")
                generated = LeanBaseGraphSampler.from_dataset(
                    params=self.params,
                    dico=self.dico,
                    dataset=generated_dataset,
                    allow_global_hyps=allow_global_hyps,
                )
                graph_sampler = LeanMixedGraphSampler(
                    human=graph_sampler,
                    generated=generated,
                    generated_prob=self.params.lean.graph.generation_prob,
                )
            elif self.params.lean.graph.mcts_min_proof_dir:
                min_proof_dataset = MinProofDataset(
                    mcts_dir=self.params.lean.graph.mcts_min_proof_dir
                )
                generated = LeanBaseGraphSampler.from_dataset(
                    params=self.params,
                    dico=self.dico,
                    dataset=min_proof_dataset,
                    allow_global_hyps=False,
                )
                graph_sampler = LeanSupervisedAndBackwardGraphSampler(
                    human=graph_sampler, generated=generated
                )
            self.graph = graph_sampler

            self.data[task] = {}
            for split in ["valid", "test"]:
                self.data[task][split] = self.create_graph_dataset(split, task)

        if LEAN_FWD_ERROR_COND_TASK in self.params.parsed_tasks("lean"):
            self.fwd_error_sampler = OnlineErrorSampler.from_trainer_args(
                params=self.params, env_name="lean", dico=self.dico
            )
            # adding error_codes to vocab
            assert not self.dico.frozen
            self.dico.add_vocab({k: 1 for k in self.fwd_error_sampler.error_codes()})

    def create_datasets(self, task: str, split: str):
        """
        Create datasets.
        """
        logger.info(f"========== Creating seq2seq {task} // {split} dataset ...")
        self.data[task] = {}

        # store datasets [task][split][samples]
        # self.data: Dict[str, Dict[str, List[Dict]]]

        lean_tasks = set(self.params.parsed_tasks("lean"))

        # conditioned goal generation
        for task in {"lean_cond_goal_gen_seq2seq"} & lean_tasks:
            self.data[task] = {}
            for split in ["valid", "test"]:
                self.data[task][split] = self.create_cond_goal_gen_dataset(split)

        # prove self
        if task == "lean_prove_self_seq2seq":
            self.data[task][split] = self.create_prove_self_dataset(split)
            return

        # MCTS subproofs
        if task.startswith("lean_subproof_mcts_bwd_"):
            self.data[task][split] = self.create_mcts_subproof_bwd_dataset(task, split)
            return

        # PACT dataset. train data will be processed by workers
        if task == "lean_pact_seq2seq":
            if split != "train":
                self.data["lean_pact_seq2seq"][split] = self.create_pact_dataset(split)
            return

        try:
            s_x, s_y, _ = split_task(task)
        except SplitException:
            logger.info(f"Ignoring SplitException on {task}")
            return

        task = task.replace(COND_TOK, "")

        for k in self.synth_tasks + ["x2y"]:
            if task.startswith(f"lean_{k}_"):
                self.data[task][split] = self.create_seq2seq_dataset(
                    split, s_x, s_y, source=k if k != "x2y" else "mathlib"
                )
                return

    def build_vocab(self):
        """
        Build vocabulary.
        """
        logger.info("\n==================== Building vocabulary ====================")

        vocab: Dict[str, int] = {}

        # error words
        vocab.update({x[0]: 0 for x in lean_error_str})

        # special words
        special_words = [
            SUCCESS_WORD,
            NO_EFFECT_WORD,
            UNK_ERROR_WORD,
            EOU_WORD,
            B_GOAL_WORD,
            E_GOAL_WORD,
            B_NS_WORD,
            E_NS_WORD,
            MASK_WORD,
            SQUID_TOK,
        ] + [self.get_pact_task_token(pact_task) for pact_task in self.PACT_TASKS]

        vocab.update({w: 0 for w in special_words})

        # hyps toks
        if self.params.lean.graph.compress_cfg.compress_hyps:
            for hyp_i in range(N_MAX_HYPS):
                tok = hyp_tok(hyp_i)
                vocab[tok] = 1

        # statement / tactic words
        for item in self.parsed_rows["mathlib"]:
            for tok in item["statement"].split() + item["tactic"].split():
                vocab[tok] = vocab.get(tok, 0) + 1

        # add BPE vocabulary to prevent UNK at test time
        logger.info(f"Found {len(vocab)} Lean words.")
        bpe_vocab = self.tokenizer.bpe.vocab()
        bpe_vocab = set(bpe_vocab[4:])  # remove YTTM special words
        extra_voc = bpe_vocab - vocab.keys()
        vocab.update({tok: 0 for tok in extra_voc})
        logger.info(f"Added {len(extra_voc)} extra words from YTTM tokenizer.")

        # update global dictionary
        self.dico.add_vocab(vocab)

    def preview_data(self, task, split):
        """
        Preview small snapshots of created datasets.
        """
        N_PREVIEW = 15
        logger.info("\n==================== Dataset preview ====================")
        content = self.data[task]
        if task != "lean_pact_seq2seq":  # done below
            logger.info(f"========== {task}")
            data = content[split]
            if len(data) == 0:
                raise RuntimeError(f"Found no data for task={task} split={split}")
            rng = np.random.RandomState(0)
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

        # preview Lean PACT dataset
        elif "lean_pact_seq2seq" in self.params.parsed_tasks("lean"):
            if split != "valid":
                return
            task = "lean_pact_seq2seq"
            valid_data = self.pact_json_data["valid"]
            n_pact_tasks = len(valid_data)
            for k, (pact_task, data) in enumerate(valid_data.items()):
                logger.info(f"========== {task}: {pact_task} ({k + 1}/{n_pact_tasks})")
                rng = np.random.RandomState(0)
                seen = 0
                for i in rng.randint(0, len(data), size=(10,)):
                    sample = self.process_pact_json(pact_task, sample=data[i], idx=i)
                    if sample is None:
                        continue
                    x_key = sample["x_key"]
                    y_key = sample["y_key"]
                    x = " ".join(self.dico[tok_id] for tok_id in sample["x"])
                    y = " ".join(self.dico[tok_id] for tok_id in sample["y"])
                    logger.info(
                        # f"(from raw sample) {x_key}: {sample['raw'][x_key]}\n"
                        # f"(from raw sample) {y_key}: {sample['raw'][y_key]}\n"
                        f"x ({x_key}): {x}\n"
                        f"y ({y_key}): {y}\n"
                    )
                    seen += 1
                    if seen == 2:  # only show 2 samples per task
                        break

    def get_x2y_sample(self, row, s_x: List[str], s_y: List[str]):
        s_xy = set(s_x + s_y)
        eos = self.dico.eos_word

        # create input / output sequences
        found_subgoals = False
        item = {"EOU": [EOU_WORD], "bwd": [BWD_WORD]}
        if "statement" in s_xy:
            assert isinstance(row["statement"], str)
            item["statement"] = row["statement"].split()
            for s in row["previous_tactics"]:
                item["statement"] += [M_STACK_WORD] + s.split()
        if "tactic" in s_xy:
            assert isinstance(row["tactic"], str)
            item["tactic"] = row["tactic"].split()
        if "subgoals" in s_xy and not row.get("is_mcts_subproof_sample", False):
            k = (
                row["name"],
                row["filename"],
                row["goal_pp"],
                row["tactic"],
            )
            found_subgoals = k in self.goal_tactic_to_subgoals
            if found_subgoals:
                subgoals = [
                    LeanTheorem(conclusion=sg_pp, context=row["context"], state=None)
                    for sg_pp in self.goal_tactic_to_subgoals[k]
                ]
                item["subgoals"] = sum([sg.tokenize() for sg in subgoals], [])
            else:
                item["subgoals"] = []
        if "subgoals" in s_xy and row.get("is_mcts_subproof_sample", False):
            item["subgoals"] = sum([sg.tokenize() for sg in row["subgoals"]], [])

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
        x = [self.dico.index(t) for t in x]
        y = [self.dico.index(t) for t in y]

        return {
            "name": row["name"],
            "x": x,
            "y": y,
            "x_subseq_pos": x_subseq_pos,
            "y_subseq_pos": y_subseq_pos,
            "found_subgoals": found_subgoals,
        }

    def print_xy_lengths_stats(self, data: List[Dict]) -> None:
        """
        Print stats about input / output sequence lengths.
        """
        x_len = [len(x["x"]) for x in data]
        y_len = [len(x["y"]) for x in data]
        logger.info(f"Avg. input len: {np.mean(x_len):.02f} (±{np.std(x_len):.02f})")
        logger.info(f"Avg. output len: {np.mean(y_len):.02f} (±{np.std(y_len):.02f})")

    def create_seq2seq_dataset(
        self, split: str, s_x: List[str], s_y: List[str], source: str
    ):
        """
        Create a sequence-to-sequence dataset.
        """
        assert split in ["train", "valid", "test"]
        assert len(s_x) == len(set(s_x)) >= 1
        assert len(s_y) == len(set(s_y)) >= 1
        assert len(set(s_x) & set(s_y)) == 0
        assert all(x in self.AVAILABLE_SUB_SEQUENCES for x in s_x + s_y), s_x + s_y

        data = []
        too_long = 0
        subgoals_not_found = 0

        for item in self.parsed_rows[source]:
            if item["split"] != split:
                continue
            to_append = self.get_x2y_sample(item, s_x, s_y)
            if to_append is None:
                too_long += 1
                continue
            if "subgoals" in s_x + s_y:
                subgoals_not_found += int(not to_append["found_subgoals"])
            data.append(to_append)

        # log data statistics
        logger.info(
            f"Created {len(data)} ({s_x} -> {s_y}) sequences. "
            f"Skipped {too_long} too long sequences."
        )
        if "subgoals" in s_x + s_y:
            logger.info(
                f"Did not find subgoals of {subgoals_not_found}/{len(data)} "
                f"({100 * subgoals_not_found / len(data):.1f}%) sequences."
            )
        self.print_xy_lengths_stats(data)

        return data

    def get_cond_goal_gen_sample(
        self, split: str, row_id: Optional[int], rng: np.random.RandomState
    ) -> Optional[Dict]:
        """
        - Take as input a split (with optionally provided row_id)
        - Sample a structured goal from that row
        - Generate an input / output for this structured goal
        """
        assert (row_id is None) == (split == "train")
        eos = self.dico.eos_word

        # retrieve row
        if row_id is None:
            row_id = rng.choice(self.row_splits["mathlib"][split])
        item = self.parsed_rows["mathlib"][row_id]
        assert item["split"] == split, (item["split"], split)

        # parse goal. randomly select one of the subgoals
        structured_goals: List[StructuredGoal] = parse_goal_structured(item["goal_pp"])
        assert len(structured_goals) > 0
        goal = structured_goals[rng.randint(len(structured_goals))]

        # input
        x = generate_goal_input(
            goal=goal,
            tactic=item["tactic"],
            tokenizer=self.tokenizer,
            p_all_toks=0.5,
            p_n_hyps=0.5,
            p_concl=0.5,
            p_tactic=0.5,
            rng=rng,
        )

        # output
        theorem = structured_goals[0].to_command()["command"]
        prefix = r"^def _root_\.[a-z]{6} "
        assert re.match(prefix, theorem)
        theorem = re.sub(prefix, "theorem NAME ", theorem)
        y = [eos, *self.tokenizer.encode(theorem), eos]

        # skip too long sequences
        if max(len(x), len(y)) > self.params.batch.max_len:
            return None

        # index sequences
        x = [self.dico.index(t) for t in x]
        y = [self.dico.index(t) for t in y]

        sample = {
            "name": item["name"],
            "x": x,
            "y": y,
        }
        return sample

    def create_cond_goal_gen_dataset(self, split: str):
        """
        </s>
        <N_HYPS>
        4
        <CONTAINS>
        </CONTAINS>
        <GENERATE_THEOREM>
        </s>
        """
        assert split in ["valid", "test"]
        rng = np.random.RandomState(0 if split == "valid" else 1)

        data = []
        too_long = 0

        for row_id in self.row_splits["mathlib"][split]:
            for _ in range(10):  # 10 random sub-sample for each sample
                to_append = self.get_cond_goal_gen_sample(
                    split=split, row_id=row_id, rng=rng
                )
                if to_append is None:
                    too_long += 1
                    continue
                data.append(to_append)

        # log data statistics
        logger.info(
            f"Created {len(data)} goal conditioned sequences. "
            f"Skipped {too_long} too long sequences."
        )
        self.print_xy_lengths_stats(data)

        return data

    def create_prove_self_dataset(self, split: str):
        assert split in ["train", "valid", "test"]

        # split train / valid / test
        n = len(self.goal_pp_data)
        idx = np.random.RandomState(0).permutation(n)
        a = int(0.90 * n)
        b = int(0.95 * n)
        indices = {"train": idx[:a], "valid": idx[a:b], "test": idx[b:]}

        data = []
        too_long = 0
        eos = self.dico.eos_word

        for i in indices[split]:

            line = self.goal_pp_data[i]

            # input / output
            theorem = LeanTheorem(
                conclusion=line["tactic_state"],
                state=None,
                context=LeanContext(namespaces=set()),
            )
            tactic = LeanTactic(tactic=f"apply {line['name']}")

            # tokenize
            x = [eos, *theorem.tokenize(), eos]
            y = [eos, *tactic.tokenize(), EOU_WORD]

            # skip too long sequences
            if max(len(x), len(y)) > self.params.batch.max_len:
                too_long += 1
                continue

            # index sequences
            x = [self.dico.index(t) for t in x]
            y = [self.dico.index(t) for t in y]

            sample = {
                "name": line["name"],
                "x": x,
                "y": y,
            }
            data.append(sample)

        # log data statistics
        logger.info(
            f"Created {len(data)} prove itself inputs/outputs. "
            f"Skipped {too_long} too long sequences."
        )
        self.print_xy_lengths_stats(data)

        return data

    def process_pact_json(self, task: str, sample: Dict, idx: int):
        """
        Preprocess a PACT sample. Input `sample` is a JSON line.
        Special SQUID_TOK (converted back in a MASK_WORD) to mark either a delimitation
        between lists or a PREDICT. Note that a MASK_WORD can either denote an actual
        mask OR a delimiter for tokens in adjacent lists.
        Good news: so far it is an exclusive OR (there is no PACT task that uses PREDICT
        within lists of sentences)
        """
        eos = self.dico.eos_word
        x_key, y_key = self.get_pact_xy(task)

        # retrieve input / output
        x = sample[x_key].split()
        y = sample[y_key].split()

        # replace placeholder tokens
        x = [MASK_WORD if tok == SQUID_TOK else tok for tok in x]
        y = [MASK_WORD if tok == SQUID_TOK else tok for tok in y]

        # add conditioning on task and sequence delimiters
        x = [eos, *x, eos]
        y = [eos, self.get_pact_task_token(task), *y, eos]

        # skip too long sequences
        if max(len(x), len(y)) > self.params.batch.max_len:
            return None

        # index sequences
        x = [self.dico.index(t) for t in x]
        y = [self.dico.index(t) for t in y]
        sample = {
            "name": f"{task}_{idx}",
            "x": x,
            "y": y,
            "x_key": x_key,
            "y_key": y_key,
            "raw": sample,
        }
        return sample

    def create_pact_dataset(self, split: str) -> List[Dict]:
        """
        Build PACT dataset. valid and test only.
        Train will be processed in dataset workers.
        Since we will go through all samples at evaluation time, we only
        need to flatten the data coming from the different PACT tasks.
        """
        assert split in {"valid", "test"}
        logger.info(f"Creating PACT dataset for {split} ...")
        data: Dict[str, List] = {}
        for task, json_data in self.pact_json_data[split].items():
            processed = [
                self.process_pact_json(task=task, sample=sample, idx=i)
                for i, sample in enumerate(json_data)
            ]
            data[task] = [x for x in processed if x is not None]
            logger.info(
                f"PACT task: {task:<25} - split:{split:<5} - Found {len(data[task])} samples. "
                f"{len(processed) - len(data[task])} were ignored because too long."
            )
        return list(itertools.chain(*data.values()))  # flatten data

    def create_graph_dataset(self, split: str, task: str):
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
            f"Created {len(data)} forward graph {split} sequences."
            f"Skipped {too_long} too long sequences."
        )
        self.print_xy_lengths_stats(data)

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

    def _get_mcts_subproof_bwd_sample(self, task: str, split: str, rng: RandomState):
        """
        Get a subproof from MCTS nodes.
        """
        samplers = self.mcts_subproof_samplers[split]
        # select a random sampler
        if self.params.lean.mcts_subproof.weight_samplers_alpha == 0:
            index = rng.randint(len(samplers))
        else:
            cumulative_scores = self.cumulative_mcts_subproofs[split]
            index = np.searchsorted(
                cumulative_scores,  # a
                rng.random() * cumulative_scores[-1],  # v
                side="right",  # a[i-1] <= v < a[i]
            )
        sampler = samplers[index]
        output = sampler.sample_lean_proof_step()
        if output is None:  # failed to generate a sample
            return None
        goal, tactic, children = output
        name = f"mcts_subproof__{sampler.name}__{sampler.n_gen_proofs}"

        # sanity check
        assert isinstance(goal, LeanTheorem)
        assert isinstance(tactic, LeanTactic)
        assert type(children) is list
        assert all(isinstance(sg, LeanTheorem) for sg in children)

        row = {
            "is_mcts_subproof_sample": True,
            "name": name,
            "statement": " ".join(goal.tokenize()),
            "tactic": " ".join(tactic.tokenize()),
            "subgoals": children,
        }
        s_x, s_y = parse_mcts_subproof_xy(task)
        return self.get_x2y_sample(row, s_x, s_y)

    def get_mcts_y_fmt(self, sample: MCTSSampleTactics):
        assert isinstance(sample.goal, LeanTheorem)
        assert all(isinstance(tactic, LeanTactic) for tactic in sample.tactics)
        item = {
            "tactic": [tactic.tokenize() for tactic in sample.tactics],
            "EOU": [[EOU_WORD]] * len(sample.tactics),
        }
        s_x, s_y = [x.split("-") for x in self.mcts["tactic"].mcts_fmt.split("--")]
        y = []
        eos = self.dico.eos_word
        for i, pi in enumerate(sample.tactics):
            this_y = [item[s][i] for s in s_y]
            this_y = [eos, *sum(this_y, []), eos]
            this_y = [self.dico.index(tok) for tok in this_y]
            y.append(this_y)
        return y

    def get_mcts_subproof_bwd_sample(self, task: str, split: str, rng: RandomState):
        while True:
            sample = self._get_mcts_subproof_bwd_sample(task=task, split=split, rng=rng)
            if sample is not None:
                return sample

    def get_mcts_subproof_online_bwd_sample(
        self, task: str, split: str, rng: RandomState
    ):
        proof_step = self.mcts_subproof.get_mcts_sample(split, None, rng)
        proof_step: ProofStepSample
        name = f"mcts_subproof_online_{self.mcts_subproof.n_gen_proofs}"
        row = {
            "is_mcts_subproof_sample": True,
            "name": name,
            "statement": " ".join(proof_step.theorem.tokenize()),
            "tactic": " ".join(proof_step.tactic.tokenize()),
            "subgoals": proof_step.children,
        }
        s_x, s_y = parse_mcts_subproof_xy(task)
        return self.get_x2y_sample(row, s_x, s_y)

    def load_pact_train_data(self) -> None:
        if self.pact_loaded:
            return
        logger.info(f"Loading PACT training data ...")
        assert set(self.PACT_TASKS) == self.pact_json_data["train"].keys()
        for pact_task, path in self.pact_json_data["train"].items():
            data, n_dupl = self.load_jsonl_data(path)
            self.pact_json_data["train"][pact_task] = data
            logger.info(
                f"Loaded {len(data)} lines ({n_dupl} duplicates) "
                f"for PACT co-training {pact_task} task from {path}"
            )
        # re-weight tasks according to their number of samples
        self.pact_sampling_probs: Optional[np.ndarray] = None
        alpha = self.materialized_dataset.pact_alpha_weight
        if alpha != 0:
            n_samples = [len(self.pact_json_data["train"][t]) for t in self.PACT_TASKS]
            p = np.array(n_samples, dtype=np.float64)
            p = p ** alpha
            p = p / p.sum()
            self.pact_sampling_probs = p
            logger.info(f"PACT sampling weights: {p}")
        self.pact_loaded = True

    def get_pact_train_sample(self, rng: np.random.RandomState):
        self.load_pact_train_data()
        # sample PACT task
        task_id = rng.choice(len(self.PACT_TASKS), p=self.pact_sampling_probs)
        pact_task = self.PACT_TASKS[task_id]
        data = self.pact_json_data["train"][pact_task]

        # sample within the PACT task
        index = rng.randint(len(data))
        return self.process_pact_json(task=pact_task, sample=data[index], idx=index)

    def get_sample(self, task: str, split: str, index: Optional[int]):
        """
        Get a data sample.
        """
        assert (split == "train") == (index is None)
        if task.startswith("lean_mcts"):
            if task not in self.MCTS_TRAINING_TASKS:
                raise Exception(f"Unknown task: {task}")
            return self.get_mcts_sample(task, split, index)
        elif task == LEAN_FWD_TASK and split == "train":
            return self.graph.get_graph_sample(split, task, rng=self.rng)
        elif task == LEAN_FWD_ERROR_COND_TASK and split == "train":
            return self.fwd_error_sampler.get_graph_sample(split, task, rng=self.rng)
        elif task.startswith("lean_subproof_mcts_bwd_") and split == "train":
            return self.get_mcts_subproof_bwd_sample(task, split, self.rng)
        elif task.startswith("lean_subproof_online_mcts_bwd_") and split == "train":
            return self.get_mcts_subproof_online_bwd_sample(task, split, self.rng)
        elif task == "lean_pact_seq2seq" and split == "train":
            return self.get_pact_train_sample(self.rng)
        elif task == "lean_cond_goal_gen_seq2seq" and split == "train":
            return self.get_cond_goal_gen_sample("train", row_id=None, rng=self.rng)
        elif COND_TOK in task:
            no_cond_task = task.replace(COND_TOK, "")
            data = self.data[no_cond_task][split]
            index = self.rng.randint(len(data)) if index is None else index
            datum = data[index]
            conditioning = None
            if (
                split in {"valid", "test"}
                or self.rng.random() > self.params.proba_random_conditioning
            ):
                conditioning = self.z_vectors.get(datum["name"], None)
            if conditioning is None:
                name = self.rng.choice(self.z_vectors_sorted_names)
                conditioning = self.z_vectors[name]
            datum["input_conditioning"] = conditioning
            return datum

        data = self.data[task][split]
        index = self.rng.randint(len(data)) if index is None else index
        return data[index]

    def get_stats(self) -> Dict[str, Number]:
        stats = super().get_stats()
        if self.graph is not None:
            stats.update(prefix_dict(self.graph.get_stats(), "graph/"))
        if stats:
            logger.info(
                f"Lean data env stats: (worker with PID: {os.getpid()}): {stats}"
            )
        if self.params.online_bwd_gen:
            stats.update(self.mcts_subproof.get_stats())
        return stats

    def close(self):
        logger.info("Closing LeanDataEnvironment ...")
        super().close()
        if self.graph:
            self.graph.close()
        logger.info("Closed LeanDataEnvironment")
