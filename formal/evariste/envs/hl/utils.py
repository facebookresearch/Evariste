# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Dict
import logging
import os
import gzip
from evariste import json as json

from evariste.backward.env.hol_light.graph import HLTheorem


def get_theorem_tokens(
    theorem: Dict, add_subgoals_tokens: bool = False, include_types: bool = False,
) -> Dict[str, int]:
    """
    Return all tokens involved in a theorem (goal + tactic + subgoals).
    Also check the data format of the theorem.
    """
    assert theorem.keys() == {"name", "filename", "line", "steps"}, theorem.keys()
    assert len(theorem["steps"]) >= 1
    assert len(theorem["steps"][-1]["subgoals"]) == 0

    tokens = {}

    def update(s: str):
        assert type(s) is str
        for tok in s.split():
            tokens[tok] = tokens.get(tok, 0) + 1

    def add_goal(goal: Dict):
        assert type(goal) is dict
        assert goal.keys() == {"hyps", "concl"}, goal.keys()
        assert type(goal["hyps"]) is list
        update(goal["concl"])
        for name, hyp in goal["hyps"]:
            if name is not None:
                assert type(name) is str and " " not in name, name
                tokens[name] = tokens.get(name, 0) + 1
            update(hyp)

    # for each step
    for step in theorem["steps"]:
        assert step.keys() == {"goal", "tactic", "subgoals"}, step.keys()
        add_goal(step["goal"])
        update(step["tactic"])
        for subgoal in step["subgoals"]:
            add_goal(subgoal)

    return tokens


def get_dag_and_th(data_dir: str, custom_dag: str = ""):
    """
    Compute DAG.
    Export DAG into a file. Reload file if available.
    """
    assert os.path.isdir(data_dir), data_dir
    dag_path = os.path.join(data_dir, "dag.gz")
    data_path = os.path.join(data_dir, "dataset.json")
    assert os.path.isfile(data_path), data_path

    # if the DAG has been dumped, reload it
    if os.path.isfile(dag_path):
        logging.info(f"Reloading DAG and theorems from {dag_path} ...")
        with gzip.open(dag_path, "rt", encoding="ascii") as f:
            saved = json.load(f)
        if len(saved) == 2:
            logging.warning(
                f'The archive "{dag_path}" does not contain the number of steps per label - needs to rebuild'
            )
            os.remove(dag_path)
            return get_dag_and_th(data_dir, custom_dag)
        dag, label_to_th, label_to_num_steps = saved
        dag = {k: set(v) for k, v in dag.items()}
        label_to_th = {
            k: HLTheorem(conclusion=conclusion, hyps=hyps, train_label=k)
            for k, (conclusion, hyps) in label_to_th.items()
        }
        logging.info(f"Reloaded DAG and theorems for {len(dag)} proofs.")

    # otherwise, build it and export it
    else:
        logging.info(f"File {dag_path} does not exist. Building DAG and theorems ...")

        # load theorems and proofs
        logging.info(f"Loading theorems and proofs from {data_path} ...")
        label_to_th = {}
        label_to_num_steps = {}
        proofs = {}
        with open(data_path, "r") as f:
            for line in f:
                theorem = json.loads(line.rstrip())
                label = theorem["name"]
                if label in label_to_th:
                    logging.warning(f"{label} ({theorem['filename']}) already found.")
                goal = theorem["steps"][0]["goal"]
                label_to_th[label] = HLTheorem(
                    conclusion=goal["concl"], hyps=goal["hyps"], train_label=label
                )
                proofs[label] = [step["tactic"] for step in theorem["steps"]]
                label_to_num_steps[label] = len(proofs[label])
        logging.info(f"Loaded {len(proofs)} theorems and proofs from {data_path}")

        # build DAG
        dag = {label: set() for label in label_to_th.keys()}
        for label, tactics in proofs.items():
            for tactic in tactics:
                for tok in tactic.split():
                    if tok in dag:
                        dag[tok].add(label)

        try:
            logging.info(f"Exporting DAG and theorems to {dag_path} ...")
            with gzip.open(dag_path, "wt", encoding="ascii") as f:
                json.dump(
                    (
                        {k: list(v) for k, v in dag.items()},
                        {k: (v.conclusion, v.hyps) for k, v in label_to_th.items()},
                        {k: v for k, v in label_to_num_steps.items()},
                    ),
                    f,
                )
            logging.info(
                f"Exported DAG and theorems for {len(dag)} labels to {dag_path}"
            )
        except PermissionError:
            logging.info(f"Couldn't write to {dag_path}")

    # finally import the custom_dag if chosen
    if custom_dag:
        custom_dag_path = os.path.join(data_dir, f"{custom_dag}.json")
        assert os.path.isfile(custom_dag_path), custom_dag_path

        logging.info(f"Loading a custom DAG from {custom_dag_path} ...")
        dag = json.load(open(custom_dag_path))
        logging.info("Reloaded the custom DAG")

    return dag, label_to_th, label_to_num_steps
