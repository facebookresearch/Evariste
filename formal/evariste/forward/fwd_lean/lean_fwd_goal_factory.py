# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
from logging import getLogger
from pathlib import Path
from typing import List

from numpy.random.mtrand import RandomState

from evariste import json
from evariste.backward.env.lean.graph import LeanTheorem, LeanContext
from evariste.backward.goal_factory import get_labels
from evariste.datasets import LeanDatasetConf
from evariste.forward.common import ForwardGoal
from evariste.forward.env_specifics.fwd_goal_factory import ForwardGoalFactory
from evariste.model.data.dictionary import Dictionary

logger = getLogger()

LeanForwardGoal = ForwardGoal[LeanTheorem]


class LeanForwardGoalFactory(ForwardGoalFactory):
    def __init__(self, dataset: LeanDatasetConf, dico: Dictionary):
        self.dataset = dataset
        self.dico = dico

    def build_forward_goals(
        self, split: str, debug: bool = False
    ) -> List[LeanForwardGoal]:

        n_to_prove = 500 if split in {"valid", "test"} else int(1e12)
        labels = get_labels(self.dataset, split, n_to_prove=n_to_prove)
        if debug:
            labels = labels[:10]
        labels = set(labels)
        logger.info(f"Using {len(labels)} labels")

        parsed_goal_path = get_parsed_goals_path(
            data_dir=Path(self.dataset.data_dir), minif2f="minif2f" in split
        )

        with parsed_goal_path.open("r") as fp:
            parsed_goals = [json.loads(line.strip()) for line in fp.readlines()]

        parsed_goals = [g for g in parsed_goals if g["label"] in labels]
        if len(parsed_goals) != len(labels):
            raise RuntimeError(
                f"len(parsed_goals) != len(labels),"
                f" {(len(parsed_goals), len(labels))}"
            )

        succeed_parsed_goals = [g for g in parsed_goals if g["succeed"]]
        n_succeed_before = len(succeed_parsed_goals)
        if self.dataset.fwd_same_fp_for_parsed:
            # we need to check that parsing didn't change the fingerprint
            succeed_parsed_goals = [
                g
                for g in succeed_parsed_goals
                if g["fingerprint"] == g["parsed_goal"]["fingerprint"]
            ]
        n_fingerint_mismatch = n_succeed_before - len(succeed_parsed_goals)
        if len(succeed_parsed_goals) != len(parsed_goals):
            logger.warning(
                f"removed {len(parsed_goals) - len(succeed_parsed_goals)} "
                f"non succeed goals "
                f"(including {n_fingerint_mismatch} fingerprint mismatch)"
            )

        goals = []
        not_in_vocab_labels = []
        not_in_vocab_tokens = set([])
        for parsed_goal in succeed_parsed_goals:
            fp = (
                parsed_goal["parsed_goal"]["conclusion"]
                if self.dataset.fwd_match_on_conclusion
                else parsed_goal["parsed_goal"]["fingerprint"]
            )
            conclusion = (
                parsed_goal["goal_pp"]
                if not self.dataset.fwd_use_parsed_pp_as_conclusion
                else parsed_goal["parsed_goal"]["conclusion"]
            )
            label = parsed_goal["label"]
            dict = copy.deepcopy(parsed_goal["parsed_goal"])
            dict["conclusion"] = conclusion
            dict["fingerprint"] = fp
            theorem = LeanTheorem.from_dict(dict)
            if self.dataset.pp_full_names:
                theorem.context = LeanContext(set([]))

            # tokens = theorem.tokenize()
            # # checking if in vocab
            # not_in_vocab = False
            # for tok in tokens:
            #     if tok not in self.dico.word2id:
            #         not_in_vocab_tokens.add(tok)
            #         not_in_vocab = True
            #
            # if not_in_vocab:
            #     not_in_vocab_labels.append(label)
            #     continue

            theorem.state = None
            # TODO: add forbidden
            fwd_goal = ForwardGoal(
                thm=theorem,
                forbidden=None,
                label=label,
                # deprecated
                statement="",
                e_hyps=[],
            )
            assert fwd_goal.is_new_fmt()
            goals.append(fwd_goal)
        if len(not_in_vocab_labels) > 0:
            logger.warning(
                f"Removed {len(not_in_vocab_labels)} goals since tokens"
                f" not in vocab detected: {not_in_vocab_tokens}. "
                f"Labels removed: {not_in_vocab_labels}. "
                f"Remaining goals: {len(goals)}"
            )
        return goals

    def build_generation_goal(self, rng: RandomState, split: str) -> LeanForwardGoal:
        raise NotImplementedError


def get_parsed_goals_path(data_dir: Path, minif2f: bool) -> Path:
    suffix = "" if not minif2f else "_mini_f2f"
    return data_dir / f"parsed_goals{suffix}.jsonl"
