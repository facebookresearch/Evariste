# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
import pickle
import os

from evariste.model.data.envs.metamath import label_remap
from evariste.backward.env.metamath.env import MMTheorem, MMTactic, MMEnvWorker
from params import ConfStore
from evariste.backward.goal_factory import get_goals_to_prove
from evariste.envs.mm.utils import Node_a_p, check_proof


data_dir = "resources/metamath/DATASET_HOLOPHRASM"


@pytest.mark.slow
class TestMMVecBackwardEnv:
    @classmethod
    def setup_class(cls):
        cls.env = MMEnvWorker(ConfStore["holophrasm"])
        cls.env.init()
        goals_to_prove = sum(
            [
                get_goals_to_prove(ConfStore["holophrasm"], split, 100_000)
                for split in ["test", "train", "valid"]
            ],
            [],
        )
        cls.goals_to_prove = {g.theorem.label: g for g in goals_to_prove}

    def test_apply_tactic(self):
        # goal = self.goal_fac.get_goal("ax-mp")
        th = MMTheorem(
            "|- ( ph -> th )",
            [
                (None, "|- ( ph -> ps )"),
                (None, "|- ( ph -> ch )"),
                (None, "|- ( ph -> ( ps -> ( ch -> th ) ) )"),
            ],
            mand_vars={"ph", "th", "ps", "ch"},
            mand_disj=set(),
        )
        tac = MMTactic("mpd", {"ph": "ph", "ch": "th", "ps": "ps"})
        subgoals, _new_disj = self.env._apply_tactic(th, tac)
        assert tac.is_valid and len(subgoals) == 5, tac

    def test_is_true(self):
        assert self.env.is_true(MMTheorem("class ( 2 + 2 )", []))

    def test_all_proofs(self):
        all_proofs = pickle.load(open(os.path.join(data_dir, "proof_trees.pkl"), "rb",))
        label_map = label_remap(self.env.mm_env)
        for i, (label, proof) in enumerate(all_proofs.items()):
            print(i, label, len(all_proofs))
            root_goal = self.goals_to_prove[label]
            root_goal.theorem = self.env.materialize_theorem(root_goal.theorem)
            to_prove = [proof]
            seen = set()
            while to_prove:
                cur_node = to_prove.pop()
                cur_th = MMTheorem(
                    conclusion=cur_node.statement_str,
                    hyps=[(None, " ".join(x)) for x in proof.e_hyps.values()],
                    mand_vars=root_goal.theorem.mand_vars,
                    mand_disj=root_goal.theorem.mand_disj,
                    train_label=label,
                )
                if cur_th in seen:
                    continue
                if not isinstance(cur_node, Node_a_p):
                    continue
                seen.add(cur_th)
                proof_tactic = MMTactic(
                    label=cur_node.label,
                    subs={x: " ".join(y) for x, y in cur_node.substitutions.items()},
                )
                has_children = (
                    hasattr(cur_node, "children") and len(cur_node.children) > 0
                )
                if not has_children:
                    continue
                expected_children = {
                    MMTheorem(
                        conclusion=c.statement_str,
                        hyps=[(None, " ".join(x)) for x in proof.e_hyps.values()],
                    ): c
                    for c in cur_node.children
                }

                assert self.env._valid_sub(
                    cur_th, proof_tactic.label, proof_tactic.subs
                )[0]

                children = set(self.env._apply_tactic(cur_th, proof_tactic)[0])
                if children != set(expected_children.keys()):
                    print(children)
                    print("---------")
                    print(expected_children)
                assert children == set(expected_children.keys())
                for c in children:
                    if self.env.is_true(c):
                        continue
                    to_prove.append(expected_children[c])

            # Check that the produced proof is metamath valid
            def get_proof(cur_node):
                cur_th = MMTheorem(
                    conclusion=cur_node.statement_str,
                    hyps=[(None, " ".join(x)) for x in proof.e_hyps.values()],
                    mand_vars=root_goal.theorem.mand_vars,
                    mand_disj=root_goal.theorem.mand_disj,
                    train_label=label,
                )
                if self.env.is_true(cur_th):
                    return cur_th, None, []

                cur_tac = MMTactic(
                    label=label_map[cur_node.label], subs=cur_node.substitutions,
                )
                return cur_th, cur_tac, [get_proof(c) for c in cur_node.children]

            try:
                check_proof(
                    self.env.to_mm_proof(get_proof(proof)),
                    ConfStore["holophrasm"].database_path,
                )
            except Exception:
                print(label)
                raise
