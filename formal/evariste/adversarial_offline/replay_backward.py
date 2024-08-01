# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from evariste.backward.env.equations.env import EQEnvWorker
from evariste.backward.env.equations.graph import EQTheorem, EQTactic
from evariste.datasets.equations import EquationsDatasetConf
from evariste.envs.eq.env import EquationEnv
from evariste.forward.fwd_eq.gen.proof_search import EqGenProofSearch
from typing import List, Tuple, Dict
from evariste.backward.graph import Theorem, Tactic
from evariste.backward.graph import BackwardGoal


class BackwardReplayer:
    def __init__(self, dataset: EquationsDatasetConf, eq_env: EquationEnv):
        self.eq_env_worker = EQEnvWorker(dataset, eq_env=eq_env)

    def replay_proof(self, goal: BackwardGoal, proofsearch: EqGenProofSearch):
        queue: List[Theorem] = [goal.theorem]

        assert isinstance(goal.theorem, EQTheorem)
        matching_node = proofsearch.next_graph.get_bwd_proof_for_node(goal.theorem)
        while queue:
            next_queue: List[Theorem] = []
            for th in queue:
                if th not in matching_node:
                    raise RuntimeError("Unexpected node not in matching_node !")
                assert isinstance(th, EQTheorem)
                expected_tac, expected_children = matching_node[th]
                result = self.eq_env_worker.apply_tactic(
                    theorem=th,
                    tactic_tokens=None,
                    tactic=expected_tac,
                    keep_if_hyp=False,  # TODO: make sure this is correct
                )
                if not result.tactic.is_valid:
                    raise RuntimeError(f"Invalid tac {result.tactic.error_msg}")
                # TODO check that expected children is good ?
                next_queue += result.children
            queue = next_queue
