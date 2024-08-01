# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pickle
import pprint
from collections import defaultdict
from logging import getLogger
from typing import Dict, List, Tuple

import numpy as np

from evariste.backward.env.lean.graph import LeanContext
from evariste.forward.fwd_lean.training.graph import (
    nodes_from_steps_v3,
    update_stats,
    sample_simple_proof,
    extract_longest_subproof,
    nodes_from_steps_v2,
)

from evariste.forward.fwd_lean.training.common import (
    LeanMetaProofNode,
    LeanProofNode,
)
from evariste.forward.common import ProofNode
from evariste.forward.training.graph_sampler import (
    GraphTrainingDataset,
    GraphTrainingSample,
)
from numpy.random.mtrand import RandomState

from evariste.forward.training.helpers import (
    sample_from_cumulative,
    count_unique_theorems,
)

logger = getLogger()


class LeanGraphTrainingDataset(GraphTrainingDataset[LeanProofNode]):
    """
    Specific to lean since it is using meta proof nodes
    """

    def __init__(
        self,
        nodes: Dict[str, List[Tuple[str, LeanMetaProofNode]]],
        allow_global_hyps: bool = False,
        choose_longest_proof: bool = False,
    ):

        self.nodes = nodes
        self.cumulatives = {
            split: np.cumsum([len(n.child_theorems()) for _, n in nodes_])
            for split, nodes_ in self.nodes.items()
        }
        self.choose_longest_proof = choose_longest_proof
        if self.choose_longest_proof:
            self.cumulatives = {
                split: np.cumsum(
                    [
                        count_unique_theorems(extract_longest_subproof(n))
                        for _, n in nodes_
                    ]
                )
                for split, nodes_ in self.nodes.items()
            }

        self._nodes_for_noise = {
            split: [
                ProofNode(
                    theorem=n.thm, tactic=n.tactics_and_children[0].tactic, children=[]
                )
                for _, n in nodes_
            ]
            for split, nodes_ in self.nodes.items()
        }
        self.allow_global_hyps = allow_global_hyps

    def get_graph_training_sample(
        self, task: str, split: str, rng: RandomState
    ) -> GraphTrainingSample[LeanProofNode]:
        nodes = self.nodes[split]
        cumulative = self.cumulatives[split]
        label, meta_root = sample_from_cumulative(cumulative, nodes, rng=rng)
        if not self.choose_longest_proof:
            root = sample_simple_proof(
                meta_root, rng=rng, allow_global_hyps=self.allow_global_hyps
            )
        else:
            root = extract_longest_subproof(meta_root)
        return GraphTrainingSample(label=label, root=root)

    def nodes_for_noise(self, split: str) -> List[LeanProofNode]:
        return self._nodes_for_noise[split]

    @classmethod
    def from_trainer_args(cls, params, parsed_rows: List[Dict]):
        from evariste.trainer.args import TrainerArgs  # circular imports

        assert isinstance(params, TrainerArgs)

        label2context = {}
        label2split = {}

        present_goals = set()
        for row in parsed_rows:
            decl_name = row["name"]
            context = row["context"]
            split = row["split"]
            goal_pp = row["goal_pp"]
            present_goals.add(goal_pp)
            if decl_name not in label2context:
                label2context[decl_name] = context
                label2split[decl_name] = split
            else:
                assert (
                    label2context[decl_name].namespaces
                    == label2context[decl_name].namespaces
                )
                assert label2split[decl_name] == split

        name = (
            "extracted_fwd.pkl"
            if not params.lean.graph.use_deprecated_apply_tactic_dataset
            else "extracted_fwd.pkl.apply_tactic"
        )

        with open(os.path.join(params.lean.dataset.data_dir, name), "rb",) as fp:
            all_steps = pickle.load(fp)
        nodes = defaultdict(list)
        stats = defaultdict(int)

        all_nodes = []
        for (label, _), steps in all_steps.items():
            if label not in label2context:
                # missing labels when debug
                assert params.debug.train
                continue

            context = label2context[label]

            if params.lean.dataset.pp_full_names:
                assert context == LeanContext(namespaces=set([]))

            if not params.lean.graph.use_deprecated_apply_tactic_dataset:
                these_nodes = nodes_from_steps_v3(
                    steps,
                    stats,
                    context=context,
                    allow_uncompleted_proofs=params.lean.graph.allow_uncompleted_proofs,
                    use_pp_as_fp=params.lean.dataset.fwd_match_on_conclusion,
                    use_parsed_pp_as_conclusion=params.lean.dataset.fwd_use_parsed_pp_as_conclusion,
                )
            else:
                assert params.lean.dataset.fwd_match_on_conclusion
                assert params.lean.dataset.fwd_use_parsed_pp_as_conclusion
                these_nodes = nodes_from_steps_v2(
                    steps,
                    stats,
                    context=context,
                    allow_uncompleted_proofs=params.lean.graph.allow_uncompleted_proofs,
                )

            if params.debug.train:
                # filter nodes not in goal_pp
                these_nodes = [
                    n for n in these_nodes if n.thm.conclusion in present_goals
                ]

            split = label2split[label]
            nodes[split].extend([(label, n) for n in these_nodes])
            all_nodes.extend(these_nodes)

        stats = update_stats(stats, all_nodes)

        logger.info("Lean Proof Node Stats:")
        logger.info(pprint.pformat(dict(sorted(stats.items(), key=lambda x: x[0]))))
        return cls(
            nodes=dict(nodes),
            allow_global_hyps=params.lean.graph.allow_uncompleted_proofs,
            choose_longest_proof=params.lean.graph.choose_longest_proof,
        )

    def close(self):
        pass
