# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pickle
from logging import getLogger
from typing import Dict, List, Set, cast, Any, Tuple

import numpy as np

from evariste.backward.env.lean.graph import LeanTactic, LeanTheorem
from evariste.backward.graph import Theorem

from evariste.forward.fwd_lean.training.common import LeanProofNode
from evariste.forward.common import ProofNode
from evariste.forward.training.generic_graph_training_dataset import (
    GenericGraphTrainingDataset,
)
from evariste.forward.training.graph_sampler import (
    GraphTrainingDataset,
    GraphTrainingSample,
)
from numpy.random.mtrand import RandomState

from evariste.forward.training.helpers import (
    sample_from_cumulative,
    postorder_traversal,
)
from evariste.model.data.mcts_subproof import (
    load_mcts_dumps,
    MCTSProofSampler,
    MCTSSubProofArgs,
    ProofNode as BwdProofNode,
    HyperTreeNode,
)
from evariste.trainer.args import TrainerArgs

logger = getLogger()

dummy_lean_tactic = LeanTactic(tactic="dummy tact", valid=True)


class LeanMCTSSubproofDataset(GraphTrainingDataset[LeanProofNode]):
    def __init__(
        self,
        subproof_params: MCTSSubProofArgs,
        samplers: Dict[str, List[MCTSProofSampler]],
        cumulatives: Dict[str, np.array],
    ):
        self.subproof_params = subproof_params

        self.samplers: Dict[str, List[MCTSProofSampler]] = samplers
        self.cumulatives: Dict[str, np.array] = cumulatives

        self._nodes_for_noise = {}
        for split, samplers in self.samplers.items():
            nodes = []
            for sampler in samplers:
                nodes.extend(
                    [
                        ProofNode(
                            theorem=cast(LeanTheorem, thm),
                            tactic=dummy_lean_tactic,
                            children=[],
                        )
                        for thm, node_ in sampler.nodes.items()
                    ]
                )
            self._nodes_for_noise[split] = nodes

    def get_graph_training_sample(
        self, task: str, split: str, rng: RandomState
    ) -> GraphTrainingSample[LeanProofNode]:
        samplers = self.samplers[split]
        if self.subproof_params.weight_samplers_alpha == 0:
            index = rng.randint(len(samplers))
            sampler = samplers[index]
        else:
            cumulatives = self.cumulatives[split]
            sampler = sample_from_cumulative(cumulatives, samplers, rng)
        goal, nodes, hyps = sampler.sample(self.subproof_params)
        name = f"mcts_subproof__{rng.randint(int(1e14))}"

        root = make_fwd_graph(goal, nodes, hyps)
        return GraphTrainingSample(label=name, root=root)

    def nodes_for_noise(self, split: str) -> List[LeanProofNode]:
        return self._nodes_for_noise[split]

    @classmethod
    def from_trainer_args(cls, params):
        from evariste.trainer.args import TrainerArgs  # circular imports

        assert isinstance(params, TrainerArgs)

        samplers, cumulatives = load_mcts_dumps(
            env_name="lean",
            params=params,
            subproof_params=params.lean.mcts_subproof,
            env_worker=None,
            returns_proof_step_sampler=False,
        )

        return cls(params.lean.mcts_subproof, samplers, cumulatives)

    def close(self):
        pass


def make_fwd_graph(
    goal: Theorem, nodes: Dict[Theorem, BwdProofNode], hyps: Set[Theorem]
) -> ProofNode:
    built: Dict[Theorem, ProofNode] = {}
    seen: Set[Theorem] = set()

    def build_node(thm: Theorem) -> ProofNode:
        if thm in built:
            return built[thm]
        if thm in hyps:
            built[thm] = ProofNode.create_hyp(theorem=thm)
            return built[thm]
        assert thm not in seen  # if thm not in built -> cycle
        seen.add(thm)
        bwd_node = nodes[thm]
        children = [build_node(c) for c in bwd_node.children]
        node = ProofNode(theorem=thm, tactic=bwd_node.tactic, children=children)
        built[thm] = node
        return node

    return build_node(thm=goal)


##############################################
# Experimental
##############################################


class LeanMCTSSolvedDataset(GenericGraphTrainingDataset[LeanProofNode]):
    @staticmethod
    def from_trainer_args(params: Any):
        # simplification of load_mcts_dumps function

        from evariste.trainer.args import TrainerArgs

        assert isinstance(params, TrainerArgs)

        paths = select_mcts_files(params.lean.mcts_subproof, params)

        seen_thms: Set[LeanTheorem] = set([])
        rng = np.random.RandomState(0)

        roots, weights = [], []
        i = 0
        for path in paths:
            these_roots, these_weights = extract_solved_proof_trees(
                path, seen_thms, rng
            )
            prev_len = len(seen_thms)
            seen_thms.update({r.theorem for _, r in these_roots})
            assert len(seen_thms) == prev_len + len(these_roots)

            roots.extend(these_roots)
            weights.extend(these_weights)
            i += 1
            if i % 250 == 0:
                logger.info(f"Loading mcts trees [{i}/{len(paths)}]")

        assert len(roots) == len(weights) == len(seen_thms)

        if params.lean.graph.solved_mcts_subproof_min_size > 0:
            roots, weights = zip(
                *(
                    (r, w)
                    for r, w in zip(roots, weights)
                    if w >= params.lean.graph.solved_mcts_subproof_min_size
                )
            )
        assert len(roots) == len(weights)

        n_multi_nodes = len([w for w in weights if w > 1])

        logger.info(
            f"LeanMCTSSolvedDataset: Adding {len(roots)} proofs. "
            f"Avg size by proof: {np.mean(weights)}, "
            f"median size: {np.median(weights)} max_size: {np.max(weights)} "
            f"n_proofs with size>1 {n_multi_nodes} "
            f"({int(100 * n_multi_nodes/len(weights))}%)"
        )

        return LeanMCTSSolvedDataset(nodes={"train": roots}, weights={"train": weights})


def select_mcts_files(
    subproof_params: MCTSSubProofArgs, params: TrainerArgs
) -> List[str]:
    # retrieve MCTS dump files
    mcts_dump_path = subproof_params.dump_path

    assert mcts_dump_path is not None and os.path.isdir(mcts_dump_path)
    logger.info(f"Beginning to parse {mcts_dump_path}")
    # using endswith instead of isfile to speedup this step when >100k files
    mcts_files = sorted(f for f in os.listdir(mcts_dump_path) if f.endswith(".pkl"))
    np.random.RandomState(0).shuffle(mcts_files)
    logger.info(f"Found {len(mcts_files)} MCTS files in {mcts_dump_path}")

    if subproof_params.max_files > 0:
        mcts_files = mcts_files[: subproof_params.max_files]
        logger.warning(
            f"Keeping only {len(mcts_files)} files (because of max_files params)"
        )

    # split across workers
    if params.slurm_conf.world_size > 1:
        mcts_files = [
            f
            for i, f in enumerate(mcts_files)
            if i % params.slurm_conf.world_size == params.slurm_conf.global_rank
        ]

    mcts_files = [os.path.join(mcts_dump_path, f) for f in mcts_files]
    return mcts_files


def extract_solved_proof_trees(
    path: str, seen_thms: Set[LeanTheorem], rng: np.random.RandomState
) -> Tuple[List[Tuple[str, LeanProofNode]], List[float]]:
    assert os.path.isfile(path)

    # reload MCTS nodes
    with open(path, "rb") as f:
        data = pickle.load(f)
        root = data["mcts_root"]
        nodes = data["mcts_nodes"]

    solved = {
        thm: HyperTreeNode(
            tactics=node.tactics,
            children_for_tactic=node.children_for_tactic,
            solved=node.solved,
            is_solved_leaf=node.is_solved_leaf,
        )
        for thm, node in nodes.items()
        if node.solved
    }

    name = os.path.basename(path)
    import re

    m = re.match(f"(?P<label>.+)\.\d+\.pkl", name)
    label = m.group("label")

    proof_trees = []
    weights = []
    for thm, node in solved.items():
        if thm in seen_thms:
            continue
        try:
            tree, size = sample_proof_tree(thm, solved, rng)
        except RecursionError:  # seems to have cycles in proofs:
            continue
        proof_trees.append((label, tree))
        weights.append(float(size))
    return proof_trees, weights


def sample_proof_tree(
    root: Theorem,
    solved_nodes: Dict[Theorem, HyperTreeNode],
    rng: np.random.RandomState,
) -> Tuple[ProofNode, int]:
    built: Dict[Theorem, ProofNode] = {}

    def _build(this_thm: Theorem) -> ProofNode:

        if this_thm in built:
            return built[this_thm]

        node = solved_nodes[this_thm]
        assert node.solved
        solving_ids = [
            i
            for i, (t, cs) in enumerate(zip(node.tactics, node.children_for_tactic))
            if (len(cs) == 0 or all([c in solved_nodes for c in cs]))
        ]
        assert len(solving_ids) > 0
        tid = rng.choice(solving_ids)
        tactic = node.tactics[tid]
        children = node.children_for_tactic[tid]

        assert all([solved_nodes[c].solved for c in children])

        children = [_build(c) for c in children]
        if this_thm in built:  # cycle, check that it's normal that it happened
            return built[this_thm]
        proof_node = ProofNode(theorem=this_thm, tactic=tactic, children=children)
        built[this_thm] = proof_node
        return proof_node

    _build(root)

    size = len(postorder_traversal(built[root], rng=None))
    assert size <= len(built)

    return built[root], size
