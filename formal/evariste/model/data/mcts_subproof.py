# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, List, Set, Dict, Type, Union, Any
from functools import lru_cache
from logging import getLogger
from copy import deepcopy
import os
import re
import sys
import math
import pickle
import traceback
import numpy as np

from evariste.backward.env.equations import EQTheorem
from evariste.backward.env.equations.env import EQEnvWorker

from leanml import parse_goal_structured
from leanml.parse_goal import Hypothesis as LeanHypothesis, StructuredGoal


from evariste.envs.mm.utils import Node, Node_a_p, Node_e, enumerate_nodes
from evariste.backward.graph import Theorem, Tactic
from evariste.backward.env.worker import EnvWorker
from evariste.backward.env.metamath import MMTheorem, MMTactic
from evariste.backward.env.metamath.env import MMEnvWorker
from evariste.backward.env.lean.graph import LeanTheorem
from evariste.backward.prover.nodes import MCTSNode
from evariste.model.data.subproof_args import MCTSSubProofArgs

logger = getLogger()


class ProofNode:
    def __init__(self, theorem: Theorem, tactic: Tactic, children: List[Theorem]):
        assert isinstance(theorem, Theorem)
        assert isinstance(tactic, Tactic)
        assert type(children) is list
        assert all(isinstance(child, Theorem) for child in children)
        self.theorem = theorem
        self.tactic = tactic
        self.children = children

    def to_json(self) -> Dict:
        return {
            "goal": self.theorem.to_dict(light=True),
            "tactic": self.tactic.to_dict(light=True),
            "children": [thm.to_dict(light=True) for thm in self.children],
        }

    @classmethod
    def from_json(
        cls, json, thm_cls: Type[Theorem], tactic_cls: Type[Tactic]
    ) -> "ProofNode":
        theorem = thm_cls.from_dict(json["goal"])
        tactic = tactic_cls.from_dict(json["tactic"])
        children = [thm_cls.from_dict(thm) for thm in json["children"]]
        return cls(theorem, tactic, children)


class ProofStepSample:
    """
    Used to describe a single proof-step with in a theorem context (hyps, goal : Theorem),
    the tactic used in that situation and the new subgoals it returned (List[Theorem]).
    """

    def __init__(self, theorem: Theorem, tactic: Tactic, children: List[Theorem]):
        assert isinstance(theorem, Theorem)
        assert isinstance(tactic, Tactic)
        self.theorem = theorem
        self.tactic = tactic
        self.children: List[Theorem] = children

    # Used to serialize it and be able to transfer it between workers
    def to_json(self):
        return {
            "goal": self.theorem.to_dict(light=True),
            "tactic": self.tactic.to_dict(light=True),
            "children": [thm.to_dict(light=True) for thm in self.children],
        }

    @staticmethod
    def from_json(json, thm_cls: Type[Theorem], tactic_cls: Type[Tactic]):
        thm = thm_cls.from_dict(json["goal"])
        tactic = tactic_cls.from_dict(json["tactic"])
        children = [thm_cls.from_dict(thm) for thm in json["children"]]
        return ProofStepSample(thm, tactic, children)


class HyperTreeNode:
    """
    Node of an hypertree, i.e. a directed graph where nodes representing
    a theorem can have different tactics applied to.
    """

    def __init__(
        self,
        tactics: List[Tactic],
        children_for_tactic: List[List[Theorem]],
        solved: bool,
        is_solved_leaf: bool,
    ):
        self.tactics: List[Tactic] = tactics
        self.children_for_tactic: List[List[Theorem]] = children_for_tactic
        self.solved = solved
        self.is_solved_leaf = is_solved_leaf
        assert len(tactics) == len(children_for_tactic)

    @classmethod
    def from_json(cls, json: Dict, thm_cls: Type[Theorem], tactic_cls: Type[Tactic]):
        tactics = [tactic_cls.from_dict(tac) for tac in json["tactics"]]
        children_for_tac = [
            [thm_cls.from_dict(child) for child in children]
            for children in json["children_for_tactic"]
        ]
        solved = bool(json["solved"])
        is_solved_leaf = bool(json["is_solved_leaf"])
        return cls(tactics, children_for_tac, solved, is_solved_leaf)

    def to_json(self):
        json = {
            "tactics": [t.to_dict(light=True) for t in self.tactics],
            "children_for_tactic": [
                [c.to_dict(light=True) for c in children]
                for children in self.children_for_tactic
            ],
            "solved": self.solved,
            "is_solved_leaf": self.is_solved_leaf,
        }
        return json


class SimplifiedMCTSState:
    def __init__(self, root: Theorem, nodes: Dict[Theorem, HyperTreeNode]):
        assert isinstance(root, Theorem)
        assert type(nodes) is dict
        assert all(
            isinstance(k, Theorem) and isinstance(v, HyperTreeNode)
            for k, v in nodes.items()
        )
        self.root = root
        self.nodes = nodes

    def to_json(self):
        return {
            "root": self.root.to_dict(light=True),
            "nodes": [
                (thm.to_dict(light=True), node.to_json())
                for thm, node in self.nodes.items()
            ],
        }

    @classmethod
    def from_json(cls, json, thm_cls, tactic_cls, hypertree_cls):
        root = thm_cls.from_dict(json["root"])
        nodes = {
            thm_cls.from_dict(thm): hypertree_cls.from_json(node, thm_cls, tactic_cls)
            for thm, node in json["nodes"]
        }
        return cls(root, nodes)

    @classmethod
    def from_mcts_data(cls, root: Theorem, nodes: Dict[Theorem, MCTSNode]):
        """
        Takes the `root` and `nodes` objects from a `MCTS` object and extract
        only what is needed to build the hypertree, i.e. no stats and no other things.
        Only save the goal and the different tactics applied.
        `deepcopy` objects for safety. TODO: check if too expensive
        """
        ht_nodes = {
            th: HyperTreeNode(
                tactics=deepcopy(node.tactics),
                children_for_tactic=deepcopy(node.children_for_tactic),
                solved=node.solved,
                is_solved_leaf=node.is_solved_leaf,
            )
            for th, node in nodes.items()
        }
        return cls(deepcopy(root), ht_nodes)


class MCTSProofSampler:
    def __init__(
        self,
        root: Theorem,
        nodes: Dict[Theorem, HyperTreeNode],
        env_worker: Optional[EnvWorker] = None,
        verbose: bool = False,
        path: Optional[str] = None,  # for debug
    ):
        """
        Samples proofs from the hypertree of an MCTS. Takes as input filtered mcts data ie HyperTree nodes.
        @param root:
        @param nodes:
        @param verbose:
        @param env_worker:
        """
        assert isinstance(root, Theorem)
        assert type(nodes) is dict
        assert all(
            isinstance(k, Theorem) and isinstance(v, HyperTreeNode)
            for k, v in nodes.items()
        )

        self.root = root
        self.nodes = nodes
        self.path = path

        self.verbose = verbose
        self.env_worker = env_worker

        # set of hypotheses generated by the environment (when provided),
        # they should correspond to the root theorem original hypotheses.
        # Metamath and Equations only
        self.theorem_hyps: Optional[Set[Theorem]] = None
        if self.env_worker is not None:
            self.theorem_hyps = set()

        # list of theorems that can be used as root goals
        self.start_goals: List[Theorem] = []

        self.process()

    def process(self):
        """
        If an environment is provided, apply tactics to theorem to check that
        we retrieve the same children. Some additional children may be found,
        in that case this corresponds to theorem hypotheses. This is only required
        for Metamath and Equations.

        Compute the number of subnodes associated to each (node, tactic)
        tuple, to optionally prioritize tactics during sampling

        Compute the list of goals that can be used as root goals.
        """
        seen: Set[Theorem] = set()
        path: Set[Theorem] = set()

        counts = {
            "n_tactics": 0,
            "n_children": 0,
            "new_children": 0,
        }

        subnodes: Dict[Theorem, Set[Theorem]] = dict()
        self.tactic_subnodes: Dict[Tuple[Theorem, Tactic], int] = dict()

        def traverse(theorem: Theorem, depth: int):
            assert isinstance(theorem, Theorem)
            if theorem in seen:
                return
            assert theorem not in path
            assert theorem not in subnodes
            seen.add(theorem)
            subnodes[theorem] = {theorem}

            # TODO: check
            if depth >= 500 and theorem in self.nodes:
                self.nodes.pop(theorem)

            if theorem not in self.nodes:
                return
            path.add(theorem)
            node = self.nodes[theorem]

            # sanity checks
            assert len(node.tactics) == len(node.children_for_tactic)
            if node.is_solved_leaf:
                assert node.solved
                assert len(node.tactics) == 1
                assert len(node.children_for_tactic[0]) == 0

            # if env is provided, re-apply tactics, check that the resulting children
            # are the same. if some are new, they must correspond to the theorem
            # original hypotheses
            if self.env_worker is not None:
                for tactic, children in zip(node.tactics, node.children_for_tactic):
                    assert tactic.is_valid
                    result = self.env_worker.apply_tactic(
                        theorem, tactic_tokens=None, tactic=tactic, keep_if_hyp=True
                    )
                    assert tactic.is_valid
                    assert all(child in result.children for child in children)
                    for child in result.children:
                        if child in children:
                            counts["n_children"] += 1
                        else:
                            counts["new_children"] += 1
                            children.append(child)
                            assert self.theorem_hyps is not None
                            self.theorem_hyps.add(child)
            counts["n_tactics"] += len(node.tactics)

            check_solved = False
            for tactic, children in zip(node.tactics, node.children_for_tactic):
                assert tactic.is_valid
                tactic_subnodes: Set[Theorem] = set()
                for child in children:
                    if child in path:
                        continue
                    traverse(child, depth + 1)
                    tactic_subnodes |= subnodes[child]
                    subnodes[theorem] |= subnodes[child]
                check_solved |= all(
                    child in self.nodes
                    and self.nodes[child].solved
                    or self.theorem_hyps is not None
                    and child in self.theorem_hyps
                    for child in children
                )
                assert (theorem, tactic) not in self.tactic_subnodes
                # 1 + len(tactic_subnodes) to prevent a score of 0 when
                # tactic don't have children
                self.tactic_subnodes[theorem, tactic] = 1 + len(tactic_subnodes)

            assert check_solved == node.solved
            path.remove(theorem)

        traverse(self.root, depth=0)

        assert len(path) == 0
        assert all(th in seen for th in self.nodes)
        assert min(len(v) for v in subnodes.values()) == 1

        # compute the list of potential root nodes, e.g. nodes that
        # can be used as root goals and from which we can sample
        for thm, node in self.nodes.items():
            tactics = [
                tactic
                for tactic, children in zip(node.tactics, node.children_for_tactic)
                if len(children) > 0 and not any(child == thm for child in children)
            ]
            if len(tactics) > 0:
                self.start_goals.append(thm)
        assert self.root in self.start_goals

        # print stats
        if self.verbose:
            print(
                f"Root node solved: {self.nodes[self.root].solved}\n"
                f"Seen {len(seen)} theorems.\n"
                f"Explored {len(self.nodes.keys() & seen)}/{len(self.nodes)} nodes.\n"
                f"Found {counts['n_tactics']} tactics.\n"
                f"Found {len(self.start_goals)} potential start goals."
            )
            if self.env_worker is not None:
                print(
                    f"{counts['n_children']} children were already there, "
                    f"{counts['new_children']} were added and come from "
                    f"original hypotheses ({len(self.theorem_hyps)} unique)."
                )

    def _sample(
        self,
        p_hyp: float,
        max_depth: int,
        solved_as_hyp: bool,
        weight_by_subgoals: bool,
        internal_nodes: bool,
        max_noise: int = 10,
    ) -> Tuple[Theorem, Dict[Theorem, ProofNode], Set[Theorem]]:

        if p_hyp == -1:
            assert max_depth >= 1, "Max depth <= 0 and phyp = -1"
        else:
            assert max_depth == -1
            assert 0 <= p_hyp <= 1, "p_hyp not a proba"

        proofs: Dict[Theorem, ProofNode] = {}
        hyps: Set[Theorem] = set()
        seen: Set[Theorem] = set()
        path: Set[Theorem] = set()

        def traverse(theorem: Theorem, depth: int):
            assert isinstance(theorem, Theorem), "Theorem not an instance of thm"
            if theorem in seen:
                return
            assert theorem not in path, "thm in path"

            seen.add(theorem)
            path.add(theorem)

            if theorem not in self.nodes and depth != 0:
                hyps.add(theorem)
                path.remove(theorem)
                return

            node = self.nodes[theorem]

            # If the goal we have has no tactic thus no subgoals we stop
            # We cant build a proof from it
            assert depth != 0 or len(node.tactics) != 0, "This can't be a theorem!"

            # ignore tactics with cycles
            no_cycle = [
                i
                for i, children in enumerate(node.children_for_tactic)
                if all(child not in path for child in children)
            ]
            children_for_tactic = [node.children_for_tactic[i] for i in no_cycle]
            tactics = [node.tactics[i] for i in no_cycle]
            assert len(children_for_tactic) == len(
                tactics
            ), "Not the same nunber of tactics and children"

            # the root goal cannot be an hypothesis
            if depth == 0:
                assert len(tactics) > 0
                add_as_hyp = False
            # if there are no valid tactics, this node needs to be an hypothesis
            elif len(tactics) == 0:
                add_as_hyp = True
            # if we do not allow solved nodes to become hypotheses
            elif node.solved and not solved_as_hyp:
                add_as_hyp = False
            # if we reached the maximum depth for an hypothesis
            elif 1 <= max_depth == depth:
                add_as_hyp = True
            # otherwise, add this node as an hypothesis with a random probability
            else:
                prob = 1 / (max_depth - depth) if (p_hyp == -1) else p_hyp
                add_as_hyp = np.random.rand() <= prob

            # this node will now be an hypothesis, no need to explore its children (if any)
            if add_as_hyp:
                hyps.add(theorem)
                path.remove(theorem)
                return

            # select a random tactic -- optionally weight it by the number of subnodes
            assert len(tactics) > 0
            if len(tactics) == 1:  # only one tactic available
                tactic_id = 0
            else:
                if weight_by_subgoals:
                    scores = [
                        self.tactic_subnodes[theorem, tactic] for tactic in tactics
                    ]
                    assert sum(scores) > 0, f"sum score <= 0 (in {self.path})"
                    p: Optional[np.ndarray] = np.array(scores, dtype=np.float64) / sum(
                        scores
                    )
                else:
                    p = None
                tactic_id = np.random.choice(len(tactics), p=p)

            # get a proof (by potentially adding hypotheses) with this tactic
            tactic = tactics[tactic_id]
            children = children_for_tactic[tactic_id]
            for child in children:
                assert child not in path
                traverse(child, depth + 1)
                assert (child in proofs) != (child in hyps)
            proofs[theorem] = ProofNode(
                theorem=theorem, tactic=tactic, children=children
            )
            path.remove(theorem)

        # define a random goal as the root, or start from the root
        if internal_nodes:
            goal: Theorem = np.random.choice(self.start_goals)  # type: ignore
        else:
            goal = self.root
        traverse(goal, 0)

        assert len(path) == 0
        assert check_proof(goal, proofs, hyps)

        if max_noise > 0:
            # only keep theorems that are not parts of the different proofs
            available_hyps: Set[Theorem] = self.nodes.keys() - seen
            if len(available_hyps) > 0:
                # sample the size of noise we want: i.e. number of useless hypotheses
                max_noise = min(max_noise, len(available_hyps))
                n_noise = np.random.randint(0, max_noise + 1)
                # add "useless" hypotheses
                hyps |= set(
                    np.random.choice(list(available_hyps), n_noise, replace=False)  # type: ignore
                )

        # also return the goal sampled by the function (potentially the root)
        return goal, proofs, hyps

    def sample(
        self, sampling_params: MCTSSubProofArgs
    ) -> Tuple[Theorem, Dict[Theorem, ProofNode], Set[Theorem]]:
        assert type(sampling_params) is MCTSSubProofArgs
        return self._sample(
            p_hyp=sampling_params.p_hyp,
            max_depth=sampling_params.max_depth,
            solved_as_hyp=sampling_params.solved_as_hyp,
            weight_by_subgoals=sampling_params.weight_by_subgoals,
            internal_nodes=sampling_params.internal_nodes,
            max_noise=sampling_params.max_noise,
        )

    @staticmethod
    def build_steps(
        goal: Theorem, proofs: Dict[Theorem, ProofNode], hyps: Set[Theorem]
    ) -> List[ProofStepSample]:

        proof_steps: List[ProofStepSample] = []
        seen: Set[Theorem] = set()
        path: Set[Theorem] = set()

        def traverse(node: ProofNode):
            assert node.theorem not in path
            if node.theorem in seen:
                return
            seen.add(node.theorem)
            path.add(node.theorem)
            assert isinstance(node, ProofNode)
            proof_steps.append(
                ProofStepSample(node.theorem, node.tactic, node.children)
            )
            for child in node.children:
                if child not in hyps:
                    traverse(proofs[child])
            path.remove(node.theorem)

        traverse(proofs[goal])

        updated_proof_steps: List[ProofStepSample] = []
        for proof_step in proof_steps:
            new_goal = update_with_hyps(proof_step.theorem, hyps)
            new_children = [update_with_hyps(c, hyps) for c in proof_step.children]
            if new_goal is None or None in new_children:
                continue
            sample = ProofStepSample(new_goal, proof_step.tactic, new_children)  # type: ignore
            updated_proof_steps.append(sample)
        return updated_proof_steps

    @classmethod
    def build_from_path(
        cls,
        path: str,
        params: MCTSSubProofArgs,
        verbose: bool,
        env_worker: Optional[EnvWorker],
    ) -> Optional["MCTSProofSampler"]:

        assert os.path.isfile(path)

        # reload MCTS nodes
        with open(path, "rb") as f:
            data = pickle.load(f)
            root = data["mcts_root"]
            nodes = data["mcts_nodes"]

        # if there are not enough nodes, skip
        assert params.min_nodes >= 2
        if len(nodes) < params.min_nodes:
            return None

        # otherwise, build proof sampler. first convert nodes to HyperTreeNo
        nodes = {
            thm: HyperTreeNode(
                tactics=deepcopy(node.tactics),
                children_for_tactic=deepcopy(node.children_for_tactic),
                solved=node.solved,
                is_solved_leaf=node.is_solved_leaf,
            )
            for thm, node in nodes.items()
        }
        return cls(
            root=root, nodes=nodes, verbose=verbose, env_worker=env_worker, path=path
        )


@lru_cache()
def parse_mcts_subproof_xy(
    s: str, allowed: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    match = re.match(
        r"[a-z]+_subproof(?:_online|)_mcts_bwd_(?P<x>.*)--(?P<y>.*)_seq2seq", s
    )
    assert match, "Task does not match"
    x = match.groupdict()["x"].split("-")
    y = match.groupdict()["y"].split("-")
    assert allowed is None or set(x + y).issubset(allowed.split(","))
    assert len(x) == len(set(x))
    assert len(y) == len(set(y))
    return x, y


def update_with_hyps_lean(
    theorem: LeanTheorem, hyps: Set[Theorem]
) -> Optional[LeanTheorem]:

    # parse theorem / hypotheses to add
    parsed_goals = parse_goal_structured(theorem.conclusion)
    parsed_hyps = [parse_goal_structured(hyp.conclusion) for hyp in hyps]

    def get_goal_hyps(goal) -> Set[str]:
        hyps: Set[str] = set()
        for i, hyp in enumerate(goal.hyps):
            hyp_content = hyp.hyp.split(" : ", maxsplit=1)[1]
            assert (hyp_content[-1] == ",") == (i < len(goal.hyps) - 1), goal.hyps
            hyps.add(hyp_content.rstrip(","))
        return hyps

    # list all hypotheses to add
    to_add: Set[str] = {h.conclusion.rstrip(",") for hyps in parsed_hyps for h in hyps}

    # add new hypotheses to each goal, by avoiding duplicates
    for goal in parsed_goals:
        goal_hyps = get_goal_hyps(goal)
        _to_add = to_add - goal_hyps
        # add missing "," on previous last hypothesis
        if len(goal.hyps) > 0 and len(_to_add) > 0 and goal.hyps[-1].hyp[-1] != ",":
            goal.hyps[-1].hyp = goal.hyps[-1].hyp + ","
        for i, hyp in enumerate(_to_add):
            hyp = f"h{i} : {hyp}"  # could be i + len(goal_hyps)
            hyp = hyp if (i == len(_to_add) - 1) else (hyp + ",")
            goal.hyps.append(LeanHypothesis(hyp=hyp, n_hyps=1))

    # do not return this sample if one goal has itself as an hypothesis
    if any(g.conclusion in get_goal_hyps(g) for g in parsed_goals):
        return None

    # merge goal hyps/conclusion
    conclusion = []
    for goal in parsed_goals:
        for goal_hyp in goal.hyps:
            conclusion.append(goal_hyp.hyp)
        conclusion.append("⊢ " + goal.conclusion)
        conclusion.append("")
    conclusion_str = "\n".join(conclusion[:-1])

    return LeanTheorem(conclusion_str, context=theorem.context, state=None)


def update_with_hyps_mm(theorem: MMTheorem, hyps: Set[Theorem]) -> MMTheorem:
    assert all(name is None for name, _ in theorem.hyps)
    return MMTheorem(
        conclusion=theorem.conclusion,
        hyps=[(None, hyp.conclusion) for hyp in hyps],
        train_label=theorem.train_label,
        mand_vars=theorem.mand_vars,
        mand_disj=theorem.mand_disj,
    )


def update_with_hyps_eq(theorem: EQTheorem, hyps: Set[Theorem]) -> EQTheorem:
    assert all(name is None for name, _ in theorem.hyps)
    # TODO: hyps should be Set[EQTheorem]
    return EQTheorem(node=theorem.eq_node, hyps=[hyp.eq_node for hyp in hyps])  # type: ignore


def update_with_hyps(theorem: Theorem, hyps: Set[Theorem]) -> Optional[Theorem]:
    """
    Given a theorem and a list of hypotheses, add the hypotheses to the theorem.
    """
    assert isinstance(theorem, Theorem)
    assert type(hyps) is set
    assert all(isinstance(hyp, Theorem) for hyp in hyps)

    if isinstance(theorem, MMTheorem):
        return update_with_hyps_mm(theorem, hyps)
    elif isinstance(theorem, LeanTheorem):
        try:
            return update_with_hyps_lean(theorem, hyps)
        except (AssertionError, IndexError) as e:
            print(
                f"SUBPROOF ERROR {e} on {theorem.conclusion}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
            return None
    elif isinstance(theorem, EQTheorem):
        return update_with_hyps_eq(theorem, hyps)
    else:
        raise NotImplementedError(f"update_with_hyps does not support {theorem}")


class MCTSProofStepSampler:
    def __init__(
        self,
        proof_sampler: MCTSProofSampler,
        params: MCTSSubProofArgs,
        name: str,
        env_worker: Optional[EnvWorker],
    ):
        self.proof_sampler = proof_sampler
        self.params = params
        self.name = name
        assert len(self.name) > 0

        self.env_worker = env_worker

        self.proof_steps: List[Tuple[Theorem, Tactic, List[Theorem]]] = []
        self.n_gen_proofs = -1

    @property
    def nodes(self) -> Dict[Theorem, HyperTreeNode]:
        return self.proof_sampler.nodes

    @classmethod
    def build_from_path(
        cls,
        path: str,
        params: MCTSSubProofArgs,
        verbose: bool,
        env_worker: Optional[EnvWorker],
    ) -> Optional["MCTSProofStepSampler"]:

        proof_sampler = MCTSProofSampler.build_from_path(
            path, params, verbose, env_worker
        )
        if proof_sampler is None:
            return None

        name = os.path.basename(path)

        # for Metamath, check that the retrieved hypotheses
        # belong to the original theorem hypotheses
        if isinstance(env_worker, MMEnvWorker):
            assert proof_sampler.theorem_hyps is not None
            s = name.split(".")
            assert len(s) >= 2
            label = ".".join(s[:-2])
            assert env_worker.mm_env is not None
            ltype, assertion = env_worker.mm_env.labels[label]
            th_hyps = {" ".join(hyp) for hyp in assertion["e_hyps"]}
            found_th_hyps = {hyp.conclusion for hyp in proof_sampler.theorem_hyps}
            if verbose:
                print(
                    f"Retrieved {len(found_th_hyps)}/{len(th_hyps)} "
                    f"of the original theorem hypotheses."
                )
            assert found_th_hyps.issubset(th_hyps), name
        elif isinstance(env_worker, EQEnvWorker):
            assert proof_sampler.theorem_hyps is not None
            assert all(name is None for name, _ in proof_sampler.root.hyps)
            th_hyps = {hyp for _, hyp in proof_sampler.root.hyps}
            found_th_hyps = {hyp.conclusion for hyp in proof_sampler.theorem_hyps}
            if verbose:
                print(
                    f"Retrieved {len(found_th_hyps)}/{len(th_hyps)} "
                    f"of the original theorem hypotheses."
                )
            assert found_th_hyps.issubset(th_hyps)
        else:
            assert proof_sampler.theorem_hyps is None
            assert env_worker is None

        return cls(proof_sampler, params=params, name=name, env_worker=env_worker)

    def refill_proof_steps(self) -> None:
        """
        Fill a cache of proof steps.
        """
        assert len(self.proof_steps) == 0

        # sample a new proof
        goal, proofs, hyps = self.proof_sampler.sample(self.params)

        _proof_steps = []
        seen: Set[Theorem] = set()

        def traverse(node: ProofNode):
            if node.theorem in seen:
                return
            seen.add(node.theorem)
            assert isinstance(node, ProofNode)
            _proof_steps.append((node.theorem, node.tactic, node.children))
            for child in node.children:
                if child not in hyps:
                    traverse(proofs[child])

        # recursively fill proof steps, starting from the root
        traverse(proofs[goal])

        # sanity check
        assert len(_proof_steps) > 0
        for theorem, tactic, children in _proof_steps:
            assert isinstance(theorem, Theorem)
            assert isinstance(tactic, Tactic)
            assert type(children) is list
            assert all(isinstance(child, Theorem) for child in children)

        # update Theorem objects with new hypotheses
        assert len(self.proof_steps) == 0
        for theorem, tactic, children in _proof_steps:
            new_goal = update_with_hyps(theorem, hyps)
            new_children = [update_with_hyps(child, hyps) for child in children]
            if new_goal is None or None in new_children:
                continue
            self.proof_steps.append((new_goal, tactic, new_children))  # type: ignore

        # shuffle proof steps
        np.random.shuffle(self.proof_steps)  # type: ignore
        self.n_gen_proofs += 1

    def sample_proof_step(self) -> Tuple[Theorem, Tactic, List[Theorem]]:
        """
        Get a random proof step. Sample it from the cache if it is not empty,
        otherwise, sample a new random subproof from the MCTS nodes, and fill
        the cache.
        """
        if len(self.proof_steps) == 0:
            self.refill_proof_steps()
        return self.proof_steps.pop()


class MCTSProofStepSamplerMM(MCTSProofStepSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mm_proof_steps: List[Tuple[Node, Node_a_p]] = []

    def __post_init__(self):
        assert self.env_worker is not None

    def refill_mm_proof_steps(self) -> None:
        assert len(self.mm_proof_steps) == 0

        # sample a new proof
        goal, proofs, hyps = self.proof_sampler.sample(self.params)

        # we need to update the hyps of the root_node
        root_node = proofs.pop(goal)
        goal = update_with_hyps(goal, hyps)  # type: ignore
        proofs[goal] = root_node

        # convert root node to Metamath format (Node_a_p)
        assert isinstance(self.env_worker, MMEnvWorker)
        root_node_ap = proof_node_to_node_ap(
            proof_node=proofs[goal],
            proofs=proofs,
            hyps=hyps,
            env_worker=self.env_worker,
        )

        # enumerate all proof steps
        self.mm_proof_steps = []
        for node in enumerate_nodes(
            root=root_node_ap, ignore_f_e=True, ignore_empty=False, no_syntactic=True
        ):
            self.mm_proof_steps.append((node, root_node_ap))
        assert len(self.mm_proof_steps) > 0

        # shuffle proof steps
        np.random.shuffle(self.mm_proof_steps)  # type: ignore
        self.n_gen_proofs += 1

    def sample_mm_proof_step(self) -> Tuple[Node, Node_a_p]:
        if len(self.mm_proof_steps) == 0:
            self.refill_mm_proof_steps()
        return self.mm_proof_steps.pop()


class MCTSProofStepSamplerLean(MCTSProofStepSampler):
    def sample_lean_proof_step(self) -> Optional[Tuple[Theorem, Tactic, List[Theorem]]]:
        remaining_attempts = 10
        while len(self.proof_steps) == 0 and remaining_attempts > 0:
            # `self.proof_steps` could still be empty after `self.refill_proof_steps`
            # if we sampled bad examples where the goal is always in the hypotheses
            self.refill_proof_steps()
            remaining_attempts -= 1
        if len(self.proof_steps) == 0:
            return None
        return self.proof_steps.pop()


def check_proof(
    theorem: Theorem, proofs: Dict[Theorem, ProofNode], hyps: Set[Theorem]
) -> bool:
    assert isinstance(theorem, Theorem)
    assert (theorem in proofs) != (
        theorem in hyps
    ), f"{theorem in proofs}, {theorem in hyps}"
    if theorem in hyps:
        return True
    assert proofs[theorem].theorem == theorem
    children = proofs[theorem].children
    return all(check_proof(child, proofs, hyps) for child in children)


def get_depth(
    theorem: Theorem, proofs: Dict[Theorem, ProofNode], hyps: Set[Theorem]
) -> int:
    assert isinstance(theorem, Theorem)
    assert (theorem in proofs) != (theorem in hyps)
    if theorem in hyps:
        return 0
    assert proofs[theorem].theorem == theorem
    children = proofs[theorem].children
    return 1 + max([get_depth(child, proofs, hyps) for child in children], default=0)


def get_size(
    theorem: Theorem, proofs: Dict[Theorem, ProofNode], hyps: Set[Theorem]
) -> int:
    assert isinstance(theorem, Theorem)
    assert (theorem in proofs) != (theorem in hyps)
    if theorem in hyps:
        return 0
    assert proofs[theorem].theorem == theorem
    children = proofs[theorem].children
    return 1 + sum(get_size(child, proofs, hyps) for child in children)


def proof_node_to_node_ap(
    proof_node: ProofNode,
    proofs: Dict[Theorem, ProofNode],
    hyps: Set[Theorem],
    env_worker: MMEnvWorker,
) -> Node_a_p:
    """
    Convert a ProofNode object to a Metamath proof node Node_a_p.
    """
    assert isinstance(proof_node, ProofNode)
    assert isinstance(proof_node.theorem, MMTheorem)
    assert isinstance(proof_node.tactic, MMTactic)

    mm_env = env_worker.mm_env
    hyp2id: Dict[str, int] = {}

    def traverse(pn: ProofNode):
        assert isinstance(pn, ProofNode)
        children = []
        for child in pn.children:
            if child in hyps:
                hyp = child.conclusion
                # label has to be different for each hypothesis
                if hyp not in hyp2id:
                    hyp2id[hyp] = len(hyp2id)
                c = Node_e(label=f"E_HYP_{hyp2id[hyp]}", statement_str=hyp)
            else:
                c = traverse(proofs[child])
            children.append(c)
        tactic: Tactic = pn.tactic
        assert isinstance(tactic, MMTactic)
        label = tactic.label
        assert mm_env is not None
        ltype, assertion = mm_env.labels[label]
        node = Node_a_p(
            ltype=ltype,
            label=label,
            disjoint=set(),  # we don't feed disjoint to the model, so should not matter
            substitutions=tactic.subs,
            statement_str=pn.theorem.conclusion,
            children=children,
        )
        # sanity check
        assert (
            " ".join(tactic.subs.get(tok, tok) for tok in assertion.tokens)
            == pn.theorem.conclusion
        )
        return node

    return traverse(proof_node)


def load_mcts_dumps(
    env_name: str,
    params,  # TODO: type TrainerArgs
    subproof_params: MCTSSubProofArgs,
    env_worker: Optional[EnvWorker],
    returns_proof_step_sampler: bool = True,
) -> Tuple[
    Dict[str, List[Union[MCTSProofStepSampler, MCTSProofSampler]]],
    Optional[Dict[str, np.ndarray]],
]:
    """
    Reload MCTS dumps.
    """
    # for Metamath we need to re-run the tactics to find removed hypotheses
    assert (env_worker is not None) == (env_name in ["mm", "eq"])
    proof_step_sampler_builders: Dict[str, Any] = {
        "mm": MCTSProofStepSamplerMM,
        "eq": MCTSProofStepSampler,
        "lean": MCTSProofStepSamplerLean,
    }

    # retrieve MCTS dump files
    mcts_dump_path = subproof_params.dump_path
    assert mcts_dump_path is not None and os.path.isdir(mcts_dump_path)
    logger.info(f"Beginning to parse {mcts_dump_path}")

    # using endswith instead of isfile to speedup this step when > 100k files
    mcts_files = sorted(f for f in os.listdir(mcts_dump_path) if f.endswith(".pkl"))
    np.random.RandomState(0).shuffle(mcts_files)
    logger.info(f"Found {len(mcts_files)} MCTS files in {mcts_dump_path}")

    # only select a subset of files
    if subproof_params.max_files > 0:
        mcts_files = mcts_files[: subproof_params.max_files]
        logger.warning(f"Selecting first {len(mcts_files)} files only.")

    # split across workers
    if params.slurm_conf.world_size > 1:
        N_SPLITS = 8
        assert params.slurm_conf.world_size % N_SPLITS == 0
        local_rank = params.slurm_conf.global_rank % N_SPLITS
        mcts_files = [f for i, f in enumerate(mcts_files) if i % N_SPLITS == local_rank]

    # load MCTS dump files
    logger.info(f"Selected {len(mcts_files)} files. Loading...")
    _samplers: Dict[str, Union[MCTSProofStepSampler, MCTSProofSampler]] = {}
    n_nodes = 0
    count_skip = 0
    count_errors = 0
    n_solved = 0
    for i, filename in enumerate(mcts_files):
        assert filename.endswith(".pkl")
        if params.debug.train:
            logger.info(f"Loading {filename} ({i + 1}/{len(mcts_files)}) ...")
        path = os.path.join(mcts_dump_path, filename)
        try:
            if returns_proof_step_sampler:
                sampler = proof_step_sampler_builders[env_name].build_from_path(
                    path=path,
                    params=subproof_params,
                    verbose=False,
                    env_worker=env_worker,
                )
            else:
                sampler = MCTSProofSampler.build_from_path(
                    path=path,
                    params=subproof_params,
                    verbose=False,
                    env_worker=env_worker,
                )
                n_solved += sampler.nodes[sampler.root].solved
        except Exception as e:
            logger.error(f"Error when loading file {path}: {e}")
            count_errors += 1
            continue
        if sampler is None:
            count_skip += 1
            continue
        _samplers[filename] = sampler
        n_nodes += len(sampler.nodes)
        if params.debug.train and len(_samplers) >= 30:
            break
    logger.info(
        f"Loaded {len(mcts_files)} MCTS dump files. "
        f"Found {n_nodes} MCTS nodes in total. "
        f"Skipped {count_skip} MCTS graphs because not enough nodes. "
        f"Found {count_errors} unexpected errors."
    )
    if not returns_proof_step_sampler:
        logger.info(f"num of solved roots {n_solved} for {len(_samplers)} files")

    # train / valid / test split
    names = sorted(_samplers.keys())
    np.random.RandomState(0).shuffle(names)
    n_valid = max(math.ceil(len(names) * 0.05), 1)
    n_train = len(names) - 2 * n_valid
    train_names = names[:n_train]
    valid_names = names[n_train : n_train + n_valid]
    test_names = names[n_train + n_valid :]
    mcts_subproof_samplers = {
        "train": [_samplers[name] for name in train_names],
        "valid": [_samplers[name] for name in valid_names],
        "test": [_samplers[name] for name in test_names],
    }
    alpha = subproof_params.weight_samplers_alpha
    cumulative_mcts_subproofs: Optional[
        Dict[str, np.ndarray]
    ] = None if alpha == 0 else {}
    for k, v in mcts_subproof_samplers.items():
        logger.info(f"Loaded {len(v)} {k} MCTS graphs.")
        assert len(v) > 0
        if alpha != 0:
            assert cumulative_mcts_subproofs is not None
            scores_lst = [len(sampler.nodes) for sampler in v]
            scores = (np.array(scores_lst, dtype=np.float64) ** alpha).cumsum()
            cumulative_mcts_subproofs[k] = scores

    return mcts_subproof_samplers, cumulative_mcts_subproofs
