# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, List, Dict
from dataclasses import asdict
from logging import getLogger
import math
import torch
import pickle

from evariste.backward.prover.policy import Policy
from params.params import flatten_dict
from evariste.utils import find_descendents
from evariste.datasets import DatasetConf
from evariste.backward.dag_factory import get_dag
from evariste.backward.env.worker import SyncEnv, EnvWorker
from evariste.backward.env.core import (
    BackwardGoal,
    BackwardEnv,
)
from evariste.backward.env.metamath.env import MMEnvWorker
from evariste.backward.graph import Theorem, UnMaterializedTheorem, Tactic
from evariste.backward.model.beam_search import BeamSearchModel
from evariste.backward.prover.dump import get_theorem_data, MCTSDump, MAX_MCTS_DEPTH
from evariste.model.transformer_args import DecodingParams
from visualizer.session.utils import parse_input_goal


logger = getLogger()


class Session:
    def __init__(
        self,
        session_name: str,
        session_type: str,
        dataset: DatasetConf,
        model: BeamSearchModel,
        env: BackwardEnv,
    ):
        self.name = session_name
        self.session_type = session_type
        self.goal: Optional[BackwardGoal] = None
        self.tree: Optional[Dict] = None
        self.dataset = dataset
        self.model: BeamSearchModel = model
        self.env = env
        assert session_type in {"mm", "hl", "eq", "lean"}
        assert model is not None and model.dico is not None
        assert isinstance(env.expander_env, SyncEnv)

        # store all goals
        self.id2goal: List[Theorem] = []

        # for MCTS sessions
        self.mcts_dump: Optional[MCTSDump] = None

        self.dag = get_dag(self.dataset)
        self.forbidden: Optional[List[str]] = None

    @property
    def env_worker(self) -> EnvWorker:
        assert self.session_type in ["mm", "eq"]
        assert isinstance(self.env.expander_env, SyncEnv)
        return self.env.expander_env.worker

    def add_theorem(self, goal: Theorem) -> int:
        """
        Return the ID of a goal. Add it if it does not exist.
        """
        self.id2goal.append(goal)
        return len(self.id2goal) - 1

    def initialize_from_goal(self, goal: BackwardGoal) -> None:
        """
        Set session goal.
        """
        assert goal.materialized
        assert self.goal is None and self.tree is None
        assert len(self.id2goal) == 0

        # create goal
        logger.info(f"Setting session {self.name} goal to: {goal}")
        if self.goal is not None:
            logger.warning(f"Goal already defined: {self.goal}")
            assert goal == self.goal
        self.goal = goal

        self.forbidden = find_descendents(self.dag, self.goal.label)
        logger.info(f"Initializing with {self.goal}")
        logger.info(f"Forbidden: {self.forbidden}")

        root_goal: Theorem = goal.theorem
        goal_id = self.add_theorem(root_goal)
        assert goal_id == 0

        self.tree = {
            "goal_id": 0,
            "goal_data": get_theorem_data(root_goal),
            "children": [],
            "is_solved": False,
        }

    def initialize_from_label(self, label: str) -> None:
        """
        Initialize from a label.
        """
        unmat_th = UnMaterializedTheorem(label=label)
        theorem = self.env_worker.materialize_theorem(unmat_th)
        goal = BackwardGoal(theorem=theorem, label=label)
        self.initialize_from_goal(goal)

    def initialize_from_statement(self, statement: str, hyps: str) -> None:
        """
        Initialize from a label.
        """
        goal = parse_input_goal(self.session_type, statement, hyps)
        self.initialize_from_goal(goal)

    def initialize_from_proof(self, proof_dump: str) -> None:
        """
        Initialize the tree from a saved proof.
        """
        assert self.goal is None and self.tree is None
        assert len(self.id2goal) == 0

        # pickable proof
        PProof = Tuple[Theorem, Tactic, List["PProof"]]  # type: ignore

        # reload proof
        with open(proof_dump, "rb") as f:
            proof: PProof = pickle.load(f)

        def build_tree(proof: PProof, depth: int):
            theorem, tactic, children = proof

            goal_data = get_theorem_data(theorem)
            goal_data["log_critic"] = -1000
            goal_data["depth"] = depth
            return {
                "goal_id": self.add_theorem(theorem),
                "goal_data": goal_data,
                "children": [
                    {
                        "is_solving": True,
                        "is_valid": True,
                        "tactic_data": {"prior": -1, "tac": tactic.to_dict()},
                        "children": [build_tree(c, depth + 1) for c in children],
                    }
                ],
                "is_solved": True,
            }

        self.goal = BackwardGoal(theorem=proof[0], label="label")
        self.tree = build_tree(proof, depth=0)
        self.enrich_tree()

    def initialize_from_mcts(self, mcts_dump: str) -> None:
        """
        Initialize the tree from a MCTS dump.
        """
        assert self.goal is None and self.tree is None
        assert len(self.id2goal) == 0

        with open(mcts_dump, "rb") as f:
            pickled = pickle.load(f)
            self.mcts_dump = MCTSDump(
                pickled["mcts_root"],
                pickled["mcts_nodes"],
                pickled["timesteps"],
                pickled["label"],
            )
        self.id2goal = [x.theorem for x in self.mcts_dump.nodes]
        self.goal = BackwardGoal(
            theorem=self.mcts_dump.root, label=self.mcts_dump.label
        )
        self.tree = self.mcts_dump.full_tree_dict(MAX_MCTS_DEPTH)
        self.enrich_tree()

    def update_mcts_policy(self, policy_type: str, exploration: float, max_depth: int):
        logger.info(f"Updating policy: type={policy_type}, exploration={exploration}")
        assert policy_type in ["alpha_zero", "other"], policy_type
        assert exploration >= 0, exploration
        assert max_depth >= 1, max_depth
        assert self.mcts_dump is not None
        self.mcts_dump._policy = Policy(
            policy_type=policy_type, exploration=exploration
        )
        self.tree = self.mcts_dump.full_tree_dict(max_depth=max_depth)
        self.enrich_tree()

    def enrich_tree(self) -> None:
        def cross(elem):
            if "tactic_data" in elem and "tac" in elem["tactic_data"]:
                if self.session_type == "mm":
                    self.enrich_results_mm(elem["tactic_data"]["tac"])
                elif self.session_type == "eq":
                    self.enrich_results_eq(elem["tactic_data"]["tac"])
            for child in elem["children"]:
                cross(child)

        cross(self.tree)

    def get_data(self) -> Dict:
        """
        Return session data.
        """
        assert (self.goal is None) == (self.tree is None)
        if self.goal is None:
            goal_data = {"statement": None, "hyps": None}
        else:
            goal_data = get_theorem_data(self.goal.theorem)
        data = {
            "session_name": self.name,
            "session_type": self.session_type,
            "goal": goal_data,
            "tree": self.tree,
        }
        if self.mcts_dump is not None:
            data["mcts"] = {"max_t": self.mcts_dump.max_t}
        return data

    def query_model(self, data: Dict):
        """
        Get suggested tactics -- ask the model for help.
        """
        logger.info(f"Querying model with {data}")

        # retrieve model / decoding parameters / dictionary
        model = self.model
        assert model is not None and model.dico is not None
        eos = model.dico.eos_word

        # read input data
        goal_id = data["goal_id"]
        settings = data["settings"]
        dec_params = {
            "use_beam": settings["use_beam"],
            "use_sampling": settings["use_sampling"],
            "n_samples": settings["n_samples"],
            "fixed_temperature": settings["sample_temperature"],
            "top_k": settings["sample_topk"],
            "prefix": None if settings["prefix"] == "" else settings["prefix"].split(),
        }

        # parameters check
        if not dec_params["use_beam"] and not dec_params["use_sampling"]:
            logger.warning("No beam + no sampling can only return 1 sample!")
            dec_params["n_samples"] = 1
            dec_params["fixed_temperature"] = None
            dec_params["top_k"] = None

        # quite hacky. TODO: add something that does this to Params
        flat_params = flatten_dict(asdict(model.decoding_params))
        flat_params.update(flatten_dict(dec_params))
        decoding_params: DecodingParams = DecodingParams.from_flat(flat_params)
        decoding_params.check_and_mutate_args()
        model.decoding_params = decoding_params
        logger.info(f"Decoding parameters: {decoding_params}")

        goal = self.id2goal[goal_id]
        logger.info(f"Querying goal ID {goal_id}")
        logger.info(f"Goal: {goal}")
        logger.info(f"Generating tactics ...")

        # prepare inputs / input device
        tok_ids = [model.dico.word2id[t] for t in [eos, *goal.tokenize(), eos]]
        src_tokens = torch.LongTensor(tok_ids).view(1, -1).to(model.device)
        src_len = torch.LongTensor([len(tok_ids)]).to(model.device)

        # Generate forbidden tokens if not None
        forbidden_lists = [
            torch.tensor(
                [model.dico.word2id[x] for x in self.forbidden]
                if self.forbidden is not None
                else []
            ).long()
        ]
        # generate tactics
        _tac_toks, _log_priors, _log_critics = model.do_batch(
            src_len=src_len,
            src_tokens=src_tokens,
            forbiddens=forbidden_lists,
            infos=None,
            params=None,
        )
        assert len(_tac_toks) == len(_log_priors) == len(_log_critics) == 1
        tac_toks = _tac_toks[0]
        log_priors = _log_priors[0]
        log_critic = _log_critics[0]
        n_tactics = len(tac_toks)
        logger.info(
            f"Generated {n_tactics} tactics:\n\t"
            + "\n\t".join(" ".join(t) for t in tac_toks)
        )

        # apply tactics
        tactic_job_results = self.env.apply_tactics(goal, tac_toks)

        # extract tactics / subgoals / sanity check
        assert len(tac_toks) == n_tactics
        assert len(tactic_job_results) == n_tactics
        assert len(log_priors) == n_tactics
        assert type(log_critic) is float, log_critic

        # for all tactics
        results = []
        for tokens, tac_job_res, log_prior in zip(
            tac_toks, tactic_job_results, log_priors
        ):
            # no children for invalid tactics
            tactic = tac_job_res.tactic
            sg = tac_job_res.children
            if not tactic.is_valid:
                assert len(sg) == 0

            # compute subgoal IDs
            subgoal_ids = []
            for _sg in sg:
                subgoal_ids.append(self.add_theorem(_sg))

            # tactic main info
            tactic_children = []
            for goal_id, child in zip(subgoal_ids, sg):
                goal_data = get_theorem_data(child)
                goal_data["log_critic"] = None
                tactic_children.append(
                    {"goal_data": goal_data, "goal_id": goal_id, "children": []}
                )
            res = {
                "hash": tactic.hash,
                "tactic_data": {"tac": tactic.to_dict(), "prior": math.exp(log_prior)},
                "is_valid": tactic.is_valid,
                "children": tactic_children,
            }

            if self.session_type == "mm":
                self.enrich_results_mm(res["tactic_data"]["tac"])
            elif self.session_type == "eq":
                self.enrich_results_eq(res["tactic_data"]["tac"])

            results.append(res)

        # log
        logger.info(f"Generated {n_tactics} tactics")
        logger.info(tactic_job_results)
        logger.info(log_priors)
        logger.info(log_critic)
        logger.info(results)

        return {"log_critic": log_critic, "tactics": results}

    def enrich_results_mm(self, tac_data) -> None:
        env_worker = self.env_worker
        assert isinstance(env_worker, MMEnvWorker)
        mm_env = env_worker.mm_env
        assert mm_env is not None
        if "label" in tac_data and tac_data["label"] in mm_env.labels:
            label_type, assertion = mm_env.labels[tac_data["label"]]
            tac_data["label_type"] = label_type
            tac_data["label_statement"] = assertion.tokens_str
            tac_data["label_e_hyps"] = [" ".join(h) for h in assertion.e_hyps]
            tac_data["pred_substs"] = [
                name for _, name in assertion.f_hyps if name in assertion.tokens
            ]

    def enrich_results_eq(self, tac_data) -> None:
        pass
