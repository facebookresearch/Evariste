# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Tuple

import pytest
from numpy.random import RandomState

from evariste.backward.env.equations import EQTheorem
from evariste.envs.eq.generation import extract_graph_steps
from evariste.envs.eq.rules import eval_assert
from evariste.envs.eq.rules_lib import ALL_A_RULES
from evariste.forward.common import (
    ForwardGraph,
    GenerationHistory,
    ForwardStep,
    MaybeForwardStep,
    SimpleFwdEnvOutput,
    PolicyOutput,
)
from evariste.forward.fwd_eq.eq_fwd_env import EqForwardTactic
from evariste.forward.fwd_eq.eq_fwd_goal_factory import eq_forward_goal
from evariste.model.data.dictionary import Dictionary
from evariste.model.data.envs.equations import EquationsEnvironment
from evariste.trainer.args import TrainerArgs
from params import ConfStore

ForwardGraphsAndTactics = List[Tuple[ForwardGraph[EQTheorem], EqForwardTactic]]


def trainer_args() -> TrainerArgs:
    trainer_args: TrainerArgs = ConfStore["default_cfg"]
    trainer_args.eq.dataset = ConfStore["eq_dataset_exp_trigo_hyper"]
    trainer_args.tasks = "eq_bwd_rwalk_seq2seq"
    return trainer_args


@pytest.fixture(scope="package")
def eq_env() -> EquationsEnvironment:
    eq_env = EquationsEnvironment(
        dico=Dictionary.create_empty(), params=trainer_args(), fast=True
    )
    if "identities" not in eq_env.labels:
        eq_env.create_identities_dataset()
    eq_env.set_rng(rng=RandomState(42))
    return eq_env


@pytest.fixture(scope="package")
def eq_theorems(eq_env: EquationsEnvironment):
    available = eq_env.labels["identities"]
    thms = []
    for name in available:
        th: EQTheorem = eq_env.label_to_eq[name]
        thms.append(th)
    return thms


@pytest.fixture(scope="package")
def eq_gen_graphs(eq_env: EquationsEnvironment) -> ForwardGraphsAndTactics:
    gen_histories = _sample_eq_gen_history(eq_env, n=3, is_generation=True, seed=43)
    return _build_graph_and_tactics(gen_histories)


@pytest.fixture(scope="package")
def eq_fwd_graphs(eq_histories: List[GenerationHistory]) -> ForwardGraphsAndTactics:
    return _build_graph_and_tactics(eq_histories)


@pytest.fixture(scope="package")
def eq_histories(eq_env: EquationsEnvironment) -> List[GenerationHistory]:
    return _sample_eq_gen_history(eq_env, n=5, is_generation=False, seed=42)


def _sample_eq_gen_history(
    eq_env: EquationsEnvironment, seed: int, n: int = 3, is_generation: bool = False,
) -> List[GenerationHistory]:
    eq_env.eq_env.set_rng(RandomState(seed))
    rng = RandomState(seed)
    # generate a random graph
    results = []
    nodes, init_hyps, graph_name = eq_env.generate_random_graph(rng)
    sampled_ids, _ = eq_env.graph_sampler.sample(graph=eq_env.egg, n_samples=n)

    for node_id in sampled_ids:
        node = nodes[node_id]
        name = f"fwd_{graph_name}_node{node_id}"

        # extract goals and tactics
        goals_with_tactics, hyps, node_ids = extract_graph_steps(node)
        final_goal = goals_with_tactics[-1][0]
        assert final_goal.eq_node.eq(node.node)
        assert isinstance(final_goal, EQTheorem)

        fwd_goal = eq_forward_goal(
            thm=final_goal, name=name, is_generation=is_generation,
        )
        thms = [thm for thm, _, _ in goals_with_tactics]
        fwd_graph = ForwardGraph(fwd_goal=fwd_goal, generated_thms=thms)
        hyps_and_generated = fwd_graph.global_hyps_and_generated

        # thm without hyp
        thm_to_id = {
            EQTheorem(node=thm.eq_node, hyps=[]): i
            for i, thm in enumerate(hyps_and_generated)
        }
        assert len(thm_to_id) == len(hyps_and_generated)

        steps: List[MaybeForwardStep] = []
        for thm, tactic, children in goals_with_tactics:
            assert isinstance(thm, EQTheorem)
            # if not missing children
            children = [
                child
                for child in children
                if not eval_assert(child, ALL_A_RULES["default"])
            ]

            try:
                children_ids = [
                    thm_to_id[EQTheorem(node=node, hyps=[])] for node in children
                ]
            except KeyError:
                print(thm_to_id)
                raise
            step = ForwardStep(
                policy_output=PolicyOutput(
                    fwd_graph,
                    command=[],
                    command_str="",
                    fwd_tactic=EqForwardTactic(next_node=thm, bwd_tactic=tactic),
                    score=-1,
                    normalized_score=-1,
                ),
                env_output=SimpleFwdEnvOutput(
                    generated=thm, tactic=tactic, children_ids=children_ids
                ),
            )
            steps.append(MaybeForwardStep(step=step, err=None))
        history = GenerationHistory(goal=fwd_goal, stack=steps)

        # sanity checks
        assert history.forward_graph() == fwd_graph
        if not is_generation:
            assert fwd_graph.generated_thms[-1] == fwd_graph.fwd_goal.thm
        for thm in fwd_graph.generated_thms:
            assert thm.hyps == final_goal.hyps

        results.append(history)
    return results


def _build_graph_and_tactics(
    eq_histories: List[GenerationHistory],
) -> ForwardGraphsAndTactics:
    """
    Removing last step from graph to check we can rebuilt it
    """
    results = []
    for history in eq_histories:
        fwd_graph = history.forward_graph()
        last_thm = fwd_graph.generated_thms.pop()
        if fwd_graph.fwd_goal.thm is not None:
            assert last_thm == fwd_graph.fwd_goal.thm
        fwd_tactic = history.forward_steps()[-1].fwd_tactic
        # fwd_graph = cast(ForwardGraph[EQTheorem], fwd_graph)
        assert isinstance(fwd_tactic, EqForwardTactic)
        results.append((fwd_graph, fwd_tactic))
    return results
