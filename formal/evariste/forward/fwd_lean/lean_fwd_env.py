# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import time
from typing import Tuple, List, cast, Dict, Optional

from evariste.backward.env.lean.env import (
    LeanExpanderEnv,
    ForwardLeanJob,
    logger,
    DeclNotFound,
)
from evariste.backward.env.lean.graph import LeanTheorem, LeanTactic
from evariste.forward.common import ForwardGraph, ForwardGoal, GenerationError
from evariste.forward.core.generation_errors import (
    InvalidTactic,
    MissingHyp,
    ParseGoalError,
    GoalError,
    NodeInGraph,
    WrongContext,
    EnvTimeout,
    ParseChildrenError,
)
from evariste.forward.core.maybe import Maybe, Fail, Ok
from evariste.forward.common import SimpleFwdEnvOutput
from evariste.forward.env_specifics.prover_env_specifics import (
    ChildrenIds,
    AsyncForwardEnv,
    SessionEnv,
)
from evariste.forward.fwd_lean.common import LeanForwardTactic

# type hints
LeanFwdEnvOutput = SimpleFwdEnvOutput[LeanTheorem, LeanTactic]
LeanForwardGraph = ForwardGraph[LeanTheorem]
LeanForwardGoal = ForwardGoal[LeanTheorem]
LeanSubGoals = List[LeanTheorem]

MAX_CONCURRENT_CREATE_SESSION = 16


class LeanAsyncForwardEnv(AsyncForwardEnv, SessionEnv):
    """
    As Lean env is stateful, this class also have a start_goals and end_goals methods
    """

    def __init__(self, expander_env: LeanExpanderEnv):
        self.expander_env = expander_env

        self.waiting: Dict[int, Tuple[LeanForwardTactic, LeanForwardGraph]] = {}

        self.closed = False

    def start_goals(
        self, goals: List[Tuple[int, LeanForwardGoal]]
    ) -> List[Tuple[int, Maybe[LeanForwardGoal]]]:
        if len(goals) == 0:
            return []
        logger.info(f"Going to start {len(goals)} lean goals")
        results: List[Tuple[int, Maybe[LeanForwardGoal]]] = []

        last_log = time.time()
        for start in range(0, len(goals), MAX_CONCURRENT_CREATE_SESSION):
            goal_chunk = goals[start : start + MAX_CONCURRENT_CREATE_SESSION]

            batch_id_to_goal = {}
            for (goal_id, fwd_goal) in goal_chunk:
                assert isinstance(fwd_goal.thm, LeanTheorem)
                try:
                    batch_id = self.expander_env.submit_create_session(fwd_goal.label)
                except DeclNotFound as err:
                    results.append((goal_id, Fail(GoalError(str(err), goal=fwd_goal))))
                else:
                    batch_id_to_goal[batch_id] = (goal_id, fwd_goal)

            for batch_id, (goal_id, fwd_goal) in batch_id_to_goal.items():
                try:
                    bwd_goal = self.expander_env.wait_for_goal(batch_id, fwd_goal.label)
                except DeclNotFound as err:
                    out = Fail(fail=GoalError(str(err), goal=fwd_goal))
                else:
                    assert isinstance(bwd_goal.theorem, LeanTheorem)
                    assert bwd_goal.theorem.state is not None
                    new_goal = copy.deepcopy(fwd_goal)
                    new_goal.thm.state = copy.deepcopy(bwd_goal.theorem.state)
                    assert new_goal.thm.state is not None, new_goal
                    out = Ok(new_goal)
                results.append((goal_id, out))
                if time.time() - last_log > 10:
                    logger.info(f"Session creation [{len(results)}/{len(goals)}]")
                    last_log = time.time()
        logger.info(
            f"Done with lean session (ok: {len([r for _, r in results if r.ok()])}/{len(goals)})"
        )
        for _, result in results:
            if not result.ok():
                assert isinstance(result.err(), GenerationError), type(result.err())
        return results

    def end_goals(self, goals: List[LeanForwardGoal]):
        if len(goals) == 0:
            return
        last_log = time.time()
        batch_ids = [
            self.expander_env.submit_del_session(g.thm.state.session) for g in goals
        ]
        for i, batch_id in enumerate(batch_ids):
            self.expander_env.wait_for_del_session(batch_id)
            if last_log - time.time() > 10:
                logger.info(f"Session deletion [{i+1}/{len(goals)}]")
                last_log = time.time()
        logger.info(f"Deleted {len(batch_ids)} sessions")

    def submit_tactic(
        self, tactic_id: int, fwd_tactic: LeanForwardTactic, graph: LeanForwardGraph
    ):
        assert tactic_id not in self.waiting
        self.waiting[tactic_id] = (fwd_tactic, graph)
        generated = fwd_tactic.next_node
        tactic = fwd_tactic.bwd_tactic
        assert graph.fwd_goal.thm.state is not None, graph
        session = graph.fwd_goal.thm.state.session

        assert generated.state is None
        job = ForwardLeanJob(theorem=generated, tactic=tactic, session=session)
        self.expander_env.process(job, batch_id=tactic_id)

    def ready_statements(self) -> List[Tuple[int, Maybe[LeanFwdEnvOutput]]]:
        ready = self.expander_env.get_all_ready()

        results = []
        for tactic_id, _, result in ready:
            fwd_tactic, graph = self.waiting.pop(tactic_id)
            if result.tactic.is_error():
                if result.tactic.error_msg.startswith("parse_goal_error:"):
                    error = ParseGoalError(
                        result.tactic.error_msg, fwd_tactic=fwd_tactic,
                    )
                elif result.tactic.error_msg.startswith("parse goal timeout"):
                    error = EnvTimeout(result.tactic.error_msg, fwd_tactic=fwd_tactic)
                elif result.tactic.error_msg.startswith("parse_children_error"):
                    error = ParseChildrenError(
                        result.tactic.error_msg, fwd_tactic=fwd_tactic
                    )
                else:
                    error = InvalidTactic(
                        result.tactic.error_msg,
                        fwd_tactic=fwd_tactic,
                        parsed_goal=result.theorem,
                    )
                results.append((tactic_id, Fail(error)))
                continue

            parsed_goal = result.theorem
            assert isinstance(parsed_goal, LeanTheorem)

            # we create a mix were conclusion is the one generated
            # by generator
            generated = copy_lean_theorem(
                parsed_goal,
                change_conclusion=None
                if self.expander_env.dataset.fwd_use_parsed_pp_as_conclusion
                else fwd_tactic.next_node.conclusion,
                change_fingerprint=parsed_goal.conclusion
                if self.expander_env.dataset.fwd_match_on_conclusion
                else None,
            )
            if (
                self.expander_env.dataset.fwd_match_on_conclusion
                and self.expander_env.dataset.fwd_use_parsed_pp_as_conclusion
            ):
                assert generated.conclusion == generated.fingerprint

            if parsed_goal.context != graph.fwd_goal.thm.context:
                results.append(
                    (
                        tactic_id,
                        Fail(WrongContext("Wrong context", fwd_tactic=fwd_tactic)),
                    )
                )
                continue

            sub_goals = cast(LeanSubGoals, result.children)
            if self.expander_env.dataset.fwd_match_on_conclusion:
                # not using real fingerprint as fingerprint but conclusion instead.
                sub_goals = [
                    copy_lean_theorem(sg, change_fingerprint=sg.conclusion)
                    for sg in sub_goals
                ]

            try:
                children_ids = check_generated_and_search_sub_goals_in_graph(
                    generated=parsed_goal,
                    fwd_tactic=fwd_tactic,
                    sub_goals=sub_goals,
                    graph=graph,
                )
            except GenerationError as err:
                out = Fail(err)
            else:
                out = LeanFwdEnvOutput(
                    generated=generated,
                    tactic=fwd_tactic.bwd_tactic,
                    children_ids=children_ids,
                )
                out = Ok(out)
            results.append((tactic_id, out))
        return results

    def close(self):
        if not self.closed:
            self.expander_env.close()
            self.closed = True

    def __del__(self):
        assert self.closed


def check_generated_and_search_sub_goals_in_graph(
    generated: LeanTheorem,
    fwd_tactic: LeanForwardTactic,
    sub_goals: LeanSubGoals,
    graph: LeanForwardGraph,
) -> ChildrenIds:

    if generated in graph.generated_thms:
        raise NodeInGraph("Node in graph!", fwd_tactic=fwd_tactic)

    thm2id = {n: i for i, n in enumerate(graph.generated_thms)}
    children_ids = []
    for i, sub_goal in enumerate(sub_goals):
        if sub_goal not in thm2id:
            raise MissingHyp(
                f"{sub_goal.conclusion}",
                fwd_tactic=fwd_tactic,
                missing=i,
                sub_goals=sub_goals,
            )
        children_ids.append(thm2id[sub_goal])
    return children_ids


def copy_lean_theorem(
    thm: LeanTheorem,
    change_fingerprint: Optional[str] = None,
    change_conclusion: Optional[str] = None,
) -> LeanTheorem:
    return LeanTheorem(
        conclusion=thm.conclusion if change_conclusion is None else change_conclusion,
        fingerprint=thm.fingerprint
        if change_fingerprint is None
        else change_fingerprint,
        context=thm.context,
        state=thm.state,
    )
