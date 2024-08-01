# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
from typing import List, Optional

from evariste.forward.common import GenerationHistory, ForwardGraph
from evariste.forward.core.generation_errors import MissingHyp, InvalidTactic
from evariste.forward.common import SimpleFwdEnvOutput
from evariste.forward.fwd_eq.conftest import ForwardGraphsAndTactics
from evariste.forward.fwd_eq.eq_fwd_env import EqForwardEnv, EqForwardTactic
from evariste.model.data.envs.equations import EquationsEnvironment


def test_eq_fwd_env(
    eq_env: EquationsEnvironment, eq_histories: List[GenerationHistory]
):
    fwd_env = EqForwardEnv(eq_env=eq_env.eq_env, rule_env="default")
    tactic_id = 0
    for eq_history in eq_histories:
        graph = ForwardGraph.from_goal(eq_history.goal)
        last_out: Optional[SimpleFwdEnvOutput] = None
        for step in eq_history.forward_steps():
            fwd_tactic = step.fwd_tactic
            assert isinstance(fwd_tactic, EqForwardTactic)
            fwd_env.submit_tactic(
                tactic_id=tactic_id, fwd_tactic=fwd_tactic, graph=graph
            )
            results = fwd_env.ready_statements()
            assert len(results) == 1
            (res_id, result) = results[0]
            assert res_id == tactic_id
            assert result.ok()
            env_output = result.unwrap()
            assert isinstance(env_output, SimpleFwdEnvOutput)
            assert env_output.generated == step.generated
            tactic_id += 1
            graph.update_with_step(step)
            last_out = env_output
        assert last_out is not None
        assert last_out.generated == graph.fwd_goal.thm


def test_eq_fwd_env_failed(
    eq_env: EquationsEnvironment, eq_fwd_graphs: ForwardGraphsAndTactics
):
    fwd_env = EqForwardEnv(eq_env=eq_env.eq_env, rule_env="default")
    _, fwd_tactic = eq_fwd_graphs[0]
    mismatch_bwd_tactic = fwd_tactic.bwd_tactic
    for i, (graph, fwd_tactic) in enumerate(eq_fwd_graphs[1:]):
        new_fwd_tactic = copy.deepcopy(fwd_tactic)
        new_fwd_tactic.bwd_tactic = copy.deepcopy(mismatch_bwd_tactic)
        fwd_env.submit_tactic(tactic_id=i, fwd_tactic=new_fwd_tactic, graph=graph)
        results = fwd_env.ready_statements()
        assert len(results) == 1
        (res_id, result) = results[0]
        assert res_id == i
        if result.ok():
            unwrapped = result.unwrap()
            assert isinstance(unwrapped, SimpleFwdEnvOutput)
            print(unwrapped.tactic)
            print(unwrapped.generated)
        assert not result.ok()
        err = result.err()
        assert isinstance(err, InvalidTactic) or isinstance(err, MissingHyp)
