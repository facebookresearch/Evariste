# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from typing import Tuple, Optional, List


from evariste.forward.core.forward_policy import (
    AsyncForwardPolicy,
    BatchConfig,
    SyncForwardPolicy,
    ForwardPolicy,
)
from evariste.forward.core.forward_stepper_env import (
    ForwardStepperEnv,
    BaseForwardStepperEnv,
)
from evariste.forward.core.maybe import Maybe, Ok
from evariste.forward.env_specifics.prover_env_specifics import (
    FwdTokenizer,
    AsyncForwardEnv,
    SessionEnv,
)
from evariste.model.data.dictionary import Dictionary
from evariste.forward.model.seq2seq_model import Seq2SeqModel
from evariste.forward.common import (
    ForwardGraph,
    ForwardStepBeam,
    ForwardGoal,
)

logger = getLogger()


class ForwardStepper:
    """
    Class that allow to sample a beam of ForwardSteps given a ForwardGraph as input.

    ForwardSteps contains the generated node statement, the tactic that generated it,
    the children ids etc...
    """

    def __init__(
        self,
        sync_policy: ForwardPolicy,
        fwd_env: AsyncForwardEnv,
        use_async_model: bool,
    ):

        if not use_async_model:
            self.policy = sync_policy
        else:
            assert isinstance(sync_policy, SyncForwardPolicy)
            self.policy = AsyncForwardPolicy(policy=sync_policy)

        self.fwd_env = fwd_env

        self.stepper_env: ForwardStepperEnv
        assert isinstance(fwd_env, AsyncForwardEnv)
        self.stepper_env = BaseForwardStepperEnv(fwd_env)
        assert isinstance(self.stepper_env, ForwardStepperEnv)

        self.closed = False

    @property
    def sync_policy(self) -> SyncForwardPolicy:
        if isinstance(self.policy, AsyncForwardPolicy):
            return self.policy.policy
        else:
            assert isinstance(self.policy, SyncForwardPolicy)
            return self.policy

    def start_goals(
        self, goals: List[Tuple[int, ForwardGoal]]
    ) -> List[Tuple[int, Maybe[ForwardGoal]]]:
        if isinstance(self.fwd_env, SessionEnv):
            return self.fwd_env.start_goals(goals)
        return [(i, Ok(goal)) for i, goal in goals]

    def end_goals(self, goals: List[ForwardGoal]):
        if isinstance(self.fwd_env, SessionEnv):
            return self.fwd_env.end_goals(goals)

    def send_graph(self, graph_id: int, graph: ForwardGraph):
        self.policy.submit_graph(graph_id, graph)

    def ready_steps(self) -> List[Tuple[int, ForwardStepBeam]]:
        ready_policy_beams = self.policy.ready_beams()
        for graph_id, policy_beam in ready_policy_beams:
            self.stepper_env.submit_policy_output_beam(graph_id, policy_beam)
        return self.stepper_env.ready_step_beams()

    @property
    def stats(self):
        stats = dict(self.policy.stats())
        stats.update(self.stepper_env.stats())
        return stats

    def reload_model_weights(self):
        self.policy.reload_model_weights()

    def close(self):
        if not self.closed:
            self.policy.close()
            self.stepper_env.close()
            self.closed = True

    def __del__(self):
        if not self.closed:
            raise RuntimeError(f"{self} was not closed properly")
