# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from typing import List, Tuple, Optional, Dict, Any

from evariste.forward.common import (
    ForwardStepBeam,
    MaybeForwardStep,
    ForwardStep,
    ForwardTactic,
    GenerationError,
)
from evariste.forward.common import FwdEnvOutput, PolicyOutputBeam, PolicyOutput
from evariste.forward.env_specifics.prover_env_specifics import AsyncForwardEnv


logger = getLogger()


class ForwardStepperEnv(ABC):
    @abstractmethod
    def submit_policy_output_beam(self, graph_id: int, policy_beam: PolicyOutputBeam):
        pass

    @abstractmethod
    def ready_step_beams(self) -> List[Tuple[int, ForwardStepBeam]]:
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        pass


class BaseForwardStepperEnv(ForwardStepperEnv):
    def __init__(self, env: AsyncForwardEnv):
        self.env = env
        self.command_id = 0

        self.graph_id2beam: Dict[int, _BeamHandler] = {}
        self.cmd_id2builder: Dict[int, _ForwardStepBuilder] = {}

        self.ready: List[Tuple[int, ForwardStepBeam]] = []
        self.command_id = 0

        self.closed = False

        self.cur_stats: Dict[str, Any] = defaultdict(float)

    def submit_policy_output_beam(self, graph_id: int, policy_beam: PolicyOutputBeam):
        start = time.time()
        builders = []
        for id_in_beam, maybe_policy_out in enumerate(policy_beam):
            builder = _ForwardStepBuilder(id_in_beam=id_in_beam, graph_id=graph_id)
            builders.append(builder)

            if not maybe_policy_out.ok():
                builder.set_error(maybe_policy_out.err())
                continue

            policy_out = maybe_policy_out.unwrap()
            builder.set_policy_output(policy_out)

            if builder.done:  # stop_tactic
                continue

            assert not builder.done
            assert isinstance(policy_out.fwd_tactic, ForwardTactic)

            cmd_id = self._cur_command_id()
            self.cmd_id2builder[cmd_id] = builder
            self.env.submit_tactic(
                cmd_id, graph=policy_out.graph, fwd_tactic=policy_out.fwd_tactic
            )

        beam = _BeamHandler(fwd_step_builders=builders)
        self.graph_id2beam[graph_id] = beam

        # only wrong tactics in beam
        self._pop_if_beam_ready(graph_id)
        self.cur_stats["time_in_async_stepper_env.submit"] += time.time() - start

    def ready_step_beams(self) -> List[Tuple[int, ForwardStepBeam]]:
        start = time.time()

        new_outputs = self.env.ready_statements()
        for cmd_id, output in new_outputs:
            builder = self.cmd_id2builder.pop(cmd_id)
            assert isinstance(builder, _ForwardStepBuilder)
            if not output.ok():
                builder.set_error(output.err())
            else:
                ok_output = output.unwrap()
                builder.set_fwd_env_output(ok_output)
            self._pop_if_beam_ready(builder.graph_id)
        ready = self.ready
        self.ready = []

        self.cur_stats["time_in_async_stepper_env.ready"] += time.time() - start
        return ready

    def _cur_command_id(self) -> int:
        cmd_id = self.command_id
        self.command_id += 1
        return cmd_id

    def _pop_if_beam_ready(self, graph_id: int):
        beam = self.graph_id2beam[graph_id]
        if beam.is_ready():
            self.ready.append((graph_id, beam.get_beam()))
            self.graph_id2beam.pop(graph_id)

    def close(self):
        if not self.closed:
            self.env.close()
            self.closed = True

    def stats(self) -> Dict[str, Any]:
        return dict(self.cur_stats)


@dataclass
class _ForwardStepBuilder:
    graph_id: int
    id_in_beam: int

    policy_output: Optional[PolicyOutput] = None
    env_output: Optional[FwdEnvOutput] = None

    error: Optional[GenerationError] = None
    done: bool = False

    def set_policy_output(self, policy_output: PolicyOutput):
        assert not self.done
        assert isinstance(policy_output, PolicyOutput)
        self.policy_output = policy_output
        if self.policy_output.is_stop():
            self.done = True

    def set_error(self, err: GenerationError):
        assert not self.done
        assert self.error is None
        self.error = err
        self.done = True

    def get_policy_output(self) -> PolicyOutput:
        """Unwrapped policy output for mypy (no Optional)"""
        assert self.policy_output is not None
        return self.policy_output

    def set_fwd_env_output(self, output: FwdEnvOutput):
        assert not self.done
        self.env_output = output
        self.done = True

    def to_forward_step(self) -> MaybeForwardStep:
        assert self.done
        if self.error:
            return MaybeForwardStep(step=None, err=self.error)

        policy_output = self.get_policy_output()
        if policy_output.is_stop():
            assert self.env_output is None
            return MaybeForwardStep(
                ForwardStep(policy_output=policy_output, env_output=None), err=None,
            )
        env_output = self.env_output
        assert env_output is not None
        assert isinstance(env_output, FwdEnvOutput)
        return MaybeForwardStep(
            step=ForwardStep(policy_output=policy_output, env_output=env_output),
            err=None,
        )


@dataclass
class _BeamHandler:
    fwd_step_builders: List["_ForwardStepBuilder"]

    def is_ready(self) -> bool:
        return all(builder.done for builder in self.fwd_step_builders)

    def get_beam(self) -> ForwardStepBeam:
        return [builder.to_forward_step() for builder in self.fwd_step_builders]
