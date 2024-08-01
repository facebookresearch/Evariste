# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import abc
from dataclasses import dataclass
from typing import Tuple, List, Optional

from evariste.forward.common import (
    ForwardGraph,
    ForwardTactic,
    EnvInfo,
    ForwardGoal,
)
from evariste.forward.common import FwdEnvOutput
from evariste.forward.core.maybe import Maybe
from evariste.model.data.dictionary import EOS_WORD, STOP_WORD
from params import Params


@dataclass
class ProverEnvSpecifics:
    fwd_params: "FwdTrainParams"
    tokenizer: "FwdTokenizer"
    env: "AsyncForwardEnv"


Statement = str
ChildrenIds = List[int]


class FwdTokenizer(abc.ABC):
    """
    Responsible for all tokenizations for forward, e.g graph -> List[str]
    and List[str] -> ForwardTactic

    Note: EOS are handled by this FwdTokenizer.
    """

    @abc.abstractmethod
    def tokenize_graph(self, graph: ForwardGraph) -> List[str]:
        pass

    @abc.abstractmethod
    def detokenize_command(
        self, command: List[str], graph: ForwardGraph
    ) -> ForwardTactic:
        pass


class ForwardEnv(abc.ABC):
    @abc.abstractmethod
    def apply_tactic(
        self, graph: ForwardGraph, tactic: ForwardTactic
    ) -> Tuple[Statement, ChildrenIds, EnvInfo]:
        pass


class AsyncForwardEnv(abc.ABC):
    @abc.abstractmethod
    def submit_tactic(
        self, tactic_id: int, fwd_tactic: ForwardTactic, graph: ForwardGraph
    ):
        pass

    @abc.abstractmethod
    def ready_statements(self) -> List[Tuple[int, Maybe[FwdEnvOutput]]]:
        pass

    @abc.abstractmethod
    def close(self):
        pass


class SessionEnv(abc.ABC):
    @abc.abstractmethod
    def start_goals(
        self, goals: List[Tuple[int, ForwardGoal]]
    ) -> List[Tuple[int, Maybe[ForwardGoal]]]:
        pass

    @abc.abstractmethod
    def end_goals(self, goals: List[ForwardGoal]):
        pass


@dataclass
class FwdTrainParams(Params):
    """
    All information parsed from TrainerArgs needed by the ForwardProver.

    Parsing of TrainerArgs depends on environment
    """

    max_inp_len: int
    stop_symbol: str
    is_generator: bool
    use_stop_action: bool
    use_critic: bool
    train_dir: str
    command_prefix: Optional[List[str]] = None
    discr_conditioning: bool = False

    @property
    def stop_command(self) -> List[str]:
        if self.command_prefix:
            return [EOS_WORD, *self.command_prefix, STOP_WORD, EOS_WORD]
        else:
            return [EOS_WORD, STOP_WORD, EOS_WORD]
