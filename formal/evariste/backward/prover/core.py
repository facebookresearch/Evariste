# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import Optional, Set, List, Any, Union, Type, Callable
from dataclasses import dataclass
from enum import unique, Enum
from evariste.backward.graph import Proof, BackwardGoal, Theorem
from evariste.backward.env.core import EnvExpansion

# todo: rename this file and reorg the entire prover folder...


@dataclass
class ProofResult:
    """
    Dataclass which store the result of a proof proven using the asynchronous backward_runner.
    """

    proof: Optional[Proof]
    goal: BackwardGoal
    exception: Optional[Exception]

    def __str__(self):
        if self.exception is not None:
            return f"NoProof : {self.exception}"
        else:
            return "Has Proof"


@unique
class ProofHandlerFailure(str, Enum):
    TERMINATED_WITHOUT_PROVING = "reach max steps without finding a proof."
    TOO_BIG = "number of leaves or nodes is too big we stopped because we dont think we will be able to prove anything."


class ProofHandlerDied(Exception):
    pass


class ProofHandler(ABC):
    """
    A ProofHandler is an object in charge of one single proof. It should provide the expander with a list of subgoal to
    expand (self.to_expand() ) and it should know what to do when provided with different possible tactics for each
    subgoal it has requested an expansion (self.apply(...)).
    Ultimatly it provides the complete proof with get_proof.
    Attribute done and fail records if the proof is finished (and thus the theorem proven) or if it should be discarded
    because of some fails or errors it can not handle.
    """

    def __init__(self, goal: BackwardGoal):
        self.done: bool = False
        self.additional_hyps: Set[Theorem] = set()
        self.goal: BackwardGoal = goal

    @abstractmethod
    def send_materialized(self, th: Theorem):
        """Materializes self.goal into th"""
        raise NotImplementedError

    @abstractmethod
    def get_theorems_to_expand(self) -> Optional[List[Theorem]]:
        """
        :return: a list of theorem=subgoals to expand
        """
        raise NotImplemented

    # Doesnt need to be abstract, if not defined we decide that the goal to aim at is the final goal
    def to_expand_by_forward(self) -> Optional[List[Theorem]]:
        """
        Used by the forward-backward prover it should retrieves goals for the forward prover to aim at
        :return: list of targets for the forward prover
        """
        return [self.goal.theorem]

    def add_hyps(self, hyps: Set[Theorem]):
        """
        Take a set of additional hypothesises to add to the proof and should use them to resolve parts of the existing
        proof if possible. *Only used in forwardbackward*
        :param hyps: hypothesis to add
        :return:
        """
        raise NotImplemented

    def restart(self):
        """
        Should reset all the parameters that control the advancement of the proof handler such as steps. *Only used in forwardbackward*
        """
        raise NotImplemented

    @abstractmethod
    def send_env_expansions(self, tactics: List[EnvExpansion]) -> None:
        """
        :param tactics: List of results of expansion: one expansion per subgoal that had to be expanded. For each of them we get possibly multiple tactics, this function should choose which to keep and add them to the proof.
        :return: None
        """
        raise NotImplemented

    @abstractmethod
    def get_result(self) -> ProofResult:
        """ 
        Extract the proof and other infos from the proof search, return the final proof result.
        :return: the proof result
        """
        pass

    def close(self):
        """called when the proof handler is discarded"""
        pass

    def get_done(self) -> bool:
        """ True if send_env_expansions has returned and done / fail is set"""
        return True

    def status(self) -> Any:
        return {}

    def stop(self) -> None:
        return


HandlerType = Union[Type[ProofHandler], Callable[[BackwardGoal], ProofHandler]]
