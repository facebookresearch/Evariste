# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Core abstractions"""

from typing import Sequence, Optional, Tuple, Dict, Union, List, Any, Callable, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib

from evariste.utils import rstr


Token = str
Hypothesis = Tuple[Optional[Token], str]


class MalformedTactic(Exception):
    pass


class MalformedTheorem(Exception):
    pass


class HashByString:
    @property
    def hash(self):
        return self._hash

    def __eq__(self, other):
        return self.hash == other.hash

    def __hash__(self):
        return self._hash

    def __lt__(self, other):
        return self.hash.__lt__(other.hash)

    def __init__(self, fingerprint: str):
        assert type(fingerprint) == str
        self._hash = int(hashlib.md5(str(fingerprint).encode()).hexdigest(), 16)


class Tactic(HashByString, ABC):
    """Elementary step in a proof"""

    def __init__(
        self, is_valid: bool, unique_str: str, error_msg: Optional[str], malformed: bool
    ):
        super().__init__(unique_str)
        self.error_msg = error_msg
        # malformed == the tactic tokens from the model couldn't be parsed
        self.malformed = malformed
        # valid == could be applied in the environment
        self.is_valid = is_valid

        self.duration: Optional[float] = None

        # sanity check
        assert type(is_valid) is bool
        assert type(unique_str) is str
        assert error_msg is None or type(error_msg) is str
        assert not (malformed and is_valid)

    @abstractmethod
    def tokenize(self) -> List[Token]:
        pass

    def tokenize_error(self) -> List[Token]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def from_tokens(tokens: List[Token]) -> "Tactic":
        pass

    @staticmethod
    @abstractmethod
    def from_error(error_msg: str, tokens: Optional[List[str]] = None) -> "Tactic":
        pass

    @abstractmethod
    def to_dict(self, light=False) -> Dict:
        """
        @param light: True if we don't require all data to be serialized, like state for hol light goals
        @return: a dictionary allowing from_dict to reconstruct the object
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict) -> "Tactic":
        pass

    def is_error(self):
        return self.error_msg is not None

    def get_error_code(self) -> str:
        assert self.error_msg is not None
        return self.error_msg


@dataclass
class NodeInfo:
    depth: Optional[int] = None


class Theorem(HashByString, ABC):
    def __init__(
        self,
        conclusion: str,
        hyps: Sequence[Tuple[Optional[Token], str]],
        train_label: Optional[str] = None,
        given_unique_str: Optional[str] = None,
    ):
        # sanity check
        assert type(conclusion) is str, type(conclusion)
        assert train_label is None or type(train_label) is str
        assert given_unique_str is None or type(given_unique_str) is str

        # sort hypotheses (str cannot be compared to None so split first)
        hyps_none = [x for x in hyps if x[0] is None]
        hyps_name = [x for x in hyps if x[0] is not None]
        hyps = sorted(hyps_none) + sorted(hyps_name)
        if given_unique_str is not None:
            unique_str = given_unique_str
        else:
            assert len(conclusion) > 0
            unique_strings: List[str] = []
            for name, hyp in hyps:
                if name is not None:
                    unique_strings.append(name)
                unique_strings.append(hyp)
            unique_strings.append(conclusion)
            assert all(type(x) is str for x in unique_strings)
            unique_str = "|||".join(unique_strings)

        super().__init__(unique_str)
        self.conclusion = conclusion
        self.hyps = hyps
        self.train_label = train_label

        # optional info about the theorem node (e.g. position in the MCTS graph / simutree)
        self.info: NodeInfo = NodeInfo()

    @abstractmethod
    def tokenize(self) -> List[Token]:
        pass

    @classmethod
    @abstractmethod
    def from_tokens(cls, tokens: List[Token]) -> "Theorem":
        pass

    @abstractmethod
    def to_dict(self, light=False):
        """
        @param light: True if we don't require all data to be serialized, like state for hol light goals
        @return: a dictionary allowing from_dict to reconstruct the object
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data):
        pass


class UnMaterializedTheorem(Theorem):
    """Represents a theorem yet to be materialized : either loaded in lean, or created by an eq worker"""

    def tokenize(self) -> List[Token]:
        raise NotImplementedError("This theorem is not materialized yet")

    @classmethod
    def from_tokens(cls, tokens: List[Token]) -> "Theorem":
        raise NotImplementedError("wrong class to call this on")

    def to_dict(self, light=False):
        raise NotImplementedError(
            "This theorem is not materialized in the ml_server yet"
        )

    @classmethod
    def from_dict(cls, data):
        raise NotImplementedError(
            "This theorem is not materialized in the ml_server yet"
        )

    def __init__(self, label: str):
        super().__init__(
            conclusion="__unmaterialized__",
            hyps=[],
            train_label=label,
            given_unique_str=label,
        )  # otherwise cannot be hashed. Used in GreedyProofHandler
        assert "__" not in label
        self.label = label

        # Lean specific: set to LeanExpanderEnv batch id once create has been called
        self.batch_id: Any = None

    def __str__(self) -> str:
        return f"UnmatTheorem({self.label})"


class NonPicklableProof:
    pass


Proof = Union[Tuple[Theorem, Tactic, List["Proof"]], NonPicklableProof]  # type: ignore ## mypy doesn't support recursive types
ProofId = int


def get_tac(p: Proof) -> Tactic:
    assert not isinstance(p, NonPicklableProof)
    return p[1]


def get_thm(p: Proof) -> Theorem:
    assert not isinstance(p, NonPicklableProof)
    return p[0]


T = TypeVar("T")


def fold_proof(p: Proof, start: T, fn: Callable[[Proof, T], T]) -> T:
    stack: List[Proof] = [p]
    x = start
    while stack:
        p = stack.pop()
        assert not isinstance(p, NonPicklableProof)
        _, _, children = p
        stack.extend(children[::-1])
        x = fn(p, x)
    return x


@dataclass
class GoalParams:

    # decoder
    temperature: Optional[float] = None
    n_samples: Optional[int] = None
    length_penalty: Optional[float] = None

    # prover
    tactic_fill: Optional[str] = None

    # MCTS
    exploration: Optional[float] = None
    depth_penalty: Optional[float] = None
    n_expansions: Optional[int] = None
    succ_expansions: Optional[int] = None
    policy: Optional[str] = None  # alpha_zero / other
    policy_temp_level: Optional[str] = None  # global / simutree / node
    policy_temperature: Optional[float] = None
    q_value_solved: Optional[int] = None

    # Conditioning
    conditioning_label: Optional[str] = None
    cond_id: Optional[int] = None

    def __post_init__(self):
        if self.n_expansions is not None:  # will be float with hyper opt
            self.n_expansions = int(self.n_expansions)
        assert self.conditioning_label is None or type(self.conditioning_label) is str


class BackwardGoal:
    def __init__(
        self,
        theorem: Theorem,
        label: Optional[str] = None,
        name: Optional[str] = None,
        split: Optional[str] = None,
        num_steps: Optional[int] = None,
        params: Optional[GoalParams] = None,
    ):
        if isinstance(theorem, UnMaterializedTheorem):
            assert label is None and name is None
            label = theorem.label
        else:
            assert isinstance(theorem, Theorem)
            assert label is not None
            assert name is None or name.startswith(f"{label}__")

        assert "__" not in label
        name = f"{label}__{rstr(10)}" if name is None else name

        self.theorem = theorem
        self.name: str = name
        self.label: str = label
        self.split = split
        self.num_steps = num_steps  # for HOL-Light proof shortening
        self.params = params  # Optimized through nevergrad

    @property
    def materialized(self):
        return not isinstance(self.theorem, UnMaterializedTheorem)

    @staticmethod
    def create_unmat(label: str, split: str):
        assert "__" not in label
        return BackwardGoal(theorem=UnMaterializedTheorem(label=label), split=split)

    def __str__(self) -> str:
        return f"{self.name} // {self.theorem} // {self.params}"


def get_proof_size(proof: Optional[Proof]) -> int:
    assert proof is not None and not isinstance(proof, NonPicklableProof)
    _, _, children = proof
    return 1 + sum([get_proof_size(c) for c in children])


def get_proof_depth(proof: Optional[Proof]) -> int:
    assert proof is not None and not isinstance(proof, NonPicklableProof)
    _, _, children = proof
    return 1 + max([get_proof_depth(c) for c in children], default=0)
