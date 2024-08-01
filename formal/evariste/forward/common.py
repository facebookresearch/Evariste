# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from dataclasses import dataclass, asdict, fields
from typing import (
    List,
    Tuple,
    Dict,
    Optional,
    NamedTuple,
    Set,
    Any,
    TypeVar,
    Generic,
    Union,
)

from evariste.forward.core.maybe import Maybe
from evariste.forward.core.generation_errors import GenerationError
from evariste.backward.graph import Tactic, Theorem
from evariste.utils import rstr


log = logging.getLogger(__name__)


class ForwardTactic:
    """No fields bwd_tactic and next_node because of MM, but this could
    change in the future (for the moment I prefer not building MMTheorem with
    missing e_hyps)"""

    pass


SomeTactic = TypeVar("SomeTactic", bound=Tactic)
SomeTheorem = TypeVar("SomeTheorem", bound=Theorem)
SomeForwardTactic = TypeVar("SomeForwardTactic", bound=ForwardTactic)


@dataclass
class GenericForwardTactic(Generic[SomeTheorem, SomeTactic], ForwardTactic):
    next_node: SomeTheorem
    bwd_tactic: SomeTactic


class EnvInfo:
    pass


class StopTactic:
    pass


class FwdEnvOutput:
    pass


@dataclass
class SimpleFwdEnvOutput(FwdEnvOutput, Generic[SomeTheorem, SomeTactic]):
    generated: SomeTheorem
    tactic: SomeTactic
    children_ids: List[int]
    is_stop: bool = False


@dataclass
class PolicyOutput:
    graph: "ForwardGraph"
    command: List[int]
    command_str: str
    fwd_tactic: Union[ForwardTactic, StopTactic]
    score: float
    normalized_score: float
    critic: Optional[float] = None

    def is_stop(self) -> bool:
        return isinstance(self.fwd_tactic, StopTactic)

    def get_fwd_tactic(self) -> ForwardTactic:
        """Unwrapped fwd tactic for mypy (not StopTactic)"""
        assert isinstance(self.fwd_tactic, ForwardTactic)
        return self.fwd_tactic


PolicyOutputBeam = List[Maybe[PolicyOutput]]

SomePolicyOutput = TypeVar("SomePolicyOutput", bound=PolicyOutput)
SomeEnvOutput = TypeVar("SomeEnvOutput", bound=FwdEnvOutput)


class ForwardStep(Generic[SomePolicyOutput, SomeEnvOutput]):
    def __init__(
        self, policy_output: SomePolicyOutput, env_output: Optional[SomeEnvOutput]
    ):
        self.policy_output = policy_output
        self.env_output = env_output

    # allows access to env_output / policy_output properties for pre-existing forward code.
    @property
    def children(self):
        assert isinstance(self.env_output, SimpleFwdEnvOutput)
        return self.env_output.children_ids

    @property
    def tactic(self):
        assert isinstance(self.env_output, SimpleFwdEnvOutput)
        return self.env_output.tactic

    @property
    def generated(self):
        assert isinstance(self.env_output, SimpleFwdEnvOutput)
        return self.env_output.generated

    @property
    def fwd_tactic(self):
        assert isinstance(self.policy_output, PolicyOutput)
        return self.policy_output.fwd_tactic

    @property
    def score(self):
        assert isinstance(self.policy_output, PolicyOutput)
        return self.policy_output.score

    def is_stop(self):
        assert isinstance(self.policy_output, PolicyOutput)
        return self.policy_output.is_stop()


@dataclass
class ForwardGoal(Generic[SomeTheorem]):
    # new fmt (None when generation)
    thm: Optional[SomeTheorem]

    # Name of the trm were this goal is extracted originally
    label: str
    name: Optional[str] = None

    # For EQ and MM, specify global_hyps here
    global_hyps: Optional[List[SomeTheorem]] = None

    # None if not enforced
    forbidden: Optional[Set[str]] = None

    # for generation, we can condition the generation with:
    # 1. a label that the generator will need to use in its generated proof
    # (to favor diversity). This label should be sampled in build_generation_goal
    # method of FwdEnvBuilder
    label_conditioning: Optional[str] = None
    # 2. a token "unproved" to force the generator to generate hard goal for the prover
    proved_conditioning: Optional[str] = None
    # 3. a token dependent on the reward_quantile we want to generate
    reward_quantile_conditioning: Optional[str] = None

    def use_global_hyps(self) -> bool:
        return self.global_hyps is not None

    def get_global_hyps(self) -> List[SomeTheorem]:
        assert self.global_hyps is not None
        return self.global_hyps

    def is_solved_by_step(self, step: ForwardStep):
        assert isinstance(step, ForwardStep)
        # relying on Theorem __equal__
        return self.thm == step.generated

    def __post_init__(self):
        # use the same naming pattern as in BackwardGoal
        assert (self.label is None) or ("__" not in self.label)
        self.name = f"{self.label}__{rstr(10)}" if self.name is None else self.name
        if self.thm is not None and self.global_hyps is not None:
            assert len(self.thm.hyps) == len(self.global_hyps)

    def to_dict(self) -> Dict[str, Any]:
        # TODO: we can probably write a custom generic function for this
        dict_ = asdict(self)
        for field in fields(self):
            name = field.name
            if isinstance(dict_[name], set):
                dict_[name] = list(dict_[name])
        return dict_


@dataclass
class ForwardGraph(Generic[SomeTheorem]):
    """
    Dataclass to store the minimal information needed to make seq2seq model inputs
    """

    fwd_goal: ForwardGoal[SomeTheorem]
    generated_thms: List[SomeTheorem]  # generated

    def is_new_fmt(self):
        pass

    @property
    def global_hyps_and_generated(self) -> List[SomeTheorem]:
        """Return global_hyps + generated nodes"""
        if self.fwd_goal.use_global_hyps():
            return self.fwd_goal.get_global_hyps() + self.generated_thms
        else:
            return self.generated_thms

    # bwd comp
    @property
    def forbidden(self) -> Optional[Set[str]]:
        return self.fwd_goal.forbidden

    def n_nodes(self):
        return len(self.generated_thms)

    @staticmethod
    def from_goal(goal: ForwardGoal) -> "ForwardGraph":
        return ForwardGraph(fwd_goal=goal, generated_thms=[])

    def update_with_step(self, step: ForwardStep) -> None:
        self.generated_thms.append(step.generated)
        for cid in step.children:
            assert cid < len(self.global_hyps_and_generated)


class MaybeForwardStep(NamedTuple):
    # this wrapper allows to gather statistics on generation errors
    step: Optional[ForwardStep]
    err: Optional[GenerationError]


ForwardStepBeam = List[MaybeForwardStep]


@dataclass
class GenerationHistory:
    goal: ForwardGoal
    stack: List[MaybeForwardStep]

    # convenient way of storing GenerationInfos
    info: Optional["GenerationInfos"] = None

    # to store beams when using beam sampling
    beams: Optional[List[ForwardStepBeam]] = None
    selected_in_beam: Optional[List[int]] = None

    def append_step(self, step: MaybeForwardStep):
        self.stack.append(step)

    def forward_steps(self) -> List[ForwardStep]:
        return [
            r.step for r in self.stack if (r.step is not None and not r.step.is_stop())
        ]

    def errors(self) -> List[GenerationError]:
        return [r.err for r in self.stack if r.err is not None]

    def forward_graph(self) -> ForwardGraph:
        graph = ForwardGraph.from_goal(self.goal)
        for step in self.forward_steps():
            graph.update_with_step(step)
        return graph

    def proof_nodes(self) -> List["ProofNode"]:
        nodes: List[ProofNode] = []
        if self.goal.global_hyps is not None:
            nodes_with_global_hyps: List[ProofNode] = [
                ProofNode.create_hyp(hyp) for hyp in self.goal.global_hyps
            ]
        else:
            nodes_with_global_hyps = []
        for step in self.forward_steps():
            assert isinstance(step.tactic, Tactic)
            node = ProofNode(
                theorem=step.generated,
                tactic=step.tactic,
                children=[nodes_with_global_hyps[cid] for cid in step.children],
            )
            nodes.append(node)
            nodes_with_global_hyps.append(node)
        return nodes

    def solving_proof_tree(self) -> "ProofNode":
        assert len(self.stack) > 0, "Goal is not solved!"
        last_step = self.stack[-1].step
        assert isinstance(last_step, ForwardStep)
        assert last_step.generated == self.goal.thm, "Goal is not solved!"
        return self.proof_nodes()[-1]

    def append_beam(self, beam: ForwardStepBeam, selected: int):
        if self.beams is None:
            assert self.selected_in_beam is None
            self.beams = []
            self.selected_in_beam = []
        assert self.selected_in_beam is not None
        self.beams.append(beam)
        self.selected_in_beam.append(selected)

    def beam_proof_nodes(self) -> List["ProofNode"]:
        nodes: List[ProofNode] = []
        if self.beams is None:
            return nodes
        assert self.selected_in_beam is not None

        assert len(self.selected_in_beam) == len(self.beams)

        current_graph: List[ProofNode] = []

        for selected_in_beam, beam in zip(self.selected_in_beam, self.beams):
            steps = sorted(
                [(i, s.step) for i, s in enumerate(beam) if s.step],
                key=lambda x: -x[1].score,
            )  # best first

            selected = False
            thm2node: Dict[Theorem, ProofNode] = {}

            for id_in_beam, step in steps:
                if step.generated in thm2node:
                    node = thm2node[step.generated]
                else:
                    node = ProofNode(
                        theorem=step.generated,
                        tactic=step.tactic,
                        children=[current_graph[cid] for cid in step.children],
                    )
                    thm2node[step.generated] = node
                    nodes.append(node)

                if selected_in_beam == id_in_beam:
                    current_graph.append(node)
                    selected = True

            if not selected:
                # if not selected, no valid stuff if beam
                assert len(steps) == 0, len(steps)
        return nodes

    # TODO: fix dependency hell
    # def to_eq_nodes(self) -> Tuple[List[GraphNode], List[GraphNode]]:
    #     return history_to_eq_nodes(self)
    # def last_node_stats(self) -> GenerationStats:
    #     return last_node_stats(self)


@dataclass
class GenerationInfos:
    n_invalid_consecutive: int = 0
    stopped: Optional[Tuple[str, Any]] = None
    solved: bool = False


class ProofNode(Generic[SomeTheorem, SomeTactic]):
    """
    Simple proof node (one tactic by node)
    """

    def __init__(
        self,
        theorem: SomeTheorem,
        tactic: Optional[SomeTactic],
        children: Optional[List["ProofNode[SomeTheorem, SomeTactic]"]],
        is_hyp: bool = False,
    ):
        self.theorem = theorem
        self.tactic = tactic
        self.children = children
        self.is_hyp = is_hyp
        if self.is_hyp:
            assert self.tactic is None
            assert self.children is None
        else:
            assert self.tactic is not None
            assert self.children is not None

    def get_tactic(self) -> SomeTactic:
        assert self.tactic is not None
        return self.tactic

    def get_children(self) -> List["ProofNode[SomeTheorem, SomeTactic]"]:
        assert self.children is not None
        return self.children

    @classmethod
    def create_hyp(cls, theorem: SomeTheorem) -> "ProofNode":
        return cls(theorem=theorem, tactic=None, children=None, is_hyp=True)

    def to_bwd_proof(self) -> Tuple[SomeTheorem, SomeTactic, List[Any]]:
        return (
            self.theorem,
            self.get_tactic(),
            [c.to_bwd_proof() for c in self.get_children()],
        )

    def __repr__(self):
        name = self.__class__.__name__
        attrs = [("theorem", self.theorem)]
        if not self.is_hyp:
            attrs.extend(
                [
                    ("tactic", self.tactic),
                    ("children", [c.theorem for c in self.children]),
                ]
            )
        else:
            attrs.append(("is_hyp", self.is_hyp))

        attr = ", ".join(f"{k}={v!r}" for k, v in attrs)
        return f"{name}({attr})"
