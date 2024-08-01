# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Set
from collections import defaultdict

from evariste.forward.common import (
    ForwardGraph,
    ForwardGoal,
    ForwardStep,
    ForwardStepBeam,
    PolicyOutput,
)
from evariste.forward.core.generation_errors import (
    GenerationError,
    InputDicoError,
    MaxLenReached,
)
from evariste.forward.core.maybe import Maybe, Fail, Ok
from evariste.forward.fwd_eq.gen.env import EqGenForwardEnv, EqGenFwdEnvOutput
from evariste.forward.fwd_eq.gen.random_forward_policy import EqGenForwardGraph
from evariste.forward.fwd_eq.gen.tactics import (
    EqGenForwardTactic,
    EqGenForwardSubstTactic,
)
from evariste.forward.proof_search import ForwardProofSearch, StopConfig
from evariste.envs.eq.generation import (
    GraphNode,
    GraphInternalNode,
    GraphHypNode,
    GraphTrueNode,
    GraphSimpNode,
    GraphNormNumNode,
)

EqStep = ForwardStep[PolicyOutput, EqGenFwdEnvOutput]


@dataclass
class MaxNodeReachedError(GenerationError):
    type: str = "max_node_reached"
    n_nodes: int = 0


@dataclass
class MaxGenReachedError(GenerationError):
    type: str = "max_gen_reached"
    n_gen: int = 0


@dataclass
class MaxConsInvError(GenerationError):
    type: str = "max_cons_inv"
    n_cons_inv: int = 0


class CycleInGeneration(Exception):
    pass


class EqGenProofSearch(ForwardProofSearch):
    def __init__(
        self,
        stop_config: StopConfig,
        env: EqGenForwardEnv,
        init_graph: EqGenForwardGraph,
    ):
        self.cfg = stop_config
        self.init_graph = init_graph
        self.init_graph.node_id_before_tac[0] = len(init_graph.nodes)
        self.prefix_to_node: Dict[str, GraphNode] = {}
        for n in init_graph.nodes:
            self.prefix_to_node[n.node.prefix()] = n

        self.all_steps: List[Maybe[EqStep]] = []
        self.steps: List[EqStep] = []

        self.finished: bool = False
        self.id_in_search = 0
        self.n_invalid_consecutive: int = 0

        self.stopped: Optional[GenerationError] = None

    def stats(self):
        counts = defaultdict(int)
        for s in self.steps:
            assert isinstance(s.env_output, EqGenFwdEnvOutput)
            counts[s.env_output.tactic.__class__.__name__] += 1
        return counts

    def clear(self):
        self.init_graph.clear()
        for s in self.steps:
            assert s.env_output is not None
            s.env_output.graph.clear()

    def graphs_and_tactics(self) -> List[Tuple[EqGenForwardGraph, EqGenForwardTactic]]:
        assert isinstance(self.steps[0].env_output, EqGenFwdEnvOutput)
        res = [(self.init_graph.cut(0), self.steps[0].env_output.tactic)]
        for i, s in enumerate(self.steps[:-1]):
            assert isinstance(s.env_output, EqGenFwdEnvOutput)
            next_env_output = self.steps[i + 1].env_output
            assert isinstance(next_env_output, EqGenFwdEnvOutput)
            res.append((s.env_output.graph.cut(i + 1), next_env_output.tactic))
        return res

    def get_forward_steps(
        self, node_id: int
    ) -> List[Tuple[EqGenForwardGraph, EqGenForwardTactic]]:

        topo_orders: List[List[Tuple[EqGenForwardTactic, GraphNode]]] = []
        subst_tacs: List[Optional[EqGenForwardSubstTactic]] = []

        root = self.prefix_to_node[self.next_graph.nodes[node_id].node.prefix()]

        next_roots: Tuple[List[GraphNode], List[GraphNode]] = ([root], [])

        visiting: Set[int] = set()
        visited: Set[int] = set()

        def visit(
            cur: GraphNode,
            next_r: List[GraphNode],
            next_topo: List[Tuple[EqGenForwardTactic, GraphNode]],
            next_substs: List[EqGenForwardSubstTactic],
        ):
            if isinstance(cur, (GraphTrueNode, GraphHypNode)):
                return
            if id(cur) in visiting:
                raise CycleInGeneration
            if id(cur) in visited:
                return
            visiting.add(id(cur))
            tac = cur.parent_tac
            if isinstance(tac, EqGenForwardSubstTactic):
                next_substs.append(tac)
                assert cur.parent is not None
                next_r.append(cur.parent)
            else:
                assert isinstance(cur, GraphInternalNode)
                for h in cur.hyps:
                    visit(h, next_r, next_topo, next_substs)
                assert cur.parent_tac is not None
                next_topo.append((cur.parent_tac, cur))
            visiting.remove(id(cur))
            visited.add(id(cur))

        i = 0
        while next_roots[i % 2]:
            topo_orders.append([])
            next_substs: List[EqGenForwardSubstTactic] = []
            for r in next_roots[i % 2]:
                visit(r, next_roots[(i + 1) % 2], topo_orders[-1], next_substs)
            next_roots[i % 2].clear()
            if len(next_substs) > 0:
                assert len({id(t) for t in next_substs}) == 1
                subst_tacs.append(next_substs[0])
            else:
                subst_tacs.append(None)
            i += 1

        assert len(topo_orders) == len(subst_tacs), (len(topo_orders), len(subst_tacs))
        result: List[Tuple[EqGenForwardGraph, EqGenForwardTactic]] = []

        cur_nodes: List[GraphNode] = []

        def mk_graph(nodes):
            return EqGenForwardGraph(
                self.init_graph.proof_id,
                self.init_graph.rule_env,
                self.init_graph.env,
                # shallow copy required
                nodes=[n for n in nodes],
                max_true_nodes=self.init_graph.max_true_nodes,
                max_created_hyps=self.init_graph.max_created_hyps,
                prob_add_hyp=self.init_graph.prob_add_hyp,
                # TODO: fix has_simps?
            )

        for subst_tac, topo_order in zip(subst_tacs[::-1], topo_orders[::-1]):
            if subst_tac is not None:
                g = mk_graph(cur_nodes)
                result.append((g, subst_tac))
                substituted, _ = g.clone(subst_tac.substs)
                cur_nodes = [n for n in substituted.nodes]

            for tac, next_node in topo_order:
                g = mk_graph(cur_nodes)
                assert isinstance(
                    next_node, (GraphInternalNode, GraphSimpNode, GraphNormNumNode)
                )
                for h in next_node.hyps:
                    if h.node.prefix() in g.prefix2id:
                        continue
                    elif isinstance(h, GraphHypNode):
                        g.add_node(h)
                    elif isinstance(h, GraphTrueNode):
                        pass
                    else:
                        raise RuntimeError("boom")
                result.append((g, tac))
                cur_nodes = [n for n in g.nodes]
                cur_nodes.append(next_node)

        return result

    @property
    def goal(self) -> ForwardGoal:
        return ForwardGoal(None, "unused")

    @property
    def next_graph(self) -> EqGenForwardGraph:
        if len(self.steps) == 0:
            return self.init_graph
        assert self.steps[-1].env_output is not None
        return self.steps[-1].env_output.graph

    def next_graphs(self) -> List[Tuple[int, ForwardGraph]]:
        # TODO: fix by adding parent type to Forward / EqGenForward
        return [(self.id_in_search, self.next_graph)]  # type: ignore

    def update_with_beam(self, id_in_search: int, step_beam: ForwardStepBeam):
        assert id_in_search == self.id_in_search  # sanity check
        self.id_in_search += 1
        assert len(step_beam) == 1, len(step_beam)
        (maybe_step,) = step_beam
        if maybe_step.err:
            self.all_steps.append(Fail(maybe_step.err))
            self.n_invalid_consecutive += 1
        else:
            assert maybe_step.step
            assert isinstance(maybe_step.step.env_output, EqGenFwdEnvOutput)

            # set this to be able to retrieve graph state before applying a tactic
            g = maybe_step.step.env_output.graph
            g.node_id_before_tac[len(self.steps) + 1] = len(g.nodes)

            self.steps.append(maybe_step.step)
            self.all_steps.append(Ok(maybe_step.step))
            self.n_invalid_consecutive = 0

            # associate any unknown node to the latest tactic
            this_tac = maybe_step.step.env_output.tactic
            for node in g.nodes:
                p = node.node.prefix()
                if p not in self.prefix_to_node:
                    self.prefix_to_node[p] = node
                if node.parent_tac is None:
                    node.parent_tac = this_tac

        if isinstance(maybe_step.err, (MaxLenReached, InputDicoError)):
            self.stopped = maybe_step.err
        elif self.next_graph.n_nodes >= self.cfg.max_nodes:
            self.stopped = MaxNodeReachedError(
                msg="max node", n_nodes=self.next_graph.n_nodes
            )
        elif len(self.all_steps) >= self.cfg.max_generations:
            self.stopped = MaxGenReachedError(
                msg="max trials", n_gen=len(self.all_steps)
            )
        elif self.n_invalid_consecutive >= self.cfg.max_cons_inv_allowed:
            self.stopped = MaxConsInvError(
                msg="max cons inv", n_cons_inv=self.n_invalid_consecutive
            )

    def should_stop(self) -> bool:
        return self.stopped is not None

    def finish(self):
        pass
