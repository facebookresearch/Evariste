# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Dict, Optional, Tuple, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.random import RandomState
from evariste.envs.eq.rules import RULE_VARS
from evariste.datasets.equations import EquationsDatasetConf
from evariste.envs.eq.generation import GraphHypNode, GraphNode

from evariste.envs.eq.graph import ONE, ZERO, CNode, Node, BNode, NodeSet, VNode
from evariste.envs.eq.rules import Rule, ARule, TRule
from evariste.envs.eq.rules_lib import LEAN__IDENTITY
from evariste.backward.env.equations.graph import (
    BWD_TOK,
    FWD_TOK,
    B_EQ_NODE_WORD,
    E_EQ_NODE_WORD,
    parse_node,
    parse_subs,
    prefix_pos_tok,
    prefix_var_name,
)
from evariste.datasets.equations import EquationsDatasetConf
from evariste.envs.eq.generation import GraphNode
from evariste.backward.env.equations.env import simplify_node


from evariste.envs.eq.graph import NodeParseError, EqInvalidVarType, Node
from evariste.envs.eq.rules import ARule, TRule
from evariste.envs.eq.rules_lib import NAME_TO_RULE

from evariste.forward.common import ForwardTactic
from evariste.forward.core.generation_errors import GenerationError, MalformedCommand
from evariste.forward.fwd_eq.gen.graph import EqGenForwardGraph, EqGenRuleEnv

from evariste.model.data.dictionary import B_SUBST_WORD, E_SUBST_WORD, M_SUBST_WORD


TTAC_TOK = "<TTAC>"
ATAC_TOK = "<ATAC>"
SUBST_TOK = "<SUBST_TAC>"
SIMP_NODE_TOK = "<SIMPNODE_TAC>"
NORMNUM_TOK = "<NORMNUM_TAC>"
HYP_MATCH_TOK = "<HYPMATCH_TAC>"
SIMP_HYP_TOK = "<SIMPHYP_TAC>"

EQ_FWD_TAC_TOKS = [
    TTAC_TOK,
    ATAC_TOK,
    SUBST_TOK,
    SIMP_NODE_TOK,
    NORMNUM_TOK,
    HYP_MATCH_TOK,
    SIMP_HYP_TOK,
]


class EqGenForwardTactic(ABC, ForwardTactic):
    @abstractmethod
    def apply(
        self, graph: EqGenForwardGraph, params: EquationsDatasetConf
    ) -> EqGenForwardGraph:
        pass

    @staticmethod
    @abstractmethod
    def sample(
        graph: EqGenForwardGraph, params: EquationsDatasetConf
    ) -> "EqGenForwardTactic":
        pass

    @abstractmethod
    def reset(self):
        pass

    @staticmethod
    def detokenize(
        command: List[str], graph: EqGenForwardGraph
    ) -> "EqGenForwardTactic":
        """
        Tokens into TTactic. Expected format:
        LABEL FWD PREFIX_POS [SUBSTS] [HYP_HOLE_TOK|NODES]        
        filled substs may be given partially
        Nodes are either given as prefix (for true nodes), or node ids.
        """

        if len(command) <= 3:
            raise MalformedCommand(f"Empty Tactic {' '.join(command)}")
        command = command[1:-1]  # remove eos
        tok_to_tac: Dict[str, Type[EqGenForwardTactic]] = {
            TTAC_TOK: EqGenForwardTTactic,
            ATAC_TOK: EqGenForwardATactic,
            SUBST_TOK: EqGenForwardSubstTactic,
            SIMP_NODE_TOK: EqGenForwardSimpTactic,
            NORMNUM_TOK: EqGenForwardNormNumTactic,
            HYP_MATCH_TOK: EqGenForwardHypMatchTactic,
            SIMP_HYP_TOK: EqGenForwardHypSimpTactic,
        }
        try:
            return tok_to_tac[command[0]].detokenize(command[1:], graph)
        except KeyError:
            raise MalformedCommand(
                f"Bad tok_to_tac {command[0]} // {' '.join(command)}"
            )

    @abstractmethod
    def tokenize(self, graph: EqGenForwardGraph) -> List[str]:
        pass


HYP_HOLE_TOK = "<HOLE_HYPS>"


def tokenize_substs(substs: Dict[str, Node]) -> List[str]:
    res = []
    for k in sorted(substs.keys()):
        res += [
            B_SUBST_WORD,
            prefix_var_name(k),
            M_SUBST_WORD,
            *substs[k].prefix_tokens(),
            E_SUBST_WORD,
        ]
    return res


def tokenize_children(graph: EqGenForwardGraph, children: List[Node],) -> List[str]:
    if len(children) == 0:
        return []
    nodes = []
    for c in children:
        nodes += [B_EQ_NODE_WORD, *c.prefix_tokens(), E_EQ_NODE_WORD]
    return nodes


@dataclass(eq=False)
class EqGenForwardTTactic(EqGenForwardTactic):
    rule: TRule
    fwd: bool
    prefix_pos: Optional[int] = None
    matched_substs: Optional[Dict[str, Node]] = None
    filled_substs: Optional[Dict[str, Node]] = None
    children: Optional[List[Node]] = None
    applied: bool = False

    def __eq__(self, other) -> bool:
        if not isinstance(other, EqGenForwardTTactic):
            return False
        if (self.rule.name, self.prefix_pos, self.fwd) != (
            other.rule.name,
            other.prefix_pos,
            other.fwd,
        ):
            return False
        both_none = self.filled_substs is None == other.filled_substs is None
        if not both_none:
            return False
        if self.filled_substs is not None and other.filled_substs is not None:
            if set(self.filled_substs.keys()) != set(other.filled_substs.keys()):
                return False
            all_eq = all(
                [
                    self.filled_substs[k].eq(other.filled_substs[k])
                    for k in self.filled_substs.keys()
                ]
            )
            if not all_eq:
                return False
        both_none = self.children is None == other.children is None
        if not both_none:
            return False
        if self.children is not None and other.children is not None:
            if len(self.children) != len(other.children):
                return False
            all_eq = all([x.eq(y) for x, y in zip(self.children, other.children)])
            if not all_eq:
                return False
        return True

    def reset(self):
        self.applied = False

    def tokenize(self, graph: EqGenForwardGraph) -> List[str]:
        assert self.prefix_pos is not None
        assert self.filled_substs is not None
        assert self.matched_substs is not None
        assert self.children is not None

        prefix = prefix_pos_tok(self.prefix_pos)
        res = [
            TTAC_TOK,
            self.rule.name,
            FWD_TOK if self.fwd else BWD_TOK,
            prefix,
            *tokenize_substs({**self.matched_substs, **self.filled_substs}),
            *tokenize_children(graph, self.children[:1]),
        ]
        return res

    @staticmethod
    def detokenize(
        command: List[str], graph: EqGenForwardGraph
    ) -> "EqGenForwardTactic":
        try:
            rule = NAME_TO_RULE[command[0]]
        except (IndexError, KeyError):
            raise MalformedCommand(f"Couldn't find TRule for {command}")
        if not isinstance(rule, TRule):
            raise MalformedCommand(f"{rule.name} not a trule {command}")
        if len(command) < 3:
            raise MalformedCommand(f"command too short for trule {command}")

        fwd = command[1] == FWD_TOK
        try:
            prefix_pos = int(command[2][len("<PREFIX_") : -len(">")])  # <PREFIX_{i}>
        except ValueError:
            raise MalformedCommand(
                f"Misunderstood token position {command[2]} in tactic {command}"
            )
        tok = 3

        try:
            substs, parsed = parse_subs(command[tok:])  # potentially partial
        except Exception as e:
            raise MalformedCommand(f"Couldn't parse subs // {e} // {tok}")
        if not set(substs.keys()) == rule.all_vars:
            raise MalformedCommand(
                f"Incorrect subs : {set(substs.keys())} // {rule.all_vars}"
            )

        try:
            children, parsed = parse_node(command[tok + parsed :])
        except (IndexError, NodeParseError, AssertionError):
            raise MalformedCommand(f"Couldn't parse src node: {command}")

        src_vars = rule.l_vars if fwd else rule.r_vars
        matched_substs = {
            x: y for x, y in substs.items() if x in (src_vars | rule.h_vars)
        }
        filled_substs = {x: y for x, y in substs.items() if x not in matched_substs}
        return EqGenForwardTTactic(
            rule,
            fwd,
            prefix_pos,
            matched_substs=matched_substs,
            filled_substs=filled_substs,
            children=[children],
        )

    def __repr__(self) -> str:
        children = (
            "\n".join([c.infix() for c in self.children])
            if self.children is not None
            else "none"
        )
        return f"{self.rule} // {self.fwd} // {self.prefix_pos} // {self.matched_substs} // {self.filled_substs} // {children}"

    @staticmethod
    def sample_rule(
        rule_env: EqGenRuleEnv, rng: RandomState, params: EquationsDatasetConf
    ) -> Tuple[TRule, bool]:
        """
        Sample a transformation rule. Optionally bias it towards rare rules.
        """
        if params.bias_rules == 0:
            rule = rule_env.rules_t[rng.randint(len(rule_env.rules_t))]
            fwd = rng.randint(2) == 0
            return rule, fwd
        rules_fwd = [(rule, fwd) for rule in rule_env.rules_t for fwd in [True, False]]
        p = np.array(
            [rule_env.rules_t_counts[(rule.name, fwd)] for rule, fwd in rules_fwd]
        )
        p = 1 / (np.maximum(p, 1).astype(np.float64) ** params.bias_rules)
        return rules_fwd[rng.choice(len(p), p=p / p.sum())]

    @staticmethod
    def sample(
        graph: EqGenForwardGraph,
        params: EquationsDatasetConf,
        src_node: Optional[Node] = None,
    ) -> "EqGenForwardTTactic":
        assert graph.rule_env is not None
        rule, fwd = EqGenForwardTTactic.sample_rule(graph.rule_env, graph.rng, params)
        res = EqGenForwardTTactic(
            rule=rule, fwd=fwd, children=None if src_node is None else [src_node]
        )
        res.apply(graph, params)  # will sample all missing params
        return res

    def apply(
        self, graph: EqGenForwardGraph, params: EquationsDatasetConf
    ) -> EqGenForwardGraph:
        if self.applied:
            return graph

        partial_to_fill: Dict[str, Node] = {}

        # No children. Find them.
        # TODO: what if matched_substs are given instead should this be supported ?
        src_node = self.rule.left if self.fwd else self.rule.right
        if self.children is None or self.prefix_pos is not None:
            substs: Dict[str, Optional[Node]] = {}
            if self.matched_substs is not None:
                substs.update(self.matched_substs)
            if self.filled_substs is not None:
                substs.update(self.filled_substs)
            src_in_graph = None if self.children is None else self.children[0]
            match = graph.search_matching_nodes(
                hyps=self.rule.hyps,
                src_node=src_node,
                bias_nodes=params.bias_nodes,
                prefix_pos=self.prefix_pos,
                substs=substs,
                src_in_graph=src_in_graph,
            )
            if match is None:
                raise GenerationError(f"No match in graph for TRule {self.rule.name}")
            children, matched_substs, self.prefix_pos = match
            if self.matched_substs is None:
                self.matched_substs = matched_substs
            self.children = [gn.node for gn in children]

            # sanity check
            assert self.matched_substs is not None
            assert self.prefix_pos is not None
            assert len(self.children) == 1 + len(
                self.rule.hyps
            )  # src node in first position
        # Rwalk
        elif len(self.children) == 1 and self.prefix_pos is None:
            valid_nodes = self.rule.eligible(self.children[0], self.fwd)
            if len(valid_nodes) == 0:
                raise GenerationError("graph gen, no valid nodes for rule")
            self.prefix_pos, _ = valid_nodes[graph.rng.randint(len(valid_nodes))]

        if not self.rule.arithmetic and self.matched_substs is not None:
            # source / target variables
            src_vars = self.rule.l_vars if self.fwd else self.rule.r_vars
            tgt_vars = self.rule.r_vars if self.fwd else self.rule.l_vars
            assert self.matched_substs.keys() == (src_vars | self.rule.h_vars), (
                set(self.matched_substs.keys()),
                (src_vars | self.rule.h_vars),
            )
            partial_to_fill.update(
                {
                    k: v
                    for k, v in self.matched_substs.items()
                    if k in (tgt_vars | self.rule.h_vars) - src_vars
                }
            )
            # partial_to_fill will often be the full fill, except for some rules
            # such as mul_eq_zero_of_right where a variable is exclusively in the target

        if self.filled_substs is not None:
            # if self.matched_substs is not None:
            #     partial_to_fill.update(self.matched_substs)
            partial_to_fill.update(self.filled_substs)

        assert self.prefix_pos is not None
        src_node = self.children[0]
        assert graph.env is not None
        try:
            applied = graph.env.apply_t_rule(
                eq=src_node,
                rule=self.rule,
                fwd=self.fwd,
                prefix_pos=self.prefix_pos,
                to_fill=partial_to_fill,
            )
        except AssertionError as e:
            raise GenerationError(f"apply_t_rule failed {type(e)}:{e}")

        if self.filled_substs is None:
            # holds all generated expressions during apply_t_rule
            self.filled_substs = {
                k: v
                for k, v in applied["to_fill"].items()
                if partial_to_fill is None or k not in partial_to_fill
            }

        res = graph.add_t_node(
            applied["eq"],
            src_node=src_node,
            hyps=applied["hyps"],
            rule=self.rule,
            substs={**applied["match"], **applied["to_fill"]},
            fwd=not self.fwd,
            prefix_pos=self.prefix_pos,
        )
        self.applied = True
        return res


@dataclass(eq=False)
class EqGenForwardATactic(EqGenForwardTactic):
    rule: ARule
    matched_substs: Optional[Dict[str, Node]] = None
    filled_substs: Optional[Dict[str, Node]] = None
    children: Optional[List[Node]] = None
    applied: bool = False

    def __eq__(self, other) -> bool:
        if not isinstance(other, EqGenForwardTTactic):
            return False
        if self.rule.name != other.rule.name:
            return False
        both_none = self.filled_substs is None == other.filled_substs is None
        if not both_none:
            return False
        if self.filled_substs is not None and other.filled_substs is not None:
            if set(self.filled_substs.keys()) != set(other.filled_substs.keys()):
                return False
            all_eq = all(
                [
                    self.filled_substs[k].eq(other.filled_substs[k])
                    for k in self.filled_substs.keys()
                ]
            )
            if not all_eq:
                return False
        both_none = self.children is None == other.children is None
        if not both_none:
            return False
        if self.children is not None and other.children is not None:
            if len(self.children) != len(other.children):
                return False
            all_eq = all([x.eq(y) for x, y in zip(self.children, other.children)])
            if not all_eq:
                return False
        return True

    def reset(self):
        self.applied = False

    def tokenize(self, graph: EqGenForwardGraph) -> List[str]:
        assert self.filled_substs is not None
        assert self.matched_substs is not None
        assert self.children is not None
        return [
            ATAC_TOK,
            self.rule.name,
            *tokenize_substs({**self.matched_substs, **self.filled_substs}),
        ]

    @staticmethod
    def detokenize(
        command: List[str], graph: EqGenForwardGraph
    ) -> "EqGenForwardTactic":
        try:
            rule = NAME_TO_RULE[command[0]]
        except (IndexError, KeyError):
            raise MalformedCommand(f"Couldn't find ARule for {command}")
        if not isinstance(rule, ARule):
            raise MalformedCommand(f"{rule.name} not a ARule {command}")
        try:
            substs, _parsed = parse_subs(command[1:])  # potentially partial
        except Exception as e:
            raise MalformedCommand(f"Couldn't parse subs // {e}")
        if not set(substs.keys()) == rule.all_vars:
            raise MalformedCommand(
                f"Incorrect subs : {set(substs.keys())} // {rule.all_vars}"
            )

        matched_substs = {x: y for x, y in substs.items() if x in rule.h_vars}
        filled_substs = {x: y for x, y in substs.items() if x not in matched_substs}
        return EqGenForwardATactic(
            rule, matched_substs=matched_substs, filled_substs=filled_substs,
        )

    def __repr__(self) -> str:
        children = (
            "\n".join([c.infix() for c in self.children])
            if self.children is not None
            else "none"
        )
        return f"{self.rule} // {self.matched_substs} // {self.filled_substs} // {children}"

    @staticmethod
    def sample_rule(graph: EqGenForwardGraph, params: EquationsDatasetConf) -> ARule:
        assert graph.rule_env is not None
        rules = graph.rule_env.rules_a
        if graph.n_true_nodes >= params.max_true_nodes:
            rules = graph.rule_env.rules_a_hyps
        if params.bias_rules == 0:
            return rules[graph.rng.randint(len(rules))]
        p_l = [graph.rule_env.rules_a_counts[rule.name] for rule in rules]
        p = 1 / (np.maximum(p_l, 1).astype(np.float64) ** params.bias_rules)
        return rules[graph.rng.choice(len(p), p=p / p.sum())]

    @staticmethod
    def sample(
        graph: EqGenForwardGraph, params: EquationsDatasetConf
    ) -> "EqGenForwardATactic":
        rule = EqGenForwardATactic.sample_rule(graph, params)
        res = EqGenForwardATactic(rule=rule)
        res.apply(graph, params)  # will sample all missing params
        return res

    def apply(
        self, graph: EqGenForwardGraph, params: EquationsDatasetConf
    ) -> EqGenForwardGraph:
        if self.applied:
            return graph

        # find subset of nodes that match the rule hypotheses
        if len(self.rule.hyps) == 0:
            if self.children is not None and len(self.children) > 0:
                raise GenerationError("non empty children for ARule with no hyps")
            if self.matched_substs is not None and len(self.matched_substs) > 0:
                raise GenerationError("non empty substs for ARule with no hyps")
            self.children = []
            self.matched_substs = {}
        else:
            if self.children is None:
                substs: Dict[str, Optional[Node]] = {}
                if self.matched_substs is not None:
                    substs.update(self.matched_substs)
                if self.filled_substs is not None:
                    substs.update(self.filled_substs)
                match = graph.search_matching_nodes(
                    hyps=self.rule.hyps, bias_nodes=params.bias_nodes, substs=substs
                )
                if match is None:
                    raise GenerationError(
                        f"No match in graph for arule {self.rule.name}"
                    )
                children, matched_substs, _ = match
                self.children = [gn.node for gn in children]
                if self.matched_substs is None:
                    self.matched_substs = matched_substs

        # sanity check
        if len(self.children) != len(self.rule.hyps):
            raise GenerationError("mismatched children / rule hyps")

        assert self.matched_substs is not None
        # set missing variables and create new statement
        if self.filled_substs is None:
            self.filled_substs = {}
        assert graph.env is not None
        for k in sorted(self.rule.n_vars):
            if k in self.matched_substs or k in self.filled_substs:
                continue
            n_ops = graph.rng.randint(graph.env.fill_max_ops + 1)
            self.filled_substs[k] = graph.env.generate_expr(n_ops)

        to_fill = {**self.matched_substs, **self.filled_substs}
        try:
            eq = self.rule.node.set_vars({k: to_fill[k] for k in self.rule.n_vars})
        except (AssertionError, EqInvalidVarType) as e:
            raise GenerationError(f"a_rule setvars failed {type(e)}:{e}")

        res = graph.add_a_node(eq, self.children, self.rule, to_fill,)
        self.applied = True
        return res


@dataclass
class EqGenForwardSubstTactic(EqGenForwardTactic):
    substs: Dict[str, Node]
    mapping: Optional[Dict[str, GraphNode]] = None

    def reset(self):
        # can't use self.applied to avoid re-applying since the graph isn't modified but cloned
        pass

    def tokenize(self, graph: EqGenForwardGraph) -> List[str]:
        return [SUBST_TOK, *tokenize_substs(self.substs)]

    @staticmethod
    def detokenize(
        command: List[str], graph: EqGenForwardGraph
    ) -> "EqGenForwardTactic":
        try:
            substs, _parsed = parse_subs(command)
        except Exception as e:
            raise MalformedCommand(f"Couldn't parse subs // {e}")
        return EqGenForwardSubstTactic(substs=substs)

    @staticmethod
    def sample_substs(
        expr: Node, rng: RandomState, params: EquationsDatasetConf
    ) -> Dict[str, Node]:
        # TODO, include in params
        inv_op = {"add": "sub", "sub": "add", "mul": "div", "div": "mul"}
        target_op = set(inv_op.keys())
        TARGETS = [ONE, ZERO]

        def _eligible_simp(expr: Node, res: List[Node]):
            # TODO improve that, in some cases we could want to include integers also, but adapt later then
            if (
                expr.value in target_op
                and expr.children[0].is_var()
                and expr.children[1].is_var()
            ):
                res.append(expr)
            for c in expr.children:
                _eligible_simp(c, res)

        eli_nodes: List[Node] = []
        _eligible_simp(expr, eli_nodes)
        if len(eli_nodes) == 0:
            raise GenerationError("no small simplification found in this node")
        to_simp = eli_nodes[rng.randint(len(eli_nodes))]
        ltg = rng.randint(2)  # 0 or 1
        child_to_simp = to_simp.children[ltg]
        target = TARGETS[rng.randint(len(TARGETS))]
        simp_node = BNode(inv_op[to_simp.value], target, to_simp.children[1 - ltg])
        return {child_to_simp.infix(): simp_node}

    @staticmethod
    def sample(
        graph: EqGenForwardGraph, params: EquationsDatasetConf
    ) -> "EqGenForwardSubstTactic":
        nodes_with_hyps = [
            node
            for node in graph.nodes
            if node.ntype not in {"true", "hyp"} and not node.is_true
        ]
        if params.bias_small_simp == 0:
            if len(nodes_with_hyps) > 0:
                node = nodes_with_hyps[graph.rng.randint(len(nodes_with_hyps))].node
            else:
                node = graph.nodes[graph.rng.randint(len(graph.nodes))].node
        else:
            raise NotImplementedError("Not yet implemented")
        substs = EqGenForwardSubstTactic.sample_substs(node, graph.rng, params)
        t = EqGenForwardSubstTactic(substs=substs)
        t.apply(graph, params)
        return t

    def apply(
        self, graph: EqGenForwardGraph, params: EquationsDatasetConf
    ) -> EqGenForwardGraph:
        if graph.has_simps:
            raise GenerationError("Tried to subst after simp")
        new_graph, mapping = graph.clone(self.substs)
        self.mapping = mapping
        return new_graph


@dataclass
class EqGenForwardSimpTactic(EqGenForwardTactic):
    """
    Apply a simplification to a node and add the simplified node(s) to the graph.
    Simplification involves all lean rules included in "simp"
    as well as a rewriting with specified simplifying_hyps.
    Simp steps are condensed into a single step and correspond to a GraphSimpNode,
    while rewrites correspond to a GraphTransformNode.
    This tactic can add several nodes to the graph if both are involved
    e.g. TODO give example.
    """

    node: GraphNode
    applied: bool = False

    def reset(self):
        self.applied = False

    def tokenize(self, graph: EqGenForwardGraph) -> List[str]:
        return [SIMP_NODE_TOK, *tokenize_children(graph, [self.node.node])]

    @staticmethod
    def detokenize(
        command: List[str], graph: EqGenForwardGraph
    ) -> "EqGenForwardTactic":
        try:
            src_node, _parsed = parse_node(command)
        except (IndexError, NodeParseError, AssertionError):
            raise MalformedCommand(f"Couldn't parse src node: {command}")
        prefix = src_node.prefix()
        if prefix not in graph.prefix2id:
            raise MalformedCommand(f"Simp source node not in graph {command}")
        return EqGenForwardSimpTactic(node=graph.nodes[graph.prefix2id[prefix]])

    @staticmethod
    def sample(
        graph: EqGenForwardGraph, params: EquationsDatasetConf
    ) -> "EqGenForwardSimpTactic":
        if params.bias_node_simp == 0:
            non_true_nodes = [node for node in graph.nodes if not node.is_true]
            if len(non_true_nodes) == 0:
                raise GenerationError("Nothing to simp")
            node_id = graph.rng.randint(len(non_true_nodes))
            node = non_true_nodes[node_id]
        else:
            raise NotImplementedError("Not yet implemented")
        t = EqGenForwardSimpTactic(node)
        t.apply(graph, params)
        return t

    def apply(
        self, graph: EqGenForwardGraph, params: EquationsDatasetConf
    ) -> EqGenForwardGraph:
        if self.applied:
            return graph
        assert graph.env is not None and graph.rule_env is not None
        last_node, rules, hyps = simplify_node(
            self.node.node,
            graph.env,
            graph.rule_env.rules_a,
            graph.rule_env.simp_rules,
            NodeSet([n.node for n in graph.nodes]),
        )
        if len(rules) == 0:
            raise GenerationError("Couldn't find any simp rule to apply")
        # parent node could be simp node if there was a subst in between. TODO: merge ?
        self.applied = True
        return graph.add_simp_node(last_node, self.node, rules, hyps=list(hyps))


@dataclass
class EqGenForwardNormNumTactic(EqGenForwardTactic):
    """
    Apply a numerical simplification to a node, similar to the tactic "norm_num" in Lean.
    If simplification works, add a GraphNormNumNode to the graph.
    e.g. (3+2)*x == 5*2+3*y -> 5*x == 10+3*y
    see also _norm_numify in evariste/envs/eq/graph.py
    """

    node: GraphNode
    applied: bool = False

    def reset(self):
        self.applied = False

    def tokenize(self, graph: EqGenForwardGraph) -> List[str]:
        return [NORMNUM_TOK, *tokenize_children(graph, [self.node.node])]

    @staticmethod
    def detokenize(
        command: List[str], graph: EqGenForwardGraph
    ) -> "EqGenForwardTactic":
        try:
            src_node, _parsed = parse_node(command)
        except (IndexError, NodeParseError, AssertionError):
            raise MalformedCommand(f"Couldn't parse src node: {command}")
        prefix = src_node.prefix()
        if prefix not in graph.prefix2id:
            raise MalformedCommand(f"Norm num source node not in graph {command}")
        return EqGenForwardNormNumTactic(node=graph.nodes[graph.prefix2id[prefix]])

    @staticmethod
    def sample(
        graph: EqGenForwardGraph, params: EquationsDatasetConf
    ) -> "EqGenForwardNormNumTactic":
        if params.bias_norm_num == 0:
            node = graph.nodes[graph.rng.randint(len(graph.nodes))]
        else:
            raise NotImplementedError("Not yet implemented")
        t = EqGenForwardNormNumTactic(node=node)
        t.apply(graph, params)
        return t

    def apply(
        self, graph: EqGenForwardGraph, params: EquationsDatasetConf
    ) -> EqGenForwardGraph:
        if self.applied:
            return graph
        assert graph.env is not None
        new_node = self.node
        ok = False
        new_children = []
        for i, expr in enumerate(self.node.node.children):
            expr1 = expr._norm_numify(graph.env.vtype)
            new_children.append(expr1)
            if expr1.eq(self.node.node.children[i]):
                continue
            ok = True
        if not ok:
            raise GenerationError("Nothing to normnumify")
        new_expr = CNode(new_node.node.str_value, *new_children)
        self.applied = True
        return graph.add_nn_node(new_expr, self.node)


@dataclass
class EqGenForwardHypMatchTactic(EqGenForwardTactic):
    """
    Select an hypothesis and, when possible,
    create a substitution of the variables
    such that this hypothesis is true.
    Fails if no such substitution is possible.
    Output a clone of the input graph with the substitution
    and the hypothesis has been removed (not stored in the
    graph currently as clone does not keep true nodes).
    """

    node: GraphHypNode
    rule: ARule
    match: Dict[str, Node]
    applied: bool = False
    mapping: Optional[Dict[str, GraphNode]] = None

    def reset(self):
        self.applied = False

    def tokenize(self, graph: EqGenForwardGraph) -> List[str]:
        return [
            HYP_MATCH_TOK,
            self.rule.name,
            *tokenize_substs(self.match),
            *tokenize_children(graph, [self.node.node]),
        ]

    @staticmethod
    def detokenize(
        command: List[str], graph: EqGenForwardGraph
    ) -> "EqGenForwardTactic":
        try:
            rule = NAME_TO_RULE[command[0]]
        except (IndexError, KeyError):
            raise MalformedCommand(f"Couldn't find ARule for {command}")
        if not isinstance(rule, ARule):
            raise MalformedCommand(f"Label {rule.name} not ARule {command}")
        try:
            match, parsed = parse_subs(command[1:])
        except Exception as e:
            raise MalformedCommand(f"Couldn't parse subs // {e}")
        try:
            src_node, _parsed = parse_node(command[1 + parsed :])
        except (IndexError, NodeParseError, AssertionError):
            raise MalformedCommand(f"Couldn't parse src node: {command}")

        prefix = src_node.prefix()
        if prefix not in graph.prefix2id:
            raise MalformedCommand(f"Simp source node not in graph {command}")
        node = graph.nodes[graph.prefix2id[prefix]]
        if not isinstance(node, GraphHypNode):
            raise MalformedCommand(f"Target node is not Hyp {command}")
        return EqGenForwardHypMatchTactic(node, rule, match)

    @staticmethod
    def sample(
        graph: EqGenForwardGraph, params: EquationsDatasetConf
    ) -> "EqGenForwardHypMatchTactic":
        assert graph.rule_env is not None
        hyp_nodes = [node for node in graph.nodes if node.ntype == "hyp"]
        if len(hyp_nodes) == 0:
            raise GenerationError("Can't hypmatch when there are no hyps")
        if params.bias_hyp_match == 0:
            node_order = graph.rng.permutation(len(hyp_nodes))
        else:
            raise NotImplementedError("Not implemented yet")
        if params.bias_hyp_match_rule == 0:
            rule_order = graph.rng.permutation(len(graph.rule_env.rules_a_no_hyps))
        else:
            raise NotImplementedError("Not implemented yet")
        res = None
        node, rule = None, None  # could be unbound otherwise
        for n_index in node_order:
            node = hyp_nodes[n_index]
            for index in rule_order:
                rule = graph.rule_env.rules_a_no_hyps[index]
                res = node.node.match(
                    rule.node, NodeSet([VNode(var) for var in node.node.get_vars()])
                )
                if res is not None:
                    break
            if res is not None:
                break
        if res is None:
            raise GenerationError("No hyp match available")
        assert isinstance(node, GraphHypNode)
        assert isinstance(rule, ARule)
        t = EqGenForwardHypMatchTactic(node=node, rule=rule, match=res)
        t.apply(graph, params)
        return t

    def apply(
        self, graph: EqGenForwardGraph, params: EquationsDatasetConf
    ) -> EqGenForwardGraph:
        if self.applied:
            return graph
        assert graph.env is not None
        # new_graph = graph.clone(self.match)
        # TODO do better than this double clone to get rid of the RULES_VAR once matched.
        # Note: clone relies on set_vars that will not work if it has a substs of the form {x0 : f(x0)}.
        all_vars = set()
        for expr in self.match.values():
            all_vars |= expr.get_vars()
        new_substs = {
            var: graph.env.generate_expr(3, avoid=set(self.match.keys()))
            for var in all_vars
        }
        final_substs = {
            x: y.set_vars(new_substs, mandatory=False) for x, y in self.match.items()
        }
        assert self.node.node.prefix() in graph.prefix2id, self.node.node.infix()
        to_ret, mapping = graph.clone(final_substs)
        self.mapping = mapping
        to_ret.make_true(self.node.node.set_vars(final_substs), self.rule, new_substs)
        self.applied = True
        return to_ret


@dataclass
class EqGenForwardHypSimpTactic(EqGenForwardTactic):
    node: GraphHypNode
    applied: bool = False

    def reset(self):
        self.applied = False

    def tokenize(self, graph: EqGenForwardGraph) -> List[str]:
        return [SIMP_HYP_TOK, *tokenize_children(graph, [self.node.node])]

    @staticmethod
    def detokenize(
        command: List[str], graph: EqGenForwardGraph
    ) -> "EqGenForwardTactic":
        try:
            src_node, _parsed = parse_node(command)
        except (IndexError, NodeParseError, AssertionError):
            raise MalformedCommand(f"Couldn't parse src node: {command}")
        prefix = src_node.prefix()
        if prefix not in graph.prefix2id:
            raise MalformedCommand(f"Simp source node not in graph {command}")
        node = graph.nodes[graph.prefix2id[prefix]]
        if not isinstance(node, GraphHypNode):
            raise MalformedCommand(f"Target node is not Hyp {command}")
        return EqGenForwardHypSimpTactic(node)

    @staticmethod
    def sample(
        graph: EqGenForwardGraph, params: EquationsDatasetConf
    ) -> "EqGenForwardHypSimpTactic":
        hyp_nodes = [node for node in graph.nodes if node.ntype == "hyp"]
        if len(hyp_nodes) == 0:
            raise GenerationError("Can't simp hypothesis when there are none")
        if params.bias_hyp_simp == 0:
            node = hyp_nodes[graph.rng.randint(len(hyp_nodes))]
        else:
            raise NotImplementedError("Not implemented yet")
        assert isinstance(node, GraphHypNode)
        t = EqGenForwardHypSimpTactic(node=node)
        t.apply(graph, params)
        return t

    def apply(
        self, graph: EqGenForwardGraph, params: EquationsDatasetConf
    ) -> EqGenForwardGraph:
        if self.applied:
            return graph
        assert graph.env is not None and graph.rule_env is not None
        last_node, rules, hyps = simplify_node(
            self.node.node,
            graph.env,
            graph.rule_env.rules_a,
            graph.rule_env.simp_rules,
            NodeSet([n.node for n in graph.nodes]),
        )
        if len(rules) == 0:
            raise GenerationError("Couldn't find any simp rule to apply")
        # HypNode -> SimpNode -> current dependents
        # NewHyp  ->  OldHyp  -> ...
        rules = rules[::-1]
        new_graph = graph.replace_hyp_with_simp(
            self.node.node, last_node, list(hyps), rules
        )
        self.applied = True
        return new_graph
