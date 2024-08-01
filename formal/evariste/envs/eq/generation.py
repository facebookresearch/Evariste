# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from functools import cached_property
import logging
from typing import Optional, Union, Tuple, List, Set, Dict, Any
from abc import ABC, abstractmethod
import itertools
import numpy as np

from evariste.envs.eq.graph import C_OPS, Node, NodeSet, CNode, RULE_VARS
from evariste.envs.eq.rules import (
    TRule,
    ARule,
    Rule,
    eval_numeric,
    eval_assert,
    eval_assert_with_rule,
)
from evariste.envs.eq.rules_lib import ALL_RULES
from evariste.backward.env.equations.graph import (
    EQTactic,
    EQTheorem,
    EQRuleTactic,
    EQSimpTactic,
    EQNormNumTactic,
)
from evariste.envs.eq.env import EquationEnv, EquationsEnvArgs, EqGraphSamplerParams


class GraphNode(ABC):
    def __init__(self, node: Node, node_type: str, is_true: bool):
        assert node_type in ["transform", "assert", "hyp", "true", "simp", "norm_num"]
        self.node = node
        self.ntype = node_type

        self.node_id: Optional[int] = None  # set by add_node
        self.parent: Optional[GraphNode] = None  # != None => parent_tac is subst
        # TODO: Any -> EqGenForwardTac, fix import cycles...
        self.parent_tac: Optional[Any] = None  # set by proof search

        self.is_true = is_true

        # cache matches with rule patterns
        self._all_matches: Dict[str, List[Tuple[int, Dict[str, Node]]]] = {}
        self._rule_counts: Optional[Dict[Union[str, Tuple[str, bool]], int]] = None
        # used for lean generation
        self.subgoal_counts: Optional[Dict[str, int]] = None

        self._depth: Optional[int] = None
        self._descendants: Optional[Set[Tuple[str, int]]] = None
        self._used_hyps: Optional[NodeSet] = None

    @property
    @abstractmethod
    def depth(self) -> int:
        pass

    @property
    @abstractmethod
    def descendants(self) -> Set[Tuple[str, Optional[int]]]:
        """
        Set of prefix expressions of all nodes descending from this one,
        including itself, with their corresponding number of children.
        Used to compute the actual proof size of a node, and the average
        number of children per node in a proof.
        """
        pass

    @property
    def size(self):
        return len(self.descendants)

    @property
    def size_by_depth(self) -> float:
        return 0.0 if self.depth == 0 else (self.size / self.depth)

    @property
    def stats(self) -> Dict[str, Union[int, float]]:
        return {
            "depth": self.depth,
            "size": self.size,
            "size_by_depth": self.size_by_depth,
            "statement_n_tok": self.node.prefix_len(),
        }

    @property
    def avg_children_per_node(self):
        n_children = [n for _, n in self.descendants if n is not None]
        return sum(n_children) / max(len(n_children), 1)

    @property
    @abstractmethod
    def used_hyps(self) -> NodeSet:
        """Return hypotheses used to prove current node."""
        pass

    def get_goal_tactics(
        self,
        goals: List["GraphNode"],
        tactics: List[EQTactic],
        children: List[List[Node]],
        used_hyps: List[Node],
        explored: Set[str],
    ):
        pass

    @abstractmethod
    def substitute(
        self, new_expr: Node, substs: Dict[str, Node], hyps: List["GraphNode"]
    ) -> "GraphNode":
        pass

    @abstractmethod
    def clone(self, hyps: List["GraphNode"]) -> "GraphNode":
        pass


class GraphInternalNode(GraphNode):
    def __init__(
        self,
        node: Node,
        node_type: str,
        hyps: List["GraphNode"],
        substs: Dict[str, Node],
        is_true: bool,
    ):
        super().__init__(node=node, node_type=node_type, is_true=is_true)

        self._hyps = hyps
        self.substs = substs

    @property
    def hyps(self):
        # prevent modification of hyps.
        # this prevents x.hyps = ... or x.hyps[0] = ...
        return tuple(self._hyps)

    @property
    def depth(self) -> int:
        if self._depth is None:
            if len(self.hyps) == 0:
                self._depth = 1
            else:
                self._depth = 1 + max(hyp.depth for hyp in self.hyps)
        return self._depth

    @property
    def descendants(self) -> Set[Tuple[str, Optional[int]]]:
        if self._descendants is None:
            self._descendants: Set[Tuple[str, Optional[int]]] = set()
            self._descendants.add((self.node.prefix(), len(self.hyps)))
            for hyp in self.hyps:
                self._descendants |= hyp.descendants
        return self._descendants

    @property
    def used_hyps(self) -> NodeSet:
        # TODO: fix all asserts here ? no idea why mypy complains :(
        if self._used_hyps is None:
            self._used_hyps = NodeSet()
            for hyp in self.hyps:
                assert self._used_hyps is not None
                self._used_hyps |= hyp.used_hyps
        assert self._used_hyps is not None
        return self._used_hyps

    def get_tactic(self) -> EQTactic:
        raise NotImplementedError

    def get_goal_tactics(
        self,
        goals: List[GraphNode],
        tactics: List[EQTactic],
        children: List[List[Node]],
        used_hyps: List[Node],
        explored: Set[str],
    ):
        prefix = self.node.prefix()
        if prefix in explored:
            return goals, tactics, children, used_hyps, explored
        for hyp in self.hyps:
            hyp.get_goal_tactics(goals, tactics, children, used_hyps, explored)
        goals.append(self)
        tactics.append(self.get_tactic())
        children.append([hyp.node for hyp in self.hyps])
        assert prefix not in explored
        explored.add(prefix)


class GraphRuleNode(GraphInternalNode):
    def __init__(
        self,
        node: Node,
        node_type: str,
        hyps: List["GraphNode"],
        rule: Rule,
        substs: Dict[str, Node],
        is_true: bool,
    ):
        super().__init__(
            node=node, node_type=node_type, hyps=hyps, substs=substs, is_true=is_true
        )
        self.rule = rule


class GraphTerminalNode(GraphNode):
    @property
    def depth(self) -> int:
        return 0

    @property
    def descendants(self) -> Set[Tuple[str, Optional[int]]]:
        return set()

    @property
    def used_hyps(self) -> NodeSet:
        return NodeSet()


class GraphTransformNode(GraphRuleNode):
    def __init__(
        self,
        node: Node,
        hyps: List["GraphNode"],
        rule: TRule,
        fwd: bool,
        prefix_pos: int,
        substs: Dict[str, Node],
        is_true: bool,
        missing_true: bool = False,
    ):
        """
        The first child is the source node (left if fwd, right otherwise).
        Applying the rule to node in the given direction should lead to hypotheses.
        """
        assert missing_true or len(hyps) == 1 + len(rule.hyps)
        assert substs.keys() == rule.all_vars
        assert missing_true or len(rule.eligible(node, fwd)) > 0
        assert missing_true or len(rule.eligible(hyps[0].node, not fwd)) > 0

        super().__init__(
            node=node,
            node_type="transform",
            hyps=hyps,
            rule=rule,
            substs=substs,
            is_true=is_true,
        )
        self.fwd = fwd
        self.prefix_pos = prefix_pos
        self.missing_true = missing_true

    def substitute(
        self, new_expr: Node, substs: Dict[str, Node], hyps: List[GraphNode]
    ) -> GraphNode:
        n_substs = {
            k: v.set_vars(subst=substs, mandatory=False) for k, v in self.substs.items()
        }
        offset = 0
        prefix_seq = self.node.prefix_tokens()[: self.prefix_pos]
        for var, var_value in substs.items():
            delta = (len(var_value.prefix_tokens()) - 1) * prefix_seq.count(var)
            offset += delta
        n_prefix_pos = self.prefix_pos + offset
        assert isinstance(self.rule, TRule)
        return GraphTransformNode(
            node=new_expr,
            hyps=hyps,
            rule=self.rule,
            substs=n_substs,
            fwd=self.fwd,
            prefix_pos=n_prefix_pos,
            is_true=False,
        )

    def clone(self, hyps: List[GraphNode]) -> GraphNode:
        assert isinstance(self.rule, TRule)
        return GraphTransformNode(
            node=self.node,
            hyps=hyps,
            rule=self.rule,
            substs=self.substs,
            fwd=self.fwd,
            prefix_pos=self.prefix_pos,
            is_true=self.is_true,
        )

    def get_tactic(self) -> EQTactic:
        assert isinstance(self.rule, TRule)
        src_vars = self.rule.l_vars if self.fwd else self.rule.r_vars
        return EQRuleTactic(
            label=self.rule.name,
            fwd=self.fwd,
            prefix_pos=self.prefix_pos,
            to_fill={k: v for k, v in self.substs.items() if k not in src_vars},
        )


class GraphAssertNode(GraphRuleNode):
    def __init__(
        self,
        node: Node,
        hyps: List["GraphNode"],
        rule: ARule,
        substs: Dict[str, Node],
        is_true: bool,
        missing_true: bool = False,
    ):
        assert len(hyps) == len(rule.hyps) or missing_true
        assert substs.keys() == rule.all_vars, (
            list(substs.keys()),
            list(rule.all_vars),
        )
        assert rule.node.match(node) is not None

        super().__init__(
            node=node,
            node_type="assert",
            hyps=hyps,
            rule=rule,
            substs=substs,
            is_true=is_true,
        )

    def get_tactic(self) -> EQTactic:
        assert isinstance(self.rule, ARule)
        return EQRuleTactic(
            label=self.rule.name,
            fwd=None,
            prefix_pos=None,
            to_fill={k: v for k, v in self.substs.items() if k not in self.rule.n_vars},
        )

    def substitute(
        self, new_expr: Node, substs: Dict[str, Node], hyps: List[GraphNode]
    ) -> GraphNode:
        n_substs = {
            k: v.set_vars(subst=substs, mandatory=False) for k, v in self.substs.items()
        }
        assert isinstance(self.rule, ARule)
        return GraphAssertNode(
            node=new_expr, hyps=hyps, rule=self.rule, substs=n_substs, is_true=False
        )

    def clone(self, hyps: List[GraphNode]) -> GraphNode:
        assert isinstance(self.rule, ARule)
        return GraphAssertNode(
            node=self.node,
            hyps=hyps,
            rule=self.rule,
            substs=self.substs,
            is_true=self.is_true,
        )


class GraphNormNumNode(GraphInternalNode):
    def __init__(self, node: Node, hyps: List["GraphNode"], is_true: bool):
        # TODO: check that this is not an issue to add substs = {}, this was None in earlier version
        super().__init__(
            node=node, node_type="norm_num", hyps=hyps, substs={}, is_true=is_true
        )

    def substitute(
        self, new_expr: Node, substs: Dict[str, Node], hyps: List[GraphNode]
    ) -> GraphNode:
        return GraphNormNumNode(node=new_expr, hyps=hyps, is_true=False)

    def clone(self, hyps: List[GraphNode]) -> GraphNode:
        return GraphNormNumNode(node=self.node, hyps=hyps, is_true=self.is_true)

    def get_tactic(self) -> EQTactic:
        return EQNormNumTactic(self.hyps[0].node)


class GraphSimpNode(GraphInternalNode):
    def __init__(
        self, node: Node, hyps: List["GraphNode"], rules: List[TRule], is_true: bool
    ):
        self.rules = rules
        super().__init__(
            node=node, node_type="simp", hyps=hyps, substs={}, is_true=is_true
        )

    def substitute(
        self, new_expr: Node, substs: Dict[str, Node], hyps: List[GraphNode]
    ) -> GraphNode:
        return GraphSimpNode(node=new_expr, hyps=hyps, rules=self.rules, is_true=False)

    def clone(self, hyps: List[GraphNode]) -> GraphNode:
        return GraphSimpNode(
            node=self.node, hyps=hyps, rules=self.rules, is_true=self.is_true
        )

    def get_tactic(self) -> EQTactic:
        return EQSimpTactic(self.hyps[0].node, hyps=[h.node for h in self.hyps[1:]])


class GraphHypNode(GraphTerminalNode):
    def __init__(
        self, node: Node,
    ):
        super().__init__(node=node, node_type="hyp", is_true=False)

    def get_goal_tactics(
        self,
        goals: List[GraphNode],
        tactics: List[EQTactic],
        children: List[List[Node]],
        used_hyps: List[Node],
        explored: Set[str],
    ):
        prefix = self.node.prefix()
        if prefix in explored:
            return goals, tactics, children, used_hyps, explored
        explored.add(prefix)
        used_hyps.append(self.node)

    def substitute(
        self, new_expr: Node, substs: Dict[str, Node], hyps: List[GraphNode]
    ) -> GraphNode:
        return GraphHypNode(new_expr)

    def clone(self, hyps: List[GraphNode]) -> GraphNode:
        return GraphHypNode(node=self.node)


class GraphTrueNode(GraphTerminalNode):
    def __init__(
        self, node: Node,
    ):
        super().__init__(node=node, node_type="true", is_true=True)

    def get_goal_tactics(
        self,
        goals: List[GraphNode],
        tactics: List[EQTactic],
        children: List[List[Node]],
        used_hyps: List[Node],
        explored: Set[str],
    ) -> Tuple[List[GraphNode], List[EQTactic], List[List[Node]], List[Node], Set[str]]:
        return goals, tactics, children, used_hyps, explored

    def substitute(
        self, new_expr: Node, substs: Dict[str, Node], hyps: List[GraphNode]
    ) -> GraphNode:
        return GraphTrueNode(new_expr)

    def clone(self, hyps: List[GraphNode]) -> GraphNode:
        return GraphTrueNode(node=self.node)


class EquationGraphStats:
    def __init__(self, nodes: List[GraphNode], rules: List[Rule]):
        self.nodes = nodes
        self.rules_t = [r for r in rules if isinstance(r, TRule)]
        self.rules_a = [r for r in rules if isinstance(r, ARule)]
        self.rules_t_counts = {
            (rule.name, fwd): 0 for rule in self.rules_t for fwd in [True, False]
        }
        self.rules_a_counts = {rule.name: 0 for rule in self.rules_a}
        self._init_rule_counts()

    def _init_rule_counts(self):
        for node in self.nodes:
            # TODO: refac
            if isinstance(node, GraphTransformNode):
                self.rules_t_counts[(node.rule.name, node.fwd)] += 1
            elif isinstance(node, GraphAssertNode):
                self.rules_a_counts[node.rule.name] += 1

    def _get_rule_counts(
        self, node: GraphNode
    ) -> Dict[Union[str, Tuple[str, bool]], int]:
        """
        Return a counts of rules used to build a graph node.
        """
        if node._rule_counts:
            return node._rule_counts

        counts: Dict[Union[str, Tuple[str, bool]], int] = {}

        def _cross(node) -> None:
            if node.ntype == "hyp" or node.ntype == "true":
                return
            if node.ntype == "transform":
                k = (node.rule.name, node.fwd)
            else:
                assert node.ntype == "assert", node.ntype
                k = node.rule.name
            counts[k] = counts.get(k, 0) + 1
            if node.hyps is not None:
                for hyp in node.hyps:
                    _cross(hyp)

        _cross(node)
        node._rule_counts = counts
        return counts

    def get_rule_score(self, node: GraphNode) -> float:
        """
        Give a score to a node according to the rules used to build it,
        e.g. give a bonus to nodes built with rare rules.
        """
        counts = self._get_rule_counts(node)
        score = 0.0
        for k, v in counts.items():
            if isinstance(k, tuple):
                score += float(v) / (self.rules_t_counts[k] + 1)
            else:
                assert isinstance(k, str)
                score += float(v) / (self.rules_a_counts[k] + 1)
        return score

    def get_rule_div_score(self, node: GraphNode) -> float:
        """
        Give a score to a node according to the rules used to build it,
        e.g. give a bonus to nodes built with rare rules.
        """
        counts = self._get_rule_counts(node)
        score = 0.0
        for k, v in counts.items():
            if isinstance(k, tuple):
                score += float(v > 0) / (self.rules_t_counts[k] + 1)
            else:
                assert isinstance(k, str)
                score += float(v > 0) / (self.rules_a_counts[k] + 1)
        return score

    def node_stats(self, node: GraphNode) -> Dict[str, Union[int, float]]:
        stats = node.stats
        stats["rule_score"] = self.get_rule_score(node)
        stats["rule_div_score"] = self.get_rule_div_score(node)
        return stats

    @cached_property
    def trees(self) -> Set[GraphNode]:
        trees = set()
        for node in self.nodes:
            add = True
            for other in self.nodes:
                if other.ntype == "hyp":
                    continue
                if isinstance(other, GraphInternalNode) and node in other.hyps:
                    add = False
                    break
            if add:
                trees.add(node)
        return trees

    @property
    def max_depth(self) -> int:
        return max(n.depth for n in self.trees)

    @property
    def max_size(self) -> int:
        return max(n.size for n in self.trees)

    @property
    def max_size_by_depth(self) -> float:
        return max(n.size_by_depth for n in self.trees)

    @property
    def max_rule_score(self) -> float:
        return max(self.get_rule_score(n) for n in self.trees)

    @property
    def max_rule_div_score(self) -> float:
        return max(self.get_rule_div_score(n) for n in self.trees)

    @property
    def stats(self) -> Dict[str, Union[int, float]]:
        return {
            "max_depth": self.max_depth,
            "max_size": self.max_size,
            "max_size_by_depth": self.max_size_by_depth,
            "max_rule_score": self.max_rule_score,
            "max_rule_div_score": self.max_rule_div_score,
            "n_trees": len(self.trees),
        }


class EquationGraphGenerator(EquationGraphStats):
    """
    Generates equation graphs and temporarily store information about them, e.g. rule counts.
    (This is why we inherit from EquationGraphStats.)
    """

    def __init__(
        self,
        env: EquationEnv,
        rules: List[Rule],
        hyp_max_ops: int = 3,
        tf_prob: float = 0.5,
        bias_rules: float = 0,
        bias_nodes: float = 0,
        max_true_nodes: int = 30,
    ):
        super().__init__(nodes=[], rules=rules)
        self.env = env
        self.node_vars = NodeSet(RULE_VARS)
        self.node_var_names: Set[str] = {node.value for node in self.node_vars}
        assert all(x.is_var() for x in self.node_vars)

        # rules
        self.rules_t_e = [r for r in self.rules_t if r.left.is_comp() is False]
        self.rules_t_c = [r for r in self.rules_t if r.left.is_comp() is True]
        self.rules_a_hyps = [rule for rule in self.rules_a if len(rule.hyps) > 0]
        self.rules_a_no_hyps = [rule for rule in self.rules_a if len(rule.hyps) == 0]

        assert all(isinstance(rule, TRule) for rule in self.rules_t_e + self.rules_t_c)
        assert all(isinstance(rule, ARule) for rule in self.rules_a)
        assert all(not rule.left.is_comp() for rule in self.rules_t_e)
        assert all(rule.left.is_comp() for rule in self.rules_t_c)
        assert all(rule.node.is_comp() for rule in self.rules_a)

        # generation parameters
        assert 0 <= tf_prob <= 1
        assert hyp_max_ops >= 0
        assert max_true_nodes >= 1
        self.tf_prob = tf_prob
        self.hyp_max_ops = hyp_max_ops
        self.max_true_nodes = max_true_nodes

        # bias generation
        self.bias_rules = bias_rules
        self.bias_nodes = bias_nodes

        self.nodes: List[GraphNode] = []
        self.prefix2id: Dict[str, int] = {}

    @property
    def rng(self) -> np.random.RandomState:
        return self.env.rng

    def generate_hypothesis(self, max_ops: int, op_values: Optional[List[str]] = None):
        """
        Generate a random hypothesis.
        """
        assert max_ops >= 0

        # hypothesis operator
        if op_values is None:
            op_values = self.env.comp_ops
        else:
            assert len(op_values) == len(set(op_values))
            assert all(op_value in C_OPS for op_value in op_values)
            assert all(op_value in self.env.comp_ops for op_value in op_values)
        op_value = op_values[self.rng.randint(len(op_values))]

        while True:

            # generate hypothesis
            n_ops1 = self.rng.randint(max_ops + 1)
            n_ops2 = self.rng.randint(max_ops + 1)
            lhs = self.env.generate_expr(n_ops1, self.env.pos_hyps, non_null=n_ops1 > 0)
            rhs = self.env.generate_expr(n_ops2, self.env.pos_hyps, non_null=n_ops2 > 0)
            eq = CNode(op_value, lhs, rhs)

            # skip hypotheses without variables
            if not eq.has_vars():
                continue

            # skip invalid nodes
            if not eq.is_valid(self.env.vtype):
                continue

            # skip hypotheses that are always true or false
            res = eval_assert(eq, self.rules_a_no_hyps, self.env.vtype)
            if res is True or res is False:
                continue

            return eq

    def select_valid_t_rule(self, eq: Node) -> Tuple[TRule, bool, int]:
        """
        Given an equation, sample a transformation rule
        that can be applied (and in which direction).
        """
        while True:
            rule, fwd = self.sample_t_rule()
            valid_nodes = rule.eligible(eq, fwd)
            # TODO: with lean_rule it sometimes takes hours to sample something
            # for ex for (abs(cos(4 / 2)) <= sqrt(-3)).
            if len(valid_nodes) > 0:
                prefix_pos, _ = valid_nodes[self.rng.randint(len(valid_nodes))]
                return rule, fwd, prefix_pos

    def create_init_hyps(self, n_init_hyps: int) -> NodeSet:
        """
        Create initial hypotheses.
        Do not only generate hypotheses of the form A != B
        """
        assert n_init_hyps >= 0
        hyps = NodeSet()
        while len(hyps) < n_init_hyps:
            hyp = self.generate_hypothesis(max_ops=self.hyp_max_ops)
            if hyp not in hyps:
                hyps.add(hyp)
            # do not generate too many hypotheses of the form A != B
            if (
                len(hyps) == n_init_hyps
                and sum(hyp.value == "!=" for hyp in hyps) > len(hyps) / 2
            ):
                hyps = NodeSet()
        return hyps

    def _random_walk(
        self,
        init_eq: Node,
        n_steps: int,
        max_created_hyps: int,
        prob_add_hyp: float,
        init_hyps: Optional[NodeSet] = None,
    ) -> Tuple[List[Dict], NodeSet]:
        """
        Perform a random walk from an equation.
        """
        assert max_created_hyps >= 0
        assert 0 <= prob_add_hyp <= 1

        eq = init_eq
        walk: List[Dict] = [{"eq": eq}]
        created_hyps = NodeSet()
        if init_hyps is None:
            init_hyps = NodeSet()

        while len(walk) <= n_steps:

            # select a random rule that matches and apply it
            rule, fwd, prefix_pos = self.select_valid_t_rule(eq)
            applied = self.env.apply_t_rule(eq, rule, fwd, prefix_pos, to_fill={})

            # skip invalid nodes
            if not applied["eq"].is_valid(self.env.vtype):
                continue

            # skip if unauthorized hypotheses
            skip = False
            temp_hyps = NodeSet()

            for hyp in applied["hyps"]:

                # check whether this is a known hypothesis
                if hyp in init_hyps or hyp in created_hyps or hyp in temp_hyps:
                    continue

                # see if we can determine whether the hypothesis is true or not
                eval_hyp = eval_assert(hyp, self.rules_a_no_hyps, self.env.vtype)
                if eval_hyp is True:
                    continue
                elif eval_hyp is False:
                    skip = True
                    break
                else:
                    assert eval_hyp is None

                # maybe add the hypothesis
                if (
                    len(created_hyps) + len(temp_hyps) < max_created_hyps
                    and self.rng.rand() < prob_add_hyp
                ):
                    temp_hyps.add(hyp)
                else:
                    skip = True
                    break

            # one of the hypothesis cannot be verified given the conditions
            if skip:
                continue

            # update created hypotheses
            for hyp in temp_hyps:
                created_hyps.add(hyp)

            eq = applied["eq"]
            self.rules_t_counts[(rule.name, fwd)] += 1

            # NOTE: Lean automatically replaces 0 + 0 by 0 (and same for all pairs of nat).
            # we could do the replacements here, but this is dangerous and it can create issues
            # with tests. if we went for this, it would have required to be done in graph as well
            # if nat_rule:
            #     eq = eq.replace((INode(0) + INode(0)), INode(0))

            # update walk
            assert applied["match"].keys() | applied["to_fill"].keys() == rule.all_vars
            walk.append(
                {
                    "rule": rule,
                    "fwd": fwd,
                    "prefix_pos": prefix_pos,
                    "match": applied["match"],
                    "to_fill": applied["to_fill"],
                    "hyps": applied["hyps"],
                    "eq": eq,
                }
            )

        assert len(created_hyps) <= max_created_hyps
        assert len(walk) == n_steps + 1
        return walk, created_hyps

    def _bi_random_walk(
        self, init_eq: Node, n_steps: int, max_created_hyps: int, prob_add_hyp: float,
    ) -> Tuple[List[Dict], NodeSet]:
        """
        Perform a bidirectional random walk from an equation.
        """
        n_steps1 = self.rng.randint(0, n_steps + 1)
        n_steps2 = n_steps - n_steps1

        walk1, created_hyps1 = self._random_walk(
            init_eq=init_eq,
            n_steps=n_steps1,
            max_created_hyps=max_created_hyps,
            prob_add_hyp=prob_add_hyp,
            init_hyps=None,
        )
        walk2, created_hyps2 = self._random_walk(
            init_eq=init_eq,
            n_steps=n_steps2,
            max_created_hyps=max_created_hyps - len(created_hyps1),
            prob_add_hyp=prob_add_hyp,
            init_hyps=created_hyps1,
        )
        created_hyps = created_hyps1 | created_hyps2
        assert len(created_hyps) <= max_created_hyps

        eq = walk1[-1]["eq"]
        walk = [{"eq": eq}]

        # revert walk 1
        for step in walk1[1:][::-1]:
            # retrieve what can be inferred, and removed it from to_fill
            rule = step["rule"]
            src_vars = rule.r_vars if step["fwd"] else rule.l_vars
            variables = {**step["match"], **step["to_fill"]}
            to_fill = {k: v for k, v in variables.items() if k not in src_vars}
            applied = self.env.apply_t_rule(
                eq, rule, not step["fwd"], step["prefix_pos"], to_fill=to_fill,
            )
            assert applied["match"].keys() | applied["to_fill"].keys() == rule.all_vars
            walk.append(
                {
                    "rule": rule,
                    "fwd": not step["fwd"],
                    "prefix_pos": step["prefix_pos"],
                    "match": applied["match"],
                    "to_fill": applied["to_fill"],
                    "hyps": applied["hyps"],
                    "eq": applied["eq"],
                }
            )
            eq = applied["eq"]

        # check that we retrieve the initial equation
        assert eq.eq(init_eq)
        assert walk[-1]["eq"].eq(init_eq)
        assert len(walk) == n_steps1 + 1

        # add the second walk
        walk = walk + walk2[1:]

        assert len(walk) == n_steps + 1
        return walk, created_hyps

    def random_walk(
        self,
        bidirectional: bool,
        n_steps: int,
        n_init_ops: int,
        max_created_hyps: int,
        prob_add_hyp: float,
        op_values: Optional[List[str]] = None,
    ) -> Dict:
        if op_values is None:
            op_values = ["==", "<=", "<"]

        assert n_init_ops >= 0
        assert max_created_hyps >= 0
        assert 0 <= prob_add_hyp <= 1
        assert len(op_values) == len(set(op_values))

        # reset stats
        self.rules_t_counts = {k: 0 for k in self.rules_t_counts.keys()}
        self.rules_a_counts = {k: 0 for k in self.rules_a_counts.keys()}

        # random walk
        init_eq = self.generate_hypothesis(max_ops=n_init_ops, op_values=op_values)

        # this wasn't doable because self._random_walk has an optional argument so signatures don't match
        # f = self._bi_random_walk if bidirectional else self._random_walk
        if bidirectional:
            walk, created_hyps = self._bi_random_walk(
                init_eq=init_eq,
                n_steps=n_steps,
                max_created_hyps=max_created_hyps,
                prob_add_hyp=prob_add_hyp,
            )
        else:
            walk, created_hyps = self._random_walk(
                init_eq=init_eq,
                n_steps=n_steps,
                max_created_hyps=max_created_hyps,
                prob_add_hyp=prob_add_hyp,
            )
        assert len(walk) == n_steps + 1
        assert bidirectional or walk[0]["eq"].eq(init_eq)
        assert walk[0]["eq"].is_comp()
        assert walk[-1]["eq"].is_comp()
        assert all(hyp.is_comp() for hyp in created_hyps)
        return {
            "init_eq": init_eq,
            "start": walk[0]["eq"],
            "end": walk[-1]["eq"],
            "steps": walk,
            "hyps": created_hyps,
        }

    def add_node(self, node: GraphNode):
        """
        Add a new node to the graph.
        """
        assert isinstance(node, GraphNode)
        prefix = node.node.prefix()
        assert prefix not in self.prefix2id
        assert len(self.nodes) == len(self.prefix2id)
        # print(len(self.nodes), node.node.infix())
        self.nodes.append(node)
        self.prefix2id[prefix] = len(self.nodes) - 1

    def get_node_id(self, node: Node) -> Optional[int]:
        """
        Node ID in the graph. None if not in the graph.
        """
        assert isinstance(node, Node)
        return self.prefix2id.get(node.prefix(), None)

    def _find_all_matches(
        self,
        node: Node,
        pattern: Node,
        prefix_pos: int,
        res: List[Tuple[int, Dict[str, Node]]],
    ):
        assert isinstance(node, Node)
        assert isinstance(pattern, Node)
        match = pattern.match(node, variables=self.node_vars)
        if match is not None:
            res.append((prefix_pos, match))
        prefix_pos += 1
        for c in node.children:
            self._find_all_matches(c, pattern, prefix_pos, res)
            prefix_pos += c.prefix_len()

    def find_all_matches(self, node: GraphNode, pattern: Node):
        """
        Find all positions where a pattern can match a node (c.f. eligible).
        """
        assert isinstance(node, GraphNode)
        prefix = pattern.prefix()
        if prefix not in node._all_matches:
            res: List[Tuple[int, Dict[str, Node]]] = []
            self._find_all_matches(node.node, pattern, prefix_pos=0, res=res)
            node._all_matches[prefix] = res
        return node._all_matches[prefix]

    def match_priority_score(self, node: Node) -> int:
        """
        Give a score to each node, in order to match in priority the
        least frequent ones, e.g. match `exp(ln(A))` before `A > 0`.
        """
        assert isinstance(node, Node)
        score = sum(self.match_priority_score(c) for c in node.children)
        if node.is_comp() or node.is_unary() or node.is_binary():
            score += 1
        return score

    def search_matching_nodes(
        self, hyps: List[Node], src_node: Optional[Node] = None
    ) -> Optional[Tuple[List[GraphNode], Dict[str, Node], Optional[int]]]:
        """
        Search nodes in the graph that can match a rule.
        """
        # nodes to match
        to_match: List[Tuple[Node, bool]] = [(hyp, True) for hyp in hyps]
        if src_node is not None:
            to_match.insert(0, (src_node, False))
        assert len(to_match) >= 1

        # try to match the most difficult nodes first
        to_match_sorted: List[Tuple[int, Tuple[Node, bool]]] = sorted(
            list(enumerate(to_match)),
            key=lambda m: self.match_priority_score(m[1][0]),
            reverse=True,
        )
        sort_ids: Dict[int, int] = {
            old_id: new_id for new_id, (old_id, _) in enumerate(to_match_sorted)
        }

        # substitutions / children
        substs: Dict[str, Optional[Node]] = {
            k: None
            for k in set.union(*[eq.get_vars() for _, (eq, _) in to_match_sorted])
            if k in self.node_var_names
        }
        children: List[Union[int, Node]] = []

        # for each equation, try to find a node in the graph
        for _, (eq, is_hyp) in to_match_sorted:

            # some variables may already have been decided
            eq_vars = eq.get_vars()
            for k in eq_vars:
                if substs[k] is not None:
                    # eq = eq.set_var(k, substs[k])
                    node = substs[k]
                    assert node is not None  # really mypy ?
                    eq = eq.set_var(k, node)

            found = False

            # if the equation to match is a hypothesis fully defined (i.e. with no more
            # variables to determine), we can skip it if we can determine that it is
            # true or if it is already in the graph
            if is_hyp and not any(substs[k] is None for k in eq_vars):
                res = self.get_node_id(eq)
                if res is not None:
                    found = True
                    children.append(res)
                else:
                    e = eval_assert(eq, self.rules_a_no_hyps, self.env.vtype)
                    if e is True:
                        found = True
                        children.append(eq)
                    elif e is False:
                        return None

            # TODO: if found: continue?

            # enumerate over nodes in the graph (optionally
            # with a bias on shallow or deep nodes)
            if self.bias_nodes != 0 and len(self.nodes) > 100:  # TODO add param
                weights_l = [
                    (node.depth + 1) ** (self.bias_nodes) for node in self.nodes
                ]
                weights = np.array(weights_l, dtype=np.float64) / sum(weights_l)
                weights = self.rng.random(len(weights_l)) ** (1 / weights)
                node_order = np.argsort(weights)[::-1]
            else:
                node_order = self.rng.permutation(len(self.nodes))

            for node_id in node_order:

                # we already found a match for this equation -- nothing to do
                if found:
                    break

                # try to match the node
                if is_hyp:
                    match = eq.match(self.nodes[node_id].node, variables=self.node_vars)
                    if match is None:
                        continue
                else:
                    matches = self.find_all_matches(
                        node=self.nodes[node_id], pattern=eq
                    )
                    if len(matches) == 0:
                        continue
                    prefix_pos, match = matches[self.rng.randint(len(matches))]

                # if this node matches, update the substitutions
                assert match is not None
                for k, v in match.items():
                    assert substs[k] is None
                    substs[k] = v

                # node found
                found = True
                children.append(node_id)

            if not found:
                return None

        # sanity check
        assert len(children) == len(to_match_sorted)
        assert not any(v is None for v in substs.values())

        # children
        children = [children[sort_ids[i]] for i in range(len(children))]
        graph_children: List[GraphNode] = [
            GraphTrueNode(c) if isinstance(c, Node) else self.nodes[c] for c in children
        ]

        if src_node is None:
            prefix_pos = None
        else:
            assert isinstance(prefix_pos, int)
        return graph_children, substs, prefix_pos  # type: ignore

    def create_transformation_node(
        self, rule: TRule, fwd: bool
    ) -> Optional[Union[GraphTransformNode, GraphTrueNode]]:
        """
        Add a transformation node to the graph.
        """
        src_node = rule.left if fwd else rule.right
        match = self.search_matching_nodes(hyps=rule.hyps, src_node=src_node)
        if match is None:
            return None

        # sanity check
        children, substs, prefix_pos = match
        assert prefix_pos is not None
        assert len(children) == 1 + len(rule.hyps)  # src node in first position

        # source / target variables
        src_vars = rule.l_vars if fwd else rule.r_vars
        tgt_vars = rule.r_vars if fwd else rule.l_vars
        assert substs.keys() == src_vars | rule.h_vars

        # partial_to_fill will often be the full fill, except for some rules
        # such as mul_eq_zero_of_right where a variable is exclusively in the target
        partial_to_fill: Dict[str, Node] = {}
        if not rule.arithmetic:
            partial_to_fill = {
                k: v
                for k, v in substs.items()
                if k in (tgt_vars | rule.h_vars) - src_vars
            }

        # apply rule
        src_node = children[0].node
        applied = self.env.apply_t_rule(
            eq=src_node,
            rule=rule,
            fwd=fwd,
            prefix_pos=prefix_pos,
            to_fill=partial_to_fill,
        )
        eq = applied["eq"]

        # skip if the node is already in the graph, or is trivially false,
        # or is trivially true and we added too many true nodes
        if self.get_node_id(eq) is not None:
            return None
        e = eval_assert(eq, self.rules_a_no_hyps, self.env.vtype)
        if e is False:
            return None
        if e is True:
            if self.n_true_nodes >= self.max_true_nodes or not eq.has_vars():
                return None
            self.n_true_nodes += 1
            return GraphTrueNode(eq)

        # create new node
        substs = {k: v for k, v in applied["match"].items()}
        assert not any(k in substs for k in applied["to_fill"].keys())
        substs.update(applied["to_fill"])
        new_node = GraphTransformNode(
            node=eq,
            hyps=children,
            rule=rule,
            fwd=not fwd,
            prefix_pos=prefix_pos,
            substs=substs,
            is_true=False,
        )
        self.rules_t_counts[(rule.name, fwd)] += 1
        return new_node

    def create_assertion_node(
        self, rule: ARule
    ) -> Optional[Union[GraphAssertNode, GraphTrueNode]]:
        """
        Add an assertion node to the graph.
        """
        # find subset of nodes that match the rule hypotheses
        if len(rule.hyps) == 0:
            children: List[GraphNode] = []
            substs: Dict[str, Node] = {}
        else:
            match = self.search_matching_nodes(hyps=rule.hyps)
            if match is None:
                return None
            children, substs, _ = match

        # sanity check
        assert len(children) == len(rule.hyps)

        # set missing variables and create new statement
        eq_vars = {}
        for k in sorted(rule.n_vars):
            v = substs.get(k, None)
            if v is None:
                n_ops = self.rng.randint(self.env.fill_max_ops + 1)
                v = self.env.generate_expr(n_ops)
            eq_vars[k] = v
        eq = rule.node.set_vars(eq_vars)

        # skip if the node is already in the graph, or is trivially false,
        # or is trivially true and we added too many true nodes
        if self.get_node_id(eq) is not None:
            return None
        e = eval_assert(eq, self.rules_a_no_hyps, self.env.vtype)
        if e is False:
            return None
        if e is True:
            if self.n_true_nodes >= self.max_true_nodes or not eq.has_vars():
                return None
            self.n_true_nodes += 1
            return GraphTrueNode(eq)

        # create new node
        substs.update(eq_vars)
        new_node = GraphAssertNode(
            node=eq, hyps=children, rule=rule, substs=substs, is_true=False
        )
        self.rules_a_counts[rule.name] += 1
        return new_node

    def sample_t_rule(self) -> Tuple[TRule, bool]:
        """
        Sample a transformation rule. Optionally bias it towards rare rules.
        """
        if self.bias_rules == 0:
            rule = self.rules_t[self.rng.randint(len(self.rules_t))]
            fwd = self.rng.randint(2) == 0
            return rule, fwd
        rules_fwd = [(rule, fwd) for rule in self.rules_t for fwd in [True, False]]
        p = np.array([self.rules_t_counts[(rule.name, fwd)] for rule, fwd in rules_fwd])
        p = 1 / (np.maximum(p, 1).astype(np.float64) ** self.bias_rules)
        return rules_fwd[self.rng.choice(len(p), p=p / p.sum())]

    def sample_a_rule(self):
        """
        Sample an assertion rule.
        Optionally bias it towards rare rules.
        """
        # print("true nodes", self.n_true_nodes)
        if self.n_true_nodes >= self.max_true_nodes:
            # print("WOWOW NEW RULES")
            rules = self.rules_a_hyps
        else:
            rules = self.rules_a
        if self.bias_rules == 0:
            # print("bias 0")
            return rules[self.rng.randint(len(rules))]
        p = [self.rules_a_counts[rule.name] for rule in rules]
        p = 1 / (np.maximum(p, 1).astype(np.float64) ** self.bias_rules)
        return rules[self.rng.choice(len(p), p=p / p.sum())]

    def generate_graph(
        self, n_nodes: int, max_trials: int, n_init_hyps: int
    ) -> Tuple[List[GraphNode], NodeSet]:
        """
        Generate a random graph by randomly applying transformation
        or assertion rules on existing nodes.
        """
        assert n_nodes >= 0
        assert max_trials >= 0
        assert 0 <= n_init_hyps < n_nodes

        # create initial hypotheses / graph
        init_hyps = self.create_init_hyps(n_init_hyps)
        self.nodes = [GraphHypNode(hyp) for hyp in init_hyps]
        self.prefix2id = {node.node.prefix(): i for i, node in enumerate(self.nodes)}

        # reset stats
        self.n_true_nodes = 0
        self.rules_t_counts = {k: 0 for k in self.rules_t_counts.keys()}
        self.rules_a_counts = {k: 0 for k in self.rules_a_counts.keys()}

        n_trials = 0
        n_fails = 0

        while len(self.nodes) < n_nodes and n_trials < max_trials:
            rule_type = "t" if self.rng.rand() <= self.tf_prob else "a"
            n_trials += 1

            new_node: Optional[GraphNode] = None

            # transformation rule
            if rule_type == "t":
                t_rule, fwd = self.sample_t_rule()
                new_node = self.create_transformation_node(rule=t_rule, fwd=fwd)
                if new_node is not None and new_node.ntype != "true":
                    assert isinstance(new_node, GraphTransformNode)
                    assert not t_rule.left.is_comp() or new_node.prefix_pos == 0

            # assertion rule
            if rule_type == "a":
                a_rule = self.sample_a_rule()
                new_node = self.create_assertion_node(rule=a_rule)
                if new_node is not None and new_node.ntype != "true":
                    assert isinstance(new_node, GraphAssertNode)

            # successfully applied -- update graph / hyps -- skip invalid nodes
            if new_node is not None and new_node.node.is_valid(self.env.vtype):
                # if t_rule is not None:
                #     print(f"accepted trule {t_rule.name}")
                # elif a_rule is not None:
                #     print(f"accepted arule {a_rule.name}")
                self.add_node(new_node)
            else:
                # if t_rule is not None:
                #     print(f"rejected trule {t_rule.name}")
                # elif a_rule is not None:
                #     print(f"rejected arule {a_rule.name}")
                n_fails += 1
        # set node IDs
        for i, node in enumerate(self.nodes):
            node.node_id = i

        return self.nodes, init_hyps


class NothingToSample(Exception):
    pass


class EquationGraphSampler:
    """
    Sample nodes from a random graph according to different heuristics.
    """

    def __init__(self, rng: np.random.RandomState, params: EqGraphSamplerParams):
        # sanity check
        assert isinstance(params, EqGraphSamplerParams)
        self.rng = rng

        # do not sample nodes with too long expressions
        self.max_prefix_len = params.max_prefix_len

        # do not samples nodes of the form A != B
        self.skip_non_equal_nodes = params.skip_non_equal_nodes

        # heuristic weights. random sampling if all zeros
        self.depth_weight = params.depth_weight
        self.size_weight = params.size_weight
        self.sd_ratio_weight = params.sd_ratio_weight
        self.prefix_len_weight = params.prefix_len_weight
        self.rule_weight = params.rule_weight

    def can_sample(self, node: GraphNode) -> bool:
        """
        Return whether a node can be sampled or not.
        """
        if (
            node.ntype == "hyp"
            or node.ntype == "true"
            or (isinstance(node, GraphInternalNode) and node.is_true)
        ):
            return False
        if 0 < self.max_prefix_len < node.node.prefix_len():
            return False
        if (
            self.skip_non_equal_nodes
            and node.node.is_comp()
            and node.node.value == "!="
        ):
            return False
        return True

    def normalize(self, x_l: Union[List[int], List[float]]) -> np.ndarray:
        """
        Normalize scores.
        """
        assert type(x_l) is list
        x = np.array(x_l, dtype=np.float64)
        assert (x >= 0).all()
        # x = (x - x.mean()) / x.std()
        m = x.max()
        x = (x / m) if m > 0 else x
        return x

    def get_scores(self) -> np.ndarray:
        """
        Compute a score for all nodes in the graph.
        Give a score of zeros to nodes that can not be sampled.
        """
        # 1 if we can sample, 0 otherwise
        sample_mask = np.array(
            [1 if self.can_sample(node) else 0 for node in self.graph.nodes],
            dtype=np.float64,
        )

        # no heuristic -> random sampling
        if (
            self.depth_weight
            == self.size_weight
            == self.sd_ratio_weight
            == self.prefix_len_weight
            == self.rule_weight
            == 0
        ):
            return sample_mask

        scores = np.zeros(len(self.graph.nodes), dtype=np.float64)

        # proof depth
        if self.depth_weight != 0:
            depths: List[int] = [
                node.depth if sample_mask[i] else 0
                for i, node in enumerate(self.graph.nodes)
            ]
            scores += self.depth_weight * self.normalize(depths)

        # proof size
        if self.size_weight != 0:
            sizes: List[int] = [
                node.size if sample_mask[i] else 0
                for i, node in enumerate(self.graph.nodes)
            ]
            scores += self.size_weight * self.normalize(sizes)

        # proof size / depth
        if self.sd_ratio_weight != 0:
            size_by_depths: List[float] = [
                (node.size / node.depth) if sample_mask[i] else 0.0
                for i, node in enumerate(self.graph.nodes)
            ]
            scores += self.sd_ratio_weight * self.normalize(size_by_depths)

        # statement prefix length
        if self.prefix_len_weight != 0:
            prefix_lengths: List[int] = [
                node.node.prefix_len() if sample_mask[i] else 0
                for i, node in enumerate(self.graph.nodes)
            ]
            scores += self.prefix_len_weight * self.normalize(prefix_lengths)

        # rule scores
        if self.rule_weight != 0:
            rule_scores = [
                self.graph.get_rule_score(node) if sample_mask[i] else 0
                for i, node in enumerate(self.graph.nodes)
            ]
            scores += self.rule_weight * self.normalize(rule_scores)

        scores -= scores.min()
        scores *= sample_mask
        return scores

    def sample(
        self, graph: EquationGraphStats, n_samples: int, greedy=False
    ) -> Tuple[List[int], List[float]]:
        """
        Samples nodes from a graph.
        """
        self.graph = graph
        nodes = graph.nodes

        # compute a score for each node
        scores = self.get_scores()
        assert (scores >= 0).all()

        # if there are not enough nodes in the graph, reduce the number of samples
        n_pos = (scores > 0).sum()
        if n_pos == 0:
            raise NothingToSample()

        if n_samples > n_pos:
            # print(f"not enough nodes! ({n_pos} / {n_samples})")
            n_samples = n_pos

        if greedy:
            # pick the argmax indices
            if n_samples == 0:
                logging.info(f"Cannot pick greedily from {scores}")
                sampled_ids = np.array([], dtype=int)
            else:
                sampled_ids = np.argpartition(scores, -n_samples)[-n_samples:]
        else:
            # sample from the score distribution
            p = scores / scores.sum()
            sampled_ids = self.rng.choice(
                len(nodes), size=n_samples, replace=False, p=p
            )
        assert all(self.can_sample(nodes[i]) for i in sampled_ids)

        return sampled_ids.tolist(), scores[sampled_ids].tolist()


def check_valid_graph(graph: List[GraphNode]):
    """
    Numerically check that all nodes in a graph are valid.
    """

    # retrieve hypotheses
    hyps = [node for node in graph if node.ntype == "hyp"]
    if hyps:
        print(f"Found {len(hyps)} hypotheses:")
        for h in hyps:
            print(f"\t{h.node}")

    # retrieve variables
    var_names = sorted(list(set.union(*[node.node.get_vars() for node in graph])))
    n_vars = len(var_names)
    print(f"Found {n_vars} variables: {var_names}")

    # create substitution tests
    values = [-0.212, 0.3924, 0.6364, 1.1313]  # , 1.414
    substs_seq = list(itertools.product(values, repeat=n_vars))
    assert all(len(x) == n_vars for x in substs_seq)
    substs = [{var_names[i]: v[i] for i in range(n_vars)} for v in substs_seq]

    # look for substitutions that satisfy hypotheses
    # NOTE: just one set of valid substitutions should be enough
    valid_substs = []
    for subst in substs:
        if all(eval_numeric(hyp.node, subst) for hyp in hyps):
            valid_substs.append(subst)
    print(f"{len(valid_substs)} / {len(substs)} substitutions satisfy the hypotheses.")

    # check that all nodes verify substitutions
    n_valid = 0
    n_fails = 0
    for node in graph:
        failed = False
        for subst in valid_substs:
            if eval_numeric(node.node, subst) is not True:
                failed = True
                break
        if failed:
            # print(f"Failed on {node.node}")
            n_fails += 1
        else:
            n_valid += 1
    print(f"{n_valid}/{len(graph)} valid nodes. {n_fails} failures.")


def extract_walk_steps(
    walk: Dict, first_as_hyp: bool = True
) -> Tuple[List[Tuple[EQTheorem, EQTactic, List[Node]]], List[Node]]:
    """
    Extract the steps required to prove a node in a random walk.
    """
    steps = walk["steps"]
    hyps = []
    if first_as_hyp:
        hyps.append(steps[0]["eq"])
    hyps += list(walk["hyps"])
    assert steps[0]["eq"].eq(walk["start"])

    goals_with_tactics: List[Tuple[EQTheorem, EQTactic, List[Node]]] = []

    for i in range(1, len(steps)):

        step = steps[i]

        # reverse the rule that was applied
        rule: TRule = step["rule"]
        src_vars = rule.r_vars if step["fwd"] else rule.l_vars
        variables = {**step["match"], **step["to_fill"]}
        tactic = EQRuleTactic(
            rule.name,
            fwd=not step["fwd"],
            prefix_pos=step["prefix_pos"],
            to_fill={k: v for k, v in variables.items() if k not in src_vars},
        )
        goal = EQTheorem(node=step["eq"], hyps=hyps)

        children = [steps[i - 1]["eq"], *steps[i]["hyps"]]
        goals_with_tactics.append((goal, tactic, children))

    assert goals_with_tactics[-1][0].eq_node.eq(walk["end"])

    return goals_with_tactics, hyps


def extract_graph_steps(
    node: GraphNode,
) -> Tuple[List[Tuple[EQTheorem, EQTactic, List[Node]]], List[Node], Set[int]]:
    """
    Extract the steps required to prove a node in a graph.
    """
    assert isinstance(node, GraphNode)
    goals: List[GraphNode] = []
    tactics: List[EQTactic] = []
    children: List[List[Node]] = []
    used_hyps: List[Node] = []
    node.get_goal_tactics(goals, tactics, children, used_hyps, set())

    # sanity check
    assert goals[-1].node.eq(node.node)

    # each node only appears once
    node_set = [goal.node.prefix() for goal in goals]
    assert len(set(node_set)) == len(node_set)

    # add only used hypotheses to goals
    assert all(goal.node_id is not None for goal in goals)
    node_ids = set([goal.node_id for goal in goals if goal.node_id is not None])
    theorems = [EQTheorem(node=goal.node, hyps=used_hyps) for goal in goals]

    assert len(theorems) == len(tactics) == len(children)
    goals_with_tactics = list(zip(theorems, tactics, children))

    return goals_with_tactics, used_hyps, node_ids


def walk_to_graph(
    walk: Dict,
    egg: EquationGraphGenerator,
    start_nodes: Optional[List[GraphNode]] = None,
    first_as_hyp: bool = True,
) -> Tuple[List[GraphNode], NodeSet]:
    """
    Convert a random walk to a random graph with possibiliy an initial root node.
    """
    goals_with_tactics, walk_hyps = extract_walk_steps(walk, first_as_hyp=first_as_hyp)
    # convert the random walk to a graph

    graph_nodes: List[GraphNode] = start_nodes or []
    prefix2gnode: Dict[str, GraphNode] = {x.node.prefix(): x for x in egg.nodes}

    def add_graph_node(gnode: GraphNode):
        prefix = gnode.node.prefix()
        assert prefix not in prefix2gnode
        gnode.node_id = len(graph_nodes)
        graph_nodes.append(gnode)
        prefix2gnode[prefix] = graph_nodes[-1]

    # add hypotheses
    for hyp in walk_hyps:
        if hyp.prefix() in prefix2gnode:
            continue
        add_graph_node(GraphHypNode(hyp))

    # add nodes from the random walk
    for goal, tactic, children in goals_with_tactics:
        assert isinstance(tactic, EQRuleTactic)
        assert tactic.rule_type == "t"
        prefix = goal.eq_node.prefix()
        if prefix in prefix2gnode:
            continue

        # re-apply tactic to get all nodes (including true nodes)
        assert isinstance(tactic.rule, TRule)
        applied = egg.env.apply_t_rule(
            eq=goal.eq_node,
            rule=tactic.rule,
            fwd=tactic.fwd,
            prefix_pos=tactic.prefix_pos,
            to_fill=tactic.to_fill,
        )  # eq / match / to_fill / hyps

        # sanity check
        assert len(applied["hyps"]) == len(children) - 1
        assert all(c1.eq(c2) for c1, c2 in zip(applied["hyps"], children[1:]))

        # add true / trivial statements
        for child in children:
            if child.prefix() in prefix2gnode:
                continue
            res = eval_assert_with_rule(child, egg.rules_a_no_hyps, egg.env.vtype)
            if res == "__NUMERIC__":
                add_graph_node(GraphTrueNode(child))
            elif isinstance(res, ARule):
                add_graph_node(GraphTrueNode(child))
            else:
                raise RuntimeError(f"{child} is not trivial (res={res})")

        node_hyps = [prefix2gnode[node.prefix()] for node in children]

        substs = {**applied["match"], **applied["to_fill"]}
        assert substs.keys() == tactic.rule.all_vars

        gnode = GraphTransformNode(
            node=goal.eq_node,
            hyps=node_hyps,
            rule=tactic.rule,
            fwd=tactic.fwd,
            prefix_pos=tactic.prefix_pos,
            substs=substs,
            missing_true=False,
            is_true=False,
        )
        # this condition is almost always verified, except when
        # the next node was in the hypotheses
        if gnode.node.prefix() not in prefix2gnode:
            add_graph_node(gnode)

        # no need to cycle around the final goal
        if goal.eq_node.eq(walk["end"]):
            break

    # each node only appears once
    node_set = [goal.node.prefix() for goal in graph_nodes]
    assert len(set(node_set)) == len(node_set)

    return graph_nodes, NodeSet(walk_hyps)


def check_bwd_proof(
    env: EquationEnv,
    rules_a: List[ARule],
    goals_with_tactics: List[Tuple[EQTheorem, EQTactic, List[Node]]],
    hyps: List[Node],
    can_loop: bool,
):
    """
    Check that a proof is valid.
    Nodes must be given in a bottom-up order.
    """
    assert type(goals_with_tactics) is list and len(goals_with_tactics) > 0
    assert type(hyps) is list
    assert all(isinstance(th, EQTheorem) for th, _, _ in goals_with_tactics)
    assert all(isinstance(tac, EQTactic) for _, tac, _ in goals_with_tactics)
    assert all(isinstance(hyp, Node) for hyp in hyps)

    from evariste.backward.env.equations.env import apply_bwd_tactic

    init_hyps = NodeSet(hyps)
    proved = {hyp.prefix(): 0 for hyp in init_hyps}

    for goal, tactic, subgoals_ in goals_with_tactics:
        subgoals = apply_bwd_tactic(
            env, goal, tactic, keep_if_hyp=True, rule_env="default"
        )
        assert subgoals is not None
        assert len(subgoals) <= len(subgoals_), (
            [sg.eq_node.infix() for sg in subgoals],
            subgoals_,
        )

        expected = NodeSet(subgoals_)
        got = NodeSet([sg.eq_node for sg in subgoals])
        for sg in got:
            assert sg in expected
        for sg in expected:
            if sg not in got:
                assert eval_assert(sg, rules_a) is True
        for sg in subgoals:
            assert isinstance(sg, EQTheorem)
            assert NodeSet(sg.eq_hyps) == init_hyps
            str_sg = sg.eq_node.prefix()
            if str_sg in proved:
                # print(f"Already proved: {sg}")
                proved[str_sg] += 1
                continue
            else:
                raise Exception(f"{sg.eq_node} has not been proved!")
        str_g = goal.eq_node.prefix()
        if str_g not in proved:
            proved[str_g] = 0

    # check that the root node is exactly the only one that is never used,
    # unless this is a random walk with a loop
    str_goal = goals_with_tactics[-1][0].eq_node.prefix()
    n_used = len([1 for v in proved.values() if v > 0])
    assert (
        n_used == len(proved) - 1
        and proved[str_goal] == 0
        or can_loop
        and n_used == len(proved)
        and proved[str_goal] > 0
    )

    return True


def build_env_egg(
    rule_env: str, seed: Optional[int], **kwargs
) -> Tuple[EquationEnv, EquationGraphGenerator]:

    # SIMPLE_OPS = [
    #     "neg",
    #     "inv",
    #     # "exp",
    #     # "ln",
    #     # "pow2",
    #     # "sqrt",
    #     # "abs",
    #     # "cos",
    #     # "sin",
    #     # "tan",
    #     # "cosh",
    #     # "sinh",
    #     # "tanh",
    # ]
    MORE_OPS = [
        "neg",
        "inv",
        "exp",
        "ln",
        "pow2",
        "sqrt",
        # "abs",
        "cos",
        "sin",
        "tan",
        "cosh",
        "sinh",
        "tanh",
    ]
    args = EquationsEnvArgs(
        vtype="real",
        unary_ops_str=",".join(MORE_OPS),
        binary_ops_str="add,sub,mul,div",
        n_vars=5,
    )

    for k, v in kwargs.items():
        assert hasattr(args, k)
        setattr(args, k, v)

    env = EquationEnv.build(args, seed=seed)

    # graph generator
    rules = [
        rule
        for rule in ALL_RULES[rule_env]
        if rule.get_unary_ops().issubset(env.unary_ops)
        and rule.get_binary_ops().issubset(env.binary_ops)
    ]
    egg = EquationGraphGenerator(
        env=env,
        rules=rules,
        hyp_max_ops=3,
        tf_prob=0.5,
        bias_nodes=0,
        bias_rules=1,
        max_true_nodes=30,
    )

    return env, egg
