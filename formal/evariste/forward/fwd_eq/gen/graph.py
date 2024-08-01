# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
from evariste.backward.env.equations.graph import EQTheorem, EQTactic

from evariste.envs.eq.env import EquationEnv
from evariste.envs.eq.graph import C_OPS, CNode
from evariste.envs.eq.generation import (
    GraphNode,
    GraphAssertNode,
    GraphInternalNode,
    GraphHypNode,
    GraphNormNumNode,
    GraphSimpNode,
    GraphTransformNode,
    GraphTrueNode,
)
from evariste.envs.eq.graph import Node, NodeSet, RULE_VARS
from evariste.envs.eq.rules import ARule, eval_assert, TRule
from evariste.backward.env.equations.env import EqGenRuleEnv
from evariste.envs.mm.utils import node_tok

from evariste.forward.common import GenerationError
from evariste.model.data.dictionary import B_NODE_WORD, E_NODE_WORD, EOS_WORD


class EqGenForwardGraph:
    def __init__(
        self,
        proof_id: int,
        rule_env: Optional[EqGenRuleEnv],
        env: Optional[EquationEnv],
        max_true_nodes: int = 0,
        max_created_hyps: int = 0,
        prob_add_hyp: float = 0,
        nodes: Optional[List[GraphNode]] = None,
        previous_nodes: Optional[Set[str]] = None,
        has_simps: bool = False,
    ):
        self.proof_id = proof_id
        self.has_simps = has_simps

        self.rule_env = rule_env
        self.nodes: List[GraphNode] = []

        self.env = env

        self.node_vars = NodeSet(RULE_VARS)
        self.node_var_names: Set[str] = {node.str_value for node in self.node_vars}

        self.prefix2id: Dict[str, int] = {}
        self.n_graphtrue_nodes = 0
        self.n_true_nodes = 0

        self.tokenized_node_id: List[int] = []
        self.node_id_before_tac: Dict[int, int] = {}
        self.max_created_hyps = max_created_hyps
        self.prob_add_hyp = prob_add_hyp
        self.max_true_nodes = max_true_nodes

        self.previous_nodes: Set[str] = set()
        if previous_nodes is not None:
            self.previous_nodes.update(previous_nodes)  # copy

        if nodes is not None:
            for n in nodes:
                self.add_node(n)

    def clear(self):
        self.rule_env = None
        self.env = None

    @property
    def forbidden(self):
        # forbidden tokens for seq2seq model
        return None

    @property
    def n_rwalk_nodes(self):
        return len([gn for gn in self.nodes if not isinstance(gn, GraphHypNode)])

    @property
    def not_true_nodes(self):
        return [gn for gn in self.nodes if not isinstance(gn, GraphTrueNode)]

    def sample_node_pred(self) -> Tuple[str, List[str]]:
        i = self.rng.choice(len(self.not_true_nodes))
        pred_tok = f"_PRED_{node_tok(i)}"
        to_pred = [
            B_NODE_WORD,
            *self.not_true_nodes[i].node.prefix_tokens(),
            E_NODE_WORD,
        ]
        return pred_tok, to_pred

    def sample_nodeid_pred(self) -> Tuple[List[str], str]:
        i = self.rng.choice(len(self.not_true_nodes))
        pred_tok = [
            B_NODE_WORD,
            *self.not_true_nodes[i].node.prefix_tokens(),
            E_NODE_WORD,
        ]
        to_pred = node_tok(i)
        return pred_tok, to_pred

    def find_true_id(self, target: Node) -> int:
        p = target.prefix()
        for i, n in enumerate(self.not_true_nodes):
            if n.node.prefix() == p:
                return i
        raise RuntimeError("Node not found")

    def get_not_true(self, id: int) -> GraphNode:
        return self.not_true_nodes[id]

    def tokenize(self) -> List[str]:
        return [
            EOS_WORD,
            *(
                tok
                for i, theorem in enumerate(self.not_true_nodes)
                for tok in (node_tok(i), *theorem.node.prefix_tokens(), node_tok(i))
            ),
            EOS_WORD,
        ]

    def get_hyps_for_node(self, node: GraphNode) -> List[GraphHypNode]:
        hyps = []
        visited = set()

        def visit(cur: GraphNode):
            p = cur.node.prefix()
            if p in visited:
                return
            visited.add(p)
            if isinstance(cur, GraphHypNode):
                hyps.append(cur)
                return
            elif isinstance(cur, (GraphInternalNode, GraphSimpNode)):
                for h in cur.hyps:
                    visit(h)

        visit(node)
        return hyps

    def get_bwd_proof_for_node(
        self, th: EQTheorem
    ) -> Dict[EQTheorem, Tuple[EQTactic, List[EQTheorem]]]:
        # maybe redundant with get nodes / tactics ?
        res = {}

        def visit(cur: GraphNode):
            cur_th = EQTheorem(cur.node, hyps=th.eq_hyps)
            if cur_th in res:
                return
            if isinstance(cur, GraphInternalNode):
                res[cur_th] = (
                    cur.get_tactic(),
                    [EQTheorem(h.node, hyps=th.eq_hyps) for h in cur.hyps],
                )
                for h in cur.hyps:
                    visit(h)

        root = self.nodes[self.prefix2id[th.eq_node.prefix()]]
        visit(root)
        return res

    def cut(self, tac_id: int) -> "EqGenForwardGraph":
        if tac_id not in self.node_id_before_tac:
            raise RuntimeError(f"tac {tac_id} didn't touch this graph")
        cut_id = self.node_id_before_tac[tac_id]
        new_graph = EqGenForwardGraph(
            self.proof_id,
            self.rule_env,
            self.env,
            max_true_nodes=self.max_true_nodes,
            max_created_hyps=self.max_created_hyps,
            prob_add_hyp=self.prob_add_hyp,
            has_simps=self.has_simps,
        )
        new_graph.nodes = self.nodes[:cut_id]
        new_graph.prefix2id = {x: y for x, y in self.prefix2id.items() if y < cut_id}
        new_graph.tokenized_node_id = self.tokenized_node_id[:cut_id]

        new_graph.n_true_nodes = sum(
            [
                1
                for gn in self.nodes[:cut_id]
                if isinstance(gn, GraphTrueNode)
                or (isinstance(gn, GraphInternalNode) and gn.is_true)
            ]
        )
        new_graph.n_graphtrue_nodes = sum(
            [1 for gn in self.nodes[:cut_id] if isinstance(gn, GraphTrueNode)]
        )
        return new_graph

    def __eq__(self, other):
        if not isinstance(other, EqGenForwardGraph):
            return False
        if len(self.nodes) != len(other.nodes):
            return False
        if self.n_true_nodes != other.n_true_nodes:
            print("NOT SAME TRUE NODES", self.n_true_nodes, other.n_true_nodes)
            return False
        if self.n_graphtrue_nodes != other.n_graphtrue_nodes:
            print(
                "NOT SAME GRAPHTRUE NODES",
                self.n_graphtrue_nodes,
                other.n_graphtrue_nodes,
            )
            return False
        if self.prefix2id != other.prefix2id:
            print("NOT SAME PREFIX ID")
            print("ME")
            for x, y in self.prefix2id.items():
                print(x, hash(x), y)
            print("OTHER")
            for x, y in other.prefix2id.items():
                print(x, hash(x), y)
            return False
        if self.tokenized_node_id != other.tokenized_node_id:
            print(
                "NOT SAME TOKENIZED NODE ID",
                self.tokenized_node_id,
                other.tokenized_node_id,
            )
            return False
        for a, b in zip(self.nodes, other.nodes):
            if not a.node.eq(b.node):
                print("DIFFERENT NODE")
                return False
        return True

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def rng(self) -> np.random.RandomState:
        assert self.env is not None
        return self.env.rng

    def get_node_id(self, node: Node) -> Optional[int]:
        """
        Node ID in the graph. None if not in the graph.
        """
        return self.prefix2id.get(node.prefix(), None)

    def clone(
        self, substs: Optional[Dict[str, Node]] = None
    ) -> Tuple["EqGenForwardGraph", Dict[str, GraphNode]]:
        """
        Returns a copy of the graph, optionally gives a copy with a substitution everywhere defined by substs
        Expensive ! Only use if needed !
        If needed and too slow: make faster by avoiding prefix() calls:
        children : List[GraphNode] -> List[int] 
        """
        new_graph = EqGenForwardGraph(
            self.proof_id,
            self.rule_env,
            self.env,
            previous_nodes=set(self.prefix2id.keys()) | self.previous_nodes,
            has_simps=self.has_simps,
            max_true_nodes=self.max_true_nodes,
            max_created_hyps=self.max_created_hyps,
            prob_add_hyp=self.prob_add_hyp,
        )
        substs = substs or {}
        mapping = {}
        for node in self.nodes:
            next_expr = node.node.set_vars(subst=substs, mandatory=False)
            evaled: Optional[bool] = node.is_true
            if len(substs) > 0:
                evaled = self.eval_assert(next_expr)
                if evaled is False:
                    raise GenerationError("Subst made node false")

            cloned_hyps: List[GraphNode] = []
            if isinstance(node, GraphInternalNode):
                cloned_hyps = []
                for hyp in node.hyps:
                    n_hyp_node = hyp.node.set_vars(subst=substs, mandatory=False)
                    try:
                        cloned_hyps.append(
                            new_graph.nodes[new_graph.prefix2id[n_hyp_node.prefix()]]
                        )
                    except KeyError as e:
                        if not self.eval_assert(n_hyp_node):
                            print(hyp.node.infix(), " --> ", n_hyp_node.infix())
                            print("IS IN PREFIX ?", hyp.node.prefix() in self.prefix2id)
                            print("IS OF TYPE ", hyp.__class__.__name__)
                            print("EVALS TO ", self.eval_assert(hyp.node))
                            # 2: ((1 / x3) <= sin(5))  -->  ((1 / x3) <= sin(5))
                            # 2: IS IN PREFIX ? False
                            # 2: IS OF TYPE  GraphHypNode
                            # 2: EVALS TO  None

                        assert self.eval_assert(n_hyp_node), e

                        cloned_hyps.append(GraphTrueNode(n_hyp_node))
            next_prefix = next_expr.prefix()
            if (
                next_prefix
                in new_graph.prefix2id
                # or next_prefix in new_graph.previous_nodes
            ):
                # raise GenerationError("Duplicated node during substs")
                continue

            mapping[next_prefix] = node
            new_node = node.substitute(next_expr, substs, cloned_hyps)
            new_node.parent = node
            new_node.is_true = evaled is True
            new_graph.add_node(new_node)

        return new_graph, mapping

    def make_true(self, node: Node, rule: ARule, substs: Dict[str, Node]):
        assert self.rule_env is not None and self.env is not None
        p = node.prefix()
        if p in self.prefix2id:
            to_replace = self.prefix2id[p]
            self.nodes[to_replace] = GraphAssertNode(
                node, hyps=[], rule=rule, substs=substs, is_true=True
            )
        else:
            assert self.eval_assert(node), node.infix()

    def add_node(self, node: GraphNode) -> "EqGenForwardGraph":
        """
        Returns a new EQGenForwardGraph with a additional new (Graph)Node.
        """
        if len(self.nodes) != len(self.prefix2id):
            raise RuntimeError(
                f"invalid size of graph, length of prefix to id is {len(self.prefix2id)} and should be {len(self.nodes)}"
            )
        prefix = node.node.prefix()
        if isinstance(node, GraphTrueNode) or (
            isinstance(node, GraphInternalNode) and node.is_true
        ):
            self.n_true_nodes += 1

        if isinstance(node, GraphTrueNode):
            self.n_graphtrue_nodes += 1
            self.tokenized_node_id.append(-1)
        else:
            self.tokenized_node_id.append(len(self.nodes) - self.n_graphtrue_nodes)
        node.node_id = len(self.nodes)
        assert prefix not in self.prefix2id
        self.prefix2id[prefix] = len(self.nodes)
        self.nodes.append(node)
        assert len(self.nodes) == len(self.prefix2id) and len(self.nodes) == len(
            self.tokenized_node_id
        ), (len(self.nodes), len(self.prefix2id), len(self.tokenized_node_id))
        return self

    def find_hyp_nodes(
        self, nodes: List[Node]
    ) -> Tuple[List[GraphNode], List[GraphNode]]:
        """
        Given a list of Node find the corresponding GraphNodes in the graph.
        If some are missing, can add them with a probability prob_add_hyp
        up to a number allowed_created_hyps
        """
        hyps_gn: List[GraphNode] = []
        n_hyps_allowed = self.max_created_hyps
        temp_hyps: List[GraphNode] = []
        temp_hyps2node: Dict[str, GraphNode] = {}
        for node in nodes:
            prefix = node.prefix()
            try:
                hyps_gn.append(self.nodes[self.prefix2id[prefix]])
            except KeyError:
                res = self.eval_assert(node)
                if res:
                    hyps_gn.append(GraphTrueNode(node))
                elif prefix in temp_hyps2node:
                    hyps_gn.append(temp_hyps2node[prefix])
                elif (
                    res is None
                    and n_hyps_allowed > 0
                    and self.rng.rand() < self.prob_add_hyp
                ):
                    created_hyp = GraphHypNode(node)
                    temp_hyps.append(created_hyp)
                    temp_hyps2node[prefix] = created_hyp
                    hyps_gn.append(created_hyp)
                    n_hyps_allowed = n_hyps_allowed - 1
                else:
                    raise GenerationError(
                        f"hyp {node} is not in the graph: res {res!r} // {n_hyps_allowed}"
                    )
        assert len(temp_hyps) == (self.max_created_hyps - n_hyps_allowed)
        self.max_created_hyps = self.max_created_hyps - len(temp_hyps)
        for hyp in temp_hyps:
            _ = self.add_node(hyp)
        return hyps_gn, temp_hyps

    def add_t_node(
        self,
        node: Node,
        src_node: Node,
        hyps: List[Node],
        rule: TRule,
        substs: Dict[str, Node],
        fwd: bool,
        prefix_pos: int,
    ) -> "EqGenForwardGraph":
        assert self.rule_env is not None
        if node.prefix() in self.prefix2id:
            raise GenerationError(f"node {node} already in prefix2id")
        is_true = self.invalid_or_true(node)
        if node.prefix() in {n.prefix() for n in hyps}:
            raise GenerationError("Transform node in hyps")
        new_hyps, created_hyps = self.find_hyp_nodes([src_node, *hyps])
        to_ret = self.add_node(
            GraphTransformNode(
                node,
                hyps=new_hyps,
                rule=rule,
                substs=substs,
                fwd=fwd,
                prefix_pos=prefix_pos,
                is_true=is_true,
            )
        )
        self.rule_env.rules_t_counts[(rule.name, fwd)] += 1
        return to_ret

    def add_a_node(
        self, node: Node, hyps: List[Node], rule: ARule, substs: Dict[str, Node],
    ):
        assert self.rule_env is not None
        if node.prefix() in self.prefix2id:
            raise GenerationError(f"node {node} already in prefix2id")
        is_true = self.invalid_or_true(node)
        if node.prefix() in {n.prefix() for n in hyps}:
            raise GenerationError("Assert node in hyps")
        new_hyps, created_hyps = self.find_hyp_nodes(hyps)
        assert len(created_hyps) == 0
        to_ret = self.add_node(
            GraphAssertNode(
                node, hyps=new_hyps, rule=rule, substs=substs, is_true=is_true
            )
        )
        self.rule_env.rules_a_counts[rule.name] += 1
        return to_ret

    def add_simp_node(
        self,
        node: Node,
        src_node: GraphNode,
        rules: List[TRule],
        hyps: Optional[List[Node]] = None,
        graph_hyps: Optional[List[GraphNode]] = None,
    ):
        is_true = self.invalid_or_true(node)
        if hyps is not None:
            assert graph_hyps is None
            if node.prefix() in {n.prefix() for n in hyps}:
                raise GenerationError("Simped node in hyps")
            graph_hyps, created_hyps = self.find_hyp_nodes(hyps)
            assert len(created_hyps) == 0
        assert graph_hyps is not None
        self.has_simps = True
        return self.add_node(
            GraphSimpNode(
                node, hyps=[src_node, *graph_hyps], rules=rules, is_true=is_true
            )
        )

    def add_nn_node(self, node: Node, src_node: GraphNode):
        is_true = self.invalid_or_true(node)
        return self.add_node(GraphNormNumNode(node, hyps=[src_node], is_true=is_true))

    def replace_hyp_with_simp(
        self, base_node: Node, new_node: Node, hyps: List[Node], rules: List[TRule]
    ) -> "EqGenForwardGraph":
        """
        Go through the graph forests and rebuild it topologically (ie from the leaves)
        """
        new_graph = EqGenForwardGraph(
            self.proof_id, self.rule_env, self.env, has_simps=self.has_simps
        )
        to_replace = base_node.prefix()
        visiting: Set[str] = set()

        def visit(node: GraphNode) -> GraphNode:
            prefix = node.node.prefix()

            if prefix in new_graph.prefix2id:
                return new_graph.nodes[new_graph.prefix2id[prefix]]
            if prefix in visiting:
                raise GenerationError("Cycle in hyp simp !")
            visiting.add(prefix)

            cloned_hyps = []
            if isinstance(node, GraphInternalNode):
                for h in node.hyps:
                    cloned_hyps.append(visit(h))
            if prefix == to_replace:
                new_hyps = []
                for h in self.find_hyp_nodes(hyps)[0]:
                    new_hyps.append(visit(h))

                if self.eval_assert(new_node) is True:
                    added_node: GraphNode = GraphTrueNode(new_node)
                else:
                    added_node = GraphHypNode(new_node)
                    new_graph.add_node(node)
                new_graph.add_simp_node(
                    base_node, src_node=added_node, graph_hyps=new_hyps, rules=rules
                )
            else:
                new_graph.add_node(node.clone(cloned_hyps))
            visiting.remove(prefix)
            return new_graph.nodes[-1]

        for n in self.nodes[::-1]:
            visit(n)

        # assert topological order is respected
        # for n in new_graph.nodes:
        #     this_id = new_graph.prefix2id[n.node.prefix()]
        #     for d, _ in n.descendants:
        #         assert new_graph.prefix2id[d] <= this_id

        return new_graph

    def eval_assert(self, expr: Node) -> Optional[bool]:
        assert self.rule_env is not None and self.env is not None
        return eval_assert(expr, self.rule_env.rules_a, self.env.vtype)

    def _find_all_matches(
        self,
        node: Node,
        pattern: Node,
        prefix_pos: int,
        res: List[Tuple[int, Dict[str, Node]]],
    ):
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
        self,
        hyps: List[Node],
        bias_nodes: float,
        src_node: Optional[Node] = None,
        where_to_apply: Optional[GraphNode] = None,
        allowed_hyps: Optional[List[str]] = None,
        prefix_pos: Optional[int] = None,
        substs: Optional[Dict[str, Optional[Node]]] = None,
        src_in_graph: Optional[Node] = None,
    ) -> Optional[Tuple[List[GraphNode], Dict[str, Node], Optional[int]]]:
        """
        Search nodes in the graph that can match a rule.
        """
        assert self.env is not None and self.rule_env is not None
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
        if substs is None:
            substs = {}
        for k in set.union(*[eq.get_vars() for _, (eq, _) in to_match_sorted]):
            if k in self.node_var_names and k not in substs:
                substs[k] = None
        final_substs: Dict[str, Node] = {}

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
                if res is not None and (
                    allowed_hyps is None or self.nodes[res].node.infix() in allowed_hyps
                ):
                    found = True
                    children.append(res)
                else:
                    e = self.eval_assert(eq)
                    if e is True:
                        found = True
                        children.append(eq)
                    elif e is False:
                        return None

            # TODO: if found: continue?

            # enumerate over nodes in the graph (optionally
            # with a bias on shallow or deep nodes)
            if where_to_apply is None:
                if bias_nodes != 0 and len(self.nodes) > 100:  # TODO add param
                    weights_l = [(node.depth + 1) ** bias_nodes for node in self.nodes]
                    weights = np.array(weights_l, dtype=np.float64) / sum(weights_l)
                    weights = self.rng.random(len(weights_l)) ** (1 / weights)
                    node_order = np.argsort(weights)[::-1]
                else:
                    node_order = self.rng.permutation(len(self.nodes))
            else:
                node_order = np.array([where_to_apply.node_id])

            src_node_id = None
            if src_in_graph is not None:
                p = src_in_graph.prefix()
                for i, n in enumerate(self.nodes):
                    if n.node.prefix() == p:
                        src_node_id = i
                        break
                if src_node_id is None:
                    raise GenerationError(f"src node {src_in_graph.prefix()} not found")

            for node_id in node_order:
                assert node_id is not None

                # we already found a match for this equation -- nothing to do
                if found:
                    break

                # try to match the node
                if is_hyp:
                    match = eq.match(self.nodes[node_id].node, variables=self.node_vars)
                    if match is None:
                        continue
                else:
                    if src_node_id is not None and node_id != src_node_id:
                        continue
                    matches = self.find_all_matches(
                        node=self.nodes[node_id], pattern=eq
                    )
                    if len(matches) == 0:
                        continue
                    if prefix_pos is not None:
                        try:
                            prefix_pos, match = [
                                (pp, m) for pp, m in matches if pp == prefix_pos
                            ][0]
                        except IndexError:
                            continue
                            # raise GenerationError("no match found at prefix pos")
                    else:
                        prefix_pos, match = matches[self.rng.randint(len(matches))]

                # if this node matches, update the substitutions
                assert match is not None
                for k, v in match.items():
                    if substs[k] is not None and not substs[k].eq(v):  # type: ignore
                        raise GenerationError("match mismatched")
                    substs[k] = v
                    final_substs[k] = v

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

        if src_node is not None:
            assert isinstance(prefix_pos, int)
        return graph_children, final_substs, prefix_pos

    def invalid_or_true(self, eq: Node) -> bool:
        assert self.env is not None
        if self.get_node_id(eq) is not None:
            raise GenerationError(f"Already exists {eq.prefix()}")
        if eq.prefix() in self.previous_nodes:
            raise GenerationError(f"Already exists in previous: {eq.prefix()}")

        # TODO: add the following ? not in old generation.
        # is_lean="lean" in graph.rule_env.rule_env
        if not eq.is_valid(self.env.vtype):
            raise GenerationError("Invalid node")
        e = self.eval_assert(eq)
        if e is False:
            raise GenerationError("Is False")
        if e is True:
            if self.n_true_nodes >= self.max_true_nodes or not eq.has_vars():
                raise GenerationError(
                    f"Too many true or no variables in the node {self.n_true_nodes} >= {self.max_true_nodes} // {eq.has_vars()}"
                )
            return True
        return False

    def generate_hypothesis(self, max_ops: int, op_values: Optional[List[str]] = None):
        assert self.env is not None and self.rule_env is not None
        assert max_ops >= 0
        rng = self.env.rng
        # hypothesis operator
        if op_values is None:
            op_values = self.env.comp_ops
        else:
            assert len(op_values) == len(set(op_values))
            assert all(op_value in C_OPS for op_value in op_values)
            assert all(op_value in self.env.comp_ops for op_value in op_values)
        op_value = op_values[rng.randint(len(op_values))]

        while True:
            # generate hypothesis
            n_ops1 = rng.randint(max_ops + 1)
            n_ops2 = rng.randint(max_ops + 1)
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
            res = self.eval_assert(eq)
            if res is True or res is False:
                continue
            return eq

    def init_rwalk(self, max_ops: int) -> "EqGenForwardGraph":
        return self.add_node(
            GraphHypNode(
                self.generate_hypothesis(max_ops=max_ops, op_values=["==", "<=", "<"])
            )
        )

    def init_graph(self, max_ops: int, n_init_hyps: int) -> "EqGenForwardGraph":
        """Avoid generating hypotheses of the form A != B"""
        assert n_init_hyps >= 0
        hyps = NodeSet()
        rand_init_hyps = self.rng.randint(1, n_init_hyps + 1)
        while len(hyps) < rand_init_hyps:
            hyp = self.generate_hypothesis(max_ops=max_ops)
            if hyp not in hyps:
                hyps.add(hyp)
            # do not generate too many hypotheses of the form A != B
            # TODO: remove these hypothesis rather than restart
            if (
                len(hyps) == rand_init_hyps
                and sum(hyp.value == "!=" for hyp in hyps) > len(hyps) / 2
            ):
                hyps = NodeSet()
        to_ret = self
        for hyp in hyps:
            to_ret = to_ret.add_node(GraphHypNode(hyp))
        return to_ret

    def __contains__(self, n: Node):
        return n.prefix() in self.prefix2id
