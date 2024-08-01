# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from operator import contains
from typing import Optional, Tuple, List, Set, Dict
import time
from copy import deepcopy

from evariste.envs.eq.graph import Node, NodeSet
from evariste.envs.eq.rules import (
    ARule,
    Rule,
    eval_assert_with_rule,
)
from evariste.envs.eq.rules_lib import RULES_DEFAULT, ALL_RULES, RULES_LEAN_IMO
from evariste.envs.eq.generation import (
    EquationGraphGenerator,
    GraphNode,
    GraphAssertNode,
    GraphHypNode,
    GraphInternalNode,
    GraphTransformNode,
    GraphTrueNode,
    build_env_egg,
    extract_graph_steps,
    walk_to_graph,
)
from evariste.utils import MyTimeoutError, timeout


RULES_EQ = RULES_DEFAULT + RULES_LEAN_IMO  # classical EQ rules
RULES_MAX_ARG = max([len(rule.n_vars) for rule in RULES_EQ if isinstance(rule, ARule)])
RULES_EQ_TYPE: List[Optional[str]] = list(set([rule.rule_type for rule in RULES_EQ]))


def _rule_eligible(
    ref_node: Node, node: Node, eligibles: List[Dict[str, Node]] = []
) -> List[Dict]:
    matched = rule_match(ref_node, node)
    if matched is not None:
        eligibles.append(matched)
    for c in node.children:
        _rule_eligible(ref_node, c, eligibles)
    return eligibles


def rule_match(ref_node: Node, node: Node) -> Optional[Dict]:
    dict: Dict[str, Node] = {}
    if len(node.children) == 2 and node.children[0].eq(node.children[1]):
        return None
    if node.value == ref_node.value and all(child.is_var() for child in node.children):
        for i in range(len(node.children)):
            dict[ref_node.children[i].infix()] = node.children[i]
    return dict if len(dict) > 0 else None


def sgIMO(
    egg: EquationGraphGenerator,
    min_rule_vars: int = 3,
    rule_type: str = "imo",
    with_hyps: bool = False,
    complexity: int = 2,
):
    assert min_rule_vars > 0
    # assert rule_type in RULES_EQ_TYPE
    assert min_rule_vars <= RULES_MAX_ARG

    # select a IMO ARule with a number of variable at least min_rule_vars
    rules = [
        rule
        for rule in ALL_RULES[rule_type]
        if isinstance(rule, ARule)
        and len(rule.n_vars) >= min_rule_vars
        and (not with_hyps or len(rule.hyps) > 0)
    ]
    assert len(rules) > 0
    rule = rules[egg.rng.randint(len(rules))]

    assert complexity > 0
    eq = rule.node
    fin_hyps: List[GraphNode] = []
    substs: Dict[str, Node] = {}
    var_set = sorted(list(eq.get_vars()), reverse=True)
    while var_set:
        rhs_str = var_set.pop()
        if rhs_str in substs.keys():
            continue
        assert rhs_str not in substs.keys(), rhs_str

        lhs = egg.env.generate_expr(egg.rng.randint(complexity))
        substs[rhs_str] = lhs
    for hyp in rule.hyps:
        hyp_var_to_replace = {
            key: substs[key] for key in substs if key in hyp.get_vars()
        }
        eval_hyp = eval_assert_with_rule(
            hyp.set_vars(hyp_var_to_replace), egg.rules_a_no_hyps
        )
        if eval_hyp is None:
            fin_hyps.append(GraphHypNode(hyp.set_vars(hyp_var_to_replace)))
        elif eval_hyp == "__NUMERIC__" or isinstance(eval_hyp, ARule):
            fin_hyps.append(GraphTrueNode(hyp.set_vars(hyp_var_to_replace)))
        elif eval_hyp is False:
            print("eval_hyp is False, statement is discarded")
            invalid_hyp = True
            fin_hyps.append(GraphHypNode(hyp.set_vars(hyp_var_to_replace)))
    state = GraphAssertNode(eq, fin_hyps, rule, substs, is_true=False)
    egg.nodes = [state]
    for i, node in enumerate(egg.nodes):
        node.node_id = i
    return egg.nodes, NodeSet(rule.hyps)


def simple_generate_graph_IMO(
    egg: EquationGraphGenerator,
    min_rule_vars: int = 3,
    rule_type: str = "imo",
    with_hyps: bool = False,
    complexity: int = 2,
) -> Optional[GraphAssertNode]:
    """
    Generate an initial statement from IMO
    """
    assert min_rule_vars > 0
    assert min_rule_vars <= RULES_MAX_ARG

    # select a IMO ARule with a number of variable at least min_rule_vars
    rules = [
        rule
        for rule in ALL_RULES[rule_type]
        if isinstance(rule, ARule)
        and len(rule.n_vars) >= min_rule_vars
        and (not with_hyps or len(rule.hyps) > 0)
    ]
    assert len(rules) > 0
    rule = rules[egg.rng.randint(len(rules))]

    assert complexity > 0
    eq = rule.node
    var_set = set(eq.get_vars())
    var_hyps: Set[str] = set()
    fin_hyps: List[GraphNode] = []
    substs: Dict[str, Node] = {}

    invalid_hyp = False
    for hyp in rule.hyps:
        var_set = var_set.union(hyp.get_vars())
        var_hyps = var_hyps.union(hyp.get_vars())

    # Sort variables in alphabetical order: need to make sure that in dico_simp the substitution of a variable
    # only depends on the variable that are before in the alphabetical order
    # ex: C can depend on A and B, B can depend on A, A cannot depend on any other variable.
    # In fact there is no real need for this as long as it is coherent ex: if a variable uses C and it wasn't prescribed, it is not prescribed later on.
    var_list = sorted(var_set, reverse=True)
    while var_list:
        rhs_str = var_list.pop()
        if rhs_str in substs.keys():
            continue
        assert rhs_str not in substs.keys(), rhs_str
        lhs = egg.env.generate_expr(egg.rng.randint(complexity))
        substs[rhs_str] = lhs
    eqn = eq.set_vars(substs, mandatory=False)
    for hyp in rule.hyps:
        hyp_var_to_replace = {
            key: substs[key] for key in substs if key in hyp.get_vars()
        }
        eval_hyp = eval_assert_with_rule(
            hyp.set_vars(hyp_var_to_replace), egg.rules_a_no_hyps
        )
        if eval_hyp is None:
            fin_hyps.append(GraphHypNode(hyp.set_vars(hyp_var_to_replace)))
        elif eval_hyp == "__NUMERIC__" or isinstance(eval_hyp, ARule):
            fin_hyps.append(GraphTrueNode(hyp.set_vars(hyp_var_to_replace)))
        elif eval_hyp is False:
            print("eval_hyp is False, statement is discarded")
            invalid_hyp = True
            fin_hyps.append(GraphHypNode(hyp.set_vars(hyp_var_to_replace)))

    assert substs.keys() == rule.all_vars
    substs2 = substs
    state = GraphAssertNode(eqn, fin_hyps, rule, substs2, is_true=False)
    if invalid_hyp:
        return None

    return state


def generate_graph_IMO(
    egg: EquationGraphGenerator,
    min_rule_vars: int = 1,
    n_nodes: int = 4,
    max_trials: int = 10,
    # max_created_hyps: int = 1,
    # prob_add_hyp: float = 0.5,
    complexity: int = 2,
    with_hyps: bool = False,
    rwalk: bool = False,
    n_steps: int = 10,
    max_created_hyps: int = 3,
    prob_add_hyp: float = 0.2,
) -> Optional[Tuple[List[GraphNode], NodeSet]]:
    """
    Generate a imo type statement with hypothesis and apply post processing in the statement.
    With or without the proof included
    """
    res = simple_generate_graph_IMO(
        egg=egg,
        min_rule_vars=min_rule_vars,
        complexity=complexity,
        with_hyps=with_hyps,
    )
    if res is None:
        return None
    state = res

    if rwalk:
        eq = state.node
        state.node_id = 0

        egg.nodes = [deepcopy(state)]
        egg.prefix2id = {node.node.prefix(): i for i, node in enumerate(egg.nodes)}

        walk, created_hyps_set = egg._random_walk(
            init_eq=eq,
            n_steps=n_steps,
            max_created_hyps=max_created_hyps,
            prob_add_hyp=prob_add_hyp,
        )

        created_hyps_list = [
            GraphHypNode(created_hyp) for created_hyp in list(created_hyps_set)
        ]
        state.hyps.extend(created_hyps_list)
        state.node = walk[-1]["eq"]
        final_walk = {
            "init_eq": eq,
            "start": walk[0]["eq"],
            "end": walk[-1]["eq"],
            "steps": walk,
            "hyps": created_hyps_set,
        }
        egg.nodes, init_hyp_set = walk_to_graph(
            final_walk, egg, start_nodes=egg.nodes, first_as_hyp=False
        )
        return egg.nodes, init_hyp_set
    assert n_nodes >= 0
    assert max_trials >= 0
    # create initial hypotheses / graph
    egg.nodes = [state]
    egg.prefix2id = {node.node.prefix(): i for i, node in enumerate(egg.nodes)}
    init_hyps: List[Node] = []
    # TODO there is an issue below, change append for add node ?
    for hyp in state.hyps:
        init_hyps.append(hyp.node)
        if hyp.node.prefix() not in egg.prefix2id.keys():
            egg.add_node(GraphHypNode(hyp.node))

    egg.prefix2id = {node.node.prefix(): i for i, node in enumerate(egg.nodes)}

    egg.n_true_nodes = 0
    egg.rules_t_counts = {k: 0 for k in egg.rules_t_counts.keys()}
    egg.rules_a_counts = {k: 0 for k in egg.rules_a_counts.keys()}

    n_trials = 0
    n_fails = 0

    while len(egg.nodes) < n_nodes and n_trials < max_trials:

        rule_type = "t" if egg.rng.rand() <= egg.tf_prob else "a"
        n_trials += 1

        new_node: Optional[GraphNode] = None

        # transformation rule
        if rule_type == "t":
            t_rule, fwd = egg.sample_t_rule()
            new_node = egg.create_transformation_node(rule=t_rule, fwd=fwd)
            if new_node is not None and isinstance(new_node, GraphTransformNode):
                assert not t_rule.left.is_comp() or new_node.prefix_pos == 0

        # assertion rule
        if rule_type == "a":
            a_rule = egg.sample_a_rule()
            new_node = egg.create_assertion_node(rule=a_rule)
            if new_node is not None:
                assert isinstance(new_node, GraphAssertNode)

        # successfully applied -- update graph / hyps -- skip invalid nodes
        if new_node is not None and new_node.node.is_valid(egg.env.vtype):
            egg.add_node(new_node)
        else:
            n_fails += 1

        # if n_trials % 100 == 0:
        #     print(len(egg.nodes) - len(init_hyps), n_fails, n_trials)

    # set node IDs
    for i, node in enumerate(egg.nodes):
        node.node_id = i

    return egg.nodes, NodeSet(init_hyps)


def display(label: GraphNode):
    problem = label.node
    if isinstance(label, GraphInternalNode):
        hyps = label.hyps
        print(
            "========= IMO type label ... \n",
            problem,
            "\n ==================== Initial hypotheses",
        )
        for hyp in hyps:
            print(hyp.node, "\n")


@timeout(seconds=10)
def basic_sample(egg: EquationGraphGenerator, nodes: List[GraphNode]) -> GraphNode:
    def _contains_root_node(egg, node: GraphNode) -> bool:
        res = False
        if node.node.eq(egg.nodes[0].node):
            return True
        elif isinstance(node, GraphInternalNode):
            for child in node.hyps:
                res = True if _contains_root_node(egg, child) else res
        return res

    while True:
        node = nodes[egg.rng.randint(len(nodes))]
        if node.ntype not in ["hyp", "true"] and _contains_root_node(egg, node):
            break
    return node


if __name__ == "__main__":

    def run_imo_generator_tests(egg: EquationGraphGenerator, n_tests: int):

        print(f"============================ Test random IMO ineq generation tests ...")

        print("============================== with proof extraction ...")
        for _ in range(n_tests):
            res = generate_graph_IMO(egg, 1, complexity=1, rwalk=True)
            # node = basic_sample(egg, nodes)

            #  res = sgIMO(egg, 1, complexity=4,)
            if res is not None:
                nodes, hyps = res
                print(f"nombre de steps dans la preuve totale = {len(nodes)}")
                node = nodes[-1]
                try:

                    def _get_hyps(node, res):
                        if node.ntype == "hyp":
                            res.add(node.node)
                        if node.ntype not in ["true", "hyp"] and node.hyps is not None:
                            for hyp in node.hyps:
                                _get_hyps(hyp, res)

                    hyps = NodeSet()
                    _get_hyps(node, hyps)

                    # node.
                    from evariste.envs.eq.generation_lean import lean_proof_from_graph

                    goals_with_tactics, init_hyps, _ = extract_graph_steps(node)
                    assert isinstance(node, GraphTransformNode) or isinstance(
                        node, GraphAssertNode
                    )
                    print(
                        "\n".join(
                            lean_proof_from_graph(
                                root=node,
                                init_hyps=init_hyps,
                                goals_with_tactics=goals_with_tactics,
                                egg=egg,
                                use_implicit=True,
                                overwrite_int_matching=True,
                                vtype="real",
                            )[0]
                        )
                    )

                except MyTimeoutError as e:
                    print(
                        "=============================ERR=================================================================",
                        e,
                    )
                    continue

    SEED = 787
    print(f"ENV SEED: {SEED}")
    _, egg = build_env_egg("lean_real", seed=SEED)

    start = time.time()
    run_imo_generator_tests(egg=egg, n_tests=1)
    print(f"Ran all tests in {time.time() - start:.2f} seconds. SEED={SEED}")
