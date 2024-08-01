# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple, List, Dict, Union

from evariste.backward.env.equations.graph import EQTheorem, EQRuleTactic
from evariste.envs.eq.generation import (
    GraphNode,
    GraphAssertNode,
    GraphHypNode,
    GraphTransformNode,
)
from evariste.envs.eq.rules import TRule, ARule
from evariste.envs.eq.rules_lib import NAME_TO_RULE
from evariste.forward.common import GenerationHistory
from evariste.forward.fwd_eq.eq_fwd_env import EqForwardTactic


def history_to_eq_nodes(
    gen: GenerationHistory,
) -> Tuple[List[GraphNode], List[GraphNode]]:

    assert gen.goal.global_hyps is not None
    hyps: List[GraphNode] = [GraphHypNode(hyp.eq_node) for hyp in gen.goal.global_hyps]

    fwd_steps = gen.forward_steps()

    nodes: List[GraphNode] = list(hyps)

    for step in fwd_steps:
        assert isinstance(step.generated, EQTheorem)
        assert isinstance(step.tactic, EQRuleTactic)
        assert isinstance(step.fwd_tactic, EqForwardTactic)

        # retrieve step data / rule
        next_node = step.generated
        tactic = step.tactic

        assert next_node == step.fwd_tactic.next_node
        assert tactic == step.fwd_tactic.bwd_tactic

        rule = NAME_TO_RULE[tactic.label]

        # children
        children = [nodes[cix] for cix in step.children]

        # retrieve substitutions (to_fill + match)
        if tactic.rule_type == "t":
            assert type(rule) is TRule
            applied = rule.apply(next_node.eq_node, tactic.fwd, tactic.prefix_pos)
        else:
            assert type(rule) is ARule
            applied = rule.apply(next_node.eq_node)
        match = applied["match"]
        substs = {k: v for k, v in tactic.to_fill.items()}
        assert not any(k in substs for k in match.keys())
        substs.update(match)

        node: GraphNode
        # build GraphNode. some children coming from True nodes may be missing
        if tactic.rule_type == "t":
            assert type(rule) is TRule
            node = GraphTransformNode(
                node=next_node.eq_node,
                hyps=children,
                rule=rule,
                fwd=tactic.fwd,  # TODO: careful about direction
                prefix_pos=tactic.prefix_pos,
                substs=substs,
                missing_true=True,
                is_true=False,
            )
        elif tactic.rule_type == "a":
            assert type(rule) is ARule
            node = GraphAssertNode(
                node=next_node.eq_node,
                hyps=children,
                rule=rule,
                substs=substs,
                missing_true=True,
                is_true=False,
            )
        else:
            raise Exception(f"Unexpected rule type: {tactic.rule_type}")

        assert node.node.eq(step.generated.eq_node)
        nodes.append(node)

    # index nodes (required by extract_graph_steps)
    for i, node in enumerate(nodes):
        node.node_id = i

    # sanity check
    n_hyps = len(hyps)
    assert all((node.ntype == "hyp") == (i < n_hyps) for i, node in enumerate(nodes))

    return nodes, hyps
