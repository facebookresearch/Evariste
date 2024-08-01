# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional, Union, Tuple, List, Dict
from evariste.envs.eq.generation import (
    EquationGraphStats,
    GraphAssertNode,
    GraphNode,
    GraphTransformNode,
)
import numpy as np

from evariste.backward.graph import BackwardGoal
from evariste.backward.env.equations.graph import EQTheorem, EQRuleTactic
from evariste.forward.env_specifics.generation_stats import GenerationStats
from evariste.forward.fwd_eq.eq_fwd_env import EqForwardTactic
from evariste.forward.fwd_eq.history_to_eq_nodes import history_to_eq_nodes
from evariste.metrics import Avg
from evariste.forward.common import GenerationHistory, ForwardStep
from evariste.forward.fwd_eq.eq_env_helper import EQFwdEnvHelper


@dataclass
class GenerationEvalStats:

    n_statements: int = 0
    n_statements_distinct: int = 0
    n_discarded: int = 0

    rule_coverage: float = 0
    rule_entropy: float = 0

    avg_proof_n_rules: Avg = field(default_factory=lambda: Avg())

    selected_node_avgs: Dict[str, Avg] = field(
        default_factory=lambda: {
            k: Avg()
            for k in [
                "depth",
                "size",
                "size_by_depth",
                "statement_n_tok",
                "rule_score",
                "rule_div_score",
            ]
        }
    )

    node_avg_avgs: Dict[str, Avg] = field(
        default_factory=lambda: {
            k: Avg()
            for k in [
                "depth",
                "size",
                "size_by_depth",
                "statement_n_tok",
                "rule_score",
                "rule_div_score",
            ]
        }
    )

    avg_n_forward_errors: Avg = field(default_factory=lambda: Avg())
    avg_n_transformations: Avg = field(default_factory=lambda: Avg())
    avg_n_hypotheses: Avg = field(default_factory=lambda: Avg())
    avg_n_assertions: Avg = field(default_factory=lambda: Avg())
    avg_n_trees: Avg = field(default_factory=lambda: Avg())
    avg_max_depth: Avg = field(default_factory=lambda: Avg())
    avg_max_size: Avg = field(default_factory=lambda: Avg())
    avg_max_size_by_depth: Avg = field(default_factory=lambda: Avg())
    avg_max_rule_score: Avg = field(default_factory=lambda: Avg())
    avg_max_rule_div_score: Avg = field(default_factory=lambda: Avg())

    def to_dict(self) -> Dict[str, Union[int, float]]:
        stats = asdict(self)
        del stats["selected_node_avgs"]
        del stats["node_avg_avgs"]
        for k, v in self.selected_node_avgs.items():
            stats[f"avg_selected_node_{k}"] = v
        for k, v in self.node_avg_avgs.items():
            stats[f"avg_avg_node_{k}"] = v
        for k, v in stats.items():
            if isinstance(v, Avg):
                stats[k] = v.stats_and_reset()
        return stats


def evaluate_diversity(
    sequences: List[List[str]], rule2id: Dict[str, int]
) -> Tuple[float, float]:
    """
    Evaluate the diversity of generated sequences.
    """
    counts = np.zeros((len(rule2id),), dtype=np.float64)
    for seq in sequences:
        for rule in seq:
            counts[rule2id[rule]] += 1

    # usage
    usage = (counts != 0).sum() / len(counts)

    # compute entropy
    p = counts / counts.sum()
    p[p == 0] = 1
    entropy = -(np.log(p) * p).sum()

    return float(usage), float(entropy)


def evaluate_eq_generations(
    histories: List[GenerationHistory],
    env_helper: EQFwdEnvHelper,
    selected_nodes: Optional[List[Optional[GraphNode]]] = None,
) -> Dict[str, Union[int, float]]:

    rules_t = env_helper.eq_data_env.rules_t
    rules_a = env_helper.eq_data_env.rules_a

    rules = sorted(
        [f"{r.name}__{fwd}" for fwd in [True, False] for r in rules_t]
        + [r.name for r in rules_a]
    )
    rule2id = {r: i for i, r in enumerate(rules)}

    all_rules: List[List[str]] = []
    statements: List[str] = []
    stats = GenerationEvalStats()

    if selected_nodes is None:
        selected_nodes = [None] * len(histories)

    assert len(selected_nodes) == len(histories), (len(selected_nodes), len(histories))

    for gen, selected_node in zip(histories, selected_nodes):
        assert isinstance(selected_node, GraphNode)
        count_types = {x: 0 for x in ["hyp", "transform", "assert"]}

        # retrieve steps
        steps = gen.forward_steps()
        if len(steps) == 0:
            stats.n_discarded += 1
            continue

        stats.n_statements += 1
        statement = selected_node.node.prefix()
        statements.append(statement)

        # build graph
        nodes, hyps = history_to_eq_nodes(gen)
        graph = EquationGraphStats(nodes, env_helper.eq_data_env.rules)

        # stats concerning terminal nodes
        stats.avg_n_trees.act(len(graph.trees))
        stats.avg_max_depth.act(graph.max_depth)
        stats.avg_max_size.act(graph.max_size)
        stats.avg_max_size_by_depth.act(graph.max_size_by_depth)
        stats.avg_max_rule_score.act(graph.max_rule_score)
        stats.avg_max_rule_div_score.act(graph.max_rule_div_score)

        # dict to collect node averages
        node_avgs: Dict[str, Avg] = {
            k: Avg()
            for k in [
                "depth",
                "size",
                "size_by_depth",
                "statement_n_tok",
                "rule_score",
                "rule_div_score",
            ]
        }

        # extract applied rules, with directions
        proof_rules: List[str] = []

        # stats on rules
        rule_counts = {r.name: 0 for r in env_helper.eq_data_env.rules}

        # collect node-wise stats
        for i, node in enumerate(nodes):
            if i < len(hyps):
                assert node.ntype == "hyp"
                count_types[node.ntype] += 1
                continue
            elif isinstance(node, GraphTransformNode):
                count_types[node.ntype] += 1
                proof_rules.append(f"{node.rule.name}__{node.fwd}")
                rule_counts[node.rule.name] += 1
            elif isinstance(node, GraphAssertNode):
                count_types[node.ntype] += 1
                proof_rules.append(node.rule.name)
                rule_counts[node.rule.name] += 1
            else:
                raise Exception(f"Unexpected node type: {node.ntype}")

            node_stats = graph.node_stats(node)
            for k, v in node_stats.items():
                node_avgs[k].act(v)

        # compute and store node averages
        for k, avg in node_avgs.items():
            stats.node_avg_avgs[k].act(avg.stats_and_reset())

        # other cumulated stats retrieved from all nodes
        assert len(proof_rules) == len(nodes) - len(hyps)
        assert all(rule in rule2id for rule in proof_rules)
        all_rules.append(proof_rules)
        stats.avg_proof_n_rules.act(len(proof_rules))
        stats.avg_n_forward_errors.act(len(gen.stack) - len(gen.forward_steps()))
        stats.avg_n_transformations.act(count_types["transform"])
        stats.avg_n_assertions.act(count_types["assert"])
        stats.avg_n_hypotheses.act(count_types["hyp"])

        # select node
        if selected_node is None:
            # compute stats for last node
            selected_node = nodes[-1]

        # stats on selected node
        node_stats = graph.node_stats(selected_node)
        for k, v in node_stats.items():
            stats.selected_node_avgs[k].act(v)

    stats.rule_coverage, stats.rule_entropy = evaluate_diversity(all_rules, rule2id)
    stats.n_statements_distinct = len(set(statements))

    return stats.to_dict()


def evaluate_eq_generation(
    goal: BackwardGoal,
    gen: GenerationHistory,
    proved: bool,
    env_helper: EQFwdEnvHelper,
) -> Dict[str, Union[int, float]]:
    """Some refactor still needed with evaluate_eq_generations.

    :param goal: selected node in gen
    :param gen: the generation history
    :param proved: whether goal was proved
    :param env_helper: an adequate eq_env_helper
    """

    # build graph
    nodes, _hyps = history_to_eq_nodes(gen)
    selected_node = None
    for node in nodes:
        assert isinstance(goal.theorem, EQTheorem)
        if goal.theorem.eq_node.prefix() == node.node.prefix():
            selected_node = node
            break
    if not selected_node:
        raise RuntimeError("Did not find node corresponding to goal.")

    stats: Dict[str, Union[int, float]] = evaluate_eq_generations(
        [gen], env_helper, [selected_node]
    )
    stats["proved"] = int(proved)

    # stats on bwd tactics
    step_avgs = {"bwd_pos": Avg(), "bwd_fill_n_tok": Avg(), "bwd_n_fill": Avg()}
    max_n_fill = 0
    for step in gen.forward_steps():
        assert isinstance(step.fwd_tactic, EqForwardTactic)
        bwd_tac = step.fwd_tactic.bwd_tactic
        assert isinstance(bwd_tac, EQRuleTactic)
        if bwd_tac.rule_type == "t":
            step_avgs["bwd_pos"].act(bwd_tac.prefix_pos)
        step_avgs["bwd_fill_n_tok"].act(
            sum(len(x) for x in bwd_tac.to_fill_prefix.values())
        )
        n_fill = len(bwd_tac.to_fill_prefix.values())
        step_avgs["bwd_n_fill"].act(n_fill)
        max_n_fill = max(max_n_fill, n_fill)

    for k, avg in step_avgs.items():
        stats["avg_" + k] = avg.stats_and_reset()
    stats["bwd_max_n_fill"] = max_n_fill

    return stats
