# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Dict
from collections import Counter
import time
import numpy as np
import pytest

from evariste.envs.eq.graph import NodeSet
from evariste.envs.eq.rules import eval_assert
from evariste.envs.eq.env import EqGraphSamplerParams
from evariste.envs.eq.generation import (
    EquationGraphGenerator,
    GraphInternalNode,
    build_env_egg,
    walk_to_graph,
    extract_walk_steps,
    check_bwd_proof,
    GraphNode,
    extract_graph_steps,
    EquationGraphSampler,
)
from evariste.envs.eq.generation import (
    GraphTransformNode,
    GraphHypNode,
    GraphTrueNode,
)
from evariste.envs.eq.graph import INode, VNode, NodeSet
from evariste.envs.eq.rules import TRule

A = VNode("A")
B = VNode("B")
x = VNode("x")
y = VNode("y")
z = VNode("z")
ZERO = INode(0)


def test_graph_hyp_node():
    node = x == ZERO
    graph_hyp_node = GraphHypNode(node)
    assert graph_hyp_node.depth == 0
    assert graph_hyp_node.used_hyps == NodeSet()
    assert graph_hyp_node.descendants == set()


def test_graph_true_node():
    node = x == x
    graph_hyp_node = GraphTrueNode(node)
    assert graph_hyp_node.depth == 0
    assert graph_hyp_node.used_hyps == NodeSet()
    assert graph_hyp_node.descendants == set()


def test_graph_transform_node():
    node = (x + y) == z
    transformed = (y + x) == z
    trule = TRule(A + B, B + A)
    fwd = True
    prefix_pos = 1
    substs = {"A": x, "B": y}
    missing_true = True

    graph_transform_node = GraphTransformNode(
        transformed, [GraphHypNode(node)], trule, fwd, prefix_pos, substs, missing_true
    )
    assert graph_transform_node.node.eq(transformed)
    assert len(graph_transform_node.hyps) == 1 and graph_transform_node.hyps[0].node.eq(
        node
    )
    assert graph_transform_node.rule == trule
    assert graph_transform_node.fwd == fwd
    assert graph_transform_node.prefix_pos == prefix_pos
    assert graph_transform_node.substs == substs
    assert graph_transform_node.depth == 1
    assert graph_transform_node.used_hyps == NodeSet()
    assert graph_transform_node.descendants == {("== add y x z", 1)}


def test_all():
    #
    # test generate hypotheses
    #
    def test_generate_hyps(egg: EquationGraphGenerator, n_hyps: int):
        print(f"========== Test generating hypotheses (n_hyps={n_hyps}) ...")
        for i in range(n_hyps):
            hyp = egg.generate_hypothesis(max_ops=i % 4, op_values=None)
            print(f"\t{hyp}")
        print("")

    #
    # test random walk
    #
    def test_random_walk_egg(
        egg: EquationGraphGenerator,
        bidirectional: bool,
        n_steps: int,
        bwd_test: bool,
        print_stats: bool,
    ):
        print(
            f"==================== Test random walk (bidirectional={bidirectional}) ..."
        )
        walk = egg.random_walk(
            bidirectional=bidirectional,
            n_steps=n_steps,
            n_init_ops=2,
            max_created_hyps=10,
            prob_add_hyp=0.5,
        )
        steps = walk["steps"]
        start = walk["start"]
        end = walk["end"]
        hyps = walk["hyps"]

        # check that each node is valid
        for hyp in hyps:
            assert hyp.is_valid()
        for eq in steps:
            assert eq["eq"].is_valid(), eq["eq"]

        print("Initial equation:")
        print(f'\t{walk["init_eq"]}')

        if hyps:
            print(f"Hypotheses ({len(hyps)}):")
            for hyp in hyps:
                print(f"\t{hyp}")

        print(f"Start: {start}")
        print(f"End  : {end}")
        print("")

        rule_counts: Dict = dict(egg.rules_t_counts)
        rule_counts.update(egg.rules_a_counts)
        if print_stats:
            print("\n========== Used rules")
            for k, count in sorted(
                list(rule_counts.items()), key=lambda x: x[1], reverse=True
            ):
                if count == 0:
                    continue
                rtype = "transform" if type(k) is tuple else "assertion"
                print(f"{count:>3} {rtype:>15}   {k}")

        # sanity check
        eq = steps[0]["eq"]
        for step in steps[1:]:
            applied = egg.env.apply_t_rule(
                eq,
                step["rule"],
                step["fwd"],
                step["prefix_pos"],
                to_fill=step["to_fill"],
            )
            eq = applied["eq"]
            assert eq.eq(step["eq"])
            for hyp in applied["hyps"]:
                assert (
                    hyp in hyps or eval_assert(hyp, egg.rules_a, egg.env.vtype) is True
                )

        # extract steps / check backward proof
        if bwd_test:

            goals_with_tactics, hyps = extract_walk_steps(walk)
            assert all(
                x is not None and y is not None for x, y, _ in goals_with_tactics
            )

            # check the proof
            check_bwd_proof(
                egg.env, egg.rules_a, goals_with_tactics, hyps, can_loop=True
            )

            # check that the proof is invalid if we remove one node. this will not
            # fail if the first tactic does not affect the goal (e.g. A * B == B * A
            # for identical values of A and B).
            if len(goals_with_tactics) > 1 and goals_with_tactics[0][0].eq_node.ne(
                goals_with_tactics[0][0].eq_hyps[0]
            ):
                try:
                    check_bwd_proof(
                        egg.env,
                        egg.rules_a,
                        goals_with_tactics[1:],
                        hyps,
                        can_loop=True,
                    )
                except Exception:
                    pass
                else:
                    raise Exception("This should have failed!")

            print("Backward test OK")

    #
    # random graph
    #
    def print_node_with_children(node: GraphNode, depth=0):
        prefix = "\t" * depth
        ntype = node.ntype[:3].upper()
        print(f"{prefix}{ntype} | {node.node}")
        # TODO fix: the line below should not be necessary,
        # "if node.hyps is not None:" should have been enough
        if isinstance(node, GraphInternalNode) and node.hyps is not None:
            for hyp in node.hyps:
                print_node_with_children(hyp, depth=depth + 1)

    def print_node(node: GraphNode, print_children: bool):
        def _get_hyps(node: GraphNode, res: NodeSet):
            if node.ntype == "hyp":
                res.add(node.node)
            if isinstance(node, GraphInternalNode) and node.hyps is not None:
                for hyp in node.hyps:
                    _get_hyps(hyp, res)

        hyps = NodeSet()
        _get_hyps(node, hyps)

        prefix = (
            f"Depth {node.depth} -- Size {node.size} -- "
            f"S/D {node.size / max(node.depth, 1)} -- "
            f"Avg. children {node.avg_children_per_node}"
        )
        if len(hyps) == 0:
            print(f"{prefix} -- 0 hypothesis")
        else:
            print(f"{prefix} -- {len(hyps)} hypotheses:")
            for hyp in hyps:
                print(f"\t{hyp}")

        if print_children:
            print_node_with_children(node)
        else:
            print(node.node)

    def test_valid_graph(
        nodes: List[GraphNode], init_hyps: NodeSet, n_bwd_tests: int, print_stats: bool
    ):

        if len(nodes) == 0:
            print(
                f"Did not generate any node! {len(init_hyps)} initial hypotheses: "
                f"{[hyp.prefix() for hyp in init_hyps]}"
            )
            return

        # check that each node is valid
        for hyp in init_hyps:
            assert hyp.is_valid()
        for node in nodes:
            assert node.node.is_valid()

        # check that each node does not appear twice in the graph,
        # and that each hypothesis is unique
        assert len(set([node.node.prefix() for node in nodes])) == len(nodes)
        assert len(init_hyps) == len(set(hyp.prefix() for hyp in init_hyps))

        # print initial hypotheses
        if init_hyps:
            print("\n========== Initial hypotheses")
            for hyp in init_hyps:
                print(f"\t{hyp}")
            print("")

        # # extract steps / check backward proof
        for _ in range(n_bwd_tests):

            # sample a node which is not an hypothesis or trivially true
            while True:
                node_id = egg.rng.randint(len(nodes))
                if nodes[node_id].ntype not in ["hyp", "true"]:
                    break

            # extract the proof steps of this node
            goals_with_tactics, hyps, _ = extract_graph_steps(nodes[node_id])
            assert all(
                x is not None and y is not None for x, y, _ in goals_with_tactics
            )

            # check that each node does not appear twice in the graph,
            # and that each hypothesis is unique
            assert len(
                set([goal.eq_node.prefix() for goal, _, _ in goals_with_tactics])
            ) == len(goals_with_tactics)
            assert len(hyps) == len(set(hyp.prefix() for hyp in hyps))

            # check the proof
            check_bwd_proof(
                egg.env, egg.rules_a, goals_with_tactics, hyps, can_loop=False
            )

            # check that the proof is invalid if we remove one node
            if len(goals_with_tactics) > 1:
                try:
                    check_bwd_proof(
                        egg.env,
                        egg.rules_a,
                        goals_with_tactics[1:],
                        hyps,
                        can_loop=False,
                    )
                except Exception:
                    pass
                else:
                    raise Exception("This should have failed!")
        if n_bwd_tests > 0:
            print(f"Backward tests OK")

        # rule counts
        rule_counts: Dict = dict(egg.rules_t_counts)
        rule_counts.update(egg.rules_a_counts)

        if print_stats:
            print("\n========== Used rules")
            for k, count in sorted(
                list(rule_counts.items()), key=lambda x: x[1], reverse=True
            ):
                if count == 0:
                    continue
                rtype = "transform" if type(k) is tuple else "assertion"
                print(f"{count:>3} {rtype:>15}   {k}")

            print("\n========== Types of nodes")
            node_types = Counter([node.ntype for node in nodes])
            for k, v in node_types.items():  # type: ignore
                print(k, v)

            print("\n========== Types of operators")
            node_types = Counter([node.node.value for node in nodes])
            for k, v in node_types.items():  # type: ignore
                print(k, v)

            print("\n========== Node evals")
            n_none = sum(
                [
                    int(eval_assert(n.node, egg.rules_a, egg.env.vtype) is None)
                    for n in nodes
                ]
            )
            n_true = sum(
                [
                    int(eval_assert(n.node, egg.rules_a, egg.env.vtype) is True)
                    for n in nodes
                ]
            )
            assert (
                sum(
                    [
                        int(eval_assert(n.node, egg.rules_a, egg.env.vtype) is False)
                        for n in nodes
                    ]
                )
                == 0
            )
            print(f"n_none = {n_none}")
            print(f"n_true = {n_true}")

            print("\n========== Random nodes")
            for i in sorted(np.random.permutation(len(nodes))[:5]):
                print(
                    f"Node {i} -- Size {nodes[i].size} -- Avg. children {nodes[i].avg_children_per_node}"
                )
                print_node(nodes[i], print_children=True)
                print("=====")

            print("\n========== Statistics")

            depths = [node.depth for node in nodes]
            total_connect = []
            for node in nodes:
                connect = 0
                for nodule in nodes:
                    if not isinstance(nodule, GraphInternalNode) or nodule.hyps is None:
                        continue
                    if node.node in NodeSet([hyp.node for hyp in nodule.hyps]):
                        connect += 1
                total_connect.append(connect)

            print("\nDepth")
            print(f"\tmean: {np.mean(depths)}")
            print(f"\tmin : {np.min(depths)}")
            print(f"\tmax : {np.max(depths)}")

            print("\nConnectivity:")
            print(f"\tmean: {np.mean(total_connect)}")
            print(f"\tmin : {np.min(total_connect)}")
            print(f"\tmax : {np.max(total_connect)}")

    def test_random_graph_egg(
        egg: EquationGraphGenerator,
        n_nodes: int,
        max_trials: int,
        max_init_hyps: int,
        n_bwd_tests: int,
        print_stats: bool,
        print_samples: int,
    ):
        print(f"==================== Test graph generator ...")
        nodes, init_hyps = egg.generate_graph(
            n_nodes=n_nodes,
            max_trials=max_trials,
            n_init_hyps=egg.rng.randint(0, max_init_hyps + 1),
        )

        test_valid_graph(
            nodes=nodes,
            init_hyps=init_hyps,
            n_bwd_tests=n_bwd_tests,
            print_stats=print_stats,
        )

        if print_samples > 0:

            def _print_samples(egg: EquationGraphGenerator, sampling_params):
                sampler = EquationGraphSampler(egg.rng, sampling_params)
                node_ids, node_scores = sampler.sample(egg, n_samples=print_samples)
                for node_id, node_score in zip(node_ids, node_scores):
                    node = nodes[node_id]
                    print(f"Node {node_id} ({node_score:.3f})")
                    print_node(node, print_children=True)
                    print("=====")

            sampler_params = [
                EqGraphSamplerParams(),
                # EqGraphSamplerParams(rule_weight=3, max_prefix_len=50),
                # EqGraphSamplerParams(max_prefix_len=50),
                # EqGraphSamplerParams(depth_weight=0, prefix_len_weight=14),
                # EqGraphSamplerParams(depth_weight=1, prefix_len_weight=-1),
                EqGraphSamplerParams(sd_ratio_weight=1, prefix_len_weight=0),
                # EqGraphSamplerParams(
                #     depth_weight=1,
                #     size_weight=1,
                #     sd_ratio_weight=1,
                #     prefix_len_weight=-0.5,
                #     rule_weight=0,
                #     # max_prefix_len=50,
                # ),
            ]

            for i, params in enumerate(sampler_params):
                print(f"\n========== Best nodes ({i})")
                _print_samples(egg, params)

    def run_generate_hyps_tests(egg: EquationGraphGenerator, n_tests: int):
        print(f"========= Test generate random hyps ...")
        for _ in range(n_tests):
            test_generate_hyps(egg=egg, n_hyps=egg.rng.randint(1, 10))
        print("OK")

    def run_random_walk_tests(egg: EquationGraphGenerator, n_tests: int):
        print(f"========= Test random walk ...")
        for _ in range(n_tests):
            test_random_walk_egg(
                egg=egg,
                bidirectional=False,
                n_steps=50,
                bwd_test=True,
                print_stats=False,
            )
            test_random_walk_egg(
                egg=egg,
                bidirectional=True,
                n_steps=50,
                bwd_test=True,
                print_stats=False,
            )
        print("OK")

    def run_random_graph_tests(egg: EquationGraphGenerator, n_tests: int):
        print(f"========= Test random graph ...")
        for _ in range(n_tests):
            test_random_graph_egg(
                egg=egg,
                n_nodes=500,
                max_trials=1000,
                max_init_hyps=10,
                n_bwd_tests=100,
                print_stats=False,
                print_samples=0,
            )
        print("OK")

    def run_random_walk2graph_tests(egg: EquationGraphGenerator, n_tests: int):
        print(f"========= Test random walk2graph tests ...")
        for bidirectional in [False, True]:
            for _ in range(n_tests):
                walk = egg.random_walk(
                    bidirectional=bidirectional,
                    n_steps=50,
                    n_init_ops=2,
                    max_created_hyps=10,
                    prob_add_hyp=0.5,
                )
                print("Converting to graph ...")
                graph_nodes, graph_hyps = walk_to_graph(walk=walk, egg=egg)
                print("Testing valid graph ...")
                test_valid_graph(
                    nodes=graph_nodes,
                    init_hyps=graph_hyps,
                    n_bwd_tests=100,
                    print_stats=False,
                )

        print("OK")

    # SEED = np.random.randint(1000)
    # SEED = 627
    # SEED = 868
    # SEED = 673
    # SEED = 534
    # SEED = 390
    # SEED = 831
    SEED = 787
    print(f"ENV SEED: {SEED}")
    _, egg = build_env_egg("default", seed=SEED)

    start = time.time()
    run_generate_hyps_tests(egg=egg, n_tests=10)
    run_random_walk_tests(egg=egg, n_tests=10)
    run_random_graph_tests(egg=egg, n_tests=10)
    run_random_walk2graph_tests(egg=egg, n_tests=10)
    print(f"Ran all tests in {time.time() - start:.2f} seconds. SEED={SEED}")
