# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple, Iterator, List

import pytest

from evariste.forward.fwd_eq.gen.tactics import EqGenForwardTactic


from evariste.datasets.equations import (
    ConfStore,
    EquationsDatasetConf,
)
from evariste.forward.common import ForwardGoal
from evariste.forward.core.forward_policy import BatchConfig
from evariste.forward.core.generation_errors import GenerationError
from evariste.forward.fwd_eq.gen.proof_search import EqGenProofSearch
from evariste.forward.fwd_eq.gen.specifics import eq_gen_env_specifics_for_gen
from evariste.forward.forward_prover import ForwardProver, ProverConfig, SearchConfig
from evariste.model.transformer_args import DecodingParams
from evariste.envs.eq.generation import GraphInternalNode, GraphHypNode


@pytest.fixture()
def eq_conf():
    eq_conf: EquationsDatasetConf = ConfStore["eq_dataset_lean"]
    eq_conf.max_init_hyps = 5
    eq_conf.bias_rules = 0
    eq_conf.gen_type = "graph"  # complex doesn't work in CI due to missing simps
    # eq_conf.prob_add_hyp = 0
    yield eq_conf


@pytest.fixture()
def gens(eq_conf: EquationsDatasetConf):
    n_gens = 50
    max_node = 50
    max_generations = 200
    # for large tests
    # n_gens = 5000
    # max_node = 500
    # max_generations = 1000
    forward_prover = ForwardProver.from_random_args(
        ProverConfig(
            # relevant
            # max_nodes = n_nodes, max_generations = max_trials, max_cons_inv does not exist in old generate
            SearchConfig(
                max_nodes=max_node,
                max_generations=max_generations,
                max_cons_inv_allowed=1 << 32,  # unused
                n_simultaneous_proofs=1,
            ),
            DecodingParams(),  # unused
            BatchConfig(max_batch_mem=1 << 32),  # unused
            name="model_free_gen",
        ),
        eq_gen_env_specifics_for_gen(eq_conf, seed=1),
        eq_conf,
    )

    def dummy_goal_stream(n_goals: int) -> Iterator[Tuple[int, ForwardGoal]]:
        for i in range(n_goals):
            yield i, ForwardGoal(thm=None, label=f"unused_{i}")

    goal_stream = dummy_goal_stream(n_gens)
    proof_stream = forward_prover.generate_proofs(goal_stream)

    def filtered():
        for _i, res in proof_stream:
            if isinstance(res, GenerationError):
                continue
            assert isinstance(res, EqGenProofSearch)
            if len(res.steps) == 0:
                continue
            yield res

    yield filtered()


class TestFwdGen:
    def test_tok_detok(
        self, eq_conf: EquationsDatasetConf, gens: Iterator[EqGenProofSearch]
    ):
        for gen in gens:
            # first check that extracting tactics and re-running works
            g = gen.init_graph.cut(0)
            for i, (graph, tac) in enumerate(gen.graphs_and_tactics()):
                tac.reset()
                assert g == graph, f"{type(tac)} -- {i}"
                g = tac.apply(g, eq_conf)
            assert g == gen.next_graph

            g = gen.init_graph.cut(0)
            for graph, tac in gen.graphs_and_tactics():
                assert g == graph
                tokenized_full = ["", *tac.tokenize(g), ""]
                new_tac_full = EqGenForwardTactic.detokenize(tokenized_full, g)
                g = new_tac_full.apply(g, eq_conf)

            assert g == gen.next_graph

    def test_sample_graphs(
        self, eq_conf: EquationsDatasetConf, gens: Iterator[EqGenProofSearch]
    ):
        for g_id, gen in enumerate(gens):
            for i in range(len(gen.next_graph.nodes)):
                if not isinstance(gen.next_graph.nodes[i], GraphInternalNode):
                    continue
                g_and_t = gen.get_forward_steps(i)
                prev_apply = None
                for j, (g, t) in enumerate(g_and_t):
                    if prev_apply is not None:
                        gn = [n for n in g.nodes if not isinstance(n, GraphHypNode)]
                        pan = [
                            n
                            for n in prev_apply.nodes
                            if not isinstance(n, GraphHypNode)
                        ]
                        try:
                            assert len(gn) == len(pan)
                            ok = True
                            for a, b in zip(gn, pan):
                                assert a.node.eq(b.node)
                        except Exception:
                            import pickle

                            pickle.dump(
                                (gen, i, j), open("repro.pkl", "wb"),
                            )
                            raise
                    assert len(g.nodes) == len(g.prefix2id), (
                        {x.node.prefix() for x in g.nodes},
                        set(g.prefix2id.keys()),
                    )
                    tokenized = ["", *t.tokenize(g), ""]
                    new_tac_full = EqGenForwardTactic.detokenize(tokenized, g)
                    prev_apply = new_tac_full.apply(g, eq_conf)

                assert prev_apply is not None
                assert prev_apply.nodes[-1].node.eq(gen.next_graph.nodes[i].node)
