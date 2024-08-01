# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List

from evariste.backward.env.equations import EQTheorem
from evariste.forward.common import GenerationHistory
from evariste.forward.fwd_eq.eq_helpers import history_to_eq_nodes


def test_history_to_eq_nodes(eq_histories: List[GenerationHistory]):
    for history in eq_histories:
        nodes, hyps = history_to_eq_nodes(history)

        assert history.goal.global_hyps is not None
        assert len(hyps) == len(history.goal.global_hyps)

        fwd_graph = history.forward_graph()
        obtained_thms = fwd_graph.global_hyps_and_generated
        assert len(nodes) == len(obtained_thms)

        for thm, node in zip(obtained_thms, nodes):
            assert isinstance(thm, EQTheorem)
            assert thm.eq_node.eq(node.node)


def test_proof_nodes_for_eq(eq_histories: List[GenerationHistory]):
    for history in eq_histories:
        proof_nodes = history.proof_nodes()
        assert all(not n.is_hyp for n in proof_nodes)
        assert len(proof_nodes) == len(history.forward_steps())
