# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from evariste import json
from evariste.backward.graph import BackwardGoal
from evariste.backward.prover.bfs_proof_graph_test import my_thm
from evariste.backward.prover.bfs_prover_test import (
    my_env_expansion,
    my_failed_env_expansion,
)
from evariste.backward.prover.prior_search_prover import PriorSearchProofHandler


def test_bfs_proved():
    """
       0
    |        \
    1       (2,      3)
    x       / \      |
           4   5     ok
           |   |
          ok   6


   """
    thms = [my_thm(f"{i}") for i in range(7)]
    handler = PriorSearchProofHandler(
        goal=BackwardGoal(theorem=thms[0], label="label"),
        n_expansions_max=10,
        n_concurrent_expansions_max=100,
    )

    assert handler.is_ready()

    step_0 = handler.get_theorems_to_expand()
    assert set(step_0) == {thms[0]}
    handler.send_env_expansions(
        [my_env_expansion(thms[0], [[thms[1]], [thms[2], thms[3]]])]
    )

    step_1 = handler.get_theorems_to_expand()
    assert set(step_1) == {thms[1], thms[2]}
    handler.send_env_expansions(
        [
            my_failed_env_expansion(thms[1]),
            my_env_expansion(thms[2], [[thms[4]], [thms[5]]]),
        ]
    )
    assert not handler.done

    step_2 = handler.get_theorems_to_expand()
    assert set(step_2) == {thms[4], thms[5]}
    handler.send_env_expansions(
        [my_env_expansion(thms[4], [[]]), my_env_expansion(thms[5], [[thms[6]]])]
    )

    assert not handler.done

    step_3 = handler.get_theorems_to_expand()
    assert set(step_3) == {thms[3]}
    handler.send_env_expansions([my_env_expansion(thms[3], [[]])])

    assert handler.done
    results = handler.get_result()
    assert handler.n_expansions_sent == handler.n_expansions_received == 6
    assert handler.stats.stopped == "proved"
    assert len(handler.tree.nodes) == 7
    assert len(handler.tree.solved) == 4
    assert len(handler.tree.failed) == 1
    assert results.proof is not None
    assert handler.proof_size == 4

    json.dumps(results.proof_stats)

    # assert 0
