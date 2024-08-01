# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List

from evariste.backward.env.core import EnvExpansion
from evariste.backward.graph import Theorem, Tactic, BackwardGoal
from evariste.backward.prover.bfs_proof_graph_test import my_tac, my_thm
from evariste.backward.prover.bfs_prover import BFSProofHandler


def my_env_expansion(thm: Theorem, sub_goals_groups: List[List[Theorem]]):
    for group in sub_goals_groups:
        assert isinstance(group, list)
        for sg in group:
            assert isinstance(sg, Theorem)
    n_groups = len(sub_goals_groups)
    return EnvExpansion(
        theorem=thm,
        exp_duration=0.0,
        gpu_duration=0.0,
        env_durations=[0.1 for _ in range(n_groups)],
        tactics=[my_tac() for _ in range(n_groups)],
        child_for_tac=sub_goals_groups,
        priors=[0.1 ** n for n in range(n_groups)],
        log_critic=-0.1,
        effects=[],
    )


def my_failed_env_expansion(thm: Theorem):
    return EnvExpansion(
        theorem=thm,
        exp_duration=0.0,
        gpu_duration=0.0,
        env_durations=[0.0],
        error="failed",
        effects=[],
    )


def test_bfs_proved():
    """
       0
    |        \
    1       (2,   3)
    x       /      |    \
           4   (2, 4)    5
           |             |
          ok             6
                         ? (not expanded)

   """
    thms = [my_thm(f"{i}") for i in range(7)]
    handler = BFSProofHandler(
        goal=BackwardGoal(theorem=thms[0], label="label"), n_expansions_max=10
    )

    assert handler.is_ready()

    step_0 = handler.get_theorems_to_expand()
    assert step_0 == [thms[0]]
    handler.send_env_expansions(
        [my_env_expansion(thms[0], [[thms[1]], [thms[2], thms[3]]])]
    )

    step_1 = handler.get_theorems_to_expand()
    assert step_1 == [thms[1], thms[2], thms[3]]
    handler.send_env_expansions(
        [
            my_failed_env_expansion(thms[1]),
            my_env_expansion(thms[3], [[thms[2], thms[4]], [thms[5]]]),
            my_env_expansion(thms[2], [[thms[4]]]),
        ]
    )
    assert not handler.done

    step_2 = handler.get_theorems_to_expand()
    assert step_2 == [thms[4], thms[5]]
    handler.send_env_expansions(
        [my_env_expansion(thms[4], [[]]), my_env_expansion(thms[5], [[thms[6]]])]
    )

    assert handler.done
    results = handler.get_result()
    assert handler.stats.n_expansions_sent == handler.stats.n_expansions_received == 6
    assert handler.stats.stopped == "proved"
    assert len(handler.graph.th_to_node) == 7
    assert len(handler.graph.failed_leafs) == 1
    assert len(handler.graph.proved_leafs) == 1
    assert len(handler.graph.leafs) == 1
    assert results.proof is not None


def test_bfs_low_n_expansions_max():
    """
       0
    |        \
    1
    x      (2,   3)
            /      \
           ok       2

   """
    thms = [my_thm(f"{i}") for i in range(4)]
    handler = BFSProofHandler(
        goal=BackwardGoal(theorem=thms[0], label="label"), n_expansions_max=3
    )

    assert handler.is_ready()

    step_0 = handler.get_theorems_to_expand()
    assert step_0 == [thms[0]]
    handler.send_env_expansions(
        [my_env_expansion(thms[0], [[thms[1]], [thms[2], thms[3]]])]
    )

    step_1 = handler.get_theorems_to_expand()
    assert step_1 == [thms[1], thms[2]]
    handler.send_env_expansions(
        [my_failed_env_expansion(thms[1]), my_env_expansion(thms[2], [[]])]
    )

    assert handler.done

    assert handler.stats.n_expansions_sent == handler.stats.n_expansions_received == 3
    assert handler.stats.stopped == "n_expansions_max"

    assert len(handler.graph.th_to_node) == 4
    assert len(handler.graph.failed_leafs) == 1
    assert len(handler.graph.proved_leafs) == 1
    assert len(handler.graph.leafs) == 1

    results = handler.get_result()
    assert results.proof is None


def test_bfs_empty_queue():
    """
       0
    |        \
    1      (2,   3)
    |       |      \
    2       1        3

   """
    thms = [my_thm(f"{i}") for i in range(4)]
    handler = BFSProofHandler(
        goal=BackwardGoal(theorem=thms[0], label="label"), n_expansions_max=100,
    )

    assert handler.is_ready()

    step_0 = handler.get_theorems_to_expand()
    assert step_0 == [thms[0]]
    handler.send_env_expansions(
        [my_env_expansion(thms[0], [[thms[1]], [thms[2], thms[3]]])]
    )

    step_1 = handler.get_theorems_to_expand()
    assert step_1 == [thms[1], thms[2], thms[3]]
    handler.send_env_expansions(
        [
            my_env_expansion(thms[1], [[thms[2]]]),
            my_env_expansion(thms[2], [[thms[1]]]),
            my_env_expansion(thms[3], [[thms[3]]]),
        ]
    )

    assert handler.done
    assert len(handler.graph.th_to_node) == 4
    assert len(handler.graph.failed_leafs) == 0
    assert len(handler.graph.proved_leafs) == 0
    assert len(handler.graph.leafs) == 0
    assert handler.stats.n_expansions_sent == handler.stats.n_expansions_received == 4
    assert handler.stats.stopped == "empty_queue"

    results = handler.get_result()
    assert results.proof is None


def test_bfs_failed():
    """
       0
    |        \
    1      (2,   3)
    |       |      \
    x       x        3

   """
    thms = [my_thm(f"{i}") for i in range(4)]
    handler = BFSProofHandler(
        goal=BackwardGoal(theorem=thms[0], label="label"), n_expansions_max=100,
    )

    assert handler.is_ready()

    step_0 = handler.get_theorems_to_expand()
    assert step_0 == [thms[0]]
    handler.send_env_expansions(
        [my_env_expansion(thms[0], [[thms[1]], [thms[2], thms[3]]])]
    )

    step_1 = handler.get_theorems_to_expand()
    assert step_1 == [thms[1], thms[2], thms[3]]
    handler.send_env_expansions(
        [
            my_failed_env_expansion(thms[1]),
            my_failed_env_expansion(thms[2]),
            my_env_expansion(thms[3], [[thms[3]]]),
        ]
    )

    assert handler.done
    assert len(handler.graph.th_to_node) == 4
    assert len(handler.graph.failed_leafs) == 2
    assert len(handler.graph.proved_leafs) == 0
    assert len(handler.graph.leafs) == 0
    assert handler.stats.n_expansions_sent == handler.stats.n_expansions_received == 4
    assert handler.stats.stopped == "failed_root"

    results = handler.get_result()
    assert results.proof is None


def test_bfs_not_useful():
    """
       0
    |         |             |
    1      (2,   3)          4
    |       |     ?          |
    x       x                x

   """
    thms = [my_thm(f"{i}") for i in range(5)]
    handler = BFSProofHandler(
        goal=BackwardGoal(theorem=thms[0], label="label"),
        n_expansions_max=100,
        n_concurrent_expansions_max=2,
    )

    assert handler.is_ready()

    step_0 = handler.get_theorems_to_expand()
    assert step_0 == [thms[0]]
    handler.send_env_expansions(
        [my_env_expansion(thms[0], [[thms[1]], [thms[2], thms[3]], [thms[4]]])]
    )

    step_1 = handler.get_theorems_to_expand()
    assert step_1 == [thms[1], thms[2]]
    handler.send_env_expansions(
        [my_failed_env_expansion(thms[1]), my_failed_env_expansion(thms[2]),]
    )

    step_2 = handler.get_theorems_to_expand()
    assert step_2 == [thms[4]]
    handler.send_env_expansions([my_failed_env_expansion(thms[4])])

    assert handler.done
    assert len(handler.graph.th_to_node) == 5
    assert len(handler.graph.failed_leafs) == 3
    assert len(handler.graph.proved_leafs) == 0
    assert len(handler.graph.leafs) == 1
    assert handler.stats.n_expansions_sent == handler.stats.n_expansions_received == 4
    assert handler.stats.not_useful_to_expand == 1
    assert handler.stats.stopped == "failed_root"

    results = handler.get_result()
    assert results.proof is None


def test_best_fs():
    """
       0
    |        \
    1      (2,   3)
            |      \
            4       5
            |       |
            ok      ok
   """
    thms = [my_thm(f"{i}") for i in range(6)]
    handler = BFSProofHandler(
        goal=BackwardGoal(theorem=thms[0], label="label"),
        n_expansions_max=100,
        best_first_search=True,
        n_concurrent_expansions_max=2,
    )

    assert handler.is_ready()

    step_0 = handler.get_theorems_to_expand()
    assert step_0 == [thms[0]]
    exp = my_env_expansion(thms[0], [[thms[1]], [thms[2], thms[3]]])
    exp.priors = [0.0000001, 0.9]
    handler.send_env_expansions([exp])

    step_1 = handler.get_theorems_to_expand()
    assert step_1 == [thms[2], thms[3]]
    exps = [
        my_env_expansion(thms[2], [[thms[4]]]),
        my_env_expansion(thms[3], [[thms[5]]]),
    ]
    exps[0].priors = [0.9]
    exps[1].priors = [1.0]
    handler.send_env_expansions(exps)
    step_2 = handler.get_theorems_to_expand()
    assert step_2 == [thms[5], thms[4]]

    exps = [
        my_env_expansion(thms[5], [[]]),
        my_env_expansion(thms[4], [[]]),
    ]
    handler.send_env_expansions(exps)

    assert handler.done
    assert len(handler.graph.th_to_node) == 6
    assert len(handler.graph.failed_leafs) == 0
    assert len(handler.graph.proved_leafs) == 2
    assert len(handler.graph.leafs) == 1
    assert handler.stats.n_expansions_sent == handler.stats.n_expansions_received == 5
    assert handler.stats.stopped == "proved"

    results = handler.get_result()
    assert results.proof is not None
