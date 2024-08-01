# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import abc
from logging import getLogger
from typing import List, Set, Optional, Tuple

from numpy.random import RandomState

from evariste.comms.store import EmptyStore
from evariste.envs.mm.utils import Node, SimpleNode

import numpy as np

from evariste.forward.fwd_mm.training.common import MMFwdTrainingProof
from evariste.forward.training.helpers import sample_from_cumulative

logger = getLogger()


class MMFwdTrainingDataset(abc.ABC):
    @abc.abstractmethod
    def sample_training_graph(self, rng: RandomState, split: str) -> MMFwdTrainingProof:
        pass


def sample_from_cumulative_mm(
    cumulative: np.array, data: List[Tuple[str, Node]], rng: RandomState,
) -> Tuple[str, Node]:
    name, proof_tree = sample_from_cumulative(cumulative, data, rng)
    assert proof_tree is not None
    return name, proof_tree


def drop_oldest_nodes(
    goal_node: Node, graph: List[Node], output_node: Node, max_len: int,
) -> Optional[List[Node]]:
    goal_size = 2 + len(goal_node.statement)
    e_hyps = []
    nodes = []
    node_sizes = []
    total_size = 2 + goal_size
    goal_hyps_size = 2 + goal_size
    for node in graph:
        size = len(node.statement)
        if node.ltype == "$e":
            e_hyps.append(node)
            goal_hyps_size += 2 + size
        else:
            nodes.append(node)
            node_sizes.append(2 + size)
        total_size += 2 + size
    # print(f"len(graph) {len(graph)}, goal_size: {goal_size}, goal_hyps_size: {goal_hyps_size} total_size: {total_size}")
    if total_size < max_len:
        return graph
    if goal_hyps_size >= max_len:
        return None

    assert total_size == goal_hyps_size + sum(node_sizes)
    children_str = {n.statement_str for n in getattr(output_node, "children", [])}

    i = 0
    while total_size >= max_len:
        assert i < len(nodes)
        to_remove = nodes[i]
        total_size -= node_sizes[i]
        i += 1
        if to_remove.statement_str in children_str:
            # problem
            # print(f" Problem! total_size: {total_size}")
            return None
    # print(f" Success: len(graph) {len(e_hyps + nodes[i:])} total_size: {total_size}")
    return e_hyps + nodes[i:]


def drop_nodes_v1(
    goal_node: Node,
    graph: List[Node],
    gsize: int,
    order: List[Node],
    max_len: int,
    rng: np.random.RandomState,
) -> List[Node]:
    def _estimated_len(goal_, graph_):
        return (
            2 + 2 + len(goal_.statement) + sum((2 + len(n.statement) for n in graph_))
        )

    # todo: if it's working do smth less hacky, if not remove this
    if _estimated_len(goal_node, graph) > max_len:
        children_required = [
            c.statement_str for n in order[gsize:] for c in getattr(n, "children", [])
        ]
        not_useful_anymore = [
            n for n in graph if n.statement_str not in children_required
        ]
        rng.shuffle(not_useful_anymore)

        # look at shortest possible graph
        not_useful = set(not_useful_anymore)
        min_graph = [n for n in graph if n not in not_useful]
        min_possible_len = _estimated_len(goal_node, min_graph)
        if min_possible_len > max_len:
            # do nothing, not useful
            pass
        else:
            cur_len = _estimated_len(goal_node, graph)
            to_remove = set()
            while not_useful_anymore and (cur_len > max_len):
                not_useful_node = not_useful_anymore.pop(0)
                to_remove.add(not_useful_node)
                cur_len -= 2 + len(not_useful_node.statement)

            graph = [n for n in graph if n not in to_remove]
            assert cur_len == _estimated_len(goal_node, graph)
    return graph


def find_path(order: List[Node]) -> List[Node]:
    """

    :param order: list of nodes in topological order. First and last nodes are supposed
    to be connected
    :return: path (list of nodes) between first and last node of order
    """
    path = [order[0]]
    for node in order[1:]:
        children = getattr(node, "children", [])
        if path[-1] in children:
            path.append(node)
    return path


def get_random_canonical_order(node: Node, rng: RandomState) -> List[Node]:
    r"""
    Canocical order but with hypothesis shuffled
    """
    order = []
    _random_cano_order(node, order, handled=set([]), rng=rng)
    return order


def _random_cano_order(
    node: Node, order: List[Node], handled: Set[str], rng: RandomState
):
    stat = node.statement_str
    if stat in handled:
        return
    children: List[Node] = getattr(node, "children", [])
    rng.shuffle(children)
    for child in children:
        _random_cano_order(child, order, handled=handled, rng=rng)
    assert stat not in handled
    handled.add(stat)
    order.append(node)


def random_order_v1(goal_node: Node, rng: RandomState) -> List[Node]:
    order = _experimental_random_topological_sort(
        goal_node, e_hyps_first=True, rng=rng, select_last_candidate=True,
    )
    order = [n.node for n in order]
    return order


def random_order_v2(goal_node: Node, rng: RandomState) -> List[Node]:
    order = get_random_canonical_order(goal_node, rng=rng)
    order = [n for n in order if n.ltype == "$e"] + [
        n for n in order if n.ltype != "$e"
    ]
    return order


def random_order_v3(goal_node: Node, rng: RandomState) -> List[Node]:
    if rng.random() < 0.5:
        order = get_random_canonical_order(goal_node, rng=rng,)
        order = [n for n in order if n.ltype == "$e"] + [
            n for n in order if n.ltype != "$e"
        ]
    else:
        order = _experimental_random_topological_sort(
            goal_node, e_hyps_first=True, rng=rng, select_last_candidate=False,
        )
        order = [n.node for n in order]
    return order


def random_order_v4(goal_node: Node, rng: RandomState, output_node: Node) -> List[Node]:
    """if output node is the last candidate inserted, then choose this candidate.
    This could help the model to generate deeper proofs (since it will learn to build
    on top of latest inserted node if possible)"""
    order = _experimental_random_topological_sort(
        goal_node,
        e_hyps_first=True,
        rng=rng,
        select_last_candidate=False,
        output_node=output_node,
    )
    order = [n.node for n in order]
    return order


def _experimental_random_topological_sort(
    root: Node,
    e_hyps_first: bool,
    rng,
    select_last_candidate: bool = False,
    output_node: Optional[Node] = None,
) -> List[SimpleNode]:
    """
    Enumerate all nodes in a proof tree, in a random topological order.

    If select_last_candidate is True, instead of sampling uniformly among candidates
    we always choose the last inserted candidate. The order is still random but less.
    This should force the network to build on top of latest nodes in the graph if it
    is possible.

    if output_node: if output_node
    is the last candidate inserted, then choose this candidate.
    This could help the model to generate deeper proofs (since it will learn to build
    on top of latest inserted node if possible)
    """
    # TODO if sample candidate is False, we can sitll sample a random candidate
    # with a given prob to make the model robust to mistake. However we need to be
    # careful not to sample the target_node in this case (the model still needs to learn
    # to generate in the good order)
    candidates = list()

    # Since the graph is a DAG and not a tree we need to keep track of where we went
    all_nodes = dict()

    def traverse(node: SimpleNode, id_in_parent: int, parent: Optional[SimpleNode]):
        statement = " ".join(node.node.statement)
        if statement in all_nodes:
            all_nodes[statement].parents.append((parent, id_in_parent))
            return
        node.parents.append((parent, id_in_parent))
        all_nodes[statement] = node
        if node.n_children == 0:
            candidates.append(node)
            return
        for id_in_parent, child in enumerate(node.node.children):
            traverse(SimpleNode(child), id_in_parent, node)

    simple_root = SimpleNode(root)
    traverse(simple_root, 0, None)

    def _update_candidates(order_, candidates_):
        to_add = []
        for parent, id_in_parent in order_[-1].parents:
            if parent is None:
                continue
            parent.children.append(
                (id_in_parent, len(order) - 1)
            )  # this points to a node in the topological ordering
            parent.n_children -= 1
            assert parent.n_children >= 0
            if parent.n_children == 0:
                to_add.append(parent)
        if to_add:
            # add more randomness for order in which we append to candidates
            rng.shuffle(to_add)
        for node in to_add:
            candidates_.append(node)

    order = []
    if e_hyps_first:
        e_hyps = [c for c in candidates if c.node.ltype == "$e"]
        candidates = [c for c in candidates if c.node.ltype != "$e"]
        rng.shuffle(e_hyps)
        for hyp in e_hyps:
            order.append(hyp)
            _update_candidates(order, candidates)

    # Then pop random candidates until list is empty
    while candidates:
        if output_node and candidates[-1].node == output_node:
            # print("output node is last candidate", len(candidates))
            pass
        elif not select_last_candidate:
            # for O(1) random pop : select index, swap with end, pop end
            to_pop = rng.randint(0, len(candidates))
            if output_node and candidates[to_pop].node == output_node:
                # print("output node is not last candidate", len(candidates))
                continue
            candidates[to_pop], candidates[-1] = candidates[-1], candidates[to_pop]
        order.append(candidates[-1])
        assert candidates[-1].n_children == 0
        candidates.pop()
        _update_candidates(order, candidates)

    # Check that all nodes are generated, that the topological ordering is correct,
    # i.e. each parent appears after all its childrens, and that the root node is
    # the last generated node.
    assert all(v.n_children == 0 for v in all_nodes.values())
    assert len(order) == len(all_nodes)
    assert order[-1].node == root
    for i, node in enumerate(order):
        for c in node.children:
            assert c[1] < i, f"child {c} after parent {i}"

    return order


def log_sizes(logger, sizes, split):
    _p = (50, 90, 95, 99)
    logger.info(
        f"Proof size stats for {split}: "
        f"n proofs: {len(sizes)}, "
        f"mean: {np.mean(sizes)}, "
        f"max: {np.max(sizes)}, "
        f"min: {np.min(sizes)}, "
        f"percentiles: {list(zip(_p, np.percentile(sizes, _p)))}"
    )
