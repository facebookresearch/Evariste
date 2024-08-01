# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, List, Set, Dict, Deque, Sequence, Iterator
from collections import deque, defaultdict
import math
import time
import numpy as np

from evariste.backward.env.core import BackwardGoal, EnvExpansion
from evariste.backward.graph import Theorem, Tactic
from evariste.backward.prover.policy import Policy
from enum import unique, Enum

from evariste.backward.prover.prover_args import ProverParams
from evariste.backward.prover.mcts import MCTSHandler, MCTSResult, MCTSNode
from evariste.datasets import DatasetConf


VIRTUAL_LOSS = 1
FPU = 0.5


@unique
class NodeStatus(str, Enum):
    Solved = "solved"
    Impossible = "impossible"
    Internal = "internal"


@unique
class NodeType(str, Enum):
    And = "and"
    Or = "or"


class AndOrNode:
    def __init__(
        self,
        value: Optional[float],
        children: List[int],
        priors: Optional[List[float]],
        status: NodeStatus,
        error: Optional[str],
        andor: NodeType,
    ):
        self.value = value
        self.children = children
        self.error = error

        self.n_actions = len(children)
        if priors is None and self.n_actions > 0:
            self.priors = np.full(self.n_actions, 1.0 / self.n_actions, np.float64)
        else:
            self.priors = np.array(priors, np.float64)
        self.virtual_loss = np.zeros((self.n_actions,), np.float64)
        self.counts = np.zeros((self.n_actions,), np.float64)
        self.q_sum = np.zeros((self.n_actions,), np.float64)

        self.status = status
        self.andor = andor

    def update(self, value: float, action: int, only_vl: bool):
        if not only_vl:
            self.q_sum[action] += value
            self.counts[action] += 1
        self.virtual_loss[action] -= VIRTUAL_LOSS

    def policy(
        self, pol: Policy, ancestors: Set[int], mu_fpu: bool, mu_fpu_at_and: bool
    ):
        mfpu = mu_fpu and (self.andor == NodeType.Or or mu_fpu_at_and)
        q = np.full(self.n_actions, FPU, dtype=np.float64)
        if np.sum(self.counts) > 0:
            if mfpu:
                q[self.counts == 0] = np.sum(self.q_sum) / np.sum(self.counts)
            seen = self.counts > 0
            q[seen] = self.q_sum[seen] / self.counts[seen]

        p = pol.get(q, self.counts + self.virtual_loss, self.priors)
        for i, x in enumerate(self.children):
            if x in ancestors:
                p[i] = -math.inf
        return p

    def terminal(self) -> bool:
        return not self.status == NodeStatus.Internal

    def only_cycles(self, ancestors: Set[int]) -> bool:
        return all(x in ancestors for x in self.children)

    @property
    def solved(self) -> bool:
        return self.status == NodeStatus.Solved

    @property
    def impossible(self) -> bool:
        return self.status == NodeStatus.Impossible


class AndNode(AndOrNode):
    def __init__(
        self,
        value: Optional[float],
        children: List[int],
        priors: Optional[List[float]],
        status: NodeStatus,
    ):
        super().__init__(value, children, priors, status, None, NodeType.And)


class OrNode(AndOrNode):
    def __init__(
        self,
        value: Optional[float],
        children: List[int],
        priors: Optional[List[float]],
        status: NodeStatus,
        error: Optional[str] = None,
    ):
        super().__init__(value, children, priors, status, error, NodeType.Or)


class Path:
    def __init__(
        self,
        node_id: int,
        parent: Optional["Path"],
        ancestors: Optional[Set[int]] = None,
    ):
        self.node_id = node_id
        self.parent = parent
        self.ancestors: Set[int] = set() if ancestors is None else ancestors
        self.ancestors.add(node_id)

        self.action: Optional[int] = None

    def next(self, next_node: int, action: int) -> "Path":
        self.action = action
        return Path(next_node, parent=self, ancestors=self.ancestors)

    def walk_to_root(self) -> Iterator["Path"]:
        cur = self
        while cur.parent is not None:
            cur = cur.parent
            yield cur

    def get_hash(self) -> int:
        return hash(tuple([x.action for x in self.walk_to_root()]))


class SimpleMCTSHandler:
    def __init__(
        self, goal: BackwardGoal, prover_params: ProverParams, process_id: int,
    ):
        self.process_id = process_id
        self.goal = goal
        self.root = goal.theorem
        self.prover_params = prover_params
        self.policy = Policy(
            prover_params.mcts.policy, exploration=prover_params.mcts.exploration
        )

        self.backup_paths: Dict[Theorem, List[Path]] = defaultdict(list)
        self.batch_size = prover_params.mcts.succ_expansions
        self.max_loop = 50

        self.n_expansions = 0
        self.start_time = time.time()
        self.done = False

        self.nodes: Dict[int, AndOrNode] = {}

        # maps a node to the set of (node_id, action_id) that leads to it
        self.ancestors: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)

        self.backuped: Set[int] = set()

        self.th_to_node_id: Dict[Theorem, int] = {self.root: 0}
        self.node_id_to_th: Dict[int, Theorem] = {0: self.root}
        self.node_id_to_tac: Dict[int, Tactic] = {}
        self.next_node_id = 1
        self.to_update: Set[int] = set()

    def backup(self, value: float, leaf: Path):
        only_vl = False
        if self.prover_params.mcts.backup_once:
            this_hash = leaf.get_hash()
            if this_hash in self.backuped:
                only_vl = True
            self.backuped.add(this_hash)
        node = leaf.parent
        while node is not None:
            value = 1 - value
            assert node.action is not None
            self.nodes[node.node_id].update(value, node.action, only_vl)
            node = node.parent

    def update_solved_and_impossible(self, leaves: Sequence[int]):
        to_update: Deque[int] = deque(leaves)
        while to_update:
            cur_id = to_update.pop()
            cur_node = self.nodes[cur_id]

            child_impossible = []
            child_solved = []
            for child_id in cur_node.children:
                if child_id not in self.nodes:
                    child_solved.append(False)
                    child_impossible.append(False)
                else:
                    node = self.nodes[child_id]
                    child_solved.append(node.solved)
                    child_impossible.append(node.impossible)
            new_status = NodeStatus.Internal
            if cur_node.andor == NodeType.And:
                if any(child_impossible):
                    new_status = NodeStatus.Impossible
                if all(child_solved):
                    new_status = NodeStatus.Solved
            if cur_node.andor == NodeType.Or:
                if all(child_impossible):
                    new_status = NodeStatus.Impossible
                if any(child_solved):
                    new_status = NodeStatus.Solved
            if new_status != cur_node.status:
                cur_node.status = new_status
                for ancestor in {a_id for a_id, _ in self.ancestors[cur_id]}:
                    to_update.appendleft(ancestor)

    def new_tac_node(
        self,
        tactic: Tactic,
        children: List[int],
        priors: Optional[List[float]],
        status: NodeStatus,
        ancestor: Tuple[int, int],
    ) -> int:
        node_id = self.next_node_id
        self.node_id_to_tac[node_id] = tactic
        self.nodes[node_id] = AndNode(
            value=None, children=children, priors=priors, status=status
        )
        self.ancestors[node_id] = {ancestor}  # only one ancestor for and nodes
        self.next_node_id += 1
        self.to_update.add(node_id)
        return node_id

    def new_th_leaf(self, th: Theorem, ancestor: Tuple[int, int]) -> int:
        maybe_id = self.th_to_node_id.get(th, None)
        if maybe_id is not None:
            self.ancestors[maybe_id].add(ancestor)
            if maybe_id in self.nodes:
                self.to_update.add(maybe_id)
            return maybe_id
        else:
            this_id = self.next_node_id
            self.th_to_node_id[th] = this_id
            self.node_id_to_th[this_id] = th
            self.next_node_id += 1
            self.ancestors[this_id].add(ancestor)
            return this_id

    def add_nodes_from_expansion(self, expansion: EnvExpansion) -> AndOrNode:
        # we expand a leaf into an OR node followed by several AND nodes
        leaf_id = self.th_to_node_id[expansion.theorem]
        #  impossible node
        if expansion.is_error:
            self.nodes[leaf_id] = OrNode(
                value=0,
                children=[],
                priors=[],
                status=NodeStatus.Impossible,
                error=expansion.error,
            )
            self.to_update.add(leaf_id)
            return self.nodes[leaf_id]
        assert (
            expansion.log_critic is not None
            and expansion.tactics is not None
            and expansion.child_for_tac is not None
            and expansion.priors is not None
        )
        #  solved node
        if len(expansion.child_for_tac[0]) == 0:
            children = []
            for tac_id, tactic in enumerate(expansion.tactics):
                children.append(
                    self.new_tac_node(
                        tactic=tactic,
                        children=[],
                        priors=None,
                        status=NodeStatus.Solved,
                        ancestor=(leaf_id, tac_id),
                    )
                )

            self.nodes[leaf_id] = OrNode(
                value=1,
                children=children,
                priors=expansion.priors,
                status=NodeStatus.Internal,  # will be updated to Solved later
            )
            self.to_update.add(leaf_id)
            return self.nodes[leaf_id]

        #  internal node with a critic
        children = []
        for tac_id, (tactic, tac_children) in enumerate(
            zip(expansion.tactics, expansion.child_for_tac)
        ):
            and_node = self.new_tac_node(
                tactic=tactic,
                children=list(range(len(tac_children))),
                priors=None,
                status=NodeStatus.Internal,
                ancestor=(leaf_id, tac_id),
            )  # created with dummy children because its ID need to exist to create children with ancestors
            tac_children_ids = [
                self.new_th_leaf(child, (and_node, c_id))
                for c_id, child in enumerate(tac_children)
            ]
            self.nodes[and_node].children = tac_children_ids
            children.append(and_node)

        self.nodes[leaf_id] = OrNode(
            value=math.exp(expansion.log_critic),
            children=children,
            priors=expansion.priors,
            status=NodeStatus.Internal,
        )
        self.to_update.add(leaf_id)
        return self.nodes[leaf_id]

    def expand_and_backup(self, expansions: List[EnvExpansion]):
        for expansion in expansions:
            node = self.add_nodes_from_expansion(expansion)
            for path in self.backup_paths.pop(expansion.theorem):
                assert node.value is not None
                self.backup(value=node.value, leaf=path)

        # filter because unexpanded theorem have ids but are not in nodes
        self.update_solved_and_impossible(list(self.to_update))
        self.to_update.clear()

        self.n_expansions += len(expansions)
        self.done |= 0 in self.nodes and self.nodes[0].terminal()
        self.done |= self.n_expansions >= self.prover_params.mcts.n_expansions
        self.done |= (time.time() - self.start_time) > 3600

    def check_expandable(self) -> bool:
        to_visit: List[int] = [0]
        ancestors: Set[int] = set()

        while to_visit:
            cur_id = to_visit.pop()
            if cur_id not in self.nodes:
                # something to expand exists, we just have to find it.
                return True
            node = self.nodes[cur_id]
            if cur_id in ancestors or node.terminal():
                continue
            ancestors.add(cur_id)
            for child in node.children:
                to_visit.append(child)
        return False

    def get_one_leaf(self) -> Optional[Path]:
        path = Path(node_id=0, parent=None)
        while True:
            if path.node_id not in self.nodes:
                return path
            cur_node = self.nodes[path.node_id]
            policy = cur_node.policy(
                self.policy,
                path.ancestors,
                self.prover_params.mcts.mu_fpu,
                self.prover_params.mcts.mu_fpu_at_and,
            )
            for cid, child in enumerate(cur_node.children):
                if child in self.nodes and self.nodes[child].terminal():
                    policy[cid] = -math.inf

            action = policy.argmax()
            if policy[action] == -math.inf:
                # if node is AND, we backup 1.
                # Parent OR node will get  1-v = 0
                self.backup(cur_node.andor == NodeType.And, path)
                return None

            cur_node.virtual_loss[action] += VIRTUAL_LOSS
            cur_th = cur_node.children[action]
            path = path.next(cur_th, action)

    def get_theorems_to_expand(self) -> List[Theorem]:
        batch: Set[Theorem] = set()
        n_loop = 0
        while len(batch) < self.batch_size and n_loop < self.max_loop:
            n_loop += 1
            maybe_path = self.get_one_leaf()
            if maybe_path is None:
                continue
            th = self.node_id_to_th[maybe_path.node_id]
            self.backup_paths[th].append(maybe_path)
            batch.add(th)
        if len(batch) == 0:
            assert self.check_expandable(), "Stuck"
        return list(batch)

    def result(self) -> MCTSResult:

        # self.update_solved_and_impossible([
        #     x for x in self.nodes
        # ])

        mcts = MCTSHandler(self.goal, self.prover_params, self.process_id)
        mcts_nodes = []
        # convert this graph into the other type to re-use result code
        for node_id, node in self.nodes.items():
            if node_id not in self.node_id_to_th:
                continue
            assert node.value is not None
            children_for_tactic = [
                [self.node_id_to_th[y] for y in self.nodes[x].children]
                for x in node.children
            ]
            try:
                log_critic = math.log(node.value) if node.value > 0 else -math.inf
            except ValueError:
                print(node.value)
                raise RuntimeError(f"{node.value} // {node.value > 0}")
            mcts_node = MCTSNode(
                theorem=self.node_id_to_th[node_id],
                time_created=0,
                tactics=[self.node_id_to_tac[x] for x in node.children],
                log_critic=log_critic,
                children_for_tactic=children_for_tactic,
                priors=node.priors.tolist(),
                exploration=self.prover_params.mcts.exploration,
                policy=self.prover_params.mcts.policy,
                error=node.error,
                effects=[],
                init_tactic_scores=FPU,
                q_value_solved=self.prover_params.mcts.q_value_solved,
            )
            mcts_nodes.append(mcts_node)
        mcts.mcts.add_nodes(mcts_nodes)

        mcts.mcts.all_simu_trees_depth = [-1]  # avoid max of empty sequence
        return mcts.result()
