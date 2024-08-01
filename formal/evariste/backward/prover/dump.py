# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, List, Set, Dict
from collections import defaultdict
from logging import getLogger
import math
import numpy as np

from evariste.backward.graph import Theorem
from evariste.backward.env.equations.graph import EQTheorem
from evariste.backward.prover.nodes import MCTSNode
from evariste.backward.prover.policy import Policy


logger = getLogger()


MAX_MCTS_DEPTH = 6


def get_theorem_data(th: Theorem) -> Dict:
    if isinstance(th, EQTheorem):
        statement = th.eq_node.infix()
        hyps = [hyp.infix() for hyp in th.eq_hyps]
    else:
        statement = th.conclusion
        hyps = [hyp for _, hyp in th.hyps]
    return {"statement": statement, "hyps": hyps}


def sanitize(x: Optional[float]):
    if x is None:
        return x
    else:
        return max(x, -1e9)


class DumpNode:
    """
    This is a storage class, simpler than MCTSNode.
    Its intended use is to visualize the process of an MCTS exploration.
    """

    def __init__(self, theorem: Theorem, node: Optional[MCTSNode]):
        self.theorem = theorem
        if node is not None:
            self.no_node = False
            self.tactics = node.tactics
            self.priors = node.priors
            self.history = node.history
            self.history.updates = [
                (t, tid, sanitize(v)) for t, tid, v in self.history.updates
            ]
            self.children_for_tactic: Optional[List[List[int]]] = None
            self.solved = node.solved
            self.mcts_node = node

            # check updates are sorted
            updates = self.history.updates
            assert all(
                updates[i][0] <= updates[i + 1][0] for i in range(len(updates) - 1)
            )
        else:
            self.no_node = True
            self.solved = False

    def get_goal_data(self) -> Dict:
        goal_data = get_theorem_data(self.theorem)
        if not self.no_node:
            counts = self.mcts_node.counts
            extra_data = {
                "log_critic": sanitize(self.mcts_node.log_critic),
                "old_log_critic": self.mcts_node.old_critic,
                "creation_time": self.history.exists_since,
                "killed_time": self.history.killed_node,
                "killed_tactic_times": self.history.killed_tactic,
                "visits": None if counts is None else counts.sum().item(),
                "history": self.history.updates,
            }
        else:
            extra_data = {
                "log_critic": None,
                "old_log_critic": None,
                "creation_time": None,
                "killed_time": None,
                "killed_tactic_times": None,
                "visits": None,
                "history": None,
            }
        assert goal_data.keys().isdisjoint(extra_data.keys())
        goal_data.update(extra_data)
        return goal_data


class MCTSDump:
    def __init__(
        self, root: Theorem, nodes: Dict[Theorem, MCTSNode], max_t: int, label: str
    ):
        self.root = root
        self.nodes: List[DumpNode] = []
        self.th_to_id: Dict[Theorem, int] = {}
        self.max_t = max_t
        self.label = label

        # count of nodes with duplicates in treeview (for stats)
        self.nodes_in_tree = 0

        def dump(cur_th: Theorem) -> int:
            if cur_th in self.th_to_id:
                return self.th_to_id[cur_th]
            cur_id = len(self.nodes)
            self.th_to_id[cur_th] = cur_id
            mcts_node = nodes.get(cur_th, None)
            self.nodes.append(DumpNode(cur_th, mcts_node))
            if mcts_node is not None:
                children_for_tactic: List[List[int]] = []
                for children in mcts_node.children_for_tactic:
                    children_id = []
                    for child in children:
                        children_id.append(dump(child))
                    children_for_tactic.append(children_id)
                self.nodes[cur_id].children_for_tactic = children_for_tactic
            return cur_id

        dump(root)

        # policy function
        self._policy = Policy(policy_type="other", exploration=5.0)

        # sanity check
        assert len(self.nodes) == len(self.th_to_id)
        assert all(self.th_to_id[x.theorem] == i for i, x in enumerate(self.nodes))
        assert self.th_to_id[self.root] == 0

        # check valid timesteps
        for node in self.nodes:
            if node.no_node:
                continue
            for timestep, _, _ in node.history.updates:
                assert 0 <= timestep < self.max_t, (timestep, self.max_t)

    def update_with_policy_values(self, node: DumpNode, goal_data: Dict):
        """
        Re-compute counts / W / Q / policy.
        Pre-compute all values, and store the associated timesteps.
        """
        if node.no_node or len(node.history.updates) == 0:
            goal_data["policy_data"] = []
            return

        n_tactics = len(node.tactics)
        assert len(node.priors) == n_tactics

        # killed tactics
        killed_tac_at = defaultdict(list)
        for tid, timestep in node.history.killed_tactic.items():
            killed_tac_at[timestep].append(tid)

        policy_data = []

        # group updates by timesteps
        updates = node.history.updates
        limits = (
            [0]
            + [i for i in range(1, len(updates)) if updates[i - 1][0] < updates[i][0]]
            + [len(updates)]
        )

        # dummy nodes to re-simulate all updates
        dn = MCTSNode(
            theorem=node.mcts_node.theorem,
            time_created=0,
            tactics=node.mcts_node.tactics,
            log_critic=node.mcts_node.log_critic,
            children_for_tactic=node.mcts_node.children_for_tactic,
            priors=node.mcts_node.priors.tolist(),
            exploration=self._policy.exploration,
            policy=self._policy.policy_type,
            error=None,
            effects=[],
            init_tactic_scores=0.5,
            q_value_solved=0,
        )
        assert dn.counts is not None
        assert dn.logW is not None

        # for each timestep group
        for a, b in zip(limits[:-1], limits[1:]):

            updates_at_t = updates[a:b]
            group_t = updates_at_t[0][0]
            assert all(x[0] == group_t for x in updates_at_t)

            # for each update in this group, update logW and counts
            for timestep, tid, value in updates_at_t:
                dn.update(tactic_id=tid, value=value)

            # compute Q
            Q = np.full(n_tactics, 0.5, dtype=np.float64)
            Q[dn.counts > 0] = np.exp(dn.logW[dn.counts > 0])
            Q[dn.counts > 0] /= dn.counts[dn.counts > 0]
            Q[dn.killed_mask] = 0

            # compute policy
            policy = dn.policy()
            assert policy[dn.killed_mask].sum() == 0, policy
            assert policy.min() >= 0, policy
            assert policy.max() <= 1, policy

            policy_data.append(
                {
                    "timestep": group_t,
                    "counts": dn.counts.tolist(),
                    "logW": np.maximum(dn.logW, -1e9).tolist(),
                    "Q": np.maximum(Q, 0).tolist(),
                    "policy": np.maximum(policy, 0).tolist(),
                    "best_tid": policy.argmax().item(),
                }
            )

            # update killed tactics
            for killed_tid in killed_tac_at[group_t]:
                dn.kill_tactic(group_t, killed_tid)

        goal_data["policy_data"] = policy_data

    def full_tree_dict(self, max_depth: int):

        logger.info(f"Building tree dict (max_depth={max_depth}) ...")

        path: Set[Theorem] = set()
        self.nodes_in_tree = 0
        assert max_depth >= 1

        def to_dict(node: DumpNode, depth: int):

            node_id = self.th_to_id[node.theorem]
            assert self.nodes[node_id] is node
            self.nodes_in_tree += 1

            # get goal data
            goal_data = node.get_goal_data()
            goal_data["depth"] = depth
            goal_data["is_cycle"] = node in path

            # display in the interface why we stopped there
            goal_data["terminal_cause"] = ""

            # get policy data. in terms of data storage, there will be a lot of
            # duplicates. could be smarter on the JS side
            self.update_with_policy_values(node, goal_data)

            data = {
                "goal_id": node_id,
                "goal_data": goal_data,
                "children": [],
                "is_solved": node.solved,
            }
            if node.no_node:
                goal_data["terminal_cause"] = "not expanded"
                return data
            if depth == max_depth:
                goal_data["terminal_cause"] = f"max depth {depth}"
                return data

            # avoid cycles (may result in inaccurate trees)
            if node.theorem in path:
                goal_data["terminal_cause"] = "cycle"
                return data
            path.add(node.theorem)

            # populate tactics
            assert node.children_for_tactic is not None
            children_for_tactic = []
            for tid, (tactic, children) in enumerate(
                zip(node.tactics, node.children_for_tactic)
            ):
                # construct children
                tactic_children = [
                    to_dict(self.nodes[child_id], depth=depth + 1)
                    for child_id in children
                ]

                # tactic data
                assert node.mcts_node.logW is not None
                assert node.mcts_node.counts is not None
                visits = node.mcts_node.counts[tid].item()
                log_W = sanitize(node.mcts_node.logW[tid].item())
                log_Q = log_W - math.log(max(visits, 1))
                assert log_Q <= 0, log_Q
                killed_time = goal_data["killed_tactic_times"].get(tid, None)
                tac_hist = [(t, v) for (t, ti, v) in goal_data["history"] if tid == ti]
                tac_policy_data = [
                    {
                        "timestep": x["timestep"],
                        "counts": x["counts"][tid],
                        "logW": x["logW"][tid],
                        "Q": x["Q"][tid],
                        "policy": x["policy"][tid],
                    }
                    for x in goal_data["policy_data"]
                ]
                tactic_data = {
                    "is_solving": tid in node.mcts_node.solving_tactics,
                    "tactic_data": {
                        "tac": tactic.to_dict(),
                        "prior": node.priors[tid].item(),
                        "visits": visits,
                        "max_visits": node.mcts_node.counts.max().item(),
                        "log_W": log_W,
                        "log_Q": log_Q,
                        "killed_time": killed_time,
                        "tac_history": tac_hist,
                        "tac_policy_data": tac_policy_data,
                    },
                    "is_valid": tactic.is_valid,
                    "children": tactic_children,
                    # "n_solved": n_solved,
                }
                children_for_tactic.append(tactic_data)
            data["children"] = children_for_tactic

            path.remove(node.theorem)

            return data

        return to_dict(self.nodes[0], depth=0)
