# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, List, Set, Dict, Iterator
from collections import deque
import math
import torch
import random
import numpy as np

from evariste.backward.graph import Tactic, Theorem
from evariste.backward.prover.args import MCTSParams
from evariste.backward.prover.graph import Node
from evariste.backward.prover.mcts_samples import MCTSSampleEffect
from evariste.backward.prover.policy import Policy


class SimuTree:
    """
    This stores the tree followed during one simulation
    """

    def __init__(
        self,
        theorem: Theorem,
        depth: int,
        parent: Optional["SimuTree"],
        message: str = "",
    ):
        self.theorem = theorem
        self.depth = depth
        self.parent: Optional["SimuTree"] = parent

        self.tactic_id: Optional[int] = None
        self.tactic_str: Optional[str] = None
        self.children: List["SimuTree"] = []
        self._value_to_backup: Optional[float] = None
        self.children_propagated = 0
        self.theorem_seen: Optional[Set[Theorem]] = {theorem}
        self.message = message
        self.solved = False
        self.virtual_count_added = False

        # useful for visualization purposes only
        self.policy: Optional[np.ndarray] = None
        self.logW: Optional[np.ndarray] = None
        self.counts: Optional[np.ndarray] = None
        self.virtual_counts: Optional[np.ndarray] = None
        self.priors: Optional[np.ndarray] = None

        if self.parent is not None:
            assert self.parent.theorem_seen is not None
            self.theorem_seen = self.theorem_seen.union(self.parent.theorem_seen)
            self.parent.children.append(self)

    def get_hash(self):
        return hash(tuple([n.tactic_id for n in self.walk()]))

    @property
    def value_to_backup(self):
        return self._value_to_backup

    @value_to_backup.setter
    def value_to_backup(self, v: float):
        assert isinstance(v, (int, float)) and v <= 0, f"Not a log prob: {v}"
        self._value_to_backup = v

    def walk(self) -> Iterator["SimuTree"]:
        to_walk = deque([self])
        while to_walk:
            cur = to_walk.pop()
            yield cur
            if cur.children is not None:
                to_walk.extend(cur.children)

    def leaves(self) -> List["SimuTree"]:
        return [x for x in self.walk() if len(x.children) == 0]

    def find_root(self) -> "SimuTree":
        cur, parent = self, self.parent
        while parent is not None:
            cur, parent = parent, parent.parent
        return cur

    def count_leaves(self) -> int:
        return len(self.leaves())

    def __eq__(self, other):
        a, b = list(self.walk()), list(other.walk())
        if len(a) != len(b):
            return False
        for aa, bb in zip(a, b):
            if aa.theorem != bb.theorem:
                return False
            if aa.tactic_id != bb.tactic_id:
                return False
        return True

    def __repr__(self):
        tabs = "\t" * self.depth
        res = (
            f"{tabs} {self.theorem.conclusion if self.theorem else self.theorem} "
            f"({self.children_propagated} - {len(self.children)})"
        )
        if self.tactic_id is not None:
            res += f" (id:{self.tactic_id}) {self.tactic_str}"
        if self.value_to_backup is not None:
            res += f" -> {str(self.value_to_backup)} {self.message}"
        res += "\n"
        for child in self.children:
            res += f"{tabs} {repr(child)}\n"
        return res

    def pre_delete(self):
        """may or may not help python gc, but cannot hurt..."""
        all_nodes = list(self.walk())
        for n in all_nodes:
            # delattr to ensure no one accesses these
            delattr(n, "parent")
            delattr(n, "children")


def display_from_leaves(leaves: List[SimuTree]):
    # first assert that all leaves come from the root
    roots = [leaf.find_root() for leaf in leaves]
    for root in roots:
        assert root.theorem == roots[0].theorem, "Mismatched root in the leaves"
    print("LEAVES: ")
    for leaf in leaves:
        print(leaf.theorem)
    print("TREE")
    print(roots[0])


class MCTSNode(Node):
    """
    Node representation used by MCTS to store statistics for each theorem at one point in time.
    """

    def __init__(
        self,
        theorem: Theorem,
        time_created: int,
        tactics: List[Tactic],
        log_critic: float,
        children_for_tactic: List[List[Theorem]],
        priors: List[float],
        exploration: float,
        policy: str,
        error: Optional[str],
        effects: List[Tuple[Tactic, Optional[List[Theorem]]]],
        init_tactic_scores: float,
        q_value_solved: int,
    ):
        super().__init__(theorem, tactics, children_for_tactic, time_created)
        self.old_critic: Optional[float] = None  # for resuscitating
        self.log_critic = log_critic
        self.priors = np.array(priors, dtype=np.float64)
        self.error = error
        self.effects = effects
        self.init_tactic_scores = init_tactic_scores
        self.q_value_solved = q_value_solved
        assert self.q_value_solved in range(0, 7)
        assert type(priors) is list
        assert all(type(p) is float and 0 <= p <= 1 for p in priors)
        assert init_tactic_scores > 0

        self.logW: Optional[np.ndarray] = None
        self.counts: Optional[np.ndarray] = None
        self.virtual_counts: Optional[np.ndarray] = None
        self.reset_mcts_stats()

        self._policy = Policy(policy_type=policy, exploration=exploration)

        # sanity check
        assert (
            len(self.children_for_tactic)
            == len(self.priors)
            == len(self.tactics)
            == self.n_tactics
        )
        if self.error is not None:
            assert not self.solved
            assert not self.is_solved_leaf
            assert self.n_tactics == 0
            assert self.log_critic == -math.inf
        else:
            assert type(self.log_critic) is float and self.log_critic <= 0
            assert abs(self.priors.sum() - 1) < 1e-6, self.priors.sum()
            assert self.n_tactics > 0
            assert any(tac.is_valid for tac in self.tactics)

    def __hash__(self):
        return self.theorem.hash

    def reset_mcts_stats(self) -> None:
        if self.error is None:
            # -math.inf here since Q = log sum exp (backprop values)
            # ensures after first backprop of 'v' at action 'a': Q[a].exp() = v
            self.logW = np.full(self.n_tactics, -math.inf, dtype=np.float64)
            self.counts = np.full(self.n_tactics, 0.0, dtype=np.float64)
            self.virtual_counts = np.full(self.n_tactics, 0.0, dtype=np.float64)
        else:
            assert len(self.tactics) == 0, f"wtf just happened"

    def should_send(self, params: MCTSParams) -> bool:
        # send if enough visit, or solved and not `use_count_threshold_for_solved`
        if self.error is not None:
            return False
        if self.counts is None:
            return False
        should_send = self.counts.sum().item() >= params.count_threshold or (
            self.solved and not params.use_count_threshold_for_solved
        )
        return should_send

    def get_train_sample_effect(self, params: MCTSParams) -> List[MCTSSampleEffect]:
        res = []
        for tac, children in self.effects:
            if random.random() > params.effect_subsampling or tac.malformed:
                continue
            if tac.is_error():
                res.append(
                    MCTSSampleEffect(goal=self.theorem, tactic=tac, children=None)
                )
            else:
                res.append(
                    MCTSSampleEffect(goal=self.theorem, tactic=tac, children=children)
                )
        return res

    def get_train_sample_critic(self, params: MCTSParams) -> Optional[Dict]:
        if not self.should_send(params) or random.random() > params.critic_subsampling:
            return None
        return {"q_estimate": math.exp(self.value())}

    def get_train_sample_tactic(
        self, params: MCTSParams, node_mask: str
    ) -> Optional[Dict]:
        if not self.should_send(params):
            return None
        stype = params.proof_stype
        is_valid = {
            "": True,
            "solving": len(self.solving_tactics) > 0,
            "proof": self.in_proof,
            "minproof": self.in_minproof[stype],
        }
        if not is_valid[node_mask]:
            return None

        assert self.counts is not None
        # filter out tactics
        # if q conditioning, I'll keep all tactics and build q_tactic
        if params.train_sample_for_q_conditioning:
            selected_ids = [
                i
                for i, tactic in enumerate(self.tactics)
                if (
                    i in self.solving_tactics
                    or not tactic.is_valid
                    or self.counts[i] >= params.count_tactic_threshold
                )
            ]
            if len(selected_ids) == 0:
                return None

            # build sample
            tactics = [self.tactics[i] for i in selected_ids]
            priors = [self.priors[i].item() for i in selected_ids]
            # NB: in the case of q conditioning, we won't use target_pi
            # to sample trainer samples, we sample uniformly
            target_pi = [-1.0 for _ in selected_ids]  # should not be used

            assert self.logW is not None  # for typing
            q_tactics: Optional[List[float]] = []
            assert q_tactics is not None
            for i in selected_ids:
                if i in self.solving_tactics:
                    q_tactics.append(1.0)
                elif not self.tactics[i].is_valid:
                    # maybe should also apply to `i in self.killed_tactics`?
                    q_tactics.append(0.0)
                else:
                    # TODO: check init_tactic_scores below
                    q_tactics.append(
                        (np.exp(self.logW[i]) / self.counts[i])
                        if self.counts[i] > 0
                        else self.init_tactic_scores
                    )
            assert len(q_tactics) == len(selected_ids)

        # if not q conditioning, I'll keep the either
        # all / solving / inproof / inminproof tactics depending on the case
        else:
            if self.all_tactics_killed:
                return None
            targets: List[float] = self.policy().tolist()
            if len(self.solving_tactics) > 0:
                selected_ids = list(self.solving_tactics)
                if params.only_learn_best_tactics or node_mask == "minproof":
                    selected_ids = self.my_minproof_tactics[stype]
                    assert len(selected_ids) > 0
            else:
                selected_ids = [
                    i
                    for i, tactic in enumerate(self.tactics)
                    if tactic.is_valid and targets[i] >= params.tactic_p_threshold
                ]
                if len(selected_ids) == 0:
                    return None
            # build sample
            tactics = [self.tactics[i] for i in selected_ids]
            priors = [self.priors[i].item() for i in selected_ids]
            # NB: in the case of q conditioning, we won't use target_pi
            # to sample trainer samples, we sample uniformly
            if len(self.solving_tactics) > 0:
                target_pi = [1.0 / len(tactics)] * len(tactics)
            else:
                target_pi = [targets[i] for i in selected_ids]
            q_tactics = None

        # used for training on proof
        # label 0 -> not in proof, 1 -> in proof, 2 -> minimal proof
        if self.in_minproof[stype]:
            inproof = 2.0
        elif self.in_proof:
            inproof = 1.0
        else:
            inproof = 0.0

        # sanity check
        assert len(tactics) == len(priors) == len(target_pi) == len(selected_ids)

        return {
            "tactics": tactics,
            "target_pi": target_pi,
            "q_tactics": q_tactics,
            "inproof": inproof,
        }

    def kill_tactic(self, t: int, tactic_id: int) -> bool:
        _kill_tactic = super().kill_tactic(t, tactic_id)
        if self.all_tactics_killed:
            assert self.log_critic is not None
            self.old_critic = self.log_critic  # allow resuscitate
            self.log_critic = -math.inf
        return _kill_tactic

    def policy(self, expansion_time: bool = False) -> np.ndarray:
        """
        `expansion_time` True if we are in find_leaves_to_expand and then tactic_is_expandable is used as mask.
        """
        # For typing
        assert self.counts is not None
        assert self.virtual_counts is not None
        assert self.logW is not None

        counts = self.counts + self.virtual_counts

        # Q_a :
        # - if (real_counts + virtual_counts) == 0 (no virtual count, no count) -> init_tactic_scores
        # - if real_counts == 0 & virtual counts > 0 -> init_tactic_scores / (real_counts + virtual_counts)
        # - if real_counts > 0  -> exp(logW) / (real_counts + virtual_counts)

        Q = np.full(self.n_tactics, self.init_tactic_scores, dtype=np.float64)
        Q[self.counts > 0] = np.exp(self.logW[self.counts > 0])
        Q[counts > 0] /= counts[counts > 0]

        # fixed_solving_tactic_score removed. now always on. #945
        # 6 options for Q value for solved nodes - chronological order + 4th new option:
        # opt 0 : 1 / (c + vc)  (logW = 0) + FPU
        # opt 1 : c / (c + vc)  (logW = log(c) --> more standard version - default) + FPU
        # opt 2 : 1             (q hardcoded = 1)
        # opt 3 : 1 / (1 + vc)  (new option)
        # opt 4 : opt 0 wo FPU
        # opt 5 : opt 1 wo FPU

        for t in self.solving_tactics:
            if self.q_value_solved == 0:
                if counts[t] > 0:
                    Q[t] = 1 / counts[t]
            elif self.q_value_solved == 1:
                if counts[t] > 0:
                    Q[t] = self.counts[t] / counts[t]
            elif self.q_value_solved == 2:
                Q[t] = 1
            elif self.q_value_solved == 3:
                Q[t] = 1 / (1 + self.virtual_counts[t])
            elif self.q_value_solved == 4:
                Q[t] = 1 / max(1, counts[t])
            elif self.q_value_solved == 5:
                Q[t] = max(1, self.counts[t]) / max(1, counts[t])
            else:
                assert self.q_value_solved == 6

        # Q[a] = -inf counts[a] = 0 ensures 'a' is ignored during policy computation
        assert self.killed_mask is not None
        valid_mask = ~self.killed_mask
        if len(self.killed_tactics) >= len(self.tactics):
            raise RuntimeError("Exploring node with all tactics killed")
        if expansion_time:
            # force policy to explore where there are still nodes to expand
            # doesn't hurt policy estimates since we have solving tactics
            # so returned policy will be uniform over solvings
            maybe_valid = valid_mask & self.tactic_is_expandable
            if maybe_valid.any():
                # only use expandable as a mask if it doesn't kill all tactics.
                # the only way for this to happen is if a sibling has things to expand and we don't
                # being non-expandable means our tactics are either :
                #  - killed
                #  - solving
                #  - lead to nodes that are already being expanded
                # In this case, it's fine to use the normal policy.
                valid_mask = maybe_valid
        mask = ~valid_mask
        Q[mask] = -math.inf
        counts[mask] = 0

        # check that at least one tactic is valid, and that
        # Q values are in [0, 1], unless invalid tactic
        if Q.max() == -math.inf or not ((Q >= 0) | (Q <= 1) | (Q == -math.inf)).all():
            raise RuntimeError(
                f"UNEXPECTED Q VALUES: "
                f"Q={Q.tolist()} "
                f"self.logW= {self.logW.tolist()} "
                f"self.counts={self.counts.tolist()} "
                f"self.virtual_counts={self.virtual_counts.tolist()} "
                f"self.killed_mask={self.killed_mask.tolist()} "
                f"self.tactic_is_expandable={self.tactic_is_expandable.tolist()} "
                f"expansion_time={expansion_time} "
                f"solving_tactics={self.solving_tactics}"
            )

        # compute policy / make sure nothing masked can be selected.
        res = self._policy.get(Q, counts, self.priors)
        if res[self.killed_mask].sum() != 0:
            raise RuntimeError(
                f"Possible to choose masked tactic! "
                f"res={res} "
                f"res[self.killed_mask]={res[self.killed_mask]} "
                f"Q={Q.tolist()} "
                f"self.logW= {self.logW.tolist()} "
                f"self.counts={self.counts.tolist()} "
                f"self.virtual_counts={self.virtual_counts.tolist()} "
                f"self.killed_mask={self.killed_mask.tolist()} "
                f"self.tactic_is_expandable={self.tactic_is_expandable.tolist()} "
                f"expansion_time={expansion_time} "
                f"solving_tactics={self.solving_tactics}"
            )

        return res

    def update(self, tactic_id: int, value: float) -> None:
        assert value <= 0, value
        assert self.logW is not None
        assert self.counts is not None
        self.counts[tactic_id] += 1
        self.logW[tactic_id] = torch.logsumexp(
            torch.DoubleTensor([self.logW[tactic_id], value]), 0
        ).item()

    def value(self) -> float:
        assert self.logW is not None
        assert self.counts is not None
        # Order here is critical, since is_solved_leaf => is_terminal. solved => is_solved_leaf
        if self.solved:
            return 0.0  # log(1)
        elif self.is_terminal():  # is_terminal == no tactics here.
            return -math.inf  # log(0)
        elif self.counts is None or self.counts.sum().item() == 0:
            to_ret = self.log_critic
        else:  # follow the policy
            # We can use default arguments for policy here since self.solved is False
            # Hence, there are no solving tactics.
            tactic_id = self.policy().argmax()
            if self.counts[tactic_id] > 0:
                to_ret = self.logW[tactic_id] - math.log(self.counts[tactic_id])
            else:
                to_ret = self.log_critic
            assert to_ret <= 0, (
                f"not a logprob",
                math.exp(self.log_critic),
                math.exp(to_ret),
                self.logW[tactic_id],
                self.counts[tactic_id],
            )
        return min(0, to_ret)
