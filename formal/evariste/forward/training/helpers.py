# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from typing import List, TypeVar, Optional, Type, Tuple, Set, Generic, Callable

import numpy as np
from numpy.random.mtrand import RandomState

from evariste.backward.graph import Theorem, Tactic, MalformedTheorem
from evariste.comms.store import EmptyStore
from evariste.forward.core.generation_errors import MalformedCommand
from evariste.forward.common import ProofNode
from evariste.model.data.dictionary import (
    B_GOAL_WORD,
    E_GOAL_WORD,
    B_NODE_WORD,
    E_NODE_WORD,
    EOS_WORD,
    B_CMD_WORD,
    E_CMD_WORD,
    EOU_WORD,
    EMPTY_NODE_WORD,
)

logger = getLogger()


T = TypeVar("T")
SomeTactic = TypeVar("SomeTactic", bound=Tactic)
SomeTheorem = TypeVar("SomeTheorem", bound=Theorem)
SomeProofNode = TypeVar("SomeProofNode", bound=ProofNode)


def sample_from_cumulative(
    cumulative: np.ndarray, data: List[T], rng: RandomState
) -> T:
    if len(data) == 0:
        raise EmptyStore
    assert len(data) == len(
        cumulative
    ), f"len(cumu):{len(cumulative)}, len(data): {len(data)}"
    index = np.searchsorted(
        cumulative,  # a
        rng.random() * cumulative[-1],  # v
        side="right",  # a[i-1] <= v < a[i]
    )
    if index == len(data):
        # if this happen, should we change side from 'right' to 'left' ?
        logger.warning(
            f"Index too big index:{index}, "
            f"len(cumu):{len(cumulative)}, len(data): {len(data)}"
        )
        index = len(data) - 1

    assert index <= len(data), f"{index} > {len(data)}"

    return data[index]


def postorder_traversal(
    root: SomeProofNode, rng: Optional[RandomState], allow_global_hyps: bool = False
) -> List[SomeProofNode]:
    r"""
    Postorder traversal of the dag. Respect topological order
    dag:
            0
          / / \
        1  /   4
       /\ /
      2 3

    should return [2, 3, 1, 4, 0]
    Optionally shuffle children.

    If 'allow_global_hyps' nodes can be global hyps (like in MM or
    when using MCTS Subproofs)
    """

    order = []
    seen = set()

    def traverse(node: SomeProofNode):
        if not allow_global_hyps:
            assert not node.is_hyp
        if node.theorem in seen:
            return
        if node.is_hyp:
            assert allow_global_hyps
            # doing nothing
        else:
            children = list(node.children)
            if rng is not None:
                rng.shuffle(children)
            for child in children:
                traverse(child)
        assert node.theorem not in seen  # no cycle
        seen.add(node.theorem)
        order.append(node)

    traverse(root)

    return order


def _tokenize(node: Theorem) -> List[str]:
    toks = node.tokenize()
    assert toks[0] == B_GOAL_WORD
    assert toks[-1] == E_GOAL_WORD
    return toks[1:-1]


def tokenize_fwd_graph(
    goal: Optional[SomeTheorem],
    graph: List[SomeTheorem],
    include_goal: bool,
    conditioning: Optional[List[str]] = None,
    tokenize_thm_fn: Optional[Callable[[SomeTheorem], List[str]]] = None,
    tokenize_goal_fn: Optional[Callable[[SomeTheorem], List[str]]] = None,
) -> List[str]:
    # global hyps are repeated, remove them in tokenization

    tokenize = tokenize_thm_fn if tokenize_thm_fn else _tokenize
    if include_goal:
        assert goal is not None
        tokenize_goal = tokenize_goal_fn if tokenize_goal_fn else _tokenize
        goal_toks = [B_GOAL_WORD, *tokenize_goal(goal), E_GOAL_WORD]
    else:
        assert goal is None
        goal_toks = []

    cond_toks = conditioning if conditioning else []

    return [
        EOS_WORD,
        *cond_toks,
        *goal_toks,
        *(
            tok
            for theorem in graph
            for tok in (B_NODE_WORD, *tokenize(theorem), E_NODE_WORD,)
        ),
        EOS_WORD,
    ]


def tokenize_command(
    target: ProofNode,
    next_node_first: bool = True,
    predict_children: bool = False,
    tokenize_thm_fn: Optional[Callable[[SomeTheorem], List[str]]] = None,
) -> List[str]:
    tokenize = tokenize_thm_fn if tokenize_thm_fn else _tokenize
    next_node = [B_NODE_WORD, *tokenize(target.theorem), E_NODE_WORD]
    next_tactic = target.tactic.tokenize()
    dec_out = next_node + next_tactic if next_node_first else next_tactic + next_node

    auxiliary_predictions = []
    eou = False
    if predict_children:
        eou = True
        children = (
            [
                t
                for c in target.children
                for t in [B_NODE_WORD, *tokenize(c.theorem), E_NODE_WORD]
            ]
            if target.children
            else [EMPTY_NODE_WORD]
        )
        auxiliary_predictions.extend(children)

    if eou:
        return [EOS_WORD, *dec_out, EOU_WORD, *auxiliary_predictions, EOS_WORD]
    else:
        assert not auxiliary_predictions
        return [EOS_WORD, *dec_out, EOS_WORD]


def detokenize_command(
    command: List[str],
    tactic_cls: Type[SomeTactic],
    theorem_cls: Type[SomeTheorem],
    parse_thm_fn: Optional[Callable[[List[str]], SomeTheorem]] = None,
    next_node_first: bool = True,
    last_token: str = EOS_WORD,
) -> Tuple[SomeTheorem, SomeTactic]:
    def error(s: str):
        return MalformedCommand(s, command=" ".join(command))

    if not command[0] == EOS_WORD:
        raise error("Invalid EOS delimitors")
    if not command[-1] == last_token:
        raise error("Invalid EOS delimitors")
    if command.count(B_NODE_WORD) != 1 or command.count(E_NODE_WORD) != 1:
        raise error("Invalid goal delimitors")

    def _default_parse_thm_fn(cmd_) -> SomeTheorem:
        return theorem_cls.from_tokens([B_GOAL_WORD, *cmd_, E_GOAL_WORD])

    parse_thm_fn = parse_thm_fn if parse_thm_fn else _default_parse_thm_fn

    if next_node_first:
        if command[1] != B_NODE_WORD:
            raise error("Invalid goal delimitors")
        i = command.index(E_NODE_WORD)
        next_node = _parse_next_node(theorem_cls, parse_thm_fn, command, command[2:i])
        tactic = _parse_tactic(tactic_cls, command, sub_command=command[i + 1 : -1])
    else:
        if command[-2] != E_NODE_WORD:
            raise error("Invalid goal delimitors")
        i = command.index(B_NODE_WORD)
        tactic = _parse_tactic(tactic_cls, command, sub_command=command[1:i])
        next_node = _parse_next_node(
            theorem_cls, parse_thm_fn, command, command[i + 1 : -2]
        )

    return next_node, tactic


def _parse_next_node(
    theorem_cls: Type[SomeTheorem],
    parse_thm_fn: Optional[Callable[[List[str]], SomeTheorem]],
    full_command: List[str],
    sub_command: List[str],
) -> SomeTheorem:
    try:
        next_node = parse_thm_fn(sub_command)
        assert isinstance(next_node, theorem_cls)
    except MalformedTheorem as e:
        raise MalformedCommand(str(e), command=" ".join(full_command))
    except AssertionError:
        print(f"Error with command: {' '.join(full_command)}")
        raise
    return next_node


def _parse_tactic(
    tactic_cls: Type[SomeTactic], full_command: List[str], sub_command: List[str]
) -> SomeTactic:

    # parse tactic
    tactic = tactic_cls.from_tokens([B_CMD_WORD, *sub_command, E_CMD_WORD])
    assert isinstance(tactic, tactic_cls)
    if not tactic.is_valid:
        raise MalformedCommand(
            f"Invalid tactic: {tactic.error_msg}", command=" ".join(full_command)
        )
    return tactic


def count_unique_theorems(root: ProofNode, allow_global_hyps: bool = False) -> bool:
    seen: Set[Theorem] = set()

    def traverse(node: ProofNode):
        if node.theorem in seen:
            return 0
        seen.add(node.theorem)
        if allow_global_hyps and node.is_hyp:
            return 1
        return 1 + sum([traverse(c) for c in node.children])

    size = traverse(root)
    assert size == len(seen)

    return size
