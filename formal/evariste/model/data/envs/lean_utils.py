# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pickle
from typing import Dict, Tuple, List, Any, Iterator

from pathlib import Path
import numpy as np

from leanml.parse_goal import StructuredGoal

from evariste.backward.env.lean.tokenizer import LeanTokenizer
from evariste.model.data.dictionary import (
    EOS_WORD,
    B_HYP_WORD,
    E_HYP_WORD,
    B_GOAL_WORD,
    E_GOAL_WORD,
    B_CMD_WORD,
    E_CMD_WORD,
)
from evariste.backward.graph import Proof, Theorem, Tactic


def select_sub_toks(toks: List[str], p_all: float):
    n = len(toks)
    assert n > 0
    assert 0 <= p_all <= 1
    if np.random.rand() <= p_all:  # take all tokens
        return toks
    start = np.random.randint(n)
    size = np.random.randint(1, n - start + 1)
    toks = toks[start : start + size]
    assert len(toks) > 0
    return toks


def generate_goal_input(
    goal: StructuredGoal,
    tactic: str,
    tokenizer: LeanTokenizer,
    p_all_toks: float,
    p_n_hyps: float,
    p_concl: float,
    p_tactic: float,
    rng: np.random.RandomState,
) -> List[str]:
    """
    </s>
    optional: N_HYPS in output
    optional hyps
    optional conclusion
    optional tactic to apply
    </s>
    """
    assert all(0 <= p <= 1 for p in [p_all_toks, p_n_hyps, p_concl, p_tactic])
    enc_tokens = [EOS_WORD]

    # specify how many hypotheses we want to see
    if rng.rand() <= p_n_hyps:
        enc_tokens.extend(list(str(len(goal.hyps))))

    # add tokens from hypotheses
    n_hyps = rng.randint(len(goal.hyps) + 1)
    hyp_ids = rng.permutation(len(goal.hyps))[:n_hyps]
    for hyp_id in hyp_ids:
        tokens = tokenizer.encode(goal.hyps[hyp_id].hyp)
        sub_tokens = select_sub_toks(tokens, p_all=p_all_toks)
        enc_tokens.extend([B_HYP_WORD, *sub_tokens, E_HYP_WORD])

    # add tokens from the conclusion
    if rng.rand() <= p_concl:
        tokens = tokenizer.encode(goal.conclusion)
        sub_tokens = select_sub_toks(tokens, p_all=p_all_toks)
        enc_tokens.extend([B_GOAL_WORD, *sub_tokens, E_GOAL_WORD])

    # add tokens from the tactic
    if rng.rand() <= p_tactic:
        assert isinstance(tactic, str)
        tokens = tactic.split()
        sub_tokens = select_sub_toks(tokens, p_all=p_all_toks)
        enc_tokens.extend([B_CMD_WORD, *sub_tokens, E_CMD_WORD])

    enc_tokens.append(EOS_WORD)

    return enc_tokens


Step = Tuple[Theorem, Tactic, List[int]]


def load_first_proofs(first_proof_dir: Path) -> Iterator[Tuple[str, Proof]]:
    for p in first_proof_dir.iterdir():
        if not p.name.endswith(".pkl"):
            continue
        with p.open("rb") as fp:
            proof = pickle.load(fp)
            decl_name = p.stem
            yield decl_name, proof


def proof_to_items(label: str, proof: Proof, split: str) -> List[Dict[str, Any]]:
    items = []
    for thm, tac, _ in post_order_traversal(proof):
        from evariste.backward.env.lean.graph import LeanTheorem
        from evariste.backward.env.lean.graph import LeanTactic

        assert isinstance(thm, LeanTheorem)
        assert isinstance(tac, LeanTactic)
        item = {
            "name": label,
            "filename": "is it needed?",
            "split": split,
            "context": thm.context,
            "goal_pp": thm.conclusion,
            "statement": " ".join(thm.tokenize()),
            "tactic": " ".join(tac.tokenize()),
        }
        items.append(item)
    return items


def post_order_traversal(root_proof: Proof) -> List[Tuple[Theorem, Tactic, List[int]]]:
    order = []

    def _traverse(node: Proof) -> int:
        thm, tac, children = node
        cids = [_traverse(c) for c in children]
        this_id = len(order)
        order.append((thm, tac, cids))
        return this_id

    _traverse(root_proof)
    return order


if __name__ == "__main__":
    for _ in range(100):
        print(select_sub_toks(list(range(20)), 0))
