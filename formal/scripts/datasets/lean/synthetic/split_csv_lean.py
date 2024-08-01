# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, List
from tqdm import tqdm
import os
import fire
import pandas as pd

import evariste.backward.env.lean.tokenizer as lean_tok
from evariste.backward.env.lean.tokenizer import LeanTokenizer
from evariste.backward.env.lean.graph import LeanContext, LeanTheorem, LeanTactic
from evariste.model.data.envs.lean import normalize_goal_pp


# initialize tokenizer
LeanTokenizer.build("bpe_arxiv_lean_utf8_20k_single_digits_no_unk")

# # sanity check
# xx = "finset α → Prop, h0 : p ∅, step : ∀ (a : α) (s : finset α), (∀ (x : α), x ∈ s → x < a) → p s → p (insert a s), n : ℕ, ihn : ∀ (s : finset α), s.card = n → p s, s : finset α, hn : s.card = n.succ, A : s.nonempty, B : s.max' A ∈ s ⊢ (s.erase (s.max' A)).card = n ⟦⟦⟦⟦⟦ ⟦⟦⟦⟦⟦⟦⟦⟦⟦⟦⟦⟦⟦⟦⟦⟦⟦ ₉₉₉₉₉₉₉₉ ₉₉"
# yy = "▁fin set ▁α ▁→ ▁Pro p, ▁h 0 ▁: ▁p ▁∅ , ▁step ▁: ▁∀ ▁(a ▁: ▁α ) ▁(s ▁: ▁fin set ▁α ), ▁( ∀ ▁(x ▁: ▁α ), ▁x ▁∈ ▁s ▁→ ▁x ▁< ▁a) ▁→ ▁p ▁s ▁→ ▁p ▁(in sert ▁a ▁s ), ▁n ▁: ▁ℕ , ▁i hn ▁: ▁∀ ▁(s ▁: ▁fin set ▁α ), ▁s. c ard ▁= ▁n ▁→ ▁p ▁s, ▁s ▁: ▁fin set ▁α, ▁h n ▁: ▁s. c ard ▁= ▁n . succ , ▁A ▁: ▁s. non empt y, ▁B ▁: ▁s. max ' ▁A ▁∈ ▁s ▁⊢ ▁( s. er ase ▁( s. max ' ▁A )). c ard ▁= ▁n ▁ ⟦ ⟦ ⟦ ⟦ ⟦ ▁ ⟦ ⟦ ⟦ ⟦ ⟦ ⟦ ⟦ ⟦ ⟦ ⟦ ⟦ ⟦ ⟦ ⟦ ⟦ ⟦ ⟦ ▁ ₉ ₉ ₉ ₉ ₉ ₉ ₉ ₉ ▁ ₉ ₉"
# assert lean_tok.tokenizer.encode(xx) == yy.split()


def split_file(path: str, n_chunks: int, chunk_id: Optional[int] = None):

    assert os.path.isfile(path)
    assert path.endswith(".csv")
    assert n_chunks > 1
    assert chunk_id is None or 0 <= chunk_id < n_chunks

    print(f"Loading data from {path} ...")
    data = pd.read_csv(path)
    print(f"Loaded {len(data)} rows.")

    fname = os.path.basename(path)
    assert fname.endswith(".csv")

    step = len(data) // n_chunks

    for cid, offset in enumerate(range(0, len(data), step)):

        # only process 1 chunk
        if chunk_id is not None and cid != chunk_id:
            continue

        # deep copy the data and reset the index (otherwise the computed
        # tokenizations will always be assigned to the first chunk)
        chunk_path = f"{path[:-4]}.{cid}.csv"
        chunk_data = data[offset : offset + step].copy(deep=True)
        chunk_data = chunk_data.reset_index(drop=True)

        # add tokenized data for fast reloading
        tok_statements: List[str] = []
        tok_tactics: List[str] = []
        print(f"Adding tokenized data ...")
        for _, row in tqdm(chunk_data.iterrows()):
            theorem = LeanTheorem(
                conclusion=normalize_goal_pp(row.goal_pp),
                context=LeanContext(namespaces=set()),
                state=None,
            )
            tactic = LeanTactic(row.human_tactic_code)
            tok_statements.append(" ".join(theorem.tokenize()))
            tok_tactics.append(" ".join(tactic.tokenize()))

        assert len(tok_statements) == len(tok_tactics) == len(chunk_data)
        chunk_data = chunk_data.assign(tok_statements=pd.Series(tok_statements))
        chunk_data = chunk_data.assign(tok_tactics=pd.Series(tok_tactics))

        print(f"Exporting {len(chunk_data)} rows in {chunk_path} ...")
        chunk_data.to_csv(chunk_path)


if __name__ == "__main__":
    fire.Fire(split_file)
