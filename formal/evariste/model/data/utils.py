# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple, List, Set
import numpy as np
import torch

from evariste.utils import COND_TOK


class SplitException(Exception):
    pass


def split_task(task: str) -> Tuple[List[str], List[str], Set[str]]:
    task = task.replace(COND_TOK, "")
    s = task.split("_")
    if (s[1], s[-1]) not in {
        ("x2y", "seq2seq"),
        ("x2y", "distillation"),
        ("syntheticrwalk", "seq2seq"),
        ("syntheticgraph", "seq2seq"),
        ("synthetic2", "seq2seq"),
        ("genwithsimp0905", "seq2seq"),  # doesn't work. too simple
        ("bwd", "rl"),
    }:
        raise SplitException
    assert len(s) == 4, s
    s_x, s_y = [t.split("-") for t in s[2].split("--")]
    s_xy = set(s_x) | set(s_y)
    return s_x, s_y, s_xy


def create_attention_mask(sequence):
    """
    Create attention mask.
    `sequence` is a list of dictionaries with `index`, `type`, `tokens` and `attn`.
    """
    # positions of each sequence [a, b[
    positions = {}
    last_pos = 0
    for x in sequence:
        slen = len(x["tokens"])
        k = (x["index"], x["type"])
        assert k not in positions
        positions[k] = (last_pos, last_pos + slen)
        last_pos += slen
    assert all(all(y in positions for y in x["attn"]) for x in sequence)

    # create attention mask
    slen = last_pos
    mask = torch.zeros((slen, slen), dtype=torch.bool)
    for x in sequence:
        index = x["index"]
        xtype = x["type"]
        a, b = positions[(index, xtype)]
        xlen = b - a
        assert xlen == len(x["tokens"])
        # attend previous tokens in the same sequence (but also the token itself)
        for i in range(xlen):
            mask[a + i, a : a + i + 1] = 1
        # attend other elements
        for to_attend in x["attn"]:
            assert to_attend != (index, xtype)
            c, d = positions[to_attend]
            mask[a:b, c:d] = 1

    return mask


if __name__ == "__main__":

    def visualize_attn_mask(sequence):
        """
        Visualize attention mask.
        """
        print("\n" + "=" * 100)
        for x in sequence:
            print(f"{x['index']}{x['type']} (size {len(x['tokens'])}): {x['attn']}")
        print("")
        labels = [f"{x['index']}{x['type']}" for x in sequence]
        mask = create_attention_mask(sequence)
        i = 0
        s = "   |  " + "".join(
            (f"{l} " + f"   " * (len(x["tokens"]) - 1))
            for x, l in zip(sequence, labels)
        )
        print(s)
        print("-" * len(s))
        for x, l in zip(sequence, labels):
            for _ in range(len(x["tokens"])):
                print(l + " | " + "".join(f" {int(y)} " for y in mask[i]))
                i += 1

    sequence1 = [
        {
            "index": 0,
            "type": "a",
            "tokens": [None for _ in range(3)],
            "attn": {(2, "a")},
        },
        {"index": 1, "type": "a", "tokens": [None for _ in range(6)], "attn": {}},
        {
            "index": 2,
            "type": "a",
            "tokens": [None for _ in range(1)],
            "attn": {(1, "b"), (3, "a")},
        },
        {
            "index": 3,
            "type": "a",
            "tokens": [None for _ in range(7)],
            "attn": {(0, "a"), (2, "a")},
        },
        {
            "index": 0,
            "type": "b",
            "tokens": [None for _ in range(2)],
            "attn": {(1, "b")},
        },
        {
            "index": 1,
            "type": "b",
            "tokens": [None for _ in range(4)],
            "attn": {(0, "b"), (1, "a")},
        },
    ]
    visualize_attn_mask(sequence1)

    sequence2 = [
        sequence1[i] for i in np.random.RandomState(0).permutation(len(sequence1))
    ]
    visualize_attn_mask(sequence2)
