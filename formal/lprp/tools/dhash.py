# Copyright (c) Meta Platforms, Inc. and affiliates.

from mpmath import mp, mpf, fmod
import hashlib
import math

# mp.dps = 50

# def hash_string_to_int(arg):
#     return int(hashlib.sha256(arg.encode("utf-8")).hexdigest(), 16) % 10**30

# def hash_string_to_float(arg):
#     x = mpf(hash_string_to_int(arg))
#     return fmod(x * mp.pi, mpf(1.))

# def get_split(arg):
#     float_hash = hash_string_to_float(arg)
#     if float_hash < 0.80: return "train"
#     elif float_hash < 0.845: return "valid"
#     else: return "test"


import itertools
import numpy as np
from mpmath import mp, mpf, fmod
import hashlib
import math
import collections
import os

mp.dps = 50


def _parse_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("decls_file")
    parser.add_argument(
        "valid_frac", type=float, help="number between 0 and 100", default=4.5
    )
    parser.add_argument(
        "test_frac", type=float, help="number between 0 and 100", default=15.5
    )
    parser.add_argument("dest_dir")
    return parser.parse_args()


def hash_string_to_int(arg):
    return int(hashlib.sha256(arg.encode("utf-8")).hexdigest(), 16) % 10 ** 30


def hash_string_to_float(arg):
    x = mpf(hash_string_to_int(arg))
    return fmod(x * mp.pi, mpf(1.0))


def get_split(arg, train_threshold=0.80, valid_threshold=0.845):
    float_hash = hash_string_to_float(arg.split()[0])
    if float_hash < train_threshold:
        return "train"
    elif float_hash < valid_threshold:
        return "valid"
    else:
        return "test"


def _main(decls_file: str, valid_frac: float, test_frac: float, dest_dir: str):
    valid_frac = 0.01 * valid_frac
    test_frac = 0.01 * test_frac
    with open(decls_file, "r") as f:
        decls = f.readlines()
    dataset = collections.defaultdict(list)
    train_threshold = 1.0 - valid_frac - test_frac
    valid_threshold = 1.0 - test_frac
    print(f"TRAIN THRESHOLD: {train_threshold} VALID_THRESHOLD: {valid_threshold}")
    float_in_unit_interval = lambda x: 0 <= x and x <= 1
    assert float_in_unit_interval(train_threshold) and float_in_unit_interval(
        valid_threshold
    )
    for decl in decls:
        dataset[get_split(decl, train_threshold, valid_threshold)].append(decl)

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for split in ["train", "valid", "test"]:
        with open(os.path.join(dest_dir, f"{split}_decls.log"), "w") as f:
            for decl in dataset[split]:
                f.write(decl)


if __name__ == "__main__":
    opts = _parse_main()
    _main(**vars(opts))
