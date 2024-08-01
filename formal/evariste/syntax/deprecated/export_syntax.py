# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import re
import logging
import argparse
import itertools
import shutil
import pdb
import io
import numpy as np

from collections import deque, Counter, OrderedDict
from syntax_engine import SyntaxEngine
from envs.metamath import get_parser

logger = logging.getLogger()

syntatic = set(["setvar", "wff", "class"])


def serialize_axiom(output, label, assertion):
    serialize_assertion(output, label, assertion, "$a")


def serialize_assertion(output, label, assertion, assertion_type="$p"):
    output.write("${\n")
    tokens = assertion["tokens"]

    for var in assertion["active_vars"]:
        if var in tokens:
            output.write(f"    $v {var} $.\n")

    ftbl = {}
    for lbl, expr in assertion["active_f_labels"].items():
        type = expr[0]
        var = expr[1]
        new_label = f"{label}.{lbl}"
        expr = f"{type} {var}"
        ftbl[expr] = new_label
        if var in tokens:
            output.write(f"    {new_label} $f {expr} $.\n")

    if assertion_type == "$p":
        # Build proof stack
        root_node = assertion["proof_tree"]
        proof = root_node.proof(ftbl)

        output.write(
            f"    {label} $p {' '.join(tokens)} $= \n    {' '.join(proof)} $.\n"
        )
    else:
        output.write(f"    {label} $a {' '.join(tokens)} $.\n")

    output.write("$}\n")


def export_syntax(args):
    dir = pathlib.Path(__file__).parent.absolute()
    # create Metamath instance
    mm_env = SyntaxEngine(filepath=args.database_path, args=args)

    mm_env.process()

    output = io.StringIO()

    constants = mm_env.fs.frames[0].c
    for c in constants:
        output.write(f"$c {c} $.\n")
    output.write("\n$( ============================ $)\n\n")

    for lbl, (lbl_type, assertion) in mm_env.labels.items():
        if (lbl_type == "$a" or lbl_type == "$p") and assertion["tokens"][
            0
        ] in syntatic:
            serialize_axiom(output, lbl, assertion)

    output.write("\n$( ============================ $)\n\n")

    for i, wff in enumerate(mm_env.wffs):
        serialize_assertion(output, f"syn.prop.{i}", wff)

    fname = f"{dir}/syn_small_test.mm"
    with open(fname, "w") as fd:
        output.seek(0)
        shutil.copyfileobj(output, fd)
    print(f"Saved {fname}")


if __name__ == "__main__":

    # parse arguments
    parser = get_parser()
    args = parser.parse_args()

    args.stop_label = "nfmod"
    args.database_path = ""
    assert os.path.isfile(args.database_path)

    logging.basicConfig(level=logging.INFO)

    export_syntax(args)
