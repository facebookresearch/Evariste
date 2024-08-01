# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import tempfile
import os
import shutil
import pytest
import pickle
from importlib import import_module
from lark import LarkError

from params import ConfStore
from evariste.syntax.parser import get_parser
from evariste.syntax.deprecated.parser import LALRParser

dummy_grammar = """#num# 0 1
$start $e $*start
$e $e * $b $*mul
$e $e + $b $*add
$e $b $*eb
$b #num#"""

dummy_grammar_2 = """#num# 0 1
$start $e $*start
$e $e * $b $*mul
$e $e - $b $*add
$e $b $*eb
$b #num#"""


@pytest.mark.slow
class TestBacktracking:
    def test_grammar(self):
        test_dir = tempfile.mkdtemp()
        g_path = os.path.join(test_dir, "grammar.in")
        try:
            with open(g_path, "w+") as f:
                f.write(dummy_grammar)
            g = get_parser(g_path)
            first_hash = g.hash
            g = get_parser(g_path)  # this should just reload the pkl
            second_hash = g.hash
            assert first_hash == second_hash

            g.parse("1 * 0 + 1 * 0".split())

            with open(g_path, "w+") as f:
                f.write(dummy_grammar_2)
            g = get_parser(g_path)
            third_hash = g.hash
            assert first_hash != third_hash
        finally:
            shutil.rmtree(test_dir)


@pytest.mark.slow
class TestBacktrackingProofs:
    def test_grammar(self):
        dataset = ConfStore["holophrasm"]

        with open(os.path.join(dataset.data_dir, "proof_trees.pkl"), "rb") as f:
            proof_trees = pickle.load(f)

        parser = get_parser("holophrasm")
        old_parser = LALRParser(
            import_module(f".holophrasm_lalr", "evariste.syntax.deprecated.grammars"),
            cache=True,
        )

        skipped = set()

        def check_parser(label, cur_node):
            cur_statement = cur_node.statement
            if cur_statement[0] == "|-":
                cur_statement[0] = "wff"

            parse_tree = parser.parse(cur_statement)
            my_proof = " ".join(parser.parse_to_proof(parse_tree))
            try:
                parse_tree = old_parser.parse(cur_statement)
                old_proof = " ".join(old_parser.parse_to_proof(parse_tree))
                assert my_proof == old_proof, (label, my_proof, old_proof)
            except LarkError:
                skipped.add(cur_node.statement_str)

            if hasattr(cur_node, "children"):
                for c in cur_node.children:
                    check_parser(label, c)

        for i, (label, root) in enumerate(proof_trees.items()):
            if i % 10 == 0:
                print(i, len(skipped))
            check_parser(label, root)
