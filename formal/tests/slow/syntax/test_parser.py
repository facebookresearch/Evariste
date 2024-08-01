# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytest
from lark.exceptions import LarkError
from evariste.syntax.deprecated.parser import Parser

parser = Parser()


def test_parse():
    # A selection of fairly complicated formulas
    # the formulas and proofs are directly extracted from set.mm
    tests = [
        (
            "wff ( -. ( ( ph -> ps ) -> -. ( ps -> ph ) ) -> ( ph <-> ps ) )",
            "wph wps wi wps wph wi wn wi wn wph wps wb wi",
        ),
        (
            "wff ( ( ps /\ ch ) /\ ( ( th /\ si ) /\ ( et /\ ze ) ) )",
            "wps wch wa wth wsi wa wet wze wa wa wa",
        ),
        (
            "wff ( z = w -> A. x ( x = y -> z = w ) )",
            "vz cv vw cv wceq vx cv vy cv wceq vz cv vw cv wceq wi vx wal wi",
        ),
        # This one is not parsable by the LALR parser
        (
            "wff ( A. z ( z e. x <-> z e. y ) -> x = y )",
            "vz cv vx cv wcel vz cv vy cv wcel wb vz wal vx cv vy cv wceq wi",
        ),
    ]

    for test in tests:
        expr = test[0].split(" ")
        proof = test[1].split(" ")
        parse_tree = parser.parse(expr)
        gen_proof = parser.parse_to_proof(parse_tree)
        assert proof == gen_proof, f"wrong proof for {test}"
    with pytest.raises(LarkError):
        parser.parse("wff ( ph".split())
