# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import unittest
from evariste.syntax.deprecated.parser_earley import EarleyParser
from evariste.syntax.deprecated.grammars import holophrasm_earley


class ParserTestEarley(unittest.TestCase):
    def setUp(self):
        print("Loading parser")
        self.parser = EarleyParser(holophrasm_earley)

    def test(self):
        # This was causing issues when the variables
        # were not length sorted in the grammar
        expr = "wff ( i =/= (/) /\ ( ( th /\ ta /\ ch ) /\ ph0 ) )".split(" ")
        parse_tree = self.parser.parse(expr)

        expr = "wff ( i =/= (/) /\ ( ( th /\ ta /\ ch ) /\ ph ) )".split(" ")
        parse_tree = self.parser.parse(expr)
