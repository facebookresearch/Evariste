# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from evariste.syntax.deprecated.parser import Parser

parser = Parser("holophrasm", cache=True)
# a = PyGrammar(b"Evariste/formal/lol.in")
a = parser.parsers["Early"]
# print(a.parse(b"wff ( 2 + 1 ) = 3"))
#
# print(a.parse(b"wffa ( 2 + 1 ) = 3"))
# print("lhs")
# print(parser.parse("wff ( -. ph -> ps )".split()))
# print(a.parse("wff ( -. ph -> ps )"))


print("rhs")
print(parser.parse("wff ( ph \\/ ps )".split()))
print(a.parse("wff ( ph \\/ ps )"))
#
# print("full")
# print(a.parse("wff ( ( -. ph -> ps ) <-> ( ph \/ ps ) )"))


# a = PyGrammar(
#     b"Evariste/formal/evariste/syntax/earley/input_2.in"
# )
#
# a.parse(b"2 + 3 * 4")
