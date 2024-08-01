# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from lark.lexer import Lexer, Token


class MetamathLexer(Lexer):
    """
        Converts a list of strings (Metamath tokens)
        into Lark Token objects, using the terminal_lookup to 
        map to the terminal's type. The types are strings of the form
        "T{i}" where i is a positive integer.
    """

    def __init__(self, lexer_conf=None, re_=None):
        self.module = None  # Set by parser
        pass

    def lex(self, data):
        pos_in_stream = 0
        for i, obj in enumerate(data):
            terminal_type = self.module.terminal_lookup[obj]
            yield Token(
                terminal_type,
                obj,
                column=i,
                line=0,
                pos_in_stream=pos_in_stream,
                end_pos=pos_in_stream + len(obj),
            )
            pos_in_stream = pos_in_stream + len(obj) + 1
