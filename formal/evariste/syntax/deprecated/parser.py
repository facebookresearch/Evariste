# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List
from collections import OrderedDict
from importlib import import_module

from evariste.syntax.deprecated.parser_earley import EarleyParser
from evariste.syntax.deprecated.parser_lalr import LALRParser
from lark.exceptions import LarkError, LexError
from logging import getLogger


class Parser:
    """
        Metamath formula parser 
        uses a cascade of progressively slower parsers until one of them works
    """

    def __init__(self, grammar="recent", earley_type="lark", cache=True):
        """
        Parameters
        ----------
        grammar : string, optional
            The name of one of the grammars in the grammar folder.
            I.e. "recent" or "holophrasm"
        cache : bool, optional
            Should the parser table be read from a cache file to reduce loading time?
            currently this feature is broken in Lark, you must use 
            Aaron's branch to enable it: https://github.com/adefazio/lark
        """
        logger = getLogger()
        logger.warning(
            "This parser is deprecated. Please use faster evariste.syntax.parser"
        )
        self._grammar = grammar
        self.parsers = OrderedDict()

        lalr_module = import_module(
            f".{grammar}_lalr", "evariste.syntax.deprecated.grammars"
        )
        if earley_type == "lark":
            earley_module = import_module(
                f".{grammar}_earley", "evariste.syntax.deprecated.grammars"
            )
        else:
            earley_module = grammar
        self.parsers["LALR"] = LALRParser(lalr_module, cache=cache)
        self.parsers["Early"] = EarleyParser(
            parser_type=earley_type, grammar_module=earley_module
        )

    @property
    def uuid(self) -> str:
        return self._grammar

    def parse_batch(self, expressions: List[List[str]]):
        to_ret = []
        for exp in expressions:
            try:
                self.parse(exp)
                to_ret.append(True)
            except LarkError:
                to_ret.append(False)
        return to_ret

    def parse(self, expression: List[str]):
        """Parse a list of Metamath tokens into a Lark parse tree.
            
        Parameters
        ----------
        expression : list of str
            Metamath expression in tokenized form (Typically from .split(" "))
        Returns
        -------
        Tree
            A Lark parse tree
        """
        for i, (pname, parser) in enumerate(self.parsers.items()):
            try:
                parse = parser.parse(expression)
                parse.parser = pname
                return parse
            except LarkError as e:
                if i == len(self.parsers) - 1:
                    raise e
            except KeyError as e:
                raise LexError(
                    f"Unexpected token {e}."
                    "Are you using the right grammar for your set.mm ?"
                )

    def parse_to_proof(self, tree, assertion=None):
        """ Extract a syntax proof from a parse tree
        Parameters
        ----------
        tree : Tree
            Metamath proof tree as output from .parse
        assertion : dict, optional
            This object corresponds to a assertion dictionary as returned 
            by the metamath proof engine. It is only needed if the tree uses
            f hypotheses that are scoped locally rather than globally.

        Returns
        -------
        list of str
            A proof in list-of-labels format as used in the Metamath engine
        """
        return self.parsers[tree.parser].parse_to_proof(tree, assertion=assertion)
