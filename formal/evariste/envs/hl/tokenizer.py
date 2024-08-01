# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import io
import os
import re
import sys
import argparse
from ply import lex
from collections import Counter
from typing import Dict


class InvalidCharacterException(Exception):
    pass


class Lexer(object):

    # List of token names.
    tokens = ("SEPARATOR", "DECIMAL", "INTEGER", "WORD", "OPERATOR")

    def t_SEPARATOR(self, t):
        r'[`"(){}\[\]]'
        return t

    def t_begin_blank(self, t):
        r"\s+"
        pass

    def t_DECIMAL(self, t):
        r"[0-9][0-9_]*\.[0-9]+"
        return t

    def t_INTEGER(self, t):
        r"[0-9][0-9_]*"
        return t

    def t_WORD(self, t):
        r"[a-zA-Z_][a-zA-Z0-9_']*"
        return t

    def t_OPERATOR(self, t):
        r'[^a-zA-Z0-9_`"(){}\[\]\s]+'
        return t

    # Special function for error handling.
    def t_ANY_error(self, t):
        raise InvalidCharacterException(
            f"Illegal character '{t.value[0]}' at line {t.lexer.lineno}\n"
        )

    # Build the lexer.
    def build(self):
        self.lexer = lex.lex(module=self)

    def run(self, s):
        self.lexer.input(s)
        tokens = [(tok.type, tok.value) for tok in self.lexer]
        return tokens


def tokenize_hl(s):
    """
    Run HOL-Light tokenizer.
    """
    assert type(s) is str
    # run the Lexer
    lexer = Lexer()
    lexer.build()
    output = lexer.run(s)

    # split integers and floats / split single quotes at the end of words
    tokens = []
    for cat, tok in output:
        if cat == "DECIMAL":
            tokens.extend(["<NUMBER>", *list(tok), "</NUMBER>"])
        elif cat == "INTEGER" and len(tok) > 1:
            tokens.extend(["<NUMBER>", *list(tok), "</NUMBER>"])
        elif cat == "WORD" and "'" in tok:
            tok = re.sub(r"'(?=[a-zA-Z0-9_]+)", "' @@@@", tok)
            tok = re.sub(r"'", " @@@@'", tok)
            tokens.append(tok)
        else:
            tokens.append(tok)

    # merge tokens
    s = " ".join(tokens)

    # handle operators in sets.ml, lines 28-32
    s = re.sub(r"<= _c(?= |$)", r"<=_c", s)
    s = re.sub(r">= _c(?= |$)", r">=_c", s)
    s = re.sub(r"< _c(?= |$)", r"<_c", s)
    s = re.sub(r"> _c(?= |$)", r">_c", s)
    s = re.sub(r"= _c(?= |$)", r"=_c", s)
    s = re.sub(r"\^ _c(?= |$)", r"^_c", s)
    s = re.sub(r"\+ _c(?= |$)", r"+_c", s)
    s = re.sub(r"\* _c(?= |$)", r"*_c", s)

    tokens = s.split()
    return tokens


def detokenize_hl(tokens):
    """
    Detokenize a sequence and make it compatible with HOL-Light.
    """
    assert type(tokens) is list

    # merge integers and floats
    _tokens = []
    current = []
    merging = False
    for t in tokens:
        if merging:
            if t == "</NUMBER>" or t == "</NUMBER>":
                _tokens.append("".join(current))
                merging = False
                del current[:]
            else:
                current.append(t)
        else:
            if t == "<NUMBER>" or t == "<NUMBER>":
                merging = True
                del current[:]
            else:
                _tokens.append(t)

    s = " ".join(_tokens)
    s = re.sub(r'" ([^"]*) "', r'"\1"', s)
    s = re.sub(r"` ([^`]*) `", r"`\1`", s)

    # merge single quotes back to their variable name
    s = s.replace(" @@@@", "")

    return s


if __name__ == "__main__":

    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="", help="Input file.")
    args = parser.parse_args()
    assert args.input_file == "" or os.path.isfile(args.input_file)

    # read from standard input, or from input file
    if args.input_file == "":
        source = sys.stdin.read()
    else:
        with io.open(args.input_file, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()

    tokens = tokenize_hl(source)
    print(f"Found {len(tokens)} tokens.")
    counts: Dict[str, int] = Counter(tokens)
    for t, c in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        if c == 1:
            break
        print(f"{t} - {c}")
