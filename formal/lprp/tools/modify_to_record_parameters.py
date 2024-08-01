# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
from pathlib import Path

from tokenize_lean_files import LeanFile

COMMANDS = [
    "theorem",
    "axiom",
    "axioms",
    "variable",
    "protected",
    "private",
    "hide",
    "definition",
    "meta",
    "mutual",
    "example",
    "noncomputable",
    "abbreviation",
    "variables",
    "parameter",
    "parameters",
    "constant",
    "constants",
    "using_well_founded",
    "[whnf]",
    "end",
    "namespace",
    "section",
    "prelude",
    "import",
    "inductive",
    "coinductive",
    "structure",
    "class",
    "universe",
    "universes",
    "local",
    "precedence",
    "reserve",
    "infixl",
    "infixr",
    "infix",
    "postfix",
    "prefix",
    "notation",
    "set_option",
    "open",
    "export",
    "@[",
    "attribute",
    "instance",
    "include",
    "omit",
    "init_quotient",
    "declare_trace",
    "add_key_equivalence",
    "run_cmd",
    "#check",
    "#reduce",
    "#eval",
    "#print",
    "#help",
    "#exit",
    "#compile",
    "#unify",
    "lemma",
    "def",
]


def next_token(lean_file, token):
    # note we use end_column here
    for t in lean_file.iter_tokens_right(token.line, token.end_column):
        return t


def prev_token(lean_file, token):
    for t in lean_file.iter_tokens_left(token.line, token.start_column):
        return t


def search_right(lean_file, start, targets):
    for t in lean_file.iter_tokens_right(start.line, start.start_column):
        if t.string in targets:
            return t
    return None


def search_left(lean_file, start, targets):
    for t in lean_file.iter_tokens_left(start.line, start.start_column):
        if t.string in targets:
            return t
    return None


def slice(lean_file, start, end):
    s = []
    for t in lean_file.iter_tokens_right(start.line, start.start_column):
        if t.line == end.line and t.start_column == end.start_column:
            return s
        s.append(t)

    return None


def main():
    assert len(sys.argv) == 2
    lean_dir = Path(sys.argv[1])
    assert lean_dir.is_dir

    for f in lean_dir.glob("**/*.lean"):
        # print(f)
        decls = set()
        lean_file = LeanFile(str(f))
        for line in lean_file.lines:
            for token in line:
                if token.string in ["parse", "itactic"]:
                    # TODO: need to check if parse or itactic is part of longer name.  If so, which name?
                    left = search_left(lean_file, token, COMMANDS)
                    if left is None or left.string != "def":
                        print(
                            "BAD LEFT:",
                            left.string,
                            " ",
                            "".join(t.string for t in line).strip(),
                        )
                        continue

                    # TODO: This search needs to skip over brakets!
                    right = search_right(
                        lean_file, next_token(lean_file, left), COMMANDS + [":=", "|"]
                    )
                    if (
                        right is None
                        or (right.line, right.start_column)
                        <= (token.line, token.start_column)
                        or right.string not in ["|", ":="]
                    ):
                        print(
                            "BAD RIGHT:",
                            right.string,
                            " ",
                            "".join(t.string for t in line).strip(),
                        )
                        continue

                    # TODO: Extract decl name.
                    # (Beware of tactic.interactive.name,
                    # and also skip anything with decl name parse or itactic)

                    # TODO: Count parameters instance
                    # TODO: Find parse arguement
                    # TODO: Replace string

                    s = slice(lean_file, left, right)
                    decl = "".join(t.string for t in s).strip()
                    if decl not in decls:
                        print(decl)
                        decls.add(decl)


if __name__ == "__main__":
    main()
