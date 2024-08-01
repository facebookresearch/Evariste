# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import subprocess
import collections
import json
import sys
from pprint import pprint
import re
import traceback

import extract_trace_data
from tokenize_lean_files import LeanFile

# Known issues:
# All of these issues show up when I use two methods to compute the tactic string.
# sometimes one method is right and sometimes another.
# I'd like the two methods to agree.  Here are the causes of the disagreement.
# * `<|>` is not recorded as it's own tactic
#   * I could find these and insert them into the tree
# * unexecuted tactics, both `t1 <|> t2` and `t1 ; t2` where t2 is not executed
#   * if I am careful I can find dead tactics on the right in `;` since the ';'
#     tactic would have only one child (on the right)
#   * same for <|> blocks (after I find the block)
# * trailing commas in begin...end or {...}
#   * since I know where the `end` or `}` is,
#     it is easy enough to check if there is a trailing comma
# * `t1 ; [t2, t3] ; [t4, t5]` is not fully handled
#   * maybe consider ";[]" to be its own type of tactic token.
#   * note the [] can only be on the right of the ";"

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

BINDERS = [
    "Pi",
    "forall",
    "∀",
    "Π",
    "Σ",
    "∑",
    "exists",
    "∃",
    "λ",
    "fun",
    "⋃",
    "∫⁻",
    "⨆",
    "∫",
    "⋂",
]
BRACKETS = [("{", "}"), ("(", ")"), ("[", "]"), ("⟨", "⟩"), ("⟪", "⟫")]


def collect_data(info_blocks):
    # store data in a two dimensional table.  Default value is None
    data = {}
    for info_block in info_blocks:
        filename = info_block["file_name"]
        for line in info_block["text"].split("\n"):
            if line.startswith("<PR>"):
                _, json_string = line.split(" ", 1)
                d = json.loads(json_string)
                d["filename"] = filename
                addr = hash((filename, d["line"], d["column"], d["depth"], d["index"]))
                if addr not in data:
                    data[addr] = collections.defaultdict(lambda: None)
                data[addr].update(d)
    return data


def add_addr_columns(data):
    for _, row in data.items():
        for key in list(row.keys()):
            if key.endswith("_line"):
                key2 = key[:-5]
                if row[key]:
                    row[key2 + "_addr"] = hash(
                        (
                            row["filename"],
                            row[key2 + "_line"],
                            row[key2 + "_column"],
                            row[key2 + "_depth"],
                            row[key2 + "_index"],
                        )
                    )


def add_forward_pointing_addrs(data):
    for addr, row in data.items():
        # first child
        if row["index"] == 1 and row["parent_addr"] is not None:
            parent_row = data[row["parent_addr"]]
            # there can be multiple "first children" in some cases
            # in that case, use the one in the first position
            if parent_row["first_child_addr"] is None or (
                row["line"],
                row["column"],
                row["index"],
            ) < (
                parent_row["first_child_line"],
                parent_row["first_child_column"],
                parent_row["first_child_index"],
            ):
                parent_row["first_child_line"] = row["line"]
                parent_row["first_child_column"] = row["column"]
                parent_row["first_child_depth"] = row["depth"]
                parent_row["first_child_index"] = row["index"]
                parent_row["first_child_addr"] = addr
        # last child
        if row["parent_addr"] is not None:
            parent_row = data[row["parent_addr"]]
            # there can be multiple "last children" in some cases
            # in that case, use the one in the last position
            if parent_row["last_child_addr"] is None or (
                row["line"],
                row["column"],
                row["index"],
            ) > (
                parent_row["last_child_line"],
                parent_row["last_child_column"],
                parent_row["last_child_index"],
            ):
                parent_row["last_child_line"] = row["line"]
                parent_row["last_child_column"] = row["column"]
                parent_row["last_child_depth"] = row["depth"]
                parent_row["last_child_index"] = row["index"]
                parent_row["last_child_addr"] = addr
        # next tactic
        if row["index"] > 1 and row["prev_addr"] is not None:
            prev_row = data[row["prev_addr"]]
            # there can be multiple "next tactics" in some cases
            # in that case, use the one in the first position
            if prev_row["next_addr"] is None or addr < prev_row["next_addr"]:
                prev_row["next_line"] = row["line"]
                prev_row["next_column"] = row["column"]
                prev_row["next_depth"] = row["depth"]
                prev_row["next_index"] = row["index"]
                prev_row["next_addr"] = addr
    # rightmost decendent
    depth_sorted_keys = sorted(data.keys(), key=lambda k: -data[k]["depth"])
    for addr in depth_sorted_keys:
        row = data[addr]
        # set to itself if no children (otherwise children will have already set it)
        if row["rightmost_decendent_addr"] is None:
            row["rightmost_decendent_line"] = row["line"]
            row["rightmost_decendent_column"] = row["column"]
            row["rightmost_decendent_depth"] = row["depth"]
            row["rightmost_decendent_index"] = row["index"]
            row["rightmost_decendent_addr"] = addr

        # set for parent
        if row["parent_addr"] is not None:
            parent_row = data[row["parent_addr"]]
            # there can be multiple "last children" in some cases
            # in that case, use the one in the last position
            if parent_row["rightmost_decendent_addr"] is None or (
                row["rightmost_decendent_line"],
                row["rightmost_decendent_column"],
                row["rightmost_decendent_index"],
            ) > (
                parent_row["rightmost_decendent_line"],
                parent_row["rightmost_decendent_column"],
                parent_row["rightmost_decendent_index"],
            ):
                parent_row["rightmost_decendent_line"] = row["rightmost_decendent_line"]
                parent_row["rightmost_decendent_column"] = row[
                    "rightmost_decendent_column"
                ]
                parent_row["rightmost_decendent_depth"] = row[
                    "rightmost_decendent_depth"
                ]
                parent_row["rightmost_decendent_index"] = row[
                    "rightmost_decendent_index"
                ]
                parent_row["rightmost_decendent_addr"] = row["rightmost_decendent_addr"]


def read_files(info_blocks):
    lean_files = {}
    for info in info_blocks:
        filename = info["file_name"]
        if filename in lean_files:
            continue

        lean_files[filename] = LeanFile(filename)
        """
        with open(filename, 'r') as f:
            block_comment = 0
            line_comment = False
            end_of_block_comment = False
            for l in f:
                new_l = ""
                for c1, c2 in zip(l, l[1:]):
                    if block_comment:
                        new_l += " "
                        if (c1, c2) == ('/', '-'):
                            block_comment += 1
                        elif (c1, c2) == ('-', '/'):
                            block_comment -= 1
                            if not block_comment:
                                end_of_block_comment = True
                    elif line_comment:
                        new_l += " "
                    elif (c1, c2) == ('/', '-'):
                        new_l += " "
                        block_comment += 1
                    elif (c1, c2) == ('-', '-'):
                        new_l += " "
                        line_comment = True
                    elif end_of_block_comment:
                        new_l += " "
                        end_of_block_comment = False
                    else:
                        new_l += c1
                file_lines[filename].append(new_l + "\n")
                line_comment = False
        """
    return lean_files


# def scan_name(s, i):
#    for j in range(i, len(s)):
#        # TODO: Make this list more complete
#        if s[j] in " \n,;{}()[]<>⟨⟩":
#            return s[i:j]
#    return s[i:]


def add_tactic_names(data, lean_files):
    for addr, row in data.items():
        lean_token = lean_files[row["filename"]].get_token(
            row["line"] - 1, row["column"] - 1
        )
        if lean_token.string == ";":
            tactic_name = "and_then"
            tactic_token = ";"
        elif lean_token.string == "}":
            tactic_name = "solve1"
            tactic_token = "{}"
        elif lean_token.string == "end":
            tactic_name = "solve1"
            tactic_token = "begin end"
        else:
            # tactic_name = scan_name(file_lines[row['filename']][row['line']-1], row['column']-1)
            tactic_name = lean_token.string
            tactic_token = lean_token.string
        row["tactic_name"] = tactic_name
        row["tactic_token"] = tactic_token


LEFT_TOKENS = ["begin", "by", ",", ";", "<|>", "{", "["]


def get_left_token(lean_file, line, column):
    combined_symbol_token = ""
    for token in lean_file.iter_tokens_left(line, column):
        if token.type == "alphanumeric" or token.type == "symbol":
            token_string = token.string
            break

    assert token_string in LEFT_TOKENS, token_string
    return (token_string, token.line, token.start_column)


def add_start_info(data, lean_files):
    depth_sorted_keys = sorted(data.keys(), key=lambda k: -data[k]["depth"])
    for addr in depth_sorted_keys:
        row = data[addr]
        if row["tactic_token"] == ";":
            assert row["first_child_addr"] is not None, pprint(row)

            # use the start, left token, and left token position of the first child instead
            first_child_row = data[row["first_child_addr"]]
            row["start_line"] = first_child_row["start_line"]
            row["start_column"] = first_child_row["start_column"]
            row["left_token"] = first_child_row["left_token"]
            row["left_token_line"] = first_child_row["left_token_line"]
            row["left_token_column"] = first_child_row["left_token_column"]

        elif row["tactic_token"] in ["{}", "begin end"]:
            assert row["first_child_addr"] is not None, pprint(row)

            # set the start to the left token position of the first child
            # get a new left token and left token position
            first_child_row = data[row["first_child_addr"]]
            row["start_line"] = first_child_row["left_token_line"]
            row["start_column"] = first_child_row["left_token_column"]

            left_token, left_token_line, left_token_column = get_left_token(
                lean_files[row["filename"]],
                row["start_line"] - 1,
                row["start_column"] - 1,
            )
            row["left_token"] = left_token
            row["left_token_line"] = left_token_line + 1
            row["left_token_column"] = left_token_column + 1

        else:
            row["start_line"] = row["line"]
            row["start_column"] = row["column"]
            left_token, left_token_line, left_token_column = get_left_token(
                lean_files[row["filename"]],
                row["start_line"] - 1,
                row["start_column"] - 1,
            )
            row["left_token"] = left_token
            row["left_token_line"] = left_token_line + 1
            row["left_token_column"] = left_token_column + 1


def add_block_type(data):
    for _, row in data.items():
        head_row = data[row["block_addr"]]
        if (
            head_row["parent_addr"] is not None
            and data[head_row["parent_addr"]]["tactic_token"] == ";"
        ):
            row["block_left_token"] = ";"
        else:
            row["block_left_token"] = head_row["left_token"]


# def split(lines, line0, col0, line1, col1):
#     c0 = col0
#     s = ""
#     for l in range(line0, line1):
#         s += lines[l][c0:]
#         c0 = 0
#     s += lines[line1][c0: col1]
#     return s


def get_text(lean_file, line0, col0, line1, col1):
    assert (line0, col0) <= (line1, col1), ((line0, col0), (line1, col1))
    s = ""
    for token in lean_file.iter_tokens_right(line0, col0):

        if (token.line, token.start_column) < (line1, col1):
            if token.type not in ["line_comment", "block_comment"]:
                s += token.string
            assert (token.line, token.end_column) <= (line1, col1)
            continue

        if (token.line, token.start_column) == (line1, col1):
            return s

    assert (
        False
    ), "Reached the end of the file at ({},{}).  Span: ({},{})-({},{})".format(
        token.line, token.end_column, line0, col0, line1, col1
    )


def scan_to_end_of_tactic(
    lean_file, start_line, start_column, pairs, min_line=None, min_column=None
):
    s = ""
    stack = []  # the symbol to go up a level in the stack
    for token in lean_file.iter_tokens_right(start_line, start_column):
        if token.type in ["line_comment", "block_comment"]:
            continue

        # check if I've reached the end of unmatched pair
        if not stack and (
            min_line is None
            or min_column is None
            or (token.line, token.start_column) > (min_line, min_column)
        ):
            for left, right in pairs:
                if token.string == right:
                    return s, token.line, token.start_column

        s += token.string

        if stack:
            # check if I've gone up the stack
            if token.string == stack[-1]:
                stack.pop()
                continue

        # check if entering new pair
        for left, right in pairs:
            if left is not None and token.string == left:
                stack.append(right)
                continue

    # reached end of file
    return s, token.line, token.start_column


def clean_whitespace(s):
    return " ".join(s.split())


def add_tactic_strings(data, lean_files):
    depth_sorted_keys = sorted(data.keys(), key=lambda k: data[k]["depth"])
    for addr in depth_sorted_keys:
        row = data[addr]
        if row["next_addr"] is not None and (row["next_line"], row["next_column"]) > (
            row["line"],
            row["column"],
        ):
            next_row = data[row["next_addr"]]
            row["end_line"] = next_row["left_token_line"]
            row["end_column"] = next_row["left_token_column"]
            row["tactic_string"] = clean_whitespace(
                get_text(
                    lean_files[row["filename"]],
                    row["start_line"] - 1,
                    row["start_column"] - 1,
                    row["end_line"] - 1,
                    row["end_column"] - 1,
                ).strip()
            )
        elif row["block_left_token"] == ";":
            parent_row = data[row["parent_addr"]]
            row["end_line"] = parent_row["end_line"]
            row["end_column"] = parent_row["end_column"]
            row["tactic_string"] = clean_whitespace(
                get_text(
                    lean_files[row["filename"]],
                    row["start_line"] - 1,
                    row["start_column"] - 1,
                    row["end_line"] - 1,
                    row["end_column"] - 1,
                ).strip()
            )
        else:
            # there is no next tactic, so scan until some end token is reached
            if row["block_left_token"] == "{":
                tactic_string, end_line, end_row = scan_to_end_of_tactic(
                    lean_files[row["filename"]],
                    row["start_line"] - 1,
                    row["start_column"] - 1,
                    pairs=[("{", "}")],
                    # min_line = row['rightmost_decendent_line']-1,
                    # min_column = row['rightmost_decendent_column']-1,
                )
            elif row["block_left_token"] == "begin":
                tactic_string, end_line, end_row = scan_to_end_of_tactic(
                    lean_files[row["filename"]],
                    row["start_line"] - 1,
                    row["start_column"] - 1,
                    pairs=BRACKETS + [("begin", "end"), ("match", "end")],
                    # min_line = row['rightmost_decendent_line']-1,
                    # min_column = row['rightmost_decendent_column']-1,
                )
            elif row["block_left_token"] in ["by", "<|>"]:
                tactic_string, end_line, end_row = scan_to_end_of_tactic(
                    lean_files[row["filename"]],
                    row["start_line"] - 1,
                    row["start_column"] - 1,
                    pairs=[(None, cmd) for cmd in COMMANDS]
                    + [(b, ",") for b in BINDERS]
                    + BRACKETS
                    + [
                        (None, ";"),
                        (None, "|"),
                        (None, "..."),
                        ("then", "else"),
                        ("begin", "end"),
                        ("match", "end"),
                    ],
                    min_line=row["rightmost_decendent_line"] - 1,
                    min_column=row["rightmost_decendent_column"] - 1,
                )
            else:
                assert None, "Unexpected left token: " + str(row["block_left_token"])

            row["end_line"] = end_line + 1
            row["end_column"] = end_row + 1
            row["tactic_string"] = clean_whitespace(tactic_string.strip())

        tactic_string2, _, _ = scan_to_end_of_tactic(
            lean_files[row["filename"]],
            row["start_line"] - 1,
            row["start_column"] - 1,
            pairs=[(None, cmd) for cmd in COMMANDS]
            + [(b, ",") for b in BINDERS]
            + BRACKETS
            + [
                (None, "|"),
                (None, ";"),
                (None, "..."),
                ("then", "else"),
                ("begin", "end"),
                ("match", "end"),
            ],
            min_line=row["rightmost_decendent_line"] - 1,
            min_column=row["rightmost_decendent_column"] - 1,
        )
        row["tactic_string2"] = clean_whitespace(tactic_string2.strip())

        # useful for debugging
        row["right_token"] = (
            lean_files[row["filename"]]
            .get_token(row["end_line"] - 1, row["end_column"] - 1)
            .string
        )


def check_results(data):
    tactic_strings = {}
    for addr, row in data.items():
        # check for completeness
        assert row["start_line"] is not None, pprint(row)
        assert row["end_line"] is not None, pprint(row)
        assert row["start_column"] is not None, pprint(row)
        assert row["end_column"] is not None, pprint(row)
        assert row["tactic_string"] is not None, pprint(row)

        if row["tactic_token"] != ";":
            if (row["line"], row["column"]) in tactic_strings:
                s1 = tactic_strings[row["line"], row["column"]]
                s2 = row["tactic_string"]
                if s1 != s2:
                    print("Different tactic strings at same location: " + s1 + " " + s2)
            else:
                tactic_strings[row["line"], row["column"]] = row["tactic_string"]

        # tactic strings don't contain comments or new lines
        assert "--" not in row["tactic_string"], pprint(row)
        assert "/-" not in row["tactic_string"], pprint(row)
        assert "\n" not in row["tactic_string"], pprint(row)

        if row["tactic_string2"] != row["tactic_string"]:
            if (
                row["tactic_string"][-1] in [",", ";", "]"]
                and row["tactic_string"][:-1].strip() == row["tactic_string2"]
            ):
                pass
            else:
                print(
                    "WARNING: string1: ",
                    row["tactic_token"][0],
                    "  ",
                    row["tactic_string"],
                )
                print(
                    "WARNING: string2: ",
                    row["tactic_token"][0],
                    "  ",
                    row["tactic_string2"],
                )
                parent_row = row
                while parent_row["parent_addr"] is not None:
                    parent_row = data[parent_row["parent_addr"]]
                    print(
                        "WARNING: parent1: ",
                        parent_row["tactic_token"][0],
                        "  ",
                        parent_row["tactic_string"],
                    )
                    print(
                        "WARNING: parent2: ",
                        parent_row["tactic_token"][0],
                        "  ",
                        parent_row["tactic_string2"],
                    )
                print()

        # assert not row['tactic_string'].endswith(";"), pprint(row)
        # assert not row['tactic_string'].endswith(","), pprint(row)
        if row["parent_addr"] is not None:
            parent_row = data[row["parent_addr"]]
            # assert row['tactic_string'] != parent_row['tactic_string'], pprint(row)

        # check that tactic_token contains correct characters
        # this doesn't consider unicode, so it might need to be updated
        # see https://leanprover.zulipchat.com/#narrow/stream/113488-general/topic/characters.20which.20can.20or.20can't.20be.20in.20names
        assert row["tactic_token"] in ("begin end", "{}", ";") or re.match(
            r"[A-Za-z_][A-Za-z0-9_']*", row["tactic_token"]
        ), pprint(row)

        # check that tactic_token is one of three special types
        # otherwise the tactic string should start with the tactic token
        if (
            row["tactic_token"] not in ("begin end", "{}", ";")
            and row["tactic_string"] is not None
        ):
            assert row["tactic_string"].startswith(row["tactic_token"]), pprint(row)

        # check that a tactic string contains all of its children
        if row["parent_addr"] is not None:
            parent_row = data[row["parent_addr"]]
            assert (parent_row["start_line"], parent_row["start_column"]) <= (
                row["start_line"],
                row["start_column"],
            )
            assert (parent_row["end_line"], parent_row["end_column"]) >= (
                row["end_line"],
                row["end_column"],
            )

        # check that a tactic string doesn't overlap any of its neighbors


def file_suffix(path_map, filename):
    for p in path_map:
        if filename.startswith(p):
            return path_map[p] + filename[len(p) :]


def fix_keys(data):
    new_data = {}
    for k, d in data.items():
        new_key = hash((d["filename"], d["line"], d["column"], d["depth"], d["index"]))
        new_data[new_key] = d
    return new_data


def main():
    import pandas as pd

    assert len(sys.argv) == 2
    data_dir = sys.argv[1]
    assert data_dir.endswith("/")

    print("Extracting traced data ...")
    data_tables, lean_files = extract_trace_data.extract_data(data_dir)
    tactic_data = data_tables["tactic_trace"]

    # process one file at a time
    for f in lean_files:
        try:
            print(f)
            data = {k: row for k, row in tactic_data.items() if row["filename"] == f}

            data = fix_keys(data)
            # print(pd.DataFrame(list(data.values())))
            add_addr_columns(data)
            add_forward_pointing_addrs(data)
            # print(pd.DataFrame(list(data.values())).info())

            add_tactic_names(data, lean_files)
            add_start_info(data, lean_files)
            add_block_type(data)
            add_tactic_strings(data, lean_files)

            sorted_rows = sorted(
                data.values(),
                key=lambda row: (
                    row["filename"],
                    row["proof_line"],
                    row["proof_column"],
                    row["depth"],
                    row["index"],
                    row["line"],
                    row["column"],
                ),
            )

            df = pd.DataFrame(list(sorted_rows))
            # df = df[df['left_token'] == 'by']
            # df = df[df['tactic_string'].str.contains('<|>')]
            for _, row in df.iterrows():
                # print(">", row['line'], row['column'], row['tactic_string'])
                # print(">>", row['right_token'])
                pass
            # print(pd.DataFrame(list(data.values()))[['prev_line', 'next_addr']])
            check_results(data)
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    main()
