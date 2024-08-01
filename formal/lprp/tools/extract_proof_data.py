# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import collections
import json
from os import pardir
from pathlib import Path
import sys
import traceback
from traceback import extract_stack
from typing import Any, Dict, List, Set, Tuple, Optional, Union

from parse_lean import AST, LeanParser
from tokenize_lean_files import LeanFile, TokenType


def get_traced_data(data_dir: Path, table: str) -> List[Dict[str, Any]]:
    with open(data_dir / "raw_traced_data" / (table + ".json"), "r") as f:
        return json.load(f)


def save_data_tables(data_tables: Dict[str, List[Dict[str, Any]]], data_dir: Path):
    dir = data_dir / "extracted_proof_data"
    dir.mkdir(exist_ok=True)
    for table_name, table in data_tables.items():
        # save each table to a file
        filename = table_name + ".json"
        with open(dir / filename, "w") as outfile:
            json.dump(table, outfile, indent=2)


class ProofExtractor:
    # inputs
    lean_file: LeanFile
    relative_file_path: Path
    tactic_instance_data: List[Dict[str, Any]]
    tactic_position_data: List[Dict[str, Any]]
    # intermediate data sets
    tactic_pos_data: List[Dict[str, Any]]
    tactic_data: Dict[str, Dict[str, Any]]
    tactic_pos_trace_keys: Dict[Tuple[str, int, int, int], str]
    # hints for parsing the proofs
    parameter_positions: Dict[Tuple[int, int], List[Tuple[int, int]]]
    tactic_block_positions: Set[Tuple[int, int]]
    # data structures for navigating the proof trees
    tactics_to_process: List[str]
    processed_tactics: Set[str]
    # end results
    proof_trees: List[Dict[str, Any]]
    proof_table: List[Dict[str, Any]]
    tactic_table: List[Dict[str, Any]]
    arg_table: List[Dict[str, Any]]

    def __init__(
        self,
        lean_file: LeanFile,
        relative_file_path: Path,
        tactic_instance_data: List[Dict[str, Any]],
        tactic_params_pos_data: List[Dict[str, Any]],
    ):
        self.lean_file = lean_file
        self.relative_file_path = relative_file_path
        self.tactic_instance_data = tactic_instance_data
        self.tactic_params_pos_data = tactic_params_pos_data

        self.tactic_pos_data = []
        self.tactic_data = {}

        self.proof_trees = []
        self.proof_table = []
        self.tactic_table = []
        self.arg_table = []

    @staticmethod
    def remove_index_from_key(key: str) -> str:
        """The key is of the form <line>:<col>:<depth>:<index>"""
        return ":".join(key.split(":")[:3])

    def build_tactic_pos_data(self) -> None:
        tactic_pos_data = {}
        for tactic_instance in self.tactic_instance_data:
            tactic_pos = {}
            tactic_pos["filename"] = tactic_instance["filename"]
            tactic_pos["key"] = self.remove_index_from_key(tactic_instance["key"])
            tactic_pos["trace_pos_line"] = tactic_instance["trace_pos_line"]
            tactic_pos["trace_pos_column"] = tactic_instance["trace_pos_column"]
            tactic_pos["line"] = tactic_instance["line"]
            tactic_pos["column"] = tactic_instance["column"]
            tactic_pos["depth"] = tactic_instance["depth"]
            tactic_pos["proof"] = self.remove_index_from_key(tactic_instance["proof"])
            tactic_pos["block"] = self.remove_index_from_key(tactic_instance["block"])
            tactic_pos["parent"] = self.remove_index_from_key(tactic_instance["parent"])

            tactic_pos_data[tactic_pos["filename"], tactic_pos["key"]] = tactic_pos

        self.tactic_pos_data = list(tactic_pos_data.values())

    def build_tactic_data(self) -> None:
        self.tactic_data = {}
        for tac in sorted(self.tactic_pos_data, key=lambda tac: tac["depth"]):
            # Tactics inside any `[...] block show up in the dataset
            # even if they are in another file.  We will remove
            # them.  They can be filtered out by checking if the
            # line and column from istep is the same as the traced
            # line and column.
            if (tac["trace_pos_line"], tac["trace_pos_column"]) != (
                tac["line"],
                tac["column"],
            ):
                continue
            data = {}
            data["filename"] = tac["filename"]
            data["key"] = tac["key"]
            data["parent"] = tac["parent"]
            data["depth"] = tac["depth"]
            symbol = self.lean_file.get_token(tac["line"] - 1, tac["column"] - 1).string

            if symbol == ";":
                data["type"] = "semicolon"
                # this info will be filled in when processing children later
                # this line-column pair will ultimately be the position of the
                # left-most symbol in the tactic.
                # It is initialized with the traced position, which is the
                # position of the left-most semicolon in a chain of semicolons.
                # Moreover, at every point in this recursive loop, the children
                # on the left side will always have a lower or equal line-column
                # pair.
                data["line"] = tac["line"]
                data["column"] = tac["column"]
                data["semicolon_reverse_depth"] = None
                data["preceeding_symbol"] = None
                data["preceeding_line"] = None
                data["preceeding_column"] = None
            else:
                if symbol == "}":
                    left = self.lean_file.find_left_bracket(
                        ["{"], ["}"], tac["line"] - 1, tac["column"] - 1
                    )
                    data["type"] = "begin_end"
                    data["line"] = left.line + 1
                    data["column"] = left.column + 1
                elif symbol == "end":
                    left = self.lean_file.find_left_bracket(
                        ["begin", "match"], ["end"], tac["line"] - 1, tac["column"] - 1
                    )
                    data["type"] = "begin_end"
                    data["line"] = left.line + 1
                    data["column"] = left.column + 1
                else:
                    data["type"] = "named"
                    data["line"] = tac["line"]
                    data["column"] = tac["column"]

                # get previous token to know if it is the start of a proof
                preceeding_token = self.lean_file.get_prev_matching_pattern(
                    data["line"] - 1,
                    data["column"] - 1,
                    [TokenType.SYMBOL, TokenType.ALPHANUMERIC],
                )

                data["preceeding_symbol"] = preceeding_token.string
                data["preceeding_line"] = preceeding_token.line + 1
                data["preceeding_column"] = preceeding_token.column + 1

                # semicolon tactics are difficult to uniquely identify by their
                # position since they are infix operators.  Moreover the tracing
                # only records the position of the first semicolon in a list of
                # semicolons.  We will recursively discover the reverse stack
                # depth of the semicolon tactic as follows:
                data[
                    "semicolon_reverse_depth"
                ] = 0  # may be changed again later by its children
                current_data = data
                rev_depth = 0
                while current_data["parent"] in self.tactic_data:
                    parent = self.tactic_data[current_data["parent"]]
                    if parent["type"] != "semicolon":
                        break
                    # check that this branch stems from the left side of the parent
                    if (data["line"], data["column"]) > (
                        parent["line"],
                        parent["column"],
                    ):
                        break

                    rev_depth += 1

                    current_data = parent
                    current_data["line"] = data["line"]
                    current_data["column"] = data["column"]
                    current_data["semicolon_reverse_depth"] = rev_depth
                    current_data["preceeding_symbol"] = data["preceeding_symbol"]
                    current_data["preceeding_line"] = data["preceeding_line"]
                    current_data["preceeding_column"] = data["preceeding_column"]

            self.tactic_data[data["key"]] = data

    def build_parser_hints(self) -> None:
        tactic_data_list = sorted(
            self.tactic_data.values(), key=lambda tac: (tac["line"], tac["column"])
        )
        self.tactic_params_pos_data.sort(
            key=lambda param: (param["line"], param["column"])
        )

        self.parameter_positions = collections.defaultdict(list)
        for param in self.tactic_params_pos_data:
            # empty parameters start and end at same position
            self.parameter_positions[param["line"] - 1, param["column"] - 1].append(
                (param["end_line"] - 1, param["end_column"] - 1)
            )
        self.tactic_block_positions = set()
        self.tactics_to_process = []
        self.tactic_pos_trace_keys = {}
        for tac in tactic_data_list:
            if tac["preceeding_symbol"] in ("by", "begin", "{"):
                self.tactic_block_positions.add(
                    (tac["preceeding_line"] - 1, tac["preceeding_column"] - 1)
                )
            self.tactics_to_process.append(tac["key"])
            self.tactic_pos_trace_keys[
                tac["filename"],
                tac["line"],
                tac["column"],
                tac["semicolon_reverse_depth"],
            ] = tac["key"]
        self.processed_tactics = set()

    def extract_ast(
        self,
        ast: AST.ASTData,
        proof_key: str,
        parent_key: str,
        parent_type: str,  # proof, tactic, or arg
        index: int,
    ) -> Dict[str, Any]:
        # each ast node will be converted into the output ast as well as added to a table
        node = {}
        node["key"] = None  # This will get filled in below

        row = {}
        row["key"] = None  # This will get filled in below
        row["filename"] = self.relative_file_path
        row["start_line"] = ast.line + 1
        row["start_column"] = ast.column + 1
        row["end_line"] = ast.end_line + 1
        row["end_column"] = ast.end_column + 1
        row["code_string"] = self.lean_file.slice_string(
            ast.line, ast.column, ast.end_line, ast.end_column, clean=True
        )
        row["class"] = None  # This will get filled in below
        row["parent_key"] = parent_key
        row["parent_type"] = parent_type
        row["index"] = index

        # Uniquely identify the node by the position of its symbol.
        # Most nodes this is the position of the leading character,
        # but forsemicolon and alternate tactics this will be updated below
        # to the position of the infix operator
        row["line"] = ast.line + 1
        row["column"] = ast.column + 1
        key = f"{row['filename']}:{row['line']}:{row['column']}"
        infix_key = 0

        if isinstance(ast, AST.ByProof):
            node["node_type"] = "proof"
            node["node_subtype"] = "by"
            node["tactic"] = self.extract_ast(
                ast.tactic,
                proof_key=proof_key,
                parent_key=key,
                parent_type=node["node_type"],
                index=0,
            )

            row["first_tactic_key"] = node["tactic"]["key"]
        elif isinstance(ast, AST.BeginProof):
            node["node_type"] = "proof"
            node["node_subtype"] = "begin"
            node["tactics"] = []
            for i, tactic in enumerate(ast.tactics):
                node["tactics"].append(
                    self.extract_ast(
                        tactic,
                        proof_key=proof_key,
                        parent_key=key,
                        parent_type=node["node_type"],
                        index=i,
                    )
                )

            row["first_tactic_key"] = node["tactics"][0]["key"]
        elif isinstance(ast, AST.BracketProof):
            node["node_type"] = "proof"
            node["node_subtype"] = "bracket"
            node["tactics"] = []
            for i, tactic in enumerate(ast.tactics):
                node["tactics"].append(
                    self.extract_ast(
                        tactic,
                        proof_key=proof_key,
                        parent_key=key,
                        parent_type=node["node_type"],
                        index=i,
                    )
                )

            row["first_tactic_key"] = node["tactics"][0]["key"]
        elif isinstance(ast, AST.SemicolonListTactic):
            node["node_type"] = "tactic"
            node["node_subtype"] = "semicolon_list"

            row["line"] = ast.semicolon_line + 1
            row["column"] = ast.semicolon_column + 1
            key = f"{row['filename']}:{row['line']}:{row['column']}"
            infix_key = ast.semicolon_count

            node["tactic1"] = self.extract_ast(
                ast.tactic1,
                proof_key=proof_key,
                parent_key=key,
                parent_type=node["node_type"],
                index=0,
            )
            node["tactics2"] = []
            for i, tactic in enumerate(ast.tactic_list):
                node["tactics2"].append(
                    self.extract_ast(
                        tactic,
                        proof_key=proof_key,
                        parent_key=key,
                        parent_type=node["node_type"],
                        index=i + 1,
                    )
                )
        elif isinstance(ast, AST.SemicolonTactic):
            node["node_type"] = "tactic"
            node["node_subtype"] = "semicolon"

            row["line"] = ast.semicolon_line + 1
            row["column"] = ast.semicolon_column + 1
            key = f"{row['filename']}:{row['line']}:{row['column']}"
            infix_key = ast.semicolon_count

            node["tactic1"] = self.extract_ast(
                ast.tactic1,
                proof_key=proof_key,
                parent_key=key,
                parent_type=node["node_type"],
                index=0,
            )
            node["tactic2"] = self.extract_ast(
                ast.tactic2,
                proof_key=proof_key,
                parent_key=key,
                parent_type=node["node_type"],
                index=1,
            )
        elif isinstance(ast, AST.AlternativeTactic):
            node["node_type"] = "tactic"
            node["node_subtype"] = "alternative"

            row["line"] = ast.alternative_line + 1
            row["column"] = ast.alternative_column + 1
            key = f"{row['filename']}:{row['line']}:{row['column']}"
            infix_key = -1  # alternative tactics are not traced, so give a dummy key

            node["tactic1"] = self.extract_ast(
                ast.tactic1,
                proof_key=proof_key,
                parent_key=key,
                parent_type=node["node_type"],
                index=0,
            )
            node["tactic2"] = self.extract_ast(
                ast.tactic2,
                proof_key=proof_key,
                parent_key=key,
                parent_type=node["node_type"],
                index=1,
            )
        elif isinstance(ast, AST.Solve1Tactic):
            node["node_type"] = "tactic"
            node["node_subtype"] = "solve1"
            node["tactics"] = []
            for i, tactic in enumerate(ast.tactics):
                node["tactics"].append(
                    self.extract_ast(
                        tactic,
                        proof_key=proof_key,
                        parent_key=key,
                        parent_type=node["node_type"],
                        index=i,
                    )
                )
        elif isinstance(ast, AST.NamedTactic):
            node["node_type"] = "tactic"
            node["node_subtype"] = "named"
            node["args"] = []
            for i, arg in enumerate(ast.args):
                node["args"].append(
                    self.extract_ast(
                        arg,
                        proof_key=proof_key,
                        parent_key=key,
                        parent_type=node["node_type"],
                        index=i,
                    )
                )
        elif isinstance(ast, AST.ITactic):
            node["node_type"] = "tactic"
            node["node_subtype"] = "itactic"
            node["tactics"] = []
            for i, tactic in enumerate(ast.tactics):
                node["tactics"].append(
                    self.extract_ast(
                        tactic,
                        proof_key=proof_key,
                        parent_key=key,
                        parent_type=node["node_type"],
                        index=i,
                    )
                )
        elif isinstance(ast, AST.CalcTactic):
            node["node_type"] = "tactic"
            node["node_subtype"] = "calc"
            # for now we don't store much special about calc tactics
            # which are fairly rare
        elif isinstance(ast, AST.ITacticTacticParam):
            node["node_type"] = "tactic_arg"
            node["node_subtype"] = "itactic"
            node["tactic"] = self.extract_ast(
                ast.tactic,
                proof_key=proof_key,
                parent_key=key,
                parent_type=node["node_type"],
                index=0,
            )
        elif isinstance(
            ast, AST.TacticParam
        ):  # TODO: Change parser here to return a subtype
            node["node_type"] = "tactic_arg"
            node["node_subtype"] = "expression"
        else:
            raise Exception(ast)

        node["key"] = row["key"] = f"{row['filename']}:{row['line']}:{row['column']}"
        row["class"] = node["node_subtype"]
        if node["node_type"] == "proof":
            self.proof_table.append(row)
        elif node["node_type"] == "tactic":
            row["proof_key"] = proof_key
            if (
                row["filename"],
                row["start_line"],
                row["start_column"],
                infix_key,
            ) in self.tactic_pos_trace_keys:
                trace_key = self.tactic_pos_trace_keys[
                    row["filename"], row["start_line"], row["start_column"], infix_key
                ]

                self.processed_tactics.add(trace_key)
                row["trace_key"] = trace_key
            else:
                row["trace_key"] = ""
            self.tactic_table.append(row)
        elif node["node_type"] == "tactic_arg":
            self.arg_table.append(row)

        return node

    def extract_proof_ast(
        self, ast: Union[AST.ByProof, AST.BeginProof, AST.BracketProof]
    ) -> Dict[str, Any]:
        return self.extract_ast(
            ast,
            proof_key=f"{self.relative_file_path}:{ast.line + 1}:{ast.column + 1}",
            parent_key="",
            parent_type="",
            index=0,
        )

    def run(self):
        self.build_tactic_pos_data()
        self.build_tactic_data()
        self.build_parser_hints()

        for key in self.tactics_to_process:
            if key in self.processed_tactics:
                continue
            tac = self.tactic_data[key]
            try:
                parser = LeanParser(
                    self.lean_file,
                    tac["preceeding_line"] - 1,
                    tac["preceeding_column"] - 1,
                    parameter_positions=self.parameter_positions,
                    tactic_block_positions=self.tactic_block_positions,
                )
                if tac["preceeding_symbol"] == "by":
                    parser_ast = parser.read_by()
                elif tac["preceeding_symbol"] == "begin":
                    parser_ast = parser.read_begin()
                elif tac["preceeding_symbol"] == "{":
                    # this should only happen in the rare case
                    # that this is a {...} proof inside a
                    # have or show in a term proof.
                    parser_ast = parser.read_bracket_proof()
                else:
                    raise Exception(
                        f"This tactic should already have been processed: {tac}"
                    )

                # besides returning a proof tree
                # this method also add each node to the output tables and
                # to the evaluated_positions set
                proof_tree = self.extract_proof_ast(parser_ast)
                self.proof_trees.append(proof_tree)

            except Exception as e:
                # print(e)
                print(self.lean_file.filename)
                traceback.print_exc()


def process_one_file(
    data_dir: Path,
    tactic_instance_data: List[Dict[str, Any]],
    tactic_params_pos_data: List[Dict[str, Any]],
    relative_file_path: str,
):
    file_path = Path(data_dir) / "lean_files" / relative_file_path
    print(file_path)
    try:
        lean_file = LeanFile(str(file_path))

        file_tactic_instance_data = [
            row for row in tactic_instance_data if row["filename"] == relative_file_path
        ]
        file_tactic_params_pos_data = [
            row
            for row in tactic_params_pos_data
            if row["filename"] == relative_file_path
        ]

        proof_extractor = ProofExtractor(
            lean_file=lean_file,
            relative_file_path=relative_file_path,
            tactic_instance_data=file_tactic_instance_data,
            tactic_params_pos_data=file_tactic_params_pos_data,
        )
        proof_extractor.run()
        return (
            proof_extractor.proof_trees,
            proof_extractor.proof_table,
            proof_extractor.tactic_table,
            proof_extractor.arg_table,
        )
    except Exception:
        print(file_path)
        traceback.print_exc()


from multiprocessing import Pool
from functools import partial


def main():
    assert len(sys.argv) == 2
    data_dir = Path(sys.argv[1])
    assert data_dir.exists(), data_dir
    assert data_dir.is_dir(), data_dir

    data_tables = collections.defaultdict(list)

    tactic_instance_data = get_traced_data(data_dir, "tactic_instances")
    tactic_params_pos_data = get_traced_data(data_dir, "tactic_param_pos")

    files: List[str] = sorted({row["filename"] for row in tactic_instance_data})
    with Pool(40) as p:
        mapped = p.map(
            partial(
                process_one_file, data_dir, tactic_instance_data, tactic_params_pos_data
            ),
            files,
        )
        for proof_trees, proof_table, tactic_table, arg_table in mapped:
            data_tables["proof_trees"].extend(proof_trees)
            data_tables["proofs"].extend(proof_table)
            data_tables["tactics"].extend(tactic_table)
            data_tables["args"].extend(arg_table)

    save_data_tables(data_tables, data_dir)


if __name__ == "__main__":
    main()
