# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from io import TextIOWrapper
from typing import Any, Dict, List, Tuple
from pprint import pprint
import sys
import json
from pathlib import Path

from tokenize_lean_files import LeanFile

# the Lean messages are returned as JSON dictionaries
LeanMessage = Dict[str, Any]

# we will store data in "DataTables", which are lists of json dictionaries
# the dictionary keys form the columns of the tabular data.
DataTable = List[Dict[str, Any]]


def seperate_lean_messages(
    stdout_file: TextIOWrapper,
) -> Tuple[List[LeanMessage], List[LeanMessage]]:
    trace_blocks = []
    other_blocks = []
    for line in stdout_file:
        if line:
            s = line
            message: LeanMessage = json.loads(s)
            if (
                message["severity"] == "information"
                and message["caption"] == "trace output"
            ):
                trace_blocks.append(message)
            else:
                other_blocks.append(message)

    return trace_blocks, other_blocks


def read_path_map(data_dir: Path) -> Dict[Path, Path]:
    with open(data_dir / "path_map.json", "r") as f:
        for line in f:
            return {Path(p1): Path(p2) for p1, p2 in json.loads(line).items()}
    raise Exception()


def read_lean_messages(data_dir: Path) -> Tuple[List[LeanMessage], List[LeanMessage]]:
    with open(data_dir / "lean_stdout.log", "r") as f:
        return seperate_lean_messages(f)


def relative_path(path_map: Dict[Path, Path], filename: Path):
    for p in path_map:
        if p in filename.parents:
            return path_map[p] / filename.relative_to(p)
    raise Exception()


def extract_data_tables(
    trace_messages: List[LeanMessage], path_map: Dict[Path, Path]
) -> Dict[str, DataTable]:
    tables = {}
    for info_message in trace_messages:
        file_path = str(relative_path(path_map, Path(info_message["file_name"])))
        pos_line = info_message["pos_line"]  # lean 1-indexes rows
        pos_column = (
            info_message["pos_col"] + 1
        )  # lean 0-indexes columns (but we 1-index)
        for line in info_message["text"].split("\n"):
            if line.startswith("<PR>"):
                _, json_string = line.split(" ", 1)
                traced_data: Dict[str, Any] = json.loads(json_string)
                assert "key" in traced_data, traced_data
                assert "table" in traced_data, traced_data
                traced_data["filename"] = file_path
                traced_data["trace_pos_line"] = pos_line
                traced_data["trace_pos_column"] = pos_column

                table_name = traced_data["table"]
                key = traced_data["key"]  # only unique within a file and table

                if table_name not in tables:
                    table = tables[table_name] = {}
                else:
                    table = tables[table_name]

                # the "key" in the trace data is unique only within a file
                if (file_path, key) not in table:
                    table[file_path, key] = {}

                table[file_path, key].update(traced_data)

    # ignore the row keys since it will just be saved to raw json
    return {name: list(t.values()) for name, t in tables.items()}


def save_other_lean_messages(messages: List[LeanMessage], data_dir: Path):
    with open(data_dir / "lean_errors.json", "w") as outfile:
        json.dump(messages, outfile, indent=4)


def save_data_tables(data_tables: Dict[str, DataTable], data_dir: Path):
    dir = data_dir / "raw_traced_data"
    dir.mkdir()
    for table_name, table in data_tables.items():
        # save each table to a file
        filename = table_name + ".json"
        with open(dir / filename, "w") as outfile:
            json.dump(table, outfile)


def extract_data(data_dir: Path) -> None:
    path_map = read_path_map(data_dir)
    trace_messages, other_messages = read_lean_messages(data_dir)

    for m in other_messages:
        print("Unexpected lean output message:")
        pprint(m)

    save_other_lean_messages(other_messages, data_dir)

    data_tables = extract_data_tables(trace_messages, path_map)
    save_data_tables(data_tables, data_dir)


def main():
    assert len(sys.argv) == 2
    data_dir = Path(sys.argv[1])
    assert data_dir.exists(), data_dir
    assert data_dir.is_dir(), data_dir

    extract_data(data_dir)


if __name__ == "__main__":
    main()
