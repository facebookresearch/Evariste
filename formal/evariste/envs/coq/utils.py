# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import re
from typing import List
from sexpdata import Symbol


logger = getLogger()


def escape_cmd(cmd):
    """
    Escape input Coq command.
    """
    return cmd.replace("\\", "\\\\").replace('"', '\\"')


def remove_comments(string: str) -> str:
    """
    Remove all comments from a Coq source file.
    """
    result = ""
    depth = 0
    in_quote = False
    for i in range(len(string)):
        if in_quote:
            if depth == 0:
                result += string[i]
            if string[i] == '"':  # and string[i - 1] != '\\':
                in_quote = False
        else:
            if string[i : i + 2] == "(*":
                depth += 1
            if depth == 0:
                result += string[i]
            if string[i - 1 : i + 1] == "*)" and depth > 0:
                depth -= 1
            if string[i] == '"' and depth == 0:
                in_quote = True
    assert depth == 0
    return result


def split_commands(string: str) -> List[str]:
    """
    Split commands within a Coq source file.
    """
    commands = []
    next_command = ""
    in_quote = False
    for i in range(len(string)):
        if in_quote:
            if string[i] == '"':  # and string[i - 1] != '\\':
                in_quote = False
        else:
            if string[i] == '"':  # and string[i - 1] != '\\':
                in_quote = True
            if re.match(r"[\{\}]", string[i]) and re.fullmatch(r"\s*", next_command):
                commands.append(string[i])
                next_command = ""
                continue
            if (
                re.match(r"[\+\-\*]", string[i])
                and string[i] != string[i + 1]
                and re.fullmatch(r"\s*[\+\-\*]*", next_command)
            ):
                next_command += string[i]
                commands.append(next_command.strip())
                next_command = ""
                continue
            if re.match(r"\.($|\s)", string[i : i + 2]) and (
                not string[i - 1] == "." or string[i - 2] == "."
            ):
                commands.append(next_command.strip() + ".")
                next_command = ""
                continue
        next_command += string[i]
    assert not any(len(c) == 0 for c in commands)
    commands = [re.sub(r"\s+", " ", c) for c in commands]
    return commands


def symbol2str(sexp, depth: int):
    """
    Convert S-Expression symbols to strings for easier matching.
    """
    if depth <= 0:
        return sexp
    if isinstance(sexp, list):
        return [symbol2str(item, depth - 1) for item in sexp]
    if isinstance(sexp, Symbol):
        return sexp.value()
    else:
        return sexp


def find_parent_coq_project(filepath, upper_path, name="_CoqProject"):
    """
    Find the parent _CoqProject file of a given Coq source file.
    For some projects (for old versions of Coq), the _CoqProject file is named Make.
    Do not search above the upper path.
    """
    assert filepath.startswith(upper_path)
    assert os.path.isfile(filepath)
    assert name in ["_CoqProject", "Make"]
    dirpath = os.path.dirname(filepath)
    while True:
        _CoqProject_path = os.path.join(dirpath, name)
        if os.path.isfile(_CoqProject_path):
            return _CoqProject_path
        dirpath = os.path.dirname(dirpath)
        if dirpath == upper_path:
            return None


def parse__CoqProject_commands(path):
    """
    Extract initialization commands from a _CoqProject file.
    """
    assert os.path.isfile(path)
    with open(path, "r") as fp:
        lines = fp.readlines()
    commands = []
    for i, line in enumerate(lines):
        line = line.strip()
        if len(line) == 0:
            continue
        if line[0] != "-":
            logger.warning(f'# Skipped: "{line}"'.format(line))
            continue
        data = line.split()
        op = data[0]
        if op == "-Q":
            assert len(data) == 3
            commands.append(f'Add LoadPath "{data[1]}" as {data[2]}.')
        elif op == "-R":
            assert len(data) >= 3
            commands.append(f'Add Rec LoadPath "{data[1]}" as {data[2]}.')
            if len(data) > 3:
                logger.warning(f'Extra parameters in line {i}: "{line[3:]}"')
        elif op == "-I":
            assert len(data) == 2
            commands.append(f'Add ML Path "{data[1]}".')
        else:
            logger.warning(f'# Unsupported operator: "{line}"')
    return commands


def retrieve__CoqProject_commands(filepath, upper_path):
    """
    Retrieve _CoqProject file above a Coq source file, and its initialization commands.
    """
    _CoqProject_path = find_parent_coq_project(
        filepath, upper_path=upper_path, name="_CoqProject"
    )
    _CoqProject_path = (
        find_parent_coq_project(filepath, upper_path=upper_path, name="Make")
        if _CoqProject_path is None
        else _CoqProject_path
    )

    # no _CoqProject file
    if _CoqProject_path is None:
        logger.warning("Did not find any _CoqProject!")
        return []

    # extract initialization commands
    logger.info(f"Found _CoqProject file: {_CoqProject_path}")
    _commands = parse__CoqProject_commands(_CoqProject_path)
    commands = []
    for _c in _commands:
        c = _c.replace('"."', f'"{os.path.dirname(_CoqProject_path)}"')
        if _c == c:
            logger.info(f'Command: "{c}"')
        else:
            logger.info(f'Command: "{_c}" -> "{c}"')
        commands.append(c)
    if len(commands) > 0:
        logger.info(
            "========== Initialization commands:\n"
            + "\n".join(commands)
            + "\n=========="
        )
    return commands
