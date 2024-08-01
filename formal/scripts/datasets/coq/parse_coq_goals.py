# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import io
import os
from evariste import json as json
import random
import argparse
import sexpdata

from evariste.logger import create_logger
from evariste.utils import search_files
from evariste.envs.ocaml.opam import init_opam_switch
from evariste.envs.coq.api import SerAPI
from evariste.envs.coq.utils import (
    remove_comments,
    split_commands,
    retrieve__CoqProject_commands,
)


IGNORED_NAMES = {"PROCESSED", "GITDIFFS"}


def get_parser():
    parser = argparse.ArgumentParser(description="Retrieve Coq goals")
    parser.add_argument(
        "--coq_projects_dir",
        type=str,
        default="YOUR_PATH/coq_projects_8.11.0",
        help="Coq projects directory",
    )
    parser.add_argument(
        "--project_name", type=str, default="math-comp", help="Project name"
    )
    parser.add_argument(
        "--opam_switch", type=str, default="coq_8_11_0_manual", help="Opam switch"
    )
    parser.add_argument(
        "--coq_timeout", type=int, default=20, help="Coq timeout (in seconds)"
    )
    parser.add_argument("--log_level", type=str, default="debug", help="Log level")
    return parser


if __name__ == "__main__":

    # create logger
    log_path = os.path.join(
        os.getcwd(), f"parse_coq_goals.{random.randint(0, 100000)}.log"
    )
    logger = create_logger(log_path)
    logger.info(f"Logging processing results in {log_path} ...")

    # parse arguments
    parser = get_parser()
    args = parser.parse_args()

    # check project path
    args.src_dir = os.path.join(args.coq_projects_dir, args.project_name)
    args.tgt_dir = os.path.join(args.coq_projects_dir, "PROCESSED", args.project_name)
    assert os.path.isdir(args.coq_projects_dir)
    assert os.path.isdir(args.src_dir)

    # initialize Opam switch
    init_opam_switch(args.opam_switch, check_ocaml=True, check_coq=True)

    # select files
    filepaths = search_files([args.src_dir], [".v"], ignore_patterns=[])
    if len(filepaths) == 0:
        logger.info(f"Did not find Coq files to process.")
        exit()
    else:
        logger.info(f"Found {len(filepaths)} Coq files to process.")
        assert len(filepaths) == len(set(filepaths)) > 0

    # read all Coq files and store all intermediary goals
    for i, path in enumerate(filepaths):

        if "ssrnum" in path:  # TODO: remove
            exit()
            # continue until fails
            # continue

        logger.info(
            f"==================== Processing file {i}/{len(filepaths)} - {path}"
        )

        # retrieve initialization commands from _CoqProject file
        init_commands = retrieve__CoqProject_commands(
            path, upper_path=args.coq_projects_dir
        )

        # retrieve commands from file
        with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        content = remove_comments(source)
        coq_commands = split_commands(content)
        n_lines = len([l for l in source.split("\n") if len(l.strip()) > 0])
        logger.info(
            f'Read {len(source)} characters, {n_lines} lines, and {len(coq_commands)} commands from "{path}".'
        )

        # goals path - skip if nothing to do (i.e. output is stored and commands are identical)
        goals_path = path.replace(args.src_dir, args.tgt_dir) + ".json"
        if os.path.isfile(goals_path):
            logger.info(f'Found existing outputs in "{goals_path}".')
            with io.open(goals_path, "r", encoding="utf-8") as f:
                reloaded = [json.loads(line.rstrip()) for line in f]
            if len(reloaded) != len(coq_commands) or not all(
                c == r["command"] for c, r in zip(coq_commands, reloaded)
            ):
                raise Exception(f"Commands in {goals_path} are not matching!")
            logger.info(
                f"{path} already processed with {len(coq_commands)} commands / outputs / goals. Nothing to do."
            )
            continue

        # create a new instance
        serapi = SerAPI(
            sertop_path="dune exec sertop --root YOUR_PATH/coq-serapi --",
            options=["--implicit", "--omit_loc", "--print0"],
            timeout=args.coq_timeout,
            log_level=args.log_level,
        )

        # run initialization commands
        logger.info(f"Running {len(init_commands)} initialization commands ...")
        for command in init_commands:
            serapi.execute(command)

        # run commands
        logger.info(f"Running {len(coq_commands)} commands ...")
        outputs = []
        for i, c in enumerate(coq_commands):
            r = serapi.execute(c)
            outputs.append(
                {
                    "command_id": i,
                    "command": c,
                    "state_id": r["state_id"],
                    "response": sexpdata.dumps(r["response"]),
                }
            )

        # query goals
        logger.info(f"Querying goals ...")
        for o in outputs:
            sid = o["state_id"]
            assert serapi.stateid2cmd[sid] == o["command"]
            goals_PpStr = serapi.query_goals(sid, print_format="PpStr")
            goals_PpSer = serapi.query_goals(sid, print_format="PpSer")
            o["goals_PpStr"] = sexpdata.dumps(goals_PpStr)
            o["goals_PpSer"] = sexpdata.dumps(goals_PpSer)

        # close serapi
        del serapi

        # export goals
        goals_dir = os.path.dirname(goals_path)
        logger.info(f"Storing goals in {goals_path} ...")
        if not os.path.isdir(goals_dir):
            os.makedirs(goals_dir)
        with io.open(goals_path, "w", encoding="utf-8") as f:
            for o in outputs:
                f.write(json.dumps(o) + "\n")
        logger.info(
            f'Stored {len(coq_commands)} commands / outputs / goals in "{goals_path}".'
        )
