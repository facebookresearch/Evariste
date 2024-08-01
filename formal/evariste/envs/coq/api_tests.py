# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import io
import os
import argparse
from pampy import match

from params.params import bool_flag
from evariste.logger import create_logger
from evariste.envs.ocaml.opam import init_opam_switch
from evariste.envs.coq.utils import symbol2str, remove_comments, split_commands
from evariste.envs.coq.api import CoqExn, CoqTimeoutInterrupted, SerAPI


def run_tests(serapi):

    successful_tests = 0

    # test command
    logger.info("==================== Command test")
    output = serapi.execute(
        "Eval compute in let f := fun x => x * 3 in f 2.", include_feedback=True
    )
    output = match(
        symbol2str(output["response"], depth=7),
        [
            ["Answer", int, "Ack"],
            ["Feedback", list],
            ["Feedback", list],
            [
                "Feedback",
                [
                    list,
                    list,
                    list,
                    ["contents", ["Message", list, list, list, ["str", str]]],
                ],
            ],
            ["Feedback", list],
            ["Answer", int, "Completed"],
        ],
        lambda *args: args,
    )
    assert output[9] == "     = S (S (S (S (S (S O)))))\n     : nat"
    logger.info(f"Retrieved expected output: {output[9]}")
    successful_tests += 1

    # test Coq exception
    logger.info("==================== Coq exception test")
    serapi.execute("Require Import mathcomp.ssreflect.ssreflect.")
    try:
        serapi.execute("Require Import BLABLABLA.")
    except CoqExn as e:
        assert "Unable to locate library BLABLABLA." in e.err_msg
        logger.info(f'Caught expected CoqExn exception: "{e.err_msg}"')
    successful_tests += 1

    # test cancel / redefine
    logger.info("==================== Cancel / Redefine test")
    serapi.execute(
        "Inductive day : Type := | monday | tuesday | wednesday | thursday | friday | saturday | sunday."
    )
    try:
        serapi.execute(
            "Inductive day : Type := | monday | tuesday | wednesday | thursday | friday | saturday | sunday."
        )
    except CoqExn as e:
        assert e.err_msg == "day already exists."
        logger.info(f'Caught expected CoqExn exception: "{e.err_msg}"')
    serapi.cancel_last()
    serapi.execute(
        "Inductive day : Type := | monday | tuesday | wednesday | thursday | friday | saturday | sunday."
    )
    logger.info("Last states:\n" + "\n".join(str(x) for x in serapi.state_ids[-5:]))
    successful_tests += 1

    # test timeout 1
    logger.info("==================== Timeout test 1")
    try:
        serapi.execute("Eval vm_compute in (1000 * 1000 * 1000).")
    except CoqTimeoutInterrupted as e:
        assert "(backtrace(Backtrace()))(exn Sys.Break)" in str(e)
        logger.info(f'Caught expected CoqTimeoutInterrupted exception: "{str(e)}"')
    logger.info("Last states:\n" + "\n".join(str(x) for x in serapi.state_ids[-5:]))
    successful_tests += 1

    # test timeout 2
    logger.info("==================== Timeout test 2")
    serapi.execute(
        "Ltac wait n := match n with | O => idtac | S ?n => wait n; wait n end."
    )
    try:
        for c in split_commands("Goal True. Proof. wait 20. reflexivity. Qed."):
            serapi.execute(c)
    except CoqTimeoutInterrupted as e:
        assert "(backtrace(Backtrace()))(exn Sys.Break)" in str(e)
        logger.info(f'Caught expected CoqTimeoutInterrupted exception: "{str(e)}"')
    logger.info("Last states:\n" + "\n".join(str(x) for x in serapi.state_ids[-5:]))
    serapi.cancel_last()  # Remove "Proof."
    serapi.cancel_last()  # Remove "Goal True."
    logger.info("Last states:\n" + "\n".join(str(x) for x in serapi.state_ids[-5:]))
    successful_tests += 1

    # test results
    logger.info("==================== Test results")
    logger.info(f"Passed {successful_tests} tests successfully.")


def get_parser():
    parser = argparse.ArgumentParser(description="Coq API tests")
    parser.add_argument("--opam_switch", type=str, default="", help="Opam switch")
    parser.add_argument(
        "--coq_timeout", type=int, default=20, help="Coq timeout (in seconds)"
    )
    parser.add_argument("--log_level", type=str, default="debug", help="Log level")
    parser.add_argument("--test", type=bool_flag, default=True, help="Run tests")
    return parser


if __name__ == "__main__":

    # create logger
    logger = create_logger(None)

    # parse arguments
    parser = get_parser()
    args = parser.parse_args()

    # initialize Opam switch
    init_opam_switch(args.opam_switch, check_ocaml=True, check_coq=True)

    # create SerAPI instance
    serapi = SerAPI(
        options=["--implicit", "--omit_loc", "--print0"],
        timeout=args.coq_timeout,
        log_level=args.log_level,
    )

    # run tests
    if args.test:
        run_tests(serapi)
        exit()

    # Coq source path
    coq_source_path = (
        "resources/coq/coq_projects_8.10.0/math-comp/mathcomp/algebra/ssrnum.v"
    )
    assert os.path.isfile(coq_source_path)

    # retrieve commands from file using split_commands

    with io.open(coq_source_path, "r", encoding="utf-8") as f:
        source = f.read()
    content = remove_comments(source)
    coq_commands = split_commands(content)
    n_lines = len([l for l in source.split("\n") if len(l.strip()) > 0])
    logger.info(
        f"Read {len(source)} characters, {n_lines} lines, and {len(coq_commands)} commands from {coq_source_path}"
    )

    # run commands
    responses = []
    for i, c in enumerate(coq_commands):
        logger.info(f"======================= {i}")
        logger.info(c)
        logger.info("")
        r = serapi.execute(c)
        responses.append(r)
        logger.info("")

    # debugger
    import ipdb

    ipdb.set_trace()
