# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import io
import os
import re
import signal
import logging
import pexpect
import sexpdata
from sexpdata import Symbol
from pexpect.popen_spawn import PopenSpawn
from pampy import match

from ..logger import LOGGING_LEVELS
from .coq_utils import escape_cmd, symbol2str, remove_comments, split_commands


logger = logging.getLogger()


class CoqTimeout(Exception):
    def __init__(self, before, after):
        super().__init__()
        self.before = before
        self.after = after

    def __str__(self):
        raise TypeError("CoqTimeout exception cannot be instantiated directly.")

    def __repr__(self):
        return str(self)


class CoqTimeoutInterrupted(CoqTimeout):
    def __str__(self):
        return (
            f"TIMEOUT exception. Successful SIGINT interruption.\n"
            f'Before: "{self.before}"\nAfter: "{self.after}"'
        )


class CoqTimeoutFailedInterruption(CoqTimeout):
    def __str__(self):
        return (
            f"TIMEOUT exception. SIGINT interruption failure.\n"
            f'Before: "{self.before}"\nAfter: "{self.after}"'
        )


class CoqExn(Exception):
    def __init__(self, err_msg, full_sexp):
        super().__init__()
        self.err_msg = err_msg
        self.full_sexp = full_sexp

    def __str__(self):
        return self.err_msg

    def __repr__(self):
        return str(self)

    def raise_exception(x):
        """
        Raise a Coq Exception returned by SerAPI.
        """
        _, r = match(
            symbol2str(x, depth=5),
            ["Answer", int, ["CoqExn", list]],
            lambda *args: args,
        )
        err_msg = match(r[-1], ["str", str], lambda x: x)
        raise CoqExn(err_msg, sexpdata.dumps(x[2]))


class SerAPI:
    def __init__(
        self,
        sertop_path=None,
        options=["--implicit", "--omit_loc", "--print0"],
        timeout=5,
        log_level="debug",
    ):
        """
        Initialize the SerAPI subprocess.
        """
        # set log level
        self.log_level = LOGGING_LEVELS[log_level]

        # look for sertop binary
        if sertop_path is None:
            sertop_path = os.popen("which sertop").read().strip()
            if os.path.isfile(sertop_path):
                self.info(f"Using sertop binary: {sertop_path}")
            else:
                raise FileNotFoundError(
                    f'Unable to locate sertop ("{sertop_path}"). Please check that sertop is '
                    f"installed or that Opam environment variables are properly set."
                )

        # sertop command
        command = " ".join([sertop_path] + options)
        self.info(f'Starting sertop with: "{command}". Log level: {log_level}')

        # run sertop
        try:
            self._proc = PopenSpawn(
                command, encoding="utf-8", timeout=timeout, maxread=10_000_000
            )
        except FileNotFoundError:
            self.error("Unable to locate sertop")
            raise

        # # Disable delays (http://pexpect.readthedocs.io/en/stable/commonissues.html?highlight=delaybeforesend)
        # self._proc.delaybeforesend = None  # TODO: check

        # store data
        self.state_ids = []
        self.ackid2response = {}
        self.stateid2cmd = {}

        # comm check
        self._proc.expect_exact(
            "(Feedback((doc_id 0)(span_id 1)(route 0)(contents Processed)))\0"
        )
        self.send(
            "Noop"
        )  # can be used to check that we are still sync - TODO: add check

        # global printing options
        self.execute("Unset Printing Notations.")  # TODO: remove
        self.execute("Unset Printing Wildcard.")
        self.execute("Set Printing Coercions.")
        self.execute("Unset Printing Allow Match Default Clause.")
        self.execute("Unset Printing Factorizable Match Patterns.")
        self.execute("Unset Printing Compact Contexts.")
        self.execute("Set Printing Implicit.")
        self.execute("Set Printing Depth 999999.")
        self.execute("Unset Printing Records.")

    def __del__(self):
        """
        Destructor.
        Kill sertop instance.
        """
        self.warning("Killing sertop instance...")
        self._proc.kill(signal.SIGKILL)

    def debug(self, msg):
        if self.log_level <= logging.DEBUG:
            logger.debug(msg)

    def info(self, msg):
        if self.log_level <= logging.INFO:
            logger.info(msg)

    def warning(self, msg):
        if self.log_level <= logging.WARNING:
            logger.warning(msg)

    def error(self, msg):
        if self.log_level <= logging.ERROR:
            logger.error(msg)

    def set_timeout(self, timeout):
        """
        Set pexpect timeout (in seconds).
        """
        self._proc.timeout = timeout

    def get_timeout(self):
        """
        Get pexpect timeout (in seconds).
        """
        return self._proc.timeout

    def parse_serapi_response(self, r, raise_on_coq_exn=True, include_feedback=False):
        """
        Parse SerAPI response.
        """
        responses = []

        # for each item
        for item in r.split("\x00"):

            # sanity check
            item = item.strip()
            assert item.startswith("(") and item.endswith(")")

            # parse response type
            item_type = re.findall(r"[a-zA-Z]+", item)[0]
            assert item_type in ["Answer", "Feedback"]  # TODO: check others

            # ignore feedback (for now)
            if item_type == "Feedback" and include_feedback is False:
                continue

            # parse response
            parsed = sexpdata.loads(item, nil=None, true=None)

            # an error occured in Coq
            if "CoqExn" in item and raise_on_coq_exn:
                CoqExn.raise_exception(parsed)

            # add response
            responses.append(parsed)

        return responses

    def send(self, cmd):
        """
        Send a command to SerAPI and retrieve the responses.
        """
        self.debug(
            f'Sending "{cmd}"' if len(cmd) <= 300 else f'Sending "{cmd[:300]} ......."'
        )

        # send command and wait for response
        self._proc.sendline(cmd)
        try:
            self._proc.expect(
                [
                    r"\(Answer \d+ Ack\)\x00.*\(Answer \d+ Completed\)\x00",
                    r"\(Answer \d+ Ack\)\x00.*\(Answer \d+\(CoqExn.*\)\x00",
                ]
            )
        except pexpect.TIMEOUT:

            # try to interrupt current command with the SIGINT signal (keyboard interrupt)
            self.error("TIMEOUT - Interrupting with signal.SIGINT (keyboard interrupt)")
            self._proc.kill(signal.SIGINT)

            # if we get an answer, check that the last command was properly interrupted, and cancel it
            try:
                self._proc.expect(
                    r"\(Answer \d+ Ack\)\x00.*\(Answer \d+\(CoqExn.*\)\x00"
                )
                r0 = self._proc.after.rstrip("\x00")
                r1 = self.parse_serapi_response(r0, raise_on_coq_exn=False)
                i1, i2, _, i3 = match(
                    symbol2str(r1, depth=4),
                    [
                        ["Answer", int, "Ack"],
                        ["Answer", int, ["CoqExn", list]],
                        ["Answer", int, "Completed"],
                    ],
                    lambda *args: args,
                )
                assert i1 == i2 == i3
                raise CoqTimeoutInterrupted(self._proc.before, self._proc.after)

            # if the SIGINT signal also times out
            except pexpect.TIMEOUT:
                raise CoqTimeoutFailedInterruption(self._proc.before, self._proc.after)

        # check that Ack IDs are matching
        r = self._proc.after.rstrip("\x00")
        ack_num = int(re.search(r"^\(Answer (?P<num>\d+)", r)["num"])
        assert all(int(num) == ack_num for num in re.findall(r"(?<=\(Answer) \d+", r))

        # return response
        self.ackid2response[ack_num] = (cmd, r)
        self.debug(f"Received ack number {ack_num}")
        return r

    def add_state(self, cmd):
        """
        Send a (Add () "XXX") command to SerAPI, and return the state ID.
        Should only add one command at a time, so that only one state is returned.
        """
        # add command
        r = self.send(f'(Add () "{escape_cmd(cmd)}")')

        # look for exceptions
        if "CoqExn" in r:
            self.parse_serapi_response(r)

        # retrieve state IDs
        state_ids = [int(sid) for sid in re.findall(r"\(Added (?P<state_id>\d+)", r)]
        assert len(state_ids) == 1, (r, state_ids)
        state_id = state_ids[-1]

        # return state ID
        self.state_ids.append((state_id, cmd))
        self.stateid2cmd[state_id] = cmd
        return state_id

    def cancel_last(self):
        """
        Cancel the last added state.
        This assumes that Coq is running until the state before that.
        """
        assert len(self.state_ids) > 0
        sid, cmd = self.state_ids.pop()
        self.info(f'Cancelling state {sid} (command "{cmd}") ...')
        r0 = self.send(f"(Cancel ({sid}))")

        # parse response
        r1 = self.parse_serapi_response(r0)
        i1, i2, removed_sid, i3 = match(
            symbol2str(r1, depth=4),
            [
                ["Answer", int, "Ack"],
                ["Answer", int, ["Canceled", [int]]],
                ["Answer", int, "Completed"],
            ],
            lambda *args: args,
        )
        assert i1 == i2 == i3
        assert removed_sid == sid
        self.info(f"Cancelled state {sid}.")

    def execute(self, cmd, include_feedback=False):
        """
        Execute a vernac command.
        First send a (Add () "XXX") command to SerAPI, retrieve the state ID, and execute it.
        """
        # add command
        state_id = self.add_state(cmd)

        # execute command - in case of a CoqTimeout exception, cancel
        # the most recent command (i.e. the last added state)
        try:
            r1 = self.send(f"(Exec {state_id})")
        except CoqTimeout:
            self.error("TIMEOUT. Cancelling last added command...")
            self.cancel_last()
            raise

        # if an exception occurred in Coq, remove the newly added state
        try:
            r2 = self.parse_serapi_response(r1, include_feedback=include_feedback)
        except CoqExn:
            self.error(f"Coq exception detected. Removing state {state_id}.")
            self.cancel_last()
            raise

        # return response
        return {"state_id": state_id, "response": r2}

    def query_goals(self, sid, print_format="PpStr"):
        """
        Query goals.
        """
        assert print_format in ["PpSer", "PpStr"]  # TODO: check 'PpTex' and 'PpCoq'
        r0 = self.send(f"(Query ((sid {sid}) (pp ((pp_format {print_format})))) Goals)")
        r1 = self.parse_serapi_response(r0)

        # parse response
        i1, i2, r2, i3 = match(
            symbol2str(r1, depth=4),
            [
                ["Answer", int, "Ack"],
                ["Answer", int, ["ObjList", list]],
                ["Answer", int, "Completed"],
            ],
            lambda *args: args,
        )
        assert i1 == i2 == i3

        # sanity check
        if print_format == "PpStr":
            assert all(x == Symbol("CoqString") and type(y) is str for [x, y] in r2)
        if print_format == "PpSer":
            assert all(x == Symbol("CoqGoal") and type(y) is list for [x, y] in r2)

        # return goals
        goals = [y for [_, y] in r2]
        return goals

    def query_command(self, sid, print_format="PpStr"):
        """
        Return the parsed sentence (i.e. the vernac command) associated to a state.
        WARNING: some commands are broken and cannot be executed once returned,
        e.g. for: https://github.com/math-comp/math-comp/blob/master/mathcomp/algebra/ssralg.v
        """
        assert print_format in ["PpSer", "PpStr"]  # TODO: check 'PpTex' and 'PpCoq'
        r0 = self.send(f"(Query ((sid {sid}) (pp ((pp_format {print_format})))) Ast)")
        r1 = self.parse_serapi_response(r0)

        # parse response
        i1, i2, r2, i3 = match(
            symbol2str(r1, depth=6),
            [
                ["Answer", int, "Ack"],
                ["Answer", int, ["ObjList", [["CoqString", str]]]],
                ["Answer", int, "Completed"],
            ],
            lambda *args: args,
        )
        assert i1 == i2 == i3

        # return vernac command
        return r2

    def parse_commands_from_file(self, filepath):
        """
        Parse vernac commands from a file.
        WARNING: see `query_command`.
        """
        # read raw file content
        assert os.path.isfile(filepath)
        with io.open(filepath, "r", encoding="utf-8") as f:
            read = f.read()
        _commands = split_commands(remove_comments(read))  # TODO: remove

        # add all commands
        r = self.send(f'(Add () "{escape_cmd(read)}")')

        # look for exceptions
        if "CoqExn" in r:
            self.parse_serapi_response(r)

        # retrieve state IDs
        state_ids = [int(sid) for sid in re.findall(r"\(Added (?P<state_id>\d+)", r)]
        self.info(f"Found {len(state_ids)} states (and {len(_commands)} commands).")
        assert state_ids == list(range(state_ids[0], state_ids[-1] + 1))

        # retrieve commands
        self.info("Retrieving commands ...")
        commands = []
        for sid in state_ids:
            command = self.query_command(sid, print_format="PpStr")
            commands.append(command)

        # return commands
        return commands

    def execute_file(
        self, filepath, execute_commands, query_goals, goals_print_format="PpStr"
    ):
        """
        Execute all vernac commands in a file.
        """
        assert type(execute_commands) is bool and type(query_goals) is bool
        assert not query_goals or execute_commands
        # read raw file content
        assert os.path.isfile(filepath)
        with io.open(filepath, "r", encoding="utf-8") as f:
            read = f.read()
        _commands = split_commands(remove_comments(read))  # TODO: remove

        # add all commands
        r = self.send(f'(Add () "{escape_cmd(read)}")')

        # look for exceptions
        if "CoqExn" in r:
            self.parse_serapi_response(r)

        # retrieve state IDs
        state_ids = [int(sid) for sid in re.findall(r"\(Added (?P<state_id>\d+)", r)]
        self.info(f"Found {len(state_ids)} states (and {len(_commands)} commands).")
        assert state_ids == list(range(state_ids[0], state_ids[-1] + 1))

        # retrieve commands
        self.info("Retrieving commands ...")
        commands = []
        for sid in state_ids:
            self.debug(f"Executing state {sid} ...")
            command = self.query_command(sid, print_format="PpStr")
            commands.append({"state_id": sid, "command": command})

        # execute commands
        if not execute_commands:
            return commands
        self.info("Executing commands ...")
        for command in commands:
            sid = command["state_id"]
            self.debug(f"Executing state {sid} ...")
            r1 = self.send(f"(Exec {sid})")
            r2 = self.parse_serapi_response(r1, include_feedback=True)
            command["response"] = r2

        # query goals
        if not query_goals:
            return commands
        self.info("Querying goals ...")
        for command in commands:
            sid = command["state_id"]
            goals = self.query_goals(sid, print_format=goals_print_format)
            command["goals"] = goals

        # return commands with responses
        return commands
