# Copyright (c) Facebook, Inc. and its affiliates.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Dict, Any, Set, IO, AnyStr, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import json
import select
import time
import os
import sys
from subprocess import Popen, PIPE, DEVNULL
import io
import re
from collections import deque
from leanml.parse_goal import parse_goal, parse_goal_structured
from copy import deepcopy


class MissingLib(Exception):
    pass


class WrongModule(Exception):
    pass


class DeadLean(Exception):
    pass


def lean_file_to_path(lean_roots: Dict[str, Path], path_to_map: str) -> str:
    """
    The lean server requires complete module path, whereas the dataset contains paths relative to mathlib/ or lean/.
    This maps one format to the other
    """
    res = None
    if path_to_map.startswith("lean/library"):
        res = lean_roots["lean"] / path_to_map.replace("lean/library/", "")
    if path_to_map.startswith("mathlib/src"):
        res = lean_roots["mathlib"] / path_to_map.replace("mathlib/src/", "")
    if path_to_map.startswith("other"):
        res = lean_roots["others"] / path_to_map.replace("others/", "")
    if path_to_map.startswith("cleaning_utils"):
        res = lean_roots["cleaning_utils"] / path_to_map.replace("cleaning_utils/", "")
    if res is None:
        raise WrongModule(
            f"Misunderstood path : {path_to_map}. Isn't mathlib or lean ?"
        )
    if not res.exists():
        raise WrongModule(f"Mapped file {res} for {path_to_map} doesn't seem to exist.")
    return str(res)


@dataclass
class PreExistingMLServer:
    stdin: IO[AnyStr]
    stdout: IO[AnyStr]

    def poll(self):
        return None


intro_regex = r"(?<!(\w|_|\d))intro[Is]?(?!(\w|_|\d))(?P<args>( [^ }\]]+)*)"
hole_in_args = r"(?<!(\w|\d))_(?!(\w|\d))"
def has_intro_without_names(tactic_str):
    for tac in re.split(",|;|<\|>", tactic_str):
        for m in re.finditer(intro_regex, tac):
            if m.group('args') == '' or re.search(hole_in_args, m.group('args')):
                return True
    return False


class LeanAPI:
    """
    Handles basic communications with the lean ml_server
    """

    def __init__(
        self,
        bin_path,
        preload: bool,
        lean_roots: Dict[str, Path],
        quiet=False,
        dump_comms=False,
        num_threads=10,
        profile_tactic_process=False,
        pre_existing: Optional[PreExistingMLServer] = None
    ):
        """
        Initialize the lean subprocess. quiet suppresses stderr from the ML Server process.
        """
        self.forbidden_tactic_answers = []
        self.lean_roots = lean_roots

        self.req_id = 0
        self.logger = logging.getLogger()
        self.dump_comms = dump_comms
        if pre_existing is None:
            os.environ["LEAN_PATH"] = ":".join(
                [str(root) for root in lean_roots.values()]
            )
            self._proc = Popen(
                [bin_path, "preload"] if preload else [bin_path],
                encoding="utf-8",
                stdin=PIPE,
                stdout=PIPE,
                stderr=DEVNULL if quiet else sys.stderr,
                # line buffering
                bufsize=1,
                universal_newlines=True
            )
            output = self._proc.stdout.readline().strip()
            self.logger.info(output)
            if output != "lean ml server ready":
                raise DeadLean(output)
        else:
            self._proc = pre_existing

        paths = []
        for resource in ['others', 'cleaning_utils']:
            p = Path.cwd() / 'runtime_loaded_lean_files' / resource
            assert p.exists(), f"{p} resource missing"
            self.lean_roots[resource] = p
            paths.append(str(p))

        self.runtime_paths = ':'.join(paths)
        self.send({
            'num_threads': num_threads,
            'profile_tactic_process': profile_tactic_process,
            'paths': self.runtime_paths
        })
        assert 'ok' in self.recv()
        self.logger.info("Lean server initialized")
        os.set_blocking(self._proc.stdout.fileno(), False)

        if pre_existing is not None:
            try:
                self.recv(timeout=0.1)
            except RuntimeError as e:
                assert str(e) == "lean ml server ready\n", repr(str(e))
            except TimeoutError:
                pass

    def new_session(self, module_path: str, decl_name: str, merge_alpha_equiv: bool, pp_opts: Optional[Dict] = None) -> str:
        return self.send(
            {
                "req_type": "new_session",
                "module_path": lean_file_to_path(self.lean_roots, module_path),
                "decl_name": decl_name,
                "merge_alpha_equiv": merge_alpha_equiv,
                "opts": pp_opts if pp_opts is not None else {},
            }
        )

    def run_cmd(self, to_run: str, module_path: str, timeout: int = 10_000):
        return self.send(
            {
                "req_type": "eval_cmd",
                "to_run": to_run,
                "module_path": lean_file_to_path(self.lean_roots, module_path),
                "timeout": timeout
            }
        )

    def del_session(self, name: str):
        return self.send({"req_type": "del_session", "name": name,})

    def parse_goal(self, goal_pp: str, session_name: str, timeout: int = 2000, max_repeated_hyps: int = 10_000):
        try:
            # pg = parse_goal(goal_pp)
            pg = [x.to_expression() for x in parse_goal_structured(goal_pp)]
        except Exception:
            print(goal_pp)
            raise
        return self.send(
            {
                "req_type": "parse_goal",
                "name": session_name,
                "parsed_goals": pg,
                "timeout": timeout,
                "max_repeated_hyps": max_repeated_hyps,
            }
        )

    def parse_goal_and_apply_tactic(
        self,
        goal_pp: str,
        session_name: str,
        tactic_str: str,
        timeout: int = 2000,
        max_size: int = 1000 ** 3,
        max_subgoals: int = 10_000,
        max_metavars: int = 10_000,
        max_repeated_hyps: int = 10_000,
        # remove extra info like "cases x:" in subgoal pretty printing
        strip_tags: bool = True
    ):
        try:
            pg = [x.to_expression() for x in parse_goal_structured(goal_pp)]
        except Exception:
            print(goal_pp)
            raise
        return self.send(
            {
                "req_type": "parse_goal_and_apply_tactic",
                "name": session_name,
                "parsed_goals": pg,
                "tactic_str": tactic_str,
                "timeout": timeout,
                "max_size": max_size,
                "max_subgoals": max_subgoals,
                "max_metavars": max_metavars,
                "max_repeated_hyps": max_repeated_hyps,
                "strip_tags": strip_tags
            }
        )

    def parse_children(
        self,
        children: List[str],
        session_name: str,
        timeout: int = 2000,
        max_metavars: int = 10_000,
        max_repeated_hyps: int = 10_000,
    ):
        preparsed_children = []
        for goal_pp in children:
            try:
                pg = [x.to_expression() for x in parse_goal_structured(goal_pp)]
            except Exception:
                print(goal_pp)
                raise
            else:
                preparsed_children.append(pg)

        return self.send(
            {
                "req_type": "parse_children",
                "name": session_name,
                "preparsed_children": preparsed_children,
                "timeout": timeout,
                "max_metavars": max_metavars,
                "max_repeated_hyps": max_repeated_hyps,
            }
        )

    def parse_command(self, goal_pp: str, session_name: str, timeout: int = 2000, max_repeated_hyps: int = 10_000):
        try:
            pg = [x.to_command() for x in parse_goal_structured(goal_pp)]
        except Exception:
            print(goal_pp)
            raise
        return self.send(
            {
                "req_type": "parse_command",
                "name": session_name,
                "parsed_goals": pg,
                "timeout": timeout,
                "max_repeated_hyps": max_repeated_hyps,
            }
        )

    def send_tactic(
        self,
        session_name: str,
        state_id: int,
        tactic_str: str,
        timeout: int = 2000,
        max_size: int = 1000 ** 3,
        max_subgoals: int = 10_000,
        max_metavars: int = 10_000,
        max_repeated_hyps: int = 10_000,
        nosplit: bool = False
    ) -> int:
        if has_intro_without_names(tactic_str):
            self.forbidden_tactic_answers.append({
                "req_id": str(self.req_id),
                "error": f"Forbidden intro tactic without explicit names in {tactic_str}"
            })
            self.req_id += 1
            return str(self.req_id - 1)
        return self.send(
            {
                "req_type": "tactic",
                "name": session_name,
                "state_id": state_id,
                "tactic_str": tactic_str,
                "timeout": timeout,
                "max_size": max_size,
                "max_subgoals": max_subgoals,
                "max_metavars": max_metavars,
                "max_repeated_hyps": max_repeated_hyps,
                "nosplit": nosplit,

                # for backward compatibility
                "type_check": True,
                "only_check_when_decrease": True,
            }
        )

    def print_cmd(self, module_path: str, decl_name: str, to_print: str) -> int:
        return self.send({
            "req_type": "print",
            "module_path": lean_file_to_path(self.lean_roots, module_path),
            "decl_name": decl_name,
            "to_print": to_print
        })

    def send(self, to_send):
        to_send["req_id"] = str(self.req_id)
        if self.dump_comms:
            print(">", to_send)
        self.req_id += 1
        try:
            self._proc.stdin.write(json.dumps(to_send) + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError:
            proc_poll = self._proc.poll()
            if proc_poll is not None:
                raise DeadLean(f"Proc.poll returned {proc_poll}")
            raise DeadLean("Broken pipe but proc_poll is None :(")
        return str(self.req_id - 1)

    def recv(self, timeout=-1) -> Dict[str, Any]:
        while self.forbidden_tactic_answers:
            return self.forbidden_tactic_answers.pop()
        start = time.time()
        while timeout < 0 or time.time() - start < timeout:
            try:
                proc_poll = self._proc.poll()
                if proc_poll is not None:
                    raise DeadLean(f"Proc.poll returned {proc_poll}")
                res = self._proc.stdout.readline().strip()
                if res == "":
                    continue
                if res is None:
                    raise DeadLean
                j = json.loads(res)
                if self.dump_comms:
                    print("<", j)
                return j
            except json.decoder.JSONDecodeError:
                if res == "lean ml server ready":
                    pass
                else:
                    raise RuntimeError(res)
        raise TimeoutError

    def close(self):
        if hasattr(self._proc, "kill"):
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass

    def __del__(self):
        self.close()


def get_api(
    path_to_server: Path,
    preload=False,
    quiet=False,
    fast=False,
    dump_comms=False,
    num_threads=10,
    profile_tactic_process=False,
    path_to_fifos: Optional[Tuple[Path]] = None,
) -> LeanAPI:
    """
    We want the following structure :
    path_to_server = /path/to/root/build/release/binary
    Where root contains:
      - lean/ submodule
      - mathlib/ submodule
      - build/release/binary
    """
    root_dir = path_to_server.parent.parent.parent
    if not all([x in os.listdir(root_dir) for x in {"lean", "mathlib"}]):
        raise MissingLib(f"Both 'lean' and 'mathlib' are expected in folder {root_dir}")
    lean_roots = {
        "lean": root_dir / "lean/library",
    }
    if not fast:
        lean_roots["mathlib"] = root_dir / "mathlib/src"
    pre_existing = None
    if path_to_fifos is not None:
        pre_existing = PreExistingMLServer(
            stdin=io.TextIOWrapper(
                io.open(path_to_fifos[0], "wb", -1),
                write_through=True,
                line_buffering=False,
                encoding="utf-8",
                errors=None,
            ),
            stdout=io.TextIOWrapper(
                io.open(path_to_fifos[1], "rb", -1), encoding="utf-8", errors=None
            ),
        )

    return LeanAPI(
        str(path_to_server),
        preload=preload,
        lean_roots=lean_roots,
        quiet=quiet,
        dump_comms=dump_comms,
        pre_existing=pre_existing,
        num_threads=num_threads,
        profile_tactic_process=profile_tactic_process
    )
