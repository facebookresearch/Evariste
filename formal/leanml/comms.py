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
import time
import os
import sys
from subprocess import Popen, PIPE, DEVNULL
import io
import re
from leanml.parse_goal import parse_goal_structured
from evariste.utils import formal_workdir


from evariste import json as json


class MissingLib(Exception):
    pass


class WrongModule(Exception):
    pass


class DeadLean(Exception):
    pass


LEAN_RESOURCES = ["others", "cleaning_utils"]


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
    for start in LEAN_RESOURCES:
        if path_to_map.startswith(start):
            res = lean_roots[start] / path_to_map.replace(f"{start}/", "")
    if path_to_map.startswith("/"):
        res = Path(path_to_map)
    if res is None:
        raise WrongModule(f"Misunderstood path : {path_to_map}")
    if not res.exists():
        raise WrongModule(f"Mapped file {res} for {path_to_map} doesn't seem to exist.")
    return str(res)


@dataclass
class PreExistingMLServer:
    stdin: IO[str]
    stdout: IO[str]

    def poll(self):
        return None


intro_regex = r"(?<!(\w|_|\d))intro[Is]?(?!(\w|_|\d))(?P<args>( [^ }\]]+)*)"
hole_in_args = r"(?<!(\w|\d))_(?!(\w|\d))"


def has_intro_without_names(tactic_str):
    for tac in re.split(",|;|<\|>", tactic_str):
        for m in re.finditer(intro_regex, tac):
            if m.group("args") == "" or re.search(hole_in_args, m.group("args")):
                return True
    return False


class LeanAPI:
    """
    Interface with Lean ml tooling checkpoint.
    
    :param bin_path: path to checkpointed (or not) lean_ml_tooling binary
    :type bin_path: str
    :param preload: whether to preload everything in lean_roots (used for checkpointing, slow otherwise)
    :type preload: bool
    :param lean_roots: absolute path for lean libraries. key: "resource" identifier such as lean, mathlib or others. value: absolute path.
    :type lean_roots: Dict[str, Path]
    :param quiet: pipes binary stderr to /dev/null, defaults to False
    :type quiet: bool, optional
    :param dump_comms: if True, dump all receive / send to stdout, defaults to False
    :type dump_comms: bool, optional
    :param num_threads: number of threads used by lean, defaults to 10
    :type num_threads: int, optional
    :param profile_tactic_process: produces lots of logs in stderr, containing all start time / end time for each tactic processed, defaults to False
    :type profile_tactic_process: bool, optional
    :param pre_existing: whether to use an already running binary with named unix fifos, defaults to None
    :type pre_existing: Optional[PreExistingMLServer], optional
    :param old: must be set to True for lean_ml_tooling_v31 and earlier. Main change is that minif2f and other statements, as well as cleaning code are loaded at runtime now. Defaults to False
    :type old: bool, optional
    :param additional_roots: If old is False, then lean will be able to load from files in these paths as well. Defaults to None
    :type additional_roots: Optional[List[Path]], optional
    """

    def __init__(
        self,
        bin_path: str,
        preload: bool,
        lean_roots: Dict[str, Path],
        quiet=False,
        dump_comms=False,
        num_threads=10,
        profile_tactic_process=False,
        pre_existing: Optional[PreExistingMLServer] = None,
        old: bool = False,
        additional_roots: Optional[List[Path]] = None,
    ) -> None:

        self.forbidden_tactic_answers: List[Dict[str, Any]] = []
        self.lean_roots = lean_roots
        if not old:
            for resource in LEAN_RESOURCES:
                p = formal_workdir() / "runtime_loaded_lean_files" / resource
                assert p.exists(), f"{p} resource missing"

        self.req_id = 0
        self.logger = logging.getLogger()
        self.dump_comms = dump_comms
        if pre_existing is None:
            os.environ["LEAN_PATH"] = ":".join(
                [str(root) for root in lean_roots.values()]
            )
            self._proc: Union[PreExistingMLServer, Popen] = Popen(
                [bin_path, "preload"] if preload else [bin_path],
                encoding="utf-8",
                stdin=PIPE,
                stdout=PIPE,
                stderr=DEVNULL if quiet else sys.stderr,
                # line buffering
                bufsize=1,
                universal_newlines=True,
            )
            assert self._proc.stdout is not None
            output = self._proc.stdout.readline().strip()
            self.logger.info(output)
            if output != "lean ml server ready":
                raise DeadLean(output)
        else:
            self._proc = pre_existing

        self.lean_roots = lean_roots

        if not old:
            paths = []
            if additional_roots is not None:
                for x in additional_roots:
                    assert x.exists()
                    paths.append(str(x))
            for resource in LEAN_RESOURCES:
                p = formal_workdir() / "runtime_loaded_lean_files" / resource
                self.lean_roots[resource] = p
                paths.append(str(p))
            self.runtime_paths = ":".join(paths)
        else:
            self.runtime_paths = ""

        self.send(
            {
                "num_threads": num_threads,
                "profile_tactic_process": profile_tactic_process,
                "paths": self.runtime_paths,
            }
        )
        assert "ok" in self.recv()
        self.logger.info("Lean server initialized")
        assert self._proc.stdout is not None
        os.set_blocking(self._proc.stdout.fileno(), False)

        if pre_existing is not None:
            try:
                self.recv(timeout=0.1)
            except RuntimeError as e:
                assert str(e) == "lean ml server ready\n", repr(str(e))
            except TimeoutError:
                pass

    def new_session(
        self,
        module_path: str,
        decl_name: str,
        merge_alpha_equiv: bool,
        pp_opts: Optional[Dict] = None,
    ) -> str:
        """Create a new "session" in the lean API. A session starts by loading the given decl_name from module_path.
        The lean API is responsible for holding all new tactic states created in this session.

        :param module_path: a path to the lean file holding `decl_name`, relative to a "resource"
        :type module_path: str
        :param decl_name: declaration to be loaded
        :type decl_name: str
        :param merge_alpha_equiv: whether alpha equivalent nodes should be merged. When set to true, two tactic states equivalent up to renaming will use the same node id.
        :type merge_alpha_equiv: bool
        :param pp_opts: pretty printer options, all options from #set_option are supported, defaults to None
        :type pp_opts: Optional[Dict], optional
        :return: req_id
        """
        return self.send(
            {
                "req_type": "new_session",
                "module_path": lean_file_to_path(self.lean_roots, module_path),
                "decl_name": decl_name,
                "merge_alpha_equiv": merge_alpha_equiv,
                "opts": pp_opts if pp_opts is not None else {},
            }
        )

    def del_session(self, name: str) -> str:
        """Delete a session in the lean API and release its resources.

        :param name: the session name
        :type name: str
        :return: req_id
        """
        return self.send({"req_type": "del_session", "name": name,})

    def parse_goal(
        self,
        goal_pp: str,
        session_name: str,
        timeout: int = 2000,
        max_repeated_hyps: int = 10_000,
    ) -> str:
        """Given a pretty-printed goal, translate it to an expression `some_expr`, and attempt to parse have `some_expr`

        :param goal_pp: the pretty-printed goal string
        :type goal_pp: str
        :param session_name: the session name with an open decl with proper imported modules and open locales
        :type session_name: str
        :param timeout: defaults to 2000
        :type timeout: int, optional
        :param max_repeated_hyps: maximum number of repeated hypothesis names, defaults to 10_000
        :type max_repeated_hyps: int, optional
        :return: req_id
        """
        try:
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
        strip_tags: bool = True,
    ) -> str:
        """
        :meth:`parse_goal` and :meth:`send_tactic` in one call. See the documentation for these functions
        :param strip_tags: remove `cases x:` from pretty printed result, defaults to True
        :type strip_tags: bool, optional
        :return: req_id
        """
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
                "strip_tags": strip_tags,
            }
        )

    def parse_children(
        self,
        children: List[str],
        session_name: str,
        timeout: int = 2000,
        max_metavars: int = 10_000,
        max_repeated_hyps: int = 10_000,
    ) -> str:
        """`api.parse_goal` on a list of pretty printed goal : `children`
        :param children: list of pretty printed goals to parse
        :type strip_tags: bool, optional
        :return: req_id
        """
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

    def parse_command(
        self,
        goal_pp: str,
        session_name: str,
        timeout: int = 2000,
        max_repeated_hyps: int = 10_000,
    ) -> str:
        """ Similar to `parse_goal` but attempts to load as a decl instead of have.
        :return: req_id
        """
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
        nosplit: bool = False,
        allow_anon_intro: bool = False,
    ) -> str:
        """Apply tactic `tactic_str` to `state_id` in `session_name`

        :param session_name: the session
        :type session_name: str
        :param state_id: node id in the session
        :type state_id: int
        :param tactic_str: tactic string
        :type tactic_str: str
        :param timeout: defaults to 2000
        :type timeout: int, optional
        :param max_size: size is roughly the number of element in proof term. avoid borked tactic states that weighs 500MB, defaults to 1000**3
        :type max_size: int, optional
        :param max_subgoals: max number of subgoals in result, defaults to 10_000
        :type max_subgoals: int, optional
        :param max_metavars: max number of metavars in result, defaults to 10_000
        :type max_metavars: int, optional
        :param max_repeated_hyps: max number of repeated hypothesis name, defaults to 10_000
        :type max_repeated_hyps: int, optional
        :param nosplit: Whether the result should be split into independent subgoals if possible, defaults to False
        :type nosplit: bool, optional
        :param allow_anon_intro: whether anonymous intros are allowed.
        :return: req_id
        """
        if not allow_anon_intro and has_intro_without_names(tactic_str):
            self.forbidden_tactic_answers.append(
                {
                    "req_id": str(self.req_id),
                    "error": f"Forbidden intro tactic without explicit names in {tactic_str}",
                }
            )
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

    def eval_cmd(
        self, to_run: str, module_path: Optional[str], timeout: int = 10_000
    ) -> str:
        """Runs a lean command like #eval to_run

        :param to_run: the command to run
        :type to_run: str
        :param module_path: path to module at the end of which "#eval to_run" would work. None for old checkpoints, where the cmd is run after `do_clean_proof` declaration.
        :type module_path: Optional[str]
        :param timeout: defaults to 10_000
        :type timeout: int, optional
        :return: req_id
        """
        return self.send(
            {
                "req_type": "eval_cmd",
                "to_run": to_run,
                "module_path": lean_file_to_path(self.lean_roots, module_path)
                if module_path is not None
                else "",
                "timeout": timeout,
            }
        )

    def send(self, to_send: Dict[str, Any]) -> str:
        """Actually send *to_send* as json to the lean server.

        :param to_send: dictionary to send to the lean_api
        :type to_send: Dict[str, Any]
        :raises DeadLean: if the lean process is dead
        :return: req_id
        """
        to_send["req_id"] = str(self.req_id)
        if self.dump_comms:
            print(">", to_send)
        self.req_id += 1
        try:
            assert self._proc.stdin is not None
            self._proc.stdin.write(json.dumps(to_send) + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError:
            proc_poll = self._proc.poll()
            if proc_poll is not None:
                raise DeadLean(f"Proc.poll returned {proc_poll}")
            raise DeadLean("Broken pipe but proc_poll is None :(")
        return str(self.req_id - 1)

    def recv(self, timeout: float = -1) -> Dict[str, Any]:
        """Poll lean process for answers to requests.

        :param timeout: defaults to -1
        :type timeout: int, optional
        :raises DeadLean: if lean process is not alive
        :raises RuntimeError: if json can't be decoded
        :raises TimeoutError: if timeout is exceeded
        :return: decoded dictionary from the lean api. Keys depend on the request answered.
        :rtype: Dict[str, Any]
        """
        while self.forbidden_tactic_answers:
            return self.forbidden_tactic_answers.pop()
        start = time.time()
        res: Optional[str] = None
        while timeout < 0 or time.time() - start < timeout:
            try:
                proc_poll = self._proc.poll()
                if proc_poll is not None:
                    raise DeadLean(f"Proc.poll returned {proc_poll}")
                assert self._proc.stdout is not None
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

    def close(self) -> None:
        """Send SIGKILL to lean process if not pre-existing."""
        if isinstance(self._proc, Popen):
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
    path_to_fifos: Optional[Tuple[Path, Path]] = None,
    old: bool = False,
    additional_roots: Optional[List[Path]] = None,
) -> LeanAPI:
    """Creates a LeanAPI. Sets up absolute paths to some resources based on lean_ml_tooling binary paths.
    Assumes the following structure:
    path_to_server = /path/to/root/build/release/binary
    Where root contains:
    - lean/ submodule
    - mathlib/ submodule
    - build/release/binary
    See LeanAPI documentation for arguments.
    :return: created lean api
    :rtype: LeanAPI
    """
    root_dir = path_to_server.parent.parent.parent
    if not all([x in os.listdir(root_dir) for x in {"lean", "mathlib"}]):
        raise MissingLib(f"Both 'lean' and 'mathlib' are expected in folder {root_dir}")
    lean_roots = {
        "lean": root_dir / "lean/library",
    }
    if not fast:
        lean_roots["mathlib"] = root_dir / "mathlib/src"
    if old:
        lean_roots["others"] = root_dir / "others"
        lean_roots["cleaning_utils"] = root_dir / "cleaning_utils"
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
        profile_tactic_process=profile_tactic_process,
        old=old,
        additional_roots=additional_roots,
    )
