# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Union, Optional, NewType, Tuple, Dict, Set
from evariste import json as json
import re
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import hashlib
from functools import wraps
import traceback
from threading import Thread, Event

from evariste.logger import create_logger, _MixinLoggerFactory
from evariste.metrics import timeit, log_timeit
from evariste.envs.ocaml.api import (
    OCamlAPI,
    OCamlError,
    OCamlErrorTimeout,
    OCamlErrorFailedRequest,
)


HOLLightToken = NewType("HOLLightToken", str)
HOLLightState = NewType("HOLLightState", str)


@dataclass
class TokensWithAnnotTypes:
    """
    This class stores the tokens as well as the actual list of annotated types
    """

    tokens: List[HOLLightToken]
    annotated_types: List[HOLLightToken]

    @staticmethod
    def from_json(data: dict) -> "TokensWithAnnotTypes":
        tokens = _ocaml_to_hol_tokens(data["tokens"])
        return TokensWithAnnotTypes(
            tokens=tokens, annotated_types=data["annotated_types"]
        )

    @property
    def systypes(self) -> Dict[HOLLightToken, int]:
        """Returns all the systypes with their number of annotations"""

        counter = defaultdict(int)
        for ty in self.annotated_types:
            if _is_systype(ty):
                counter[ty] += 1
        return counter

    @property
    def caps_in_use(self) -> Set[str]:
        caps = {ty for ty in self.systypes if ty in _ALL_CAPS}
        return caps

    def replace_systypes(
        self, systypes_replacements: Optional[Dict[HOLLightToken, HOLLightToken]] = None
    ) -> "TokensWithAnnotTypes":
        """
        Replace systypes according to the 'systypes_replacements' mapping.
        """
        if systypes_replacements is None:
            return TokensWithAnnotTypes(
                tokens=self.tokens, annotated_types=self.annotated_types
            )
        if not all(_is_systype(tok) for tok in systypes_replacements):
            raise HOLLightException(
                f"Some tokens are not system-defined types: "
                f"{[k for k in systypes_replacements]}"
            )
        tokens = [systypes_replacements.get(tok, tok) for tok in self.tokens]
        annotated_types = [
            systypes_replacements.get(tok, tok) for tok in self.annotated_types
        ]
        return TokensWithAnnotTypes(tokens=tokens, annotated_types=annotated_types)


class HOLLightHypothesis(TokensWithAnnotTypes):
    def __init__(
        self,
        tokens: List[HOLLightToken],
        annotated_types: List[HOLLightToken],
        name: Optional[HOLLightToken] = None,
    ):
        TokensWithAnnotTypes.__init__(
            self, tokens=tokens, annotated_types=annotated_types
        )
        self.name = name

    @staticmethod
    def from_json(data: dict) -> "HOLLightHypothesis":
        tokens_with_annot_types = TokensWithAnnotTypes.from_json(data)
        return HOLLightHypothesis(
            name=data["name"] if data["name"] else None,
            tokens=tokens_with_annot_types.tokens,
            annotated_types=tokens_with_annot_types.annotated_types,
        )

    def replace_systypes(
        self, systypes_replacements: Optional[Dict[HOLLightToken, HOLLightToken]] = None
    ) -> "HOLLightHypothesis":
        """
        Replace systypes according to the 'systypes_replacements' mapping
        Args
            systypes_replacements: Dict[HOLLightToken, HOLLightToken]
        Returns
            hyp: HOLLightHypothesis
        """
        tokens_with_annot_ty = TokensWithAnnotTypes.replace_systypes(
            self, systypes_replacements=systypes_replacements
        )
        return HOLLightHypothesis(
            tokens=tokens_with_annot_ty.tokens,
            annotated_types=tokens_with_annot_ty.annotated_types,
            name=self.name,
        )


@dataclass
class RawHOLLightGoal:
    hyps_tokens_no_systypes: List[HOLLightHypothesis]
    hyps_tokens_with_systypes: List[HOLLightHypothesis]
    concl_tokens_no_systypes: TokensWithAnnotTypes
    concl_tokens_with_systypes: TokensWithAnnotTypes
    state: HOLLightState
    pprint: str
    _hash: int = field(init=False)

    @staticmethod
    def from_json(data: dict) -> Union["RawHOLLightGoal", None]:
        if not data:
            return None
        hyps_tokens_no_systypes = [
            HOLLightHypothesis.from_json(hyp) for hyp in data["hyps_tokens_no_systypes"]
        ]
        hyps_tokens_with_systypes = [
            HOLLightHypothesis.from_json(hyp)
            for hyp in data["hyps_tokens_with_systypes"]
        ]
        if len(hyps_tokens_no_systypes) != len(hyps_tokens_with_systypes):
            raise HOLLightException(
                f"Not the same number of hypotheses: len(hyps_tokens_no_systypes)={len(hyps_tokens_no_systypes)}"
                f" vs len(hyps_tokens_with_systypes)={len(hyps_tokens_with_systypes)}"
            )
        concl_tokens_no_systypes = TokensWithAnnotTypes.from_json(
            data["concl_tokens_no_systypes"]
        )
        concl_tokens_with_systypes = TokensWithAnnotTypes.from_json(
            data["concl_tokens_with_systypes"]
        )
        return RawHOLLightGoal(
            hyps_tokens_no_systypes=hyps_tokens_no_systypes,
            hyps_tokens_with_systypes=hyps_tokens_with_systypes,
            concl_tokens_no_systypes=concl_tokens_no_systypes,
            concl_tokens_with_systypes=concl_tokens_with_systypes,
            state=data["state"],
            pprint=data["pprint"],
        )

    @property
    def systypes(self) -> Dict[HOLLightToken, int]:
        """Returns all the systypes with their number of annotations"""
        counter = Counter(self.concl_tokens_with_systypes.systypes)
        for hyp in self.hyps_tokens_with_systypes:
            counter.update(hyp.systypes)
        return dict(counter)

    @property
    def caps_in_use(self) -> Set[str]:
        caps_in_use = self.concl_tokens_no_systypes.caps_in_use
        for hyp in self.hyps_tokens_with_systypes:
            caps_in_use.update(hyp.caps_in_use)
        return caps_in_use

    def __hash__(self):
        return self._hash

    def __post_init__(self):
        self._hash = int(hashlib.md5(str(self.state).encode()).hexdigest(), 16)


@dataclass
class HOLLightGoalstack:
    raw_subgoals: List[RawHOLLightGoal]
    raw_top_goal: Optional[RawHOLLightGoal]

    @staticmethod
    def from_json(data: dict) -> "HOLLightGoalstack":
        raw_subgoals = [RawHOLLightGoal.from_json(sg) for sg in data["subgoals"]]
        raw_top_goal = RawHOLLightGoal.from_json(data["top_goal"])
        return HOLLightGoalstack(raw_subgoals=raw_subgoals, raw_top_goal=raw_top_goal)


@dataclass
class HOLLightGoal:
    hyps_tokens: List[
        Tuple[Optional[HOLLightToken], List[HOLLightToken]]
    ]  # (name, tokens)
    concl_tokens: List[HOLLightToken]
    raw: Optional[RawHOLLightGoal] = None

    @staticmethod
    def from_raw(
        raw_goal: RawHOLLightGoal,
        concl_with_systypes: bool = True,
        hyps_with_systypes: Optional[List[bool]] = None,  # default to [True, True, ...]
        systypes_replacements: Optional[Dict[HOLLightToken, HOLLightToken]] = None,
    ) -> "HOLLightGoal":
        if hyps_with_systypes is None:
            hyps_with_systypes = [True] * len(raw_goal.hyps_tokens_with_systypes)
        if len(hyps_with_systypes) != len(raw_goal.hyps_tokens_with_systypes):
            raise HOLLightException(
                f"Not the same number of hypotheses: len(hyps_with_systypes)={len(hyps_with_systypes)}"
                f" vs len(raw_goal.hyps_tokens_with_systypes)={len(raw_goal.hyps_tokens_with_systypes)}"
            )
        concl_tokens = (
            raw_goal.concl_tokens_with_systypes.replace_systypes(
                systypes_replacements
            ).tokens
            if concl_with_systypes
            else raw_goal.concl_tokens_no_systypes.tokens
        )
        hyps_tokens = [
            (
                hyp_sty.name,
                hyp_sty.replace_systypes(systypes_replacements).tokens
                if with_sty
                else hyp_no_sty.tokens,
            )
            for with_sty, hyp_no_sty, hyp_sty in zip(
                hyps_with_systypes,
                raw_goal.hyps_tokens_no_systypes,
                raw_goal.hyps_tokens_no_systypes,
            )
        ]
        return HOLLightGoal(
            hyps_tokens=hyps_tokens, concl_tokens=concl_tokens, raw=raw_goal
        )

    def __eq__(self, goal):
        if not isinstance(goal, type(self)):
            return False
        return (
            self.hyps_tokens == goal.hyps_tokens
            and self.concl_tokens == goal.concl_tokens
        )

    @property
    def light(self) -> Dict:
        """
        'light' export of goal for the dataset
        returns
            light_goal: {
                "hyps": List[Tuple[Optional[HOLLightToken], str]],
                "concl": str
            }
        """
        hyps = [(name, " ".join(tokens)) for name, tokens in self.hyps_tokens]
        concl = " ".join(self.concl_tokens)
        return {"hyps": hyps, "concl": concl}


@dataclass
class HOLLightSample:
    """Goal + Tactic -> Subgoals, with systypes normalized"""

    goal: HOLLightGoal
    tactic: List[HOLLightToken]
    subgoals: List[HOLLightGoal]
    raw_goal: Optional[RawHOLLightGoal] = None
    raw_subgoals: Optional[List[HOLLightGoal]] = None
    origin: Optional[Tuple[str, int, str]] = None  # filename, line_no, theorem name

    @staticmethod
    def from_raw(
        raw_goal: RawHOLLightGoal,
        tactic: List[HOLLightToken],
        raw_subgoals: List[RawHOLLightGoal],
        origin: Optional[Tuple[str, int, str]] = None,
        systypes_replacements: Optional[Dict[HOLLightToken, HOLLightToken]] = None,
    ) -> "HOLLightSample":
        goal = HOLLightGoal.from_raw(
            raw_goal=raw_goal, systypes_replacements=systypes_replacements
        )
        subgoals = [
            HOLLightGoal.from_raw(
                raw_goal=raw_sg, systypes_replacements=systypes_replacements
            )
            for raw_sg in raw_subgoals
        ]
        return HOLLightSample(
            goal=goal,
            tactic=tactic,
            subgoals=subgoals,
            origin=origin,
            raw_goal=raw_goal,
            raw_subgoals=raw_subgoals,
        )

    def normalize(self) -> "HOLLightSample":
        systypes_replacements = _map_systypes_with_caps(
            raw_goals=[self.raw_goal, *self.raw_subgoals],
        )
        return HOLLightSample.from_raw(
            raw_goal=self.raw_goal,
            tactic=self.tactic,
            raw_subgoals=self.raw_subgoals,
            systypes_replacements=systypes_replacements,
            origin=self.origin,
        )

    @staticmethod
    def normalize_proof(
        goalstacks: List[HOLLightGoalstack],
        tactics: List[List[HOLLightToken]],
        origin: Optional[Tuple[str, int, str]] = None,
    ) -> List["HOLLightSample"]:
        if len(goalstacks) != (len(tactics) + 1):
            raise HOLLightException(
                f"Error in proof normalization: len(goalstacks) ({len(goalstacks)}) != len(tactics) + 1 ({len(tactics)} + 1)"
            )
        if goalstacks[0].raw_subgoals:
            raise HOLLightException("The initial goalstack should not have subgoals")
        # check the list of goalstack is a proof
        samples = []
        raw_top_goal = goalstacks[0].raw_top_goal
        stack = [raw_top_goal]
        systypes_replacements_per_subgoals = {
            raw_top_goal: {}
        }  # as the top goal is never a subgoal
        for goalstack, tactic in zip(goalstacks[1:], tactics):
            # check the list of goalstacks is a proof
            if not stack or raw_top_goal != stack.pop():
                raise HOLLightException(
                    "The list of goalstacks is not a valid sequance of goalstacks"
                )
            raw_subgoals = goalstack.raw_subgoals
            stack.extend(raw_subgoals)
            systypes_replacements = _map_systypes_with_caps(
                raw_goals=[raw_top_goal, *raw_subgoals],
                prev_systypes_replacements=systypes_replacements_per_subgoals[
                    raw_top_goal
                ],
            )
            sample = HOLLightSample.from_raw(
                raw_goal=raw_top_goal,
                tactic=tactic,
                raw_subgoals=raw_subgoals,
                systypes_replacements=systypes_replacements,
                origin=origin,
            )
            systypes_replacements_per_subgoals.update(
                {sg: systypes_replacements for sg in raw_subgoals}
            )
            samples.append(sample)
            raw_top_goal = goalstack.raw_top_goal
        # check the list of goalstacks is a proof
        if stack or raw_top_goal:
            raise HOLLightException("Not a proof")
        return samples

    def __eq__(self, sample):
        if not isinstance(sample, type(self)):
            return False
        return (
            self.goal == sample.goal
            and self.tactic == sample.tactic
            and self.subgoals == sample.subgoals
        )

    @property
    def light(self) -> Dict:
        """
        'light' export of sample for the dataset
        returns
            light_sample: {
                "goal": {
                    "hyps": List[Tuple[Optional[HOLLightToken], str]],
                    "concl": str
                },
                "tactic: str,
                "subgoals": List[
                    {
                        "hyps": List[Tuple[Optional[HOLLightToken], str]],
                        "concl": str
                    }
                ]
            }
        """
        return {
            "goal": self.goal.light,
            "tactic": " ".join(self.tactic),
            "subgoals": [sg.light for sg in self.subgoals],
        }


class HOLLightException(Exception):
    def __init__(
        self, *args, logged: bool = False, origin: Optional[Exception] = None
    ) -> None:
        super().__init__(*args)
        self.logged = logged
        self.origin = origin


def wrap_error(fn):
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except (OCamlError, HOLLightException) as e:
            msg = f"{str(e)}\n{traceback.format_exc()}"
            if isinstance(e, OCamlError):
                msg = f"(from {type(e).__name__}): {msg}"
                exc = HOLLightException(msg, origin=e)
            else:
                exc = e
            if not exc.logged:
                if isinstance(exc.origin, OCamlErrorTimeout) or isinstance(
                    exc.origin, OCamlErrorFailedRequest
                ):
                    # too many logs that are not for actual errors, more for failed tactics
                    log_level = "debug"
                else:
                    log_level = "error"
                self.log(f"HOLLightException {msg}", log_level)
                exc.logged = True
            if isinstance(e, OCamlErrorFailedRequest):
                pass
            elif isinstance(e, OCamlError):
                self.make_env()
            raise exc

    return wrapped


def _log_formatter(obj, _, msg):
    return f'HOLLightAPI{" #" + str(obj.rank) if obj.rank is not None else ""}: {msg}'


class HOLLightAPI(_MixinLoggerFactory("debug", _log_formatter)):
    """Interface with a single HOL-Light process"""

    def __init__(
        self,
        checkpoint_path: str,
        timeout: float,
        logger=None,
        rank: Optional[int] = None,
    ):
        logger = logger if logger is not None else create_logger(None)
        self.set_logger(logger)
        self.rank = rank
        self.checkpoint_path = checkpoint_path
        self.timeout = timeout
        self.env_ready = Event()
        self.make_env()  # creates the self.env: OCamlAPI

    @timeit
    def wait_till_ready(self):
        self.env_ready.wait()

    @timeit
    def _make_env(self) -> None:
        if not hasattr(self, "_env"):
            self.log("booting OCaml process ...")
        else:
            self.log("rebooting OCaml process ...")
        while True:
            try:
                if hasattr(self, "_env"):
                    self._env.hard_kill()
                self._env = OCamlAPI(
                    checkpoint_path=self.checkpoint_path,
                    timeout=self.timeout,
                    logger=self.logger,
                )
            except OCamlError:
                self.log("failed to boot OCaml process - retrying ...")
            else:
                break
        self.log(f"OCaml process booted: PID#{self._env.rank}")
        self.env_ready.set()

    @timeit
    def make_env(self) -> None:
        self.env_ready.clear()
        # thread in order not to block wrap_error if there is a reboot
        Thread(target=HOLLightAPI._make_env, name="_make_env", args=(self,)).start()

    @timeit
    def _send(self, cmd: str) -> str:
        self.env_ready.wait()
        self.log(f'sending to OCaml process: "{cmd}"')
        s = self._env.send(cmd)
        self.log(f'getting from OCaml process: "{s}"')
        return s

    @timeit
    @wrap_error
    def bwd_apply_tactic(
        self,
        tactic_tokens: List[HOLLightToken],
        concl_tokens: Optional[List[HOLLightToken]] = None,
        hyps_tokens: Optional[
            List[Tuple[Optional[HOLLightToken], List[HOLLightToken]]]
        ] = None,
        state: Optional[HOLLightState] = None,
    ) -> HOLLightSample:
        """
        Apply a tactic (expressed as tokens) to a goal (expressed as tokens or state)
        """
        # restore goal
        goalstack = self.set_bwd_proving(
            concl_tokens=concl_tokens, hyps_tokens=hyps_tokens, state=state
        )
        raw_goal = goalstack.raw_top_goal
        # apply tactic
        goalstack = self.bwd_apply_tactic_to_goalstack(tactic_tokens=tactic_tokens)
        raw_subgoals = goalstack.raw_subgoals
        return HOLLightSample.from_raw(
            raw_goal=raw_goal, tactic=tactic_tokens, raw_subgoals=raw_subgoals
        )

    @timeit
    @wrap_error
    def set_bwd_proving(
        self,
        concl_tokens: Optional[List[HOLLightToken]] = None,
        hyps_tokens: Optional[
            List[Tuple[Optional[HOLLightToken], List[HOLLightToken]]]
        ] = None,
        state: Optional[HOLLightState] = None,
    ) -> HOLLightGoalstack:
        """
        Set the goalstack with a top_goal, in order to apply a sequence of tactics
        """
        if (state is None and concl_tokens is None) or (
            state is not None and (concl_tokens is not None or hyps_tokens is not None)
        ):
            raise HOLLightException(
                "Either a state or tokens should be provided, but not both"
            )
        s = self._send(
            _cmd_set_goal(
                concl_tokens=concl_tokens, hyps_tokens=hyps_tokens, state=state
            )
        )
        # check that the goal set is the same as the input
        goalstack = _parse_output_as_goalstack(s)
        raw_goal_set = goalstack.raw_top_goal
        if state is not None and state != raw_goal_set.state:
            raise HOLLightException(
                "Goal set different from input goal (different states)"
            )
        elif (
            concl_tokens is not None
            and (hyps_tokens or [])
            != [(hyp.name, hyp.tokens) for hyp in raw_goal_set.hyps_tokens_no_systypes]
            and concl_tokens != raw_goal_set.concl_tokens_no_systypes.tokens
        ):
            raise HOLLightException(
                "Goal set different from input goal (different tokens): "
                f"INPUT hyps={[(name, ' '.join(tokens)) for name, tokens in hyps_tokens]}, concl=\"{' '.join(concl_tokens)}\" != "
                f"SET hyps={[(hyp.name, ' '.join(hyp.tokens)) for hyp in raw_goal_set.hyps_tokens_no_systypes]},"
                f" concl=\"{' '.join(raw_goal_set.concl_tokens_no_systypes.tokens)}\""
            )
        return goalstack

    @timeit
    @wrap_error
    def bwd_apply_tactic_to_goalstack(
        self, tactic_tokens: List[HOLLightToken]
    ) -> HOLLightGoalstack:
        """
        Apply a tactic to the current goalstack
        Args:
            tactic_tokens: List[HOLLightToken]
        Returns:
            goalstack: HOLLightGoalstack
        """
        tact_str = _build_tactic_from_tokens(tactic_tokens).strip()
        cmd = f"e ({tact_str});;"
        self.log(f'HOLLightAPI apply to the top_goal the tactic "{tact_str}"')
        s = self._send(cmd)
        goalstack = _parse_output_as_goalstack(s)
        return goalstack

    def __del__(self):
        self.close()

    @log_timeit
    def close(self):
        print("Destroying HOLLightAPI...")
        if hasattr(self, "_env"):
            print("Destroying HOLLightAPI.env...")
            self._env.hard_kill()


#################################################################################################
# Internal helpers
#################################################################################################

# The next two functions are similar in spirit to tokenize_hl and detokenize_hl
# (see evariste/envs/hol_light_tokenizer)
# however, since the splitting with blank space is actually delimiting tokens directly in HOL Light,
# there is no need to code tokenization rules, EXCEPT FOR NUMBERS


def _ocaml_to_hol_tokens(ocaml_tokens: List[str]) -> List[HOLLightToken]:
    """
    Currently, for a number of the form `#12.34`, the HOL Light printer would tokenize it as
    '#', '12', '.', '34'
    So the tokenization of this expression would return:
    '#', '<NUMBER>', '1', '2', '</NUMBER>', '.', '<NUMBER>', '3', '4', '</NUMBER>'
    TODO: An improvement would be to have HOL Light directly tokenize as '#', '12.34',
    or even better as the expected final tokenization:
    '#', '<NUMBER>', '1', '2', '.', '3', '4', '</NUMBER>'
    """
    tokens = []
    for tok in ocaml_tokens:
        if re.fullmatch(r"\d+", tok) is not None:
            tokens.extend(["<NUMBER>", *list(tok), "</NUMBER>"])
        else:
            tokens.append(tok)
    return tokens


def _hol_tokens_to_ocaml(tokens: List[HOLLightToken]) -> str:
    ocaml_str = ""
    is_number = False
    for tok in tokens:
        if tok == "<NUMBER>":
            if is_number:
                raise HOLLightException(f"Wrong number tokenization: {tokens}")
            is_number = True
            continue
        if tok == "</NUMBER>":
            if not is_number:
                raise HOLLightException(f"Wrong number tokenization: {tokens}")
            is_number = False
            ocaml_str += " "
            continue
        ocaml_str += tok + ("" if is_number else " ")
    if is_number:
        raise HOLLightException(f"Wrong number tokenization: {tokens}")
    return ocaml_str.strip()


def _cmd_set_goal(
    concl_tokens: Optional[List[HOLLightToken]] = None,
    hyps_tokens: Optional[
        List[Tuple[Optional[HOLLightToken], List[HOLLightToken]]]
    ] = None,
    state: HOLLightState = None,
) -> str:
    """
    Builds the cli command that restores the goal as the top goal
    in the HOL-Light process.
    """
    if state is not None:
        return f'restore_as_top_goal "{state}";;'
    cmd = "set_goal_with_labeled_hyps(["
    if hyps_tokens:
        hyps_tokens = hyps_tokens[::-1]  # in reverse
        cmd += (
            f'("{hyps_tokens[0][0] if hyps_tokens[0][0] else ""}", parse_term "'
            + _hol_tokens_to_ocaml(hyps_tokens[0][1])
            + '")'
        )
        for hyp in hyps_tokens[1:]:
            cmd += (
                f'; ("{hyp[0] if hyp[0] else ""}", parse_term "'
                + _hol_tokens_to_ocaml(hyp[1])
                + '")'
            )
    cmd += "], "
    cmd += ' parse_term "' + _hol_tokens_to_ocaml(concl_tokens) + '"'
    cmd += ");;"
    return cmd


def _build_tactic_from_tokens(tactic_tokens: List[HOLLightToken]) -> str:
    # BUGFIX: for 'distinctiveness "list"' it is tokenized as 'distinctiveness', '"'", 'list', '""
    # hence it is detokenized as 'distinctiveness " list "' which does not work...
    # hence between quotes and backticks there should be no whitespaces
    # another corner case: SUBGOAL_TAC "" `..` is tokenized as 'SUBGOAL_TAC', '"", '"", '``, ...
    tact_str = _hol_tokens_to_ocaml(tactic_tokens)
    return re.sub(r'"\s*(?P<inside>[^"]*)\s+"', r'"\g<inside>"', tact_str)


_PATTERN_OUT = re.compile(
    r"val\sit\s:\s(?P<type_of_result>\w+)\s=\s*\x00?(?P<result>.+)", re.DOTALL
)


def _parse_output_as_json(s: str) -> Tuple[str, Union[int, float, str, List, Dict]]:
    """
    Parses an input string from HOL-Light as if it were a json
    Args:
        s: str
    Returns:
        type_of_result: str
        result: dict  # json-like object
    """
    matches = _PATTERN_OUT.search(s)
    if not matches:
        raise HOLLightException(f'Output not parsable: "{s}"')
    try:
        result_match = matches.group("result")
        result = json.loads(result_match)
    except json.JSONDecodeError as e:
        raise HOLLightException(
            f'from exception "{type(e).__name__}: '
            f'{str(e)}" while trying to load "{result_match}"'
        )
    return matches.group("type_of_result"), result


def _parse_output_as_goalstack(s: str) -> HOLLightGoalstack:
    """
    Parse HOL-Light output as if it were a goalstack
        - `top_goal` is the current goal to prove. If empty, it means the initial goal is proved.
        - `subgoals` is the result of the previous tactic on the previous top goal. If empty, this
          means the previous goal has been solved by the previous tactic.
    The system-defined types are added only when there is some ambiguities
    (for instance when a given systype is in hyps and in the concl)
    Args:
        s: str
    Returns:
        goalstack: HOLLightGoalstack
    """
    type_of_result, result_as_json = _parse_output_as_json(s)
    try:
        assert type_of_result == "goalstack", "Result not a goalstack"
        return HOLLightGoalstack.from_json(result_as_json)
    except Exception as e:
        raise HOLLightException(f'from exception "{type(e).__name__}: {str(e)}"')


def _is_systype(ty: str) -> bool:
    return ty.startswith("?")


_ALL_CAPS = [chr(i) for i in range(65, 91)]


def _map_systypes_with_caps(
    raw_goals: List[RawHOLLightGoal],
    prev_systypes_replacements: Optional[Dict[HOLLightToken, HOLLightToken]] = None,
) -> Dict[HOLLightToken, HOLLightToken]:
    """
    Disambiguate system-defined types (of the form `:?43234`) by
    replacing them by `:A`, `:B`, etc. when necessary
    """
    systypes_to_caps: Dict[
        HOLLightToken, HOLLightToken
    ] = {}  # systype to be normalized
    if prev_systypes_replacements is None:
        prev_systypes_replacements = {}
    if not all(c in _ALL_CAPS for c in prev_systypes_replacements.values()):
        raise HOLLightException(
            f"Not all previous systype replacements are capital letters: {prev_systypes_replacements.values()}"
        )

    # list all systypes with their previous mapping if any
    for raw_goal in raw_goals:
        for hyp in raw_goal.hyps_tokens_with_systypes:
            systypes_to_caps.update(
                {ty: prev_systypes_replacements.get(ty, None) for ty in hyp.systypes}
            )
        systypes_to_caps.update(
            {
                ty: prev_systypes_replacements.get(ty, None)
                for ty in raw_goal.concl_tokens_with_systypes.systypes
            }
        )

    # before normalizing systype as capital letters, see if there are already some in use
    caps_in_use = set()
    for raw_goal in raw_goals:
        for hyp in raw_goal.hyps_tokens_with_systypes:
            caps_in_use.update(hyp.caps_in_use)
        caps_in_use.update(raw_goal.concl_tokens_with_systypes.caps_in_use)

    # now relabel systypes with capital letters not in use with types already annotated
    available_caps = sorted(set(_ALL_CAPS) - caps_in_use)
    if len(systypes_to_caps) > len(available_caps):
        raise HOLLightException(
            "No capital letters available to replace system-defined types"
        )
    i = 0
    for sty in sorted(systypes_to_caps):  # sorted for determinism
        if systypes_to_caps[sty] is None:
            systypes_to_caps[sty] = available_caps[i]
            i += 1
    return systypes_to_caps
