# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Tuple, Optional, Dict, NamedTuple
import re

from evariste.forward.common import ProofNode
from evariste.logger import create_logger
from evariste.model.data.dictionary import (
    B_CMD_WORD,
    E_CMD_WORD,
    B_HYP_WORD,
    M_HYP_WORD,
    E_HYP_WORD,
    B_GOAL_WORD,
    E_GOAL_WORD,
    SPECIAL_WORDS,
)
from evariste.backward.graph import Tactic, Theorem, Token
from evariste.backward.graph import MalformedTactic, MalformedTheorem
from evariste.envs.hl.api import HOLLightException, _build_tactic_from_tokens


_SPECIAL_WORDS = set(SPECIAL_WORDS)


logger = create_logger(None)


hol_error_str = [
    ("TIMEOUT", "OCamlErrorTimeout"),
    ("NOPA", "Got a nopa status"),
    ("FAILURE_FIND", 'Failure "find"'),
    ("FAILURE_TRYFIND", 'Failure "tryfind"'),
    ("FAILURE_FIND_TERM", 'Failure "find_term"'),
    ("DEST_EQ", 'Failure "dest_eq"'),
    ("DEST_REALINTCONST", 'Failure "dest_realintconst"'),
    ("DEST_NUMERAL", 'Failure "dest_numeral"'),
    ("LINEAR_INEQS_NO_CONTRADICTION", 'Failure "linear_ineqs: no contradiction"'),
    ("GOAL_TOO_DEEP", 'Failure "solve_goal: Too deep"'),
    ("NO_ASSUMPTION_TO_POP", 'Failure "POP_ASSUM: No assumption to pop"'),
    ("MATCH_MP_TAC_NO_MATCH", 'Failure "MATCH_MP_TAC: No match"'),
    ("MATCH_MP_TAC_BAD_THEOREM", 'Failure "MATCH_MP_TAC: Bad theorem"'),
    ("X_GEN_TAC_INVALID_VARIABLE", 'Failure "X_GEN_TAC: invalid variable"'),
    ("TAC_PROOF_UNSOLVED_GOALS", 'Failure "TAC_PROOF: Unsolved goals"'),
    ("TAC_FAILURE", 'Failure "[0-9A-Z_]+"'),
]
hol_error_regexes = [(error_name, re.compile(rgx)) for error_name, rgx in hol_error_str]


def get_error(error: str):
    for error_name, regex in hol_error_regexes:
        if regex.search(error):
            return error_name
    return "UNK_ERROR"


class HLTactic(Tactic):
    def tokenize(self) -> List[Token]:
        assert self.is_valid
        return self._tactic.split()

    def get_error_code(self) -> str:
        return get_error(self.error_msg)

    @classmethod
    def from_tokens(cls, tokens: List[Token]) -> "HLTactic":
        try:
            if len(tokens) == 0:
                raise MalformedTactic("Empty tactic")
            if (tokens[0], tokens[-1]) != (B_CMD_WORD, E_CMD_WORD):
                raise MalformedTactic(f"Missing b_e cmd {' '.join(tokens)}")
            tactic_tokens = tokens[1:-1]
            if _SPECIAL_WORDS.intersection(set(tactic_tokens)):
                raise MalformedTactic(f"SPECIAL WORDS in {' '.join(tactic_tokens)}")
            try:
                _build_tactic_from_tokens(tactic_tokens)
            except HOLLightException as e:
                raise MalformedTactic(f"(from {type(e).__name__}): {e}")
            return cls(" ".join(tactic_tokens))
        except MalformedTactic as e:
            return cls(
                "", valid=False, error_msg=f"MalformedTactic: {str(e)}", tokens=tokens
            )

    def to_dict(self, light=True):
        if self.is_valid:
            return {
                "str": self._tactic,
                # "warning": self.warning,
            }
        return {"error_msg": self.error_msg, "str": self._tactic}

    @classmethod
    def from_dict(cls, data):
        return cls(data["str"])  # , warning=data["warning"]

    def __init__(
        self,
        tactic: str,
        valid=True,
        error_msg=None,
        # warning=None,
        tokens: Optional[List[str]] = None,
    ):
        assert type(tactic) is str
        assert valid == (error_msg is None)
        assert valid == (tokens is None)
        self.is_valid = valid
        # self.warning = warning
        self._tactic = tactic
        unique_str = self._tactic if self.is_valid else " ".join(tokens)
        Tactic.__init__(self, valid, unique_str=unique_str, error_msg=error_msg)

    def __repr__(self):
        return self._tactic if self.is_valid else "INVALID_TACTIC"


class HLTheorem(Theorem):
    def to_dict(self, light=False):
        return {
            "conclusion": self.conclusion,
            "hyps": self.hyps,
            "state": self.state if not light else None,
        }

    def to_tokens_dict(self, light=False):
        concl_tokens = self.conclusion.split()
        hyps_tokens = []
        for name, hyp in self.hyps:
            if name is not None:
                assert type(name) is str and " " not in name
            hyps_tokens.append((name, hyp.split()))
        return {
            "concl_tokens": concl_tokens,
            "hyps_tokens": hyps_tokens,
            "state": self.state if not light else None,
        }

    def tokenize(self) -> List[Token]:
        """
        B_GOAL_WORD
        [conclusion]
        B_HYP_WORD [hyp_0] E_HYP_WORD
        ...
        B_HYP_WORD [hyp_n] E_HYP_WORD
        E_GOAL_WORD
        """
        tokens_dict = self.to_tokens_dict(light=True)
        tokens = [B_GOAL_WORD]
        tokens.extend(tokens_dict["concl_tokens"])
        for name, hyp_tokens in tokens_dict["hyps_tokens"]:
            tokens.append(B_HYP_WORD)
            if name is not None:
                tokens.append(name)
            tokens.extend([M_HYP_WORD, *hyp_tokens, E_HYP_WORD])
        tokens.append(E_GOAL_WORD)
        return tokens

    @classmethod
    def from_tokens(cls, tokens: List[Token]) -> "HLTheorem":
        assert type(tokens) is list

        def exception(s: str):
            return MalformedTheorem(f"{s}: {' '.join(tokens)}")

        if len(tokens) == 0 or (tokens[0], tokens[-1]) != (B_GOAL_WORD, E_GOAL_WORD):
            raise exception("invalid theorem delimitors")

        if not (
            tokens.count(B_HYP_WORD)
            == tokens.count(M_HYP_WORD)
            == tokens.count(E_HYP_WORD)
        ):
            raise exception("invalid hypotheses delimitors")

        # remove goal token delimitors
        tokens = tokens[1:-1]

        # end of goal conclusion
        concl_idx = tokens.index(B_HYP_WORD) if B_HYP_WORD in tokens else len(tokens)
        concl_tokens = tokens[:concl_idx]

        hyps_tokens: List[Tuple[Token, List[Token]]] = []
        i = concl_idx
        try:
            while i < len(tokens):
                if tokens[i] != B_HYP_WORD:
                    raise exception("was expecting beginning of hypothesis")
                i += 1

                # parse hypothesis name
                if tokens[i] == M_HYP_WORD:
                    hyp_name = None
                    i += 1
                elif tokens[i + 1] == M_HYP_WORD:
                    hyp_name = tokens[i]
                    i += 2
                else:
                    raise exception("invalid hypothesis name")

                # parse hypothesis content
                hyp_content = []
                if tokens[i + 1] == E_HYP_WORD:
                    raise exception("empty hypothesis")
                try:
                    while tokens[i] != E_HYP_WORD:
                        hyp_content.append(tokens[i])
                        i += 1
                except IndexError:
                    raise exception("unfinished hypothesis")
                hyps_tokens.append((hyp_name, hyp_content))
                i += 1
            if i != len(tokens):
                raise exception("unfinished hypothesis")
        except IndexError:
            raise exception("unfinished hypothesis")

        # sanity check
        uniq_tok = set(concl_tokens)
        for name, hyp_tokens in hyps_tokens:
            if name is not None:
                uniq_tok.add(name)
            uniq_tok |= set(hyp_tokens)
        special_words = _SPECIAL_WORDS.intersection(uniq_tok)
        if len(special_words) > 0:
            raise exception(f"unexpected special words ({special_words})")

        conclusion = " ".join(concl_tokens)
        hyps = [(name, " ".join(hyp_tokens)) for name, hyp_tokens in hyps_tokens]
        return cls(conclusion, hyps=hyps)

    @classmethod
    def from_dict(cls, data):
        return cls(data["conclusion"], data["hyps"], state=data["state"])

    def __init__(
        self,
        conclusion: str,
        hyps: List[Tuple[Optional[Token], str]],
        train_label: Optional[Token] = None,
        state: Optional[str] = None,
    ):
        assert type(conclusion) is str
        for name, hyp in hyps:
            assert name is None or type(name) is str and " " not in name, name
            assert type(hyp) is str
        super().__init__(conclusion, hyps, train_label)
        self._state = state

    def __repr__(self):
        s = []
        if len(self.hyps) > 0:
            s = ["Hypotheses:"]
            for name, hyp in self.hyps:
                s.append(f"\t{hyp}" if name is None else f"\t{name} {hyp}")
        s.append(f"Conclusion: {self.conclusion}")
        return "\n".join(s)

    @property
    def state(self) -> Optional[str]:
        return self._state


class HLProofNode(ProofNode[HLTheorem, HLTactic]):
    """Simple hl proof node"""

    pass


class DebugInfo(NamedTuple):
    cmds_that_dont_change_goal: List[HLTactic]
    th_with_not_found_subgoals: List[HLTheorem]


def has_cycle(root: HLProofNode) -> bool:
    assert isinstance(root, HLProofNode)
    path = set()

    def traverse(node: HLProofNode):
        path.add(node.theorem)
        for child in node.children:
            if child.theorem in path or traverse(child):
                return True
        path.remove(node.theorem)
        return False

    return traverse(root)


def build_hl_proof_graphs(
    theorems: Dict[str, Dict]
) -> Tuple[Dict[str, HLProofNode], DebugInfo]:

    n_cmds = 0
    cmds_that_dont_change_goal = []
    th_with_not_found_subgoals = []
    proofs = {}

    # for each theorem
    for name, theorem in theorems.items():
        assert theorem.keys() == {"name", "filename", "line", "steps"}

        # create nodes
        theorem_to_node = {}
        goal_tactic_subgoals = []

        for step in theorem["steps"]:

            # goal
            goal = step["goal"]
            goal = HLTheorem(conclusion=goal["concl"], hyps=goal["hyps"])

            # tactic
            tactic = HLTactic(tactic=step["tactic"])
            n_cmds += 1

            # subgoals
            subgoals = [
                HLTheorem(conclusion=sg["concl"], hyps=sg["hyps"])
                for sg in step["subgoals"]
            ]

            # tactic has on effect on the goal -> skip
            if len(subgoals) == 1 and subgoals[0] == goal:
                cmds_that_dont_change_goal.append(tactic)

            # add node -- NOTE: sometimes the goal is repeated. in that case we consider
            # the last one. since steps are ordered, the last one will necessarily lead
            # to a proof, while the first ones may result in a cycle.
            theorem_to_node[goal] = HLProofNode(goal, tactic, children=[])
            goal_tactic_subgoals.append((goal, tactic, subgoals))

        # populate children
        try:
            for goal, tactic, subgoals in goal_tactic_subgoals:
                node = theorem_to_node[goal]
                node.children = [theorem_to_node[sg] for sg in subgoals]
        except KeyError:
            logger.info(f"Subgoal not found in {name}")
            th_with_not_found_subgoals.append(name)
            continue

        # save proof
        root = theorem_to_node[goal_tactic_subgoals[0][0]]
        assert not has_cycle(root)
        proofs[name] = root

    logger.info(
        f"Extracted {len(proofs)} proofs with {n_cmds} commands.\n"
        f"{len(cmds_that_dont_change_goal)} commands had no effect on the goal.\n"
        f"{len(th_with_not_found_subgoals)} theorems with not found subgoals were ignored."
    )

    return (
        proofs,
        DebugInfo(
            cmds_that_dont_change_goal=cmds_that_dont_change_goal,
            th_with_not_found_subgoals=th_with_not_found_subgoals,
        ),
    )
