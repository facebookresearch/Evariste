# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Union, List, Dict, Any, Tuple

from evariste.model.data.dictionary import (
    B_CMD_WORD,
    E_CMD_WORD,
    B_SUBST_WORD,
    M_SUBST_WORD,
    E_SUBST_WORD,
    B_HYP_WORD,
    E_HYP_WORD,
    B_GOAL_WORD,
    E_GOAL_WORD,
    SPECIAL_WORDS,
)
from evariste.backward.graph import Tactic, Theorem, Token
from evariste.backward.graph import MalformedTactic, MalformedTheorem
from evariste.envs.eq.graph import Node, NodeParseError
from evariste.envs.eq.rules import ARule, TRule
from evariste.envs.eq.rules_lib import NAME_TO_RULE, NAME_TO_TYPE


SERIALIZE_NODES = False

_SPECIAL_WORDS = set(SPECIAL_WORDS)

T_RULE_TOK = "<TRANFORM_RULE>"
A_RULE_TOK = "<ASSERT_RULE>"
SIMP_TAC_TOK = "<SIMP>"
NN_TAC_TOK = "<NORMNUM>"
FWD_TOK = "<RULE_FORWARD>"
BWD_TOK = "<RULE_BACKWARD>"

N_PREFIX_TOKS = 4096


def prefix_pos_tok(i: int):
    return f"<PREFIX_{i}>"


def prefix_var_name(name: str):
    assert type(name) is str
    return f"EQ_VAR_{name}"


eq_error_str = [
    ("was expecting beginning of hypothesis", "NO_BEGINNING_OF_HYP"),
    ("empty hypothesis", "EMPTY_HYPOTHESIS"),
    ("Empty tactic", "EMPTY_TACTIC"),
    ("unfinished hypothesis", "UNFINISHED_HYPOTHESIS"),
    ("unexpected special words", "UNEXPECTED_SPECIAL_WORDS"),
    ("from_prefix_tokens failure", "FROM_PREFIX_TOKENS_FAILURE"),
    ("some hypotheses are not comparisons", "HYPS_NOT_COMP"),
    ("conclusion is not a comparison!", "CONCL_NOT_COMP"),
    ("EqMatchException", "EQ_MATCH_EXCEPTION"),
    ("Didn't parse everything", "DIDNT_PARSE_EVERYTHING"),
    ("is always False", "SUBGOAL_ALWAYS_FALSE"),
    ("Malformed tactic", "MALFORMED_TACTIC"),
    ('assert to_fill.keys() == res["to_fill"]', "MISMATCH_TO_FILL_VARS"),
    ("Too short tactic", "TOO_SHORT_TACTIC"),
    ("Invalid rule name/type", "INVALID_RULE_NAME_TYPE"),
    ("Too short transform tactic", "TOO_SHORT_TRANSFORM_TACTIC"),
    ("Misunderstood token position", "MISUNDERSTOOD_TOKEN_POS"),
    ("Multiple subs for", "MULTIPLE_SUBS"),
    ("Unexpected token of", "UNEXPECTED_TOKEN"),
    ("Missing token", "MISSING_TOKEN"),
    ("Missing digit", "MISSING_DIGIT"),
    ("Invalid substitution", "INVALID_SUBSTITUTION"),
    ("Invalid child", "INVALID_CHILD"),
    ("RecursionError", "RECURSION_ERROR"),
]


def get_error(error: str):
    for substr, name in eq_error_str:
        if substr in error:
            return name
    print("UNKNOWN EQUATION ERROR", error)
    return "UNK_ERROR"


def parse_subs(tokens: List[str]) -> Tuple[Dict[str, Node], int]:
    subs: Dict[str, List[str]] = {}
    i = 0
    try:
        while i < len(tokens) and tokens[i] == B_SUBST_WORD:
            old_token = tokens[i + 1]
            if tokens[i + 2] != M_SUBST_WORD:
                raise RuntimeError("Malformed tactic - m_subst")
            i += 3
            new_tokens = []
            while tokens[i] != E_SUBST_WORD:
                new_tokens.append(tokens[i])
                i += 1
            if old_token in subs:
                raise RuntimeError(f"Multiple subs for '{old_token}'")
            subs[old_token] = new_tokens
            i += 1
    except IndexError:
        raise RuntimeError(f"Malformed tactic - index")

    # build nodes
    parsed_subs: Dict[str, Node] = {}
    for x, y in subs.items():
        sub = Node.from_prefix_tokens(y)
        parsed_subs[x[7:]] = sub  # EQ_VAR_{name}
        if not sub.is_valid():
            raise RuntimeError(f"Invalid substitution: {sub}")

    return parsed_subs, i


B_EQ_NODE_WORD = "<EQNODE>"
E_EQ_NODE_WORD = "</EQNODE>"


def parse_node(command: List[str]) -> Tuple[Node, int]:
    assert len(command) > 0, f"empty command"
    tok = 0
    assert command[tok] == B_EQ_NODE_WORD, "b_eq_node_word not found"
    b_tok = tok
    while tok < len(command) and command[tok] != E_EQ_NODE_WORD:
        tok += 1
    assert tok <= len(command)
    # full node
    return Node.from_prefix_tokens(command[b_tok + 1 : tok]), tok


class EQTactic(Tactic):
    @staticmethod
    def from_tokens(tokens: List[Token]) -> "EQTactic":
        if len(tokens) == 0 or (tokens[0], tokens[-1]) != (B_CMD_WORD, E_CMD_WORD):
            return EQRuleTactic.from_tokens(tokens)
        if tokens[1] == SIMP_TAC_TOK:
            return EQSimpTactic.from_tokens(tokens[2:-1])
        if tokens[1] == NN_TAC_TOK:
            return EQNormNumTactic.from_tokens(tokens[2:-1])
        return EQRuleTactic.from_tokens(tokens)

    @classmethod
    def from_dict(cls, data):
        return {"rule": EQRuleTactic, "simp": EQSimpTactic, "nn": EQNormNumTactic,}[
            data["kind"]
        ].from_dict(data)


class EQSimpTactic(EQTactic):
    @staticmethod
    def from_tokens(tokens: List[Token]) -> "EQTactic":
        # parse list of node.
        try:
            # first is target node, others are hyps required for simp rules
            target, parsed = parse_node(tokens)
            hyps = []
            while parsed < len(tokens):
                n_node, n_parsed = parse_node(tokens[parsed:])
                parsed += n_parsed
                hyps.append(n_node)
            return EQSimpTactic(target, hyps)
        except Exception as e:
            return EQSimpTactic.from_error(error_msg=str(e), tokens=tokens)

    def to_dict(self):
        assert self.target is not None
        return {
            "kind": "simp",
            "tgt": self.target.prefix(),
            "hyps": [h.prefix() for h in self.hyps],
        }

    @classmethod
    def from_dict(cls, data):
        return EQSimpTactic(
            target=Node.from_prefix_tokens(data["tgt"].split()),
            hyps=[Node.from_prefix_tokens(h.split()) for h in data["hyps"]],
        )

    def tokenize(self) -> List[Token]:
        assert self.target is not None
        nodes: List[Token] = sum(
            [
                [B_EQ_NODE_WORD, *n.prefix_tokens(), E_EQ_NODE_WORD]
                for n in [self.target, *self.hyps]
            ],
            [],
        )
        return [SIMP_TAC_TOK, *nodes]

    @staticmethod
    def from_error(
        error_msg: str, tokens: Optional[List[str]] = None
    ) -> "EQSimpTactic":
        return EQSimpTactic(
            target=None,
            hyps=[],
            valid=False,
            malformed=True,
            error_msg=error_msg,
            tokens=tokens,
        )

    def __init__(
        self,
        target: Optional[Node],
        hyps: List[Node],
        valid=True,
        malformed=False,
        error_msg=None,
        tokens: Optional[List[str]] = None,
    ):
        self.target = target
        self.hyps = hyps
        self.is_valid = valid
        if self.is_valid:
            unique_str = " ".join(self.tokenize())
        else:
            assert tokens is not None
            unique_str = " ".join(tokens)
        Tactic.__init__(
            self, valid, unique_str=unique_str, error_msg=error_msg, malformed=malformed
        )


class EQNormNumTactic(EQTactic):
    @staticmethod
    def from_tokens(tokens: List[Token]) -> "EQTactic":
        # parse list of node.
        try:
            # first is target node, others are hyps required for simp rules
            target, parsed = parse_node(tokens)
            return EQNormNumTactic(target)
        except Exception as e:
            return EQNormNumTactic.from_error(error_msg=str(e), tokens=tokens)

    def to_dict(self):
        assert self.target is not None
        return {
            "kind": "nn",
            "tgt": self.target.prefix(),
        }

    @classmethod
    def from_dict(cls, data):
        return EQNormNumTactic(target=Node.from_prefix_tokens(data["tgt"].split()))

    def tokenize(self) -> List[Token]:
        assert self.target is not None
        return [
            NN_TAC_TOK,
            B_EQ_NODE_WORD,
            *self.target.prefix_tokens(),
            E_EQ_NODE_WORD,
        ]

    @staticmethod
    def from_error(
        error_msg: str, tokens: Optional[List[str]] = None
    ) -> "EQNormNumTactic":
        return EQNormNumTactic(
            target=None,
            valid=False,
            malformed=True,
            error_msg=error_msg,
            tokens=tokens,
        )

    def __init__(
        self,
        target: Optional[Node],
        valid=True,
        malformed=False,
        error_msg=None,
        tokens: Optional[List[str]] = None,
    ):
        self.target = target
        self.is_valid = valid
        if self.is_valid:
            unique_str = " ".join(self.tokenize())
        else:
            assert tokens is not None
            unique_str = " ".join(tokens)
        Tactic.__init__(
            self, valid, unique_str=unique_str, error_msg=error_msg, malformed=malformed
        )


class EQRuleTactic(EQTactic):

    LABEL_TO_ID = {label: i for i, label in enumerate(sorted(NAME_TO_RULE.keys()))}

    @staticmethod
    def from_error(
        error_msg: str, tokens: Optional[List[str]] = None
    ) -> "EQRuleTactic":
        return EQRuleTactic(
            label="",
            fwd=None,
            prefix_pos=None,
            to_fill={},
            valid=False,
            malformed=True,
            error_msg=error_msg,
            tokens=tokens,
        )

    def get_error_code(self) -> str:
        assert self.error_msg is not None
        return get_error(self.error_msg)

    def tokenize(self) -> List[Token]:
        """
        Transformation rule:
            [B_CMD_WORD, RULE_TYPE_T, LABEL, FWD, POS, *SUBSTS, E_CMD_WORD]
        Assertion rule:
            [B_CMD_WORD, RULE_TYPE_A, LABEL, *SUBSTS, E_CMD_WORD]
        NOTE: the B_CMD_WORD / E_CMD_WORD tokens are actually added by
              the beam search decoder
        """
        assert self.is_valid

        # substitutions
        substs = []
        for name, prefix in sorted(self.to_fill_prefix.items()):
            substs.extend(
                [
                    B_SUBST_WORD,
                    prefix_var_name(name),
                    M_SUBST_WORD,
                    *prefix.split(),
                    E_SUBST_WORD,
                ]
            )

        # transform / assert rule
        if self.rule_type == "t":
            assert self.prefix_pos is not None
            tokens = [
                T_RULE_TOK,
                self.label,
                FWD_TOK if self.fwd else BWD_TOK,
                prefix_pos_tok(self.prefix_pos),
                *substs,
            ]
        else:
            assert self.rule_type == "a"
            tokens = [A_RULE_TOK, self.label, *substs]

        return tokens

    @staticmethod
    def from_tokens(tokens: List[Token]) -> "EQTactic":
        def exception(s: str):
            return MalformedTactic(f"{s} {' '.join(tokens)}")

        assert type(tokens) is list
        try:
            if len(tokens) == 0:
                raise exception("Empty tactic")
            if (tokens[0], tokens[-1]) != (B_CMD_WORD, E_CMD_WORD):
                raise exception("Malformed tactic")
            if len(tokens) < 4:
                raise exception("Too short tactic")
            if tokens[1] not in [T_RULE_TOK, A_RULE_TOK]:
                raise exception("Malformed tactic")

            rule_type = "t" if tokens[1] == T_RULE_TOK else "a"
            label = tokens[2]
            if NAME_TO_TYPE.get(label, None) != rule_type:
                raise exception("Invalid rule name/type")

            prefix_pos: Optional[int]
            if rule_type == "t":
                if len(tokens) < 6:
                    raise exception("Too short transform tactic")
                fwd: Optional[bool] = tokens[3] == FWD_TOK
                try:
                    prefix_pos = int(tokens[4][8:-1])  # <PREFIX_{i}>
                except ValueError:
                    raise exception(
                        f"Misunderstood token position {tokens[4]} in tactic"
                    )
                i = 5
            else:
                fwd = None
                prefix_pos = None
                i = 3

            # parse substitutions
            try:
                parsed_subs, j = parse_subs(tokens[i:])
            except Exception as e:
                raise exception(str(e))
            if i + j != len(tokens) - 1:
                raise exception("Malformed tactic")

            return EQRuleTactic(
                label=label,
                fwd=fwd,
                prefix_pos=prefix_pos,
                to_fill=parsed_subs,
                valid=True,
            )

        except MalformedTactic as e:
            return EQRuleTactic.from_error(error_msg=str(e), tokens=tokens)

    def to_dict(self, light=False) -> Dict:
        if self.is_valid:
            res: Dict[str, Any] = {
                "kind": "rule",
                "rule_type": self.rule_type,
                "label": self.label,
                "to_fill": {k: v.prefix_tokens() for k, v in self.to_fill.items()},
            }
            if not light:
                if self.rule_type == "t":
                    assert isinstance(self.rule, TRule)
                    res["rule_type"] = "transformation"
                    res["fwd"] = self.fwd
                    res["prefix_pos"] = self.prefix_pos
                    res["left"] = self.rule.left.infix()
                    res["right"] = self.rule.right.infix()
                else:
                    assert self.rule_type == "a"
                    assert isinstance(self.rule, ARule)
                    res["node"] = self.rule.node.infix()
                    res["rule_type"] = "assertion"
                res["hyps"] = [hyp.infix() for hyp in self.rule.hyps]
                res["to_fill_infix"] = {k: v.infix() for k, v in self.to_fill.items()}
            return res
        return {"error_msg": self.error_msg, "str": self.label}

    @classmethod
    def from_dict(cls, data):
        return cls(
            label=data["label"],
            fwd=data["fwd"],
            prefix_pos=data["prefix_pos"],
            to_fill={k: Node.from_prefix_tokens(v) for k, v in data["to_fill"].items()},
        )

    @property
    def rule(self) -> Union[ARule, TRule]:
        return NAME_TO_RULE[self.label]

    @property
    def fwd(self) -> bool:
        assert self.rule_type == "t"
        assert self._fwd is not None
        return self._fwd

    @property
    def prefix_pos(self) -> int:
        assert self.rule_type == "t"
        assert self._prefix_pos is not None
        return self._prefix_pos

    def __init__(
        self,
        label: str,
        fwd: Optional[bool],
        prefix_pos: Optional[int],
        to_fill: Dict[str, Node],
        valid=True,
        malformed=False,
        error_msg=None,
        tokens: Optional[List[str]] = None,
    ):
        assert type(label) is str and type(to_fill) is dict
        assert all(type(k) is str and isinstance(v, Node) for k, v in to_fill.items())
        assert valid == (error_msg is None)
        assert valid == (tokens is None)

        self.rule_type = None if label == "" else NAME_TO_TYPE[label]
        self.label = label
        self._fwd = fwd
        self._prefix_pos = prefix_pos
        self._to_fill = to_fill
        self.to_fill_prefix: Dict[str, str] = {
            k: v.prefix() for k, v in to_fill.items()
        }
        self.is_valid = valid

        # sanity check
        if valid:
            assert self.rule_type in ["t", "a"]
            if self.rule_type == "t":
                assert type(fwd) is bool and type(prefix_pos) is int
            else:
                assert fwd is None and prefix_pos is None
        else:
            assert self.rule_type is None
        if self.is_valid:
            unique_str = " ".join(self.tokenize())
        else:
            assert tokens is not None
            unique_str = " ".join(tokens)
        Tactic.__init__(
            self, valid, unique_str=unique_str, error_msg=error_msg, malformed=malformed
        )

    @property
    def to_fill(self) -> Dict[str, Node]:
        if self._to_fill is None:
            self._to_fill = {
                k: Node.from_prefix_tokens(v.split())
                for k, v in self.to_fill_prefix.items()
            }
        return self._to_fill

    def __getstate__(self):
        if SERIALIZE_NODES:
            return self.__dict__
        else:
            state = dict(self.__dict__)
            state["_to_fill"] = None
            return state

    def __repr__(self):
        name = self.__class__.__name__
        if not self.is_valid:
            attrs = [("is_valid", False), ("error_msg", self.error_msg)]
        else:
            attrs = [("rule_type", self.rule_type), ("label", self.label)]
            if self.rule_type == "t":
                attrs.extend([("fwd", self.fwd), ("prefix pos", self.prefix_pos)])
            if len(self.to_fill_prefix) > 0:
                attrs.append(("to_fill_prefix", self.to_fill_prefix))

        attr = ", ".join(f"{k}={v!r}" for k, v in attrs)
        return f"{name}({attr})"


class EQTheorem(Theorem):
    def to_dict(self, light=False):
        return {
            "node": self.conclusion.split(),
            "hyps": [hyp.split() for _, hyp in self.hyps],
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            node=Node.from_prefix_tokens(data["node"]),
            hyps=[Node.from_prefix_tokens(hyp) for hyp in data["hyps"]],
        )

    def conc_in_hyp(self) -> bool:
        return any(self.conclusion == hyp for _, hyp in self.hyps)

    def tokenize(self) -> List[Token]:
        """
        B_GOAL_WORD
        [conclusion]
        B_HYP_WORD [hyp_0] E_HYP_WORD
        ...
        B_HYP_WORD [hyp_n] E_HYP_WORD
        E_GOAL_WORD
        """
        tokens = [B_GOAL_WORD]
        tokens.extend(self.conclusion.split())
        for _, hyp in self.hyps:
            tokens.append(B_HYP_WORD)
            tokens.extend(hyp.split())
            tokens.append(E_HYP_WORD)
        tokens.append(E_GOAL_WORD)
        return tokens

    @classmethod
    def from_tokens(cls, tokens: List[Token]) -> "EQTheorem":
        assert type(tokens) is list

        def exception(s: str):
            return MalformedTheorem(f"{s}: {' '.join(tokens)}")

        if len(tokens) == 0 or (tokens[0], tokens[-1]) != (B_GOAL_WORD, E_GOAL_WORD):
            raise exception("invalid theorem delimitors")

        if tokens.count(B_HYP_WORD) != tokens.count(E_HYP_WORD):
            raise exception("invalid hypotheses delimitors")

        # remove goal token delimitors
        tokens = tokens[1:-1]

        # end of goal conclusion
        concl_idx = tokens.index(B_HYP_WORD) if B_HYP_WORD in tokens else len(tokens)
        conclusion = tokens[:concl_idx]

        hyps: List[List[str]] = []
        i = concl_idx
        try:
            while i < len(tokens):
                if tokens[i] != B_HYP_WORD:
                    raise exception("was expecting beginning of hypothesis")
                hyp_content = []
                if tokens[i + 1] == E_HYP_WORD:
                    raise exception("empty hypothesis")
                i += 1
                try:
                    while tokens[i] != E_HYP_WORD:
                        hyp_content.append(tokens[i])
                        i += 1
                except IndexError:
                    raise exception("unfinished hypothesis")
                hyps.append(hyp_content)
                i += 1
            if i != len(tokens):
                raise exception("unfinished hypothesis")
        except IndexError:
            raise exception("unfinished hypothesis")

        tok_hyps: List[str] = sum(hyps, [])
        all_toks = set(tok_hyps + conclusion)
        special_words = _SPECIAL_WORDS.intersection(all_toks)
        if len(special_words) > 0:
            raise exception(f"unexpected special words ({special_words})")

        try:
            conclusion_node = Node.from_prefix_tokens(conclusion)
            node_hyps = [Node.from_prefix_tokens(hyp) for hyp in hyps]
        except NodeParseError as e:
            raise exception(f"from_prefix_tokens failure ({str(e)}")

        # conclusion and hypotheses must be comparisons
        # (i.e. expressions of the form A == B, A < B, etc.)
        if not conclusion_node.is_comp():
            raise exception(f"conclusion is not a comparison! {conclusion_node}")
        if not all(hyp.is_comp() for hyp in node_hyps):
            raise exception(f"some hypotheses are not comparisons! {node_hyps}")

        return cls(node=conclusion_node, hyps=node_hyps)

    def __init__(self, node: Node, hyps: List[Node]):
        """
        Store the flattened tokens of the conclusion / hypotheses.
        """
        assert isinstance(node, Node)
        assert node.is_comp()
        assert type(hyps) is list
        assert all(isinstance(hyp, Node) and hyp.is_comp() for hyp in hyps)
        self._eq_node = node
        self._eq_hyps = hyps
        # self.unpickle_node = 0
        # self.unpickle_hyps = 0
        super().__init__(node.prefix(), [(None, hyp.prefix()) for hyp in hyps])

    @property
    def eq_node(self) -> Node:
        if self._eq_node is None:
            assert not SERIALIZE_NODES
            # assert self.unpickle_node <= 1  # should happen at most twice
            # self.unpickle_node += 1
            self._eq_node = Node.from_prefix_tokens(self.conclusion.split())
        return self._eq_node

    @property
    def eq_hyps(self) -> List[Node]:
        if self._eq_hyps is None:
            assert not SERIALIZE_NODES
            # assert self.unpickle_hyps <= 1  # should happen at most twice
            # self.unpickle_hyps += 1
            self._eq_hyps = [Node.from_prefix_tokens(h.split()) for _, h in self.hyps]
        return self._eq_hyps

    def __getstate__(self):
        if SERIALIZE_NODES:
            return self.__dict__
        else:
            state = dict(self.__dict__)
            state["_eq_node"] = None
            state["_eq_hyps"] = None
            return state

    # def __setstate__(self, state: Dict[str, Any]):
    #     state["_eq_node"] = Node.from_prefix_tokens(state["conclusion"].split())
    #     state["_eq_hyps"] = [Node.from_prefix_tokens(h.split()) for _, h in state["hyps"]]
    #     self.__dict__.update(state)

    def __repr__(self):
        name = self.__class__.__name__
        attr = ", ".join(
            f"{k}={v!r}"
            for k, v in [
                ("concl", self.eq_node.infix()),
                ("hyps", [hyp.infix() for hyp in self.eq_hyps]),
            ]
        )
        return f"{name}({attr})"

    def old_repr(self):
        s = [self.eq_node.infix()]
        if len(self.eq_hyps) > 0:
            s.append("  Hypotheses:")
            for hyp in self.eq_hyps:
                s.append(f"    {hyp.infix()}")
        return "\n".join(s)
