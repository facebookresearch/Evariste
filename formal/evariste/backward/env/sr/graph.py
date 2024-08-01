# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from params import ConfStore

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
from evariste.envs.eq.rules import TRule
from evariste.envs.sr.rules import NAME_TO_RULE
from evariste.envs.sr.env import SREnv, Transition, XYValues
from evariste.envs.sr.tokenizer import FloatTokenizer
from evariste.envs.sr.env import SREnv, SREnvArgs, Transition, XYValues

SERIALIZE_NODES = False

_SPECIAL_WORDS = set(SPECIAL_WORDS)

N_PREFIX_TOKS = 4096


def prefix_pos_tok(i: int):
    return f"<PREFIX_{i}>"


def prefix_var_name(name: str):
    assert type(name) is str
    return f"EQ_VAR_{name}"


sr_error_str = [
    ("Empty tactic", "EMPTY_TACTIC"),
    ("unexpected special words", "UNEXPECTED_SPECIAL_WORDS"),
    ("from_prefix_tokens failure", "FROM_PREFIX_TOKENS_FAILURE"),
    ("conclusion is a comparison!", "CONCL_IS_COMP"),
    ("EqMatchException", "EQ_MATCH_EXCEPTION"),
    ("Didn't parse everything", "DIDNT_PARSE_EVERYTHING"),
    ("is always False", "SUBGOAL_ALWAYS_FALSE"),
    ("Malformed tactic", "MALFORMED_TACTIC"),
    ('assert to_fill.keys() == res["to_fill"]', "MISMATCH_TO_FILL_VARS"),
    ("Too short tactic", "TOO_SHORT_TACTIC"),
    ("Invalid rule name", "INVALID_RULE_NAME_TYPE"),
    ("Too short transform tactic", "TOO_SHORT_TRANSFORM_TACTIC"),
    ("Misunderstood token position", "MISUNDERSTOOD_TOKEN_POS"),
    ("Multiple subs for", "MULTIPLE_SUBS"),
    ("Unexpected token of", "UNEXPECTED_TOKEN"),
    ("Missing token", "MISSING_TOKEN"),
    ("Missing digit", "MISSING_DIGIT"),
    ("Invalid substitution", "INVALID_SUBSTITUTION"),
    ("Invalid child", "INVALID_CHILD"),
    ("RecursionError", "RECURSION_ERROR"),
    ("invalid forward token", "INVALID_FWD_TOKEN"),
    ("unexpected hypothesis", "UNEXPECTED_HYPOTHESIS"),
    ("end of conclusion not found", "END_OF_CONCL_NOT_FOUND"),
    ("unexpected to_fill variables", "UNEXPECTED_TO_FILL_VARIABLES"),
]


def get_error(error: str):
    for substr, name in sr_error_str:
        if substr in error:
            return name
    print("UNKNOWN EQUATION ERROR", error)
    return "UNK_ERROR"


class SRTactic(Tactic):

    LABEL_TO_ID = {label: i for i, label in enumerate(sorted(NAME_TO_RULE.keys()))}

    @staticmethod
    def from_error(error_msg: str, tokens: Optional[List[str]] = None) -> "SRTactic":
        return SRTactic(
            label="",
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
            [B_CMD_WORD, LABEL, FWD, POS, *SUBSTS, E_CMD_WORD]
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

        tokens = [
            # B_CMD_WORD,  # NOTE: no need because added in beam_search.py
            self.label,
            prefix_pos_tok(self.prefix_pos),
            *substs,
            # E_CMD_WORD,
        ]

        return tokens

    @staticmethod
    def from_tokens(tokens: List[Token]) -> "SRTactic":
        def exception(s: str):
            return MalformedTactic(f"{s} {' '.join(tokens)}")

        assert type(tokens) is list
        try:
            if len(tokens) == 0:
                raise exception("Empty tactic")
            if (tokens[0], tokens[-1]) != (B_CMD_WORD, E_CMD_WORD):
                raise exception("Malformed tactic")
            if len(tokens) < 3:
                raise exception("Too short tactic")

            label = tokens[1]
            if label not in NAME_TO_RULE:
                raise exception("Invalid rule name")

            if len(tokens) < 4:
                raise exception("Too short transform tactic")

            try:
                prefix_pos = int(tokens[2][8:-1])  # <PREFIX_{i}>
            except ValueError:
                raise exception(f"Misunderstood token position {tokens[2]} in tactic")
            i = 3

            # parse substitutions
            subs: Dict[str, List[str]] = {}
            try:
                while tokens[i] == B_SUBST_WORD:
                    old_token = tokens[i + 1]
                    if tokens[i + 2] != M_SUBST_WORD:
                        raise exception("Malformed tactic")
                    i += 3
                    new_tokens = []
                    while tokens[i] != E_SUBST_WORD:
                        new_tokens.append(tokens[i])
                        i += 1
                    if old_token in subs:
                        raise exception(f"Multiple subs for '{old_token}'")
                    subs[old_token] = new_tokens
                    i += 1
            except IndexError:
                raise exception("Malformed tactic")
            if i != len(tokens) - 1:
                raise exception("Malformed tactic")

            # build nodes
            parsed_subs: Dict[str, Node] = {}
            for x, y in subs.items():
                try:
                    sub = Node.from_prefix_tokens(y)
                    parsed_subs[x[7:]] = sub  # EQ_VAR_{name}
                    if not sub.is_valid():
                        raise exception(f"Invalid substitution: {sub}")
                except NodeParseError as e:
                    raise exception(f"Malformed substitution {str(e)}")

            # some variables that only appear in the target (e.g. rhs) of the rule are not provided
            rule = NAME_TO_RULE[label]
            if parsed_subs.keys() != rule.r_vars - rule.l_vars:
                raise exception(f"unexpected to_fill variables")

            return SRTactic(
                label=label, prefix_pos=prefix_pos, to_fill=parsed_subs, valid=True
            )

        except MalformedTactic as e:
            return SRTactic.from_error(error_msg=str(e), tokens=tokens)

    def to_dict(self, light=False) -> Dict:
        if self.is_valid:
            res: Dict[str, Any] = {
                "label": self.label,
                "to_fill": {k: v.prefix_tokens() for k, v in self.to_fill.items()},
            }
            if not light:
                assert isinstance(self.rule, TRule)
                res["prefix_pos"] = self.prefix_pos
                res["left"] = self.rule.left.infix()
                res["right"] = self.rule.right.infix()
                res["to_fill_infix"] = {k: v.infix() for k, v in self.to_fill.items()}
            return res
        return {"error_msg": self.error_msg, "str": self.label}

    @classmethod
    def from_dict(cls, data):
        return cls(
            label=data["label"],
            prefix_pos=data["prefix_pos"],
            to_fill={k: Node.from_prefix_tokens(v) for k, v in data["to_fill"].items()},
        )

    @property
    def rule(self) -> TRule:
        return NAME_TO_RULE[self.label]

    @property
    def prefix_pos(self) -> int:
        assert self._prefix_pos is not None
        return self._prefix_pos

    # def __setstate__(self, state):
    #     """
    #     Used to reload nodes dumped before `fwd` and `prefix_pos` were properties.
    #     """
    #     if "fwd" in state:
    #         state["_fwd"] = state.pop("fwd")
    #     if "prefix_pos" in state:
    #         state["_prefix_pos"] = state.pop("prefix_pos")
    #     self.__dict__.update(state)

    def __init__(
        self,
        label: str,
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

        self.label = label
        self._prefix_pos = prefix_pos
        self._to_fill = to_fill
        self.to_fill_prefix: Dict[str, str] = {
            k: v.prefix() for k, v in to_fill.items()
        }
        self.is_valid = valid

        # sanity check
        if valid:
            assert type(prefix_pos) is int
            assert (
                self.to_fill.keys() == self.rule.r_vars - self.rule.l_vars
            ), "to_fill: {}\nrule: {}\nr_vars:{}, diff_Vars:{}".format(
                to_fill,
                self.rule,
                self.rule.r_vars,
                self.rule.r_vars - self.rule.l_vars,
            )

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
        if not self.is_valid:
            return f"<INVALID_TACTIC> {self.error_msg}"
        s = (
            f"Tranformation tactic ({self.label})\n"
            f"  prefix pos: {self.prefix_pos}\n"
        )
        if len(self.to_fill_prefix) > 0:
            s += f"  substitutions:\n"
            s += "".join(f"    {k} -> {v}\n" for k, v in self.to_fill_prefix.items())
        return s


class SRTheorem(Theorem):
    def to_dict(self, light=False):
        return {"node": self.conclusion.split()}

    @staticmethod
    def from_dict(data):
        ##TODO: complete
        raise NotImplementedError
        return SRTheorem(node=Node.from_prefix_tokens(data["node"]))

    def tokenize(self) -> List[Token]:
        """
        B_GOAL_WORD
        *current_function.prefix_tokens()
        B_SUBST_WORD
        *tokenize_float(X_0)
        *tokenize_float(Y_0)
        *tokenize_float(f(X_0))
        *tokenize_float(f(X_0) - Y_0)
        M_SUBST_WORD
        ...
        M_SUBST_WORD
        *tokenize_float(X_n)
        *tokenize_float(Y_n)
        *tokenize_float(f(X_n))
        *tokenize_float(f(X_n) - Y_n)
        E_SUBST_WORD
        *tokenize_float(|f(X) - Y|)
        E_GOAL_WORD
        """
        tokens = [B_GOAL_WORD]
        tokens.extend(self.conclusion.split())

        x = self.true_xy.x
        y = self.true_xy.y
        y_tilde = self.curr_xy.y
        for (x_i, y_i, ytilde_i) in zip(x, y, y_tilde):
            tokens.append(B_SUBST_WORD)
            tokens.extend(self.tokenize_float(x_i))
            tokens.extend(self.tokenize_float(y_i))
            tokens.extend(self.tokenize_float(ytilde_i))
            # TODO: add error when NaN, Nones are handled correctly
            # error = y_i - ytilde_i
            # tokens.extend(self.tokenize_float(error))
        # TODO: global error
        tokens.append(E_GOAL_WORD)

        return tokens

    @classmethod
    def from_tokens(cls, tokens: List[Token]) -> "SRTheorem":
        assert type(tokens) is list

        def exception(s: str):
            return MalformedTheorem(f"{s}: {' '.join(tokens)}")

        if len(tokens) == 0 or (tokens[0], tokens[-1]) != (B_GOAL_WORD, E_GOAL_WORD):
            raise exception("invalid theorem delimitors")

        if tokens.count(B_HYP_WORD) + tokens.count(E_HYP_WORD) > 0:
            raise exception("unexpected hypothesis")

        # remove goal token delimitors
        tokens = tokens[1:-1]

        # end of goal conclusion
        if B_SUBST_WORD not in tokens:
            raise exception("end of conclusion not found")
        concl_idx = tokens.index(B_SUBST_WORD)

        conclusion = tokens[:concl_idx]

        special_words = _SPECIAL_WORDS.intersection(set(conclusion))
        if len(special_words) > 0:
            raise exception(f"unexpected special words ({special_words})")

        try:
            conclusion_node = Node.from_prefix_tokens(conclusion)
        except NodeParseError as e:
            raise exception(f"from_prefix_tokens failure ({str(e)}")

        data = tokens[concl_idx + 1 :]
        encoder = FloatTokenizer()
        decodeds = []
        while len(data) > 0:
            if B_SUBST_WORD not in data:
                end = -1
                values = data
                data = []
            else:
                end = data.index(B_SUBST_WORD)
                values = data[:end]
                data = data[end + 1 :]
            decoded = []
            while len(values) > 0:
                if values[0] in ["+", "-"]:
                    curr_float_vals, values = values[:3], values[3:]
                else:
                    curr_float_vals, values = values[:1], values[1:]
                curr_float_str = " ".join(curr_float_vals)
                decoded_float = encoder.detokenize(curr_float_str)
                decoded.append(decoded_float)
            decodeds.append(decoded)

        x = [decoded[0] for decoded in decodeds]
        y = [decoded[1] for decoded in decodeds]
        y_tilde = [decoded[2] for decoded in decodeds]
        true_xy = XYValues(x, y)
        curr_xy = XYValues(x, y_tilde)

        # conclusion should not be a comparaison
        if conclusion_node.is_comp():
            raise exception(f"conclusion is a comparison! {conclusion_node}")

        return cls(node=conclusion_node, true_xy=true_xy, curr_xy=curr_xy,)

    def __init__(
        self,
        node: Node,
        true_xy: XYValues,
        curr_xy: XYValues,
        target_node: Optional[Node] = None,
    ):
        """
        Store the flattened tokens of the conclusion.  # TODO: finish
        """
        assert isinstance(node, Node)
        assert isinstance(target_node, Node) or target_node is None

        assert not node.is_comp()
        assert ((np.array(true_xy.x) - np.array(curr_xy.x)) ** 2).mean() == 0.0

        self._curr_node = node
        self._target_node = target_node
        super().__init__(node.prefix(), hyps=[])
        self.true_xy = true_xy
        self.curr_xy = curr_xy
        self.tokenizer = FloatTokenizer()

    @property
    def curr_node(self) -> Node:
        if self._curr_node is None:
            assert not SERIALIZE_NODES
            self._curr_node = Node.from_prefix_tokens(self.conclusion.split())
        return self._curr_node

    @property
    def target_node(self) -> Optional[Node]:
        return self._target_node

    def tokenize_float(self, v: float) -> List[str]:
        return self.tokenizer.tokenize(v).split()

    def __getstate__(self):
        if SERIALIZE_NODES:
            return self.__dict__
        else:
            state = dict(self.__dict__)
            state["_curr_node"] = None
            return state

    def __repr__(self):
        return self.curr_node.infix()


if __name__ == "__main__":

    # python -m evariste.backward.env.sr.graph

    import evariste.datasets

    sr_args = SREnvArgs(
        eq_env=ConfStore["sr_eq_env_default"], max_backward_steps=100, max_n_points=100
    )
    env = SREnv.build(sr_args)

    def test_large_ints():
        return

    def test_theorem_tokenization():
        expr = env.eq_env.generate_expr(n_ops=10)
        true_xy: XYValues = env.sample_dataset(expr)
        step: Transition = env.sample_transition(expr)
        print("Eq: {}\nx: {}\ny: {}".format(step.eq, true_xy.x, true_xy.y))

        curr_y = env.evaluate_at(step.eq, true_xy.x)
        curr_xy = XYValues(x=true_xy.x, y=curr_y)
        theorem = SRTheorem(node=step.eq, true_xy=true_xy, curr_xy=curr_xy)
        tokenized_theorem = theorem.tokenize()
        theorem_from_tokens = SRTheorem.from_tokens(tokenized_theorem)
        decoded_true_xy = theorem_from_tokens.true_xy
        decoded_curr_xy = theorem_from_tokens.curr_xy

        assert np.isclose(
            true_xy.y, decoded_true_xy.y, atol=1e-3, rtol=1e-3, equal_nan=True
        ).all(), (true_xy.y, decoded_true_xy.y)
        assert np.isclose(
            curr_xy.y, decoded_curr_xy.y, atol=1e-3, rtol=1e-3, equal_nan=True
        ).all(), (curr_xy.y, decoded_curr_xy.y)

    def test_tactic_tokenization():
        expr = env.eq_env.generate_expr(n_ops=10)
        step: Transition = env.sample_transition(expr)
        print(step)
        print("Eq: {}".format(step.eq))
        rule: TRule = step.rule
        tactic = SRTactic(rule.name, prefix_pos=step.prefix_pos, to_fill=step.tgt_vars,)
        tokenized_tactic = tactic.tokenize()
        tactic_from_tokens = SRTactic.from_tokens(
            [B_CMD_WORD] + tokenized_tactic + [E_CMD_WORD]
        )
        assert tactic_from_tokens.rule == rule, "Excepted {} but got {} rule".format(
            rule, tactic_from_tokens.rule
        )
        assert (
            tactic_from_tokens.prefix_pos == step.prefix_pos
        ), "Excepted {} but got {} prefix_pos".format(
            step.prefix_pos, tactic_from_tokens.prefix_pos
        )
        assert (
            tactic_from_tokens.to_fill.keys() == step.tgt_vars.keys()
        ), "Wrong tgt vars"
        for k, v in step.tgt_vars.items():
            assert tactic_from_tokens.to_fill[k].eq(
                v
            ), "Excepted {} but got {} to_fill".format(v, tactic_from_tokens.to_fill[k])
        print("\n")

    for _ in range(10):
        test_tactic_tokenization()
        test_theorem_tokenization()
        print("test {} passed".format(_))
