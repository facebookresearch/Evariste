# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Set


from evariste.backward.env.metamath import MMTheorem
from evariste.envs.mm.utils import node_tok, Node
from evariste.forward.common import ForwardTactic
from evariste.forward.env_specifics.prover_env_specifics import FwdTokenizer
from evariste.forward.fwd_mm.mm_fwd_tasks import (
    LEGACY_FWD_INP_FMT,
    FWD_INP_FMT,
    GEN_INP_FMT,
    MMFwdFormat,
    LEGACY_CMD_FMT,
    LABEL_SUBST_FMT,
    DEPRECATED_PROVED_LABEL_SUBST_FMT,
)
from evariste.model.data.dictionary import (
    B_STACK_WORD,
    B_GOAL_WORD,
    E_GOAL_WORD,
    E_STACK_WORD,
    E_NODE_WORD,
    B_HYP_WORD,
    E_HYP_WORD,
    B_NODE_WORD,
    B_HEAD_WORD,
    E_HEAD_WORD,
    EMPTY_GOAL_WORD,
    EOS_WORD,
    UNPROVED_WORD,
    B_THEOREM_WORD,
    E_THEOREM_WORD,
    M_SUBST_WORD,
    E_SUBST_WORD,
    B_SUBST_WORD,
    B_CMD_WORD,
    E_CMD_WORD,
    PROVED_WORD,
)

from typing import Optional, Dict, List, Tuple, Any

from evariste.envs.mm.env import MetamathEnv
from evariste.forward.common import ForwardGraph
from evariste.forward.core.generation_errors import (
    NonExistentTheorem,
    MalformedCommand,
    MalformedChildren,
)
from evariste.forward.fwd_mm.mm_helpers import MMForwardTactic


class MMFwdTokenizer(FwdTokenizer):
    def __init__(self, fmt: MMFwdFormat):
        self.inp_fmt = fmt.inp_fmt
        if self.inp_fmt not in [LEGACY_FWD_INP_FMT, FWD_INP_FMT, GEN_INP_FMT]:
            raise NotImplementedError(self.inp_fmt)
        self.fmt = fmt

    def tokenize_graph(self, graph: ForwardGraph) -> List[str]:

        legacy_fmt = self.inp_fmt == LEGACY_FWD_INP_FMT
        is_generation = self.inp_fmt == GEN_INP_FMT
        assert is_generation == self.fmt.is_generation, f"Wrong {self.fmt}"
        if is_generation:
            assert graph.goal_statement == "", graph.goal_statement

        conditioning = []
        if self.fmt.label_conditioning:
            assert self.fmt.is_generation, f"Wrong {self.fmt}"
            # the label to condition with is supposed to be in the generation goal
            assert graph.fwd_goal.label_conditioning is not None
            conditioning.append(graph.fwd_goal.label_conditioning)
        if self.fmt.proved_conditioning:
            assert self.fmt.is_generation, f"Wrong {self.fmt}"
            # the token to condition with is supposed to be in the generation goal
            assert graph.fwd_goal.proved_conditioning is not None
            assert graph.fwd_goal.proved_conditioning in {PROVED_WORD, UNPROVED_WORD}
            conditioning.append(graph.fwd_goal.proved_conditioning)
        if self.fmt.reward_quantile_conditioning:
            assert self.fmt.is_generation, f"Wrong {self.fmt}"
            assert graph.fwd_goal.reward_quantile_conditioning is not None
            conditioning.append(graph.fwd_goal.reward_quantile_conditioning)

        enc_inp = tokenize_mm_graph(
            goal_statement=graph.goal_statement,
            node_statements=graph.nodes,
            is_generation=is_generation,
            legacy_fmt=legacy_fmt,
            conditioning=conditioning,
        )

        return [EOS_WORD, *enc_inp, EOS_WORD]

    def detokenize_command(
        self, command: List[str], graph: ForwardGraph
    ) -> ForwardTactic:
        if self.fmt.cmd_fmt == LEGACY_CMD_FMT:
            label, subst = parse_graph2subst_command(command)
            children_ids = None
        else:
            label, subst, children_ids = parse_fwd_command(
                command,
                cmd_fmt=self.fmt.cmd_fmt,
                last_token=self.fmt.stop_symbol,
                th_to_label={},
            )
        return MMForwardTactic(
            label=label, substitutions=subst, children_ids=children_ids
        )

    @classmethod
    def from_trainer_args(cls, params: Any) -> "MMFwdTokenizer":
        from evariste.trainer.args import TrainerArgs  # cyclic import

        assert isinstance(params, TrainerArgs)
        return cls(fmt=MMFwdFormat.from_trainer_args(params))


def tokenize_mm_graph(
    goal_statement: str,
    node_statements: List[str],
    is_generation: bool,
    legacy_fmt: bool,
    conditioning: Optional[List[str]] = None,
) -> List[str]:

    enc_inp = []
    if conditioning:
        assert is_generation, "conditioning only supported for generation"
        enc_inp.extend(conditioning)

    if is_generation:
        enc_inp.append(EMPTY_GOAL_WORD)
    else:
        enc_inp.extend([B_GOAL_WORD, *goal_statement.split(), E_GOAL_WORD])

    start_tok = B_NODE_WORD if not legacy_fmt else B_STACK_WORD
    end_tok = E_NODE_WORD if not legacy_fmt else E_STACK_WORD
    for statement in node_statements:
        enc_inp.extend([start_tok] + statement.split() + [end_tok])
    return enc_inp


def _deprecated_make_fwd_x(
    graph: List[Node], goal: Node, s_x: Tuple[str, ...], noises: Set[str],
):
    assert s_x in {
        ("goal", "stack"),
        ("goal", "igraph"),
        ("graph:v0",),
        ("graph:v1",),
        ("graph:v2",),
    }, f"{s_x} is not implemented"

    if s_x == ("goal", "stack"):
        inp = [B_GOAL_WORD] + goal.statement + [E_GOAL_WORD]
        for i, node in enumerate(graph):
            inp.extend([B_STACK_WORD] + node.statement + [E_STACK_WORD])
    elif s_x == ("goal", "graph"):
        inp = [B_GOAL_WORD] + goal.statement + [E_GOAL_WORD]
        for i, node in enumerate(graph):
            inp.extend([B_NODE_WORD] + node.statement + [E_NODE_WORD])
    elif s_x == ("goal", "igraph"):
        inp = [B_GOAL_WORD] + goal.statement + [E_GOAL_WORD]
        for i, node in enumerate(graph):
            inp.extend([node_tok(i)] + node.statement + [E_NODE_WORD])
    elif s_x == ("graph:v0",):
        inp = _featurize_v0(
            goal_statement=goal.statement_str,
            node_statements=[n.statement_str for n in graph],
            add_node_id=False,
        )
    elif s_x == ("graph:v1",):
        is_e_hyp = []
        node_statements = []
        for node in graph:
            is_noise = node.statement_str in noises
            if node.ltype == "$e" and not is_noise:
                is_e_hyp.append(True)
            else:
                is_e_hyp.append(False)
            node_statements.append(node.statement_str)

        inp = _featurize_v1(
            goal_statement=goal.statement_str,
            node_statements=node_statements,
            is_e_hyp=is_e_hyp,
        )
    elif s_x == ("graph:v2",):
        is_head = []
        is_e_hyp = []
        node2id = {}
        for node in graph:
            is_noise = node.statement_str in noises
            node2id[node.statement_str] = len(node2id)
            if node.ltype == "$e" and not is_noise:
                is_e_hyp.append(True)
                is_head.append(False)
            else:
                is_e_hyp.append(False)
                is_head.append(True)
                if is_noise:
                    continue
                for child in node.children:
                    is_head[node2id[child.statement_str]] = False

        inp = _featurize_v2(
            goal_statement=goal.statement_str,
            node_statements=[n.statement_str for n in graph],
            is_e_hyp=is_e_hyp,
            is_head=is_head,
        )
    else:
        raise NotImplementedError
    return inp


def _featurize_v0(goal_statement: str, node_statements: List[str], add_node_id: bool):
    goal_statement = goal_statement.split()
    node_statements = [n.split() for n in node_statements]
    enc_inp = [B_GOAL_WORD] + goal_statement + [E_GOAL_WORD]
    for i, statement in enumerate(node_statements):
        start_tok = B_STACK_WORD if not add_node_id else node_tok(i)
        end_tok = E_STACK_WORD if not add_node_id else E_NODE_WORD
        enc_inp.extend([start_tok] + statement + [end_tok])
    return enc_inp


def _featurize_v1(
    goal_statement: str, node_statements: List[str], is_e_hyp: List[bool]
):
    assert len(node_statements) == len(is_e_hyp)
    goal_statement = goal_statement.split()
    node_statements = [n.split() for n in node_statements]
    enc_inp = [B_GOAL_WORD] + goal_statement + [E_GOAL_WORD]
    other_nodes = []
    for i, statement in enumerate(node_statements):
        if is_e_hyp[i]:
            enc_inp.extend([B_HYP_WORD] + statement + [E_HYP_WORD])
        else:
            other_nodes.extend(reversed([B_NODE_WORD] + statement + [E_NODE_WORD]))
    enc_inp.extend(list(reversed(other_nodes)))
    return enc_inp


def _featurize_v2(
    goal_statement: str,
    node_statements: List[str],
    is_e_hyp: List[bool],
    is_head: List[bool],
):
    assert len(node_statements) == len(is_e_hyp) == len(is_head)
    goal_statement = goal_statement.split()
    node_statements = [n.split() for n in node_statements]
    enc_inp = [B_GOAL_WORD] + goal_statement + [E_GOAL_WORD]
    heads = []
    other_nodes = []
    for i, statement in enumerate(node_statements):
        if is_e_hyp[i]:
            enc_inp.extend([B_HYP_WORD] + statement + [E_HYP_WORD])
        elif is_head[i]:
            heads.extend(reversed([B_HEAD_WORD] + statement + [E_HEAD_WORD]))
        else:
            other_nodes.extend(reversed([B_NODE_WORD] + statement + [E_NODE_WORD]))
    enc_inp.extend(list(reversed(heads)))
    enc_inp.extend(list(reversed(other_nodes)))
    return enc_inp


def parse_fwd_command(
    command: List[str],
    cmd_fmt: Tuple[str, ...],
    last_token: str,
    th_to_label: Dict[MMTheorem, str],
) -> Tuple[str, Dict[str, str], Optional[List[int]]]:
    return _parse_fwd_x2y_command(command, EOS_WORD, last_token, cmd_fmt, th_to_label)


def _parse_fwd_x2y_command(
    command: List[str],
    first_token: str,
    last_token: str,
    cmd_fmt: Tuple[str, ...],
    th_to_label: Dict[MMTheorem, str],
) -> Tuple[str, Dict[str, str], Optional[List[int]]]:
    assert cmd_fmt in {
        LABEL_SUBST_FMT,
        DEPRECATED_PROVED_LABEL_SUBST_FMT,
        ("theorem", "subst"),
        ("label", "children", "subst"),
    }

    full_cmd = " ".join(command)

    def _raise():
        raise MalformedCommand(
            f"Malformed subst", command=full_cmd,
        )

    if len(command) < 3 or command[0] != first_token or command[-1] != last_token:
        _raise()
    command = command[1:-1]

    if cmd_fmt[0] == "proved":
        assert command[0] == UNPROVED_WORD, command
        command = command[1:]
        if len(command) < 1:
            _raise()
        cmd_fmt = tuple(list(cmd_fmt)[1:])

    if cmd_fmt[0] == "label":
        label = command[0]
        command = command[1:]
    elif cmd_fmt[0] == "theorem":
        label, end_thm = parse_theorem(command, full_cmd, th_to_label=th_to_label)
        command = command[end_thm + 1 :]
    else:
        raise ValueError(cmd_fmt[0])

    children = None
    if cmd_fmt[1] == "children":
        children = []
        prefix = "<NODE_"
        sufix = ">"
        while len(command) > 0:
            tok = command[0]
            if command[0] == B_SUBST_WORD:
                break
            if not (tok.startswith(prefix) and tok.endswith(sufix)):
                raise MalformedChildren(f"{tok} is invalid")
            try:
                cid = int(tok[len(prefix) : -len(sufix)])
            except ValueError:
                raise MalformedChildren(f"{tok} is invalid")
            children.append(cid)
            command = command[1:]

    subst = parse_subst(command, full_cmd)
    return label, subst, children


def parse_theorem(
    command: List[str], full_cmd: str, th_to_label: Dict[MMTheorem, str]
) -> Tuple[str, int]:
    _err = MalformedCommand(f"Malformed theorem", command=full_cmd)

    if len(command) < 1 or command[0] != B_THEOREM_WORD:
        raise _err
    if E_THEOREM_WORD not in command:
        raise _err
    end_th = command.index(E_THEOREM_WORD)
    thm_tokens = [B_GOAL_WORD] + command[1:end_th] + [E_GOAL_WORD]
    try:
        theorem = MMTheorem(thm_tokens)
    except RuntimeError:
        raise _err
    label = th_to_label.get(theorem, None)
    if label is None:
        raise NonExistentTheorem(
            f"Theorem doesn't exist: {theorem.conclusion}, hyps: {theorem.hyps}"
        )
    return label, end_th


def parse_subst(command: List[str], full_cmd: str) -> Dict[str, str]:
    def _raise():
        raise MalformedCommand(
            f"Malformed subst", command=full_cmd,
        )

    subst = {}
    while len(command) > 0:
        if E_SUBST_WORD not in command:
            _raise()
        idx = command.index(E_SUBST_WORD)
        new_subst = command[: idx + 1]
        if (
            len(new_subst) < 5
            or new_subst[0] != B_SUBST_WORD
            or new_subst[2] != M_SUBST_WORD
            or new_subst[-1] != E_SUBST_WORD
        ):
            _raise()
        subst[new_subst[1]] = " ".join(new_subst[3:-1])
        command = command[idx + 1 :]
    return subst


def make_th_to_label(mm_env: MetamathEnv) -> Dict[MMTheorem, str]:
    th_to_label = {}
    for label, (_, assertion) in mm_env.labels.items():
        th_to_label[
            MMTheorem(
                conclusion=assertion.tokens_str,
                hyps=[(None, " ".join(x)) for x in assertion.e_hyps],
            )
        ] = label
    return th_to_label


def parse_graph2subst_command(command: List[str]) -> Tuple[str, Dict[str, str]]:
    command_str = " ".join(command)
    if not command[0] == command[-1] == EOS_WORD:
        raise MalformedCommand(
            f"Wrong eos in command", command=command_str,
        )
    if len(command) < 3:
        raise MalformedCommand(
            f"Too short command", command=command_str,
        )
    command = command[1:-1]
    if len(command) < 1:
        raise MalformedCommand(
            f"Invalid command", command=command_str,
        )
    if command[0] != B_CMD_WORD or command[-1] != E_CMD_WORD:
        raise MalformedCommand(
            f"Invalid command", command=command_str,
        )
    command = command[1:-1]
    if len(command) < 1:
        raise MalformedCommand(
            f"Invalid command", command=command_str,
        )
    token = command[0]
    subst = {}
    command = command[1:]
    while len(command) > 0:
        if E_SUBST_WORD not in command:
            raise MalformedCommand(
                f"Invalid command", command=command_str,
            )
        idx = command.index(E_SUBST_WORD)
        new_subst = command[: idx + 1]
        if len(new_subst) < 4:
            raise MalformedCommand(
                f"Invalid substitution: {new_subst}", command=command_str,
            )
        if (
            new_subst[0] != B_SUBST_WORD
            or new_subst[2] != M_SUBST_WORD
            or new_subst[-1] != E_SUBST_WORD
        ):
            raise MalformedCommand(
                f"Invalid substitution: {new_subst}", command=command_str,
            )
        subst[new_subst[1]] = " ".join(new_subst[3:-1])
        command = command[idx + 1 :]
    return token, subst
