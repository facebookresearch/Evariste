# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from evariste.backward.env.equations import EQTheorem
from evariste.backward.graph import BackwardGoal
from evariste.envs.eq.graph import Node, NodeParseError
from evariste.envs.eq.utils import infix_to_node


def parse_eq_input_goal(statement_: str, hyps_: str) -> BackwardGoal:
    # input can be given in the prefix form
    try:
        goal = Node.from_prefix_tokens(statement_.split())
        hyps = [Node.from_prefix_tokens(h.split()) for h in hyps_.split("\n") if h]
    except NodeParseError:
        # or in the infix form
        goal = infix_to_node(statement_)
        hyps = [infix_to_node(h) for h in hyps_.split("\n") if h]
    return BackwardGoal(theorem=EQTheorem(goal, hyps), label="custom")


def parse_input_goal(lang: str, statement: str, hyps: str) -> BackwardGoal:
    if lang == "eq":
        return parse_eq_input_goal(statement, hyps)
    else:
        raise NotImplementedError
