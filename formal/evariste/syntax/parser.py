# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sys
from evariste import json as json
from collections import defaultdict
import pickle
import os
import hashlib
from logging import getLogger

from evariste.clusters.utils import clusterify_path


class ParseError(Exception):
    pass


class Tree(object):
    def __init__(self, data, children, start, end, string):
        self.data = data
        self.children = children
        self.start_pos = start
        self.end_pos = end
        self.string = string

    def __str__(self):
        return f"Tree(data: {self.data}, string={self.string}, nchildren={len(self.children)})"


def adapt(node, to_parse):
    next_left = 0
    left, right = [], [0]
    for tok in to_parse:
        left.append(next_left)
        next_left += len(tok) + 1
        right.append(left[-1] + len(tok))

    def change_in_tree(cur_node):
        cur_node.start_pos = left[cur_node.start_pos]
        cur_node.end_pos = right[cur_node.end_pos]
        for c in cur_node.children:
            change_in_tree(c)

    change_in_tree(node)
    return node


Rule = Tuple[Optional[str], List[str]]


class Parser:
    def __init__(self, path: str):
        self.rules: Dict[str, List[Rule]] = defaultdict(list)
        self.vars: Dict[str, str] = {}
        self.symbols: Dict[str, int] = {}

        with open(path, "r") as f:
            self.hash = hashlib.sha256(f.read().encode()).hexdigest()

        with open(path, "r") as f:
            for line in f:
                tokens = line.split()
                if tokens[0][0] == "#":
                    for tok in tokens[1:]:
                        self.vars[tok] = tokens[0]
                elif tokens[0] == "axiom_hyp_order":
                    self.axiom_hyp_order = json.loads(" ".join(tokens[1:]))
                elif tokens[0] == "f_table":
                    self.f_table = json.loads(" ".join(tokens[1:]))
                else:
                    non_terminal = tokens[0]
                    assert non_terminal[0] == "$"
                    new_rule, rule_name = None, None
                    if tokens[-1].startswith("$*"):
                        rule_name = tokens[-1]
                        new_rule = tokens[1:-1]
                    else:
                        new_rule = tokens[1:]
                    for x in new_rule:
                        if x not in self.symbols:
                            self.symbols[x] = len(self.symbols)
                    if non_terminal not in self.symbols:
                        self.symbols[non_terminal] = len(self.symbols)
                    self.rules[non_terminal].append((rule_name, new_rule))

        self.symbols["EOF"] = len(self.symbols)
        self.rules["$root"].append(("root_rule", ["$start", "EOF"]))
        self.final = self.build_tables()

    def tokenize(self, to_parse: List[str]):
        return [self.vars.get(x, x) for x in to_parse]

    def close_state(self, state):
        total = set()
        to_add = state
        while to_add:
            cur_state = to_add.pop()
            cur_term, rule_id, position = cur_state
            # print(
            #     f"cur_state: cur_term = {cur_term} rule_id = {rule_id} position = {position}"
            # )
            if cur_state in total:
                continue
            total.add(cur_state)
            rule = self.rules[cur_term][rule_id][1]
            # If we're not at the end of this rule
            if position < len(rule):
                # If next symbol is a non-terminal
                if rule[position] in self.rules:
                    for rule_id, next_rule in enumerate(self.rules[rule[position]]):
                        to_add.append((rule[position], rule_id, 0))
        return tuple(sorted(total))

    def build_tables(self):
        all_states = set()
        transitions = defaultdict(dict)
        start = self.close_state([("$root", 0, 0)])
        to_visit = [start]
        n_transitions = 0
        while to_visit:
            cur = to_visit.pop()
            if cur in all_states:
                continue
            all_states.add(cur)

            for symbol in self.symbols.keys():
                next_state = []
                for non_terminal, rule_id, position in cur:
                    rule = self.rules[non_terminal][rule_id][1]
                    if position < len(rule) and rule[position] == symbol:
                        next_state.append((non_terminal, rule_id, position + 1))
                if len(next_state) > 0:
                    next_closed_state = self.close_state(next_state)
                    # print(f"TRANSITION : {cur} -> {symbol} -> {next_closed_state}")
                    transitions[cur][symbol] = next_closed_state
                    n_transitions += 1
                    to_visit.append(next_closed_state)
        print(f"Found {len(all_states)} states and {n_transitions} transitions.")

        final = [[[] for _ in range(len(self.symbols))] for _ in range(len(all_states))]

        state_to_id = {start: 0}
        id_to_state = [start]
        visited = set()
        to_visit = [0]

        while to_visit:
            cur = to_visit.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for symbol, state in transitions[id_to_state[cur]].items():
                id = state_to_id.get(state, None)
                if id is None:
                    id = len(state_to_id)
                    state_to_id[state] = id
                    id_to_state.append(state)

                # If symbol is non terminal
                if symbol in self.rules:
                    final[cur][self.symbols[symbol]].append(("Goto", id))
                # If symbol is terminal
                else:
                    final[cur][self.symbols[symbol]].append(("Shift", id))
                to_visit.append(id)

        for state in all_states:
            state_id = state_to_id[state]
            # print(state_id, state)
            for symbol, rule_id, position in state:
                rule_name, rule = self.rules[symbol][rule_id]
                if position == len(rule):
                    for terminal, t_id in self.symbols.items():
                        if terminal not in self.rules:
                            final[state_id][t_id].append(("Reduce", (symbol, rule_id)))
                if position == len(rule) - 1 and rule[position] == "EOF":
                    final[state_id][self.symbols["EOF"]].append(("Accept", None))
        #
        # for state_id in range(len(all_states)):
        #     print(f"{state_id}:")
        #     for symbol, s_id in self.symbols.items():
        #         if len(final[state_id][s_id]) > 0:
        #             print(f"\t{symbol}:{final[state_id][s_id]}")
        return final

    def parse(self, to_parse: List[str]):
        try:
            tokenized = self.tokenize(to_parse) + ["EOF"]
            call_stack: List[Tuple[int, List[int], List]] = [(0, [0], [])]
            while call_stack:
                position, parser_state, output = call_stack.pop()
                if position >= len(tokenized):
                    continue
                cur_state = parser_state[-1]
                next_token = tokenized[position]
                actions = self.final[cur_state][self.symbols[next_token]]
                if len(actions) == 0:
                    continue
                for action in actions:
                    if action[0] == "Shift":
                        call_stack.append(
                            (position + 1, parser_state + [action[1]], output)
                        )
                    elif action[0] == "Reduce":
                        symbol, r_id = action[1]
                        rule_name, rule = self.rules[symbol][r_id]
                        new_state = parser_state[: -len(rule)]
                        right = position
                        left = position
                        children = []
                        to_grab = output
                        for token in rule[::-1]:
                            if token in self.rules:
                                children.append(to_grab[-1])
                                to_grab = to_grab[:-1]
                                left = children[-1].start_pos
                            else:
                                left -= 1
                        tree = Tree(
                            rule_name[2:] if rule_name is not None else "*",
                            children[::-1],
                            left,
                            right,
                            to_parse[left:right],
                        )
                        to_output = to_grab + [tree]

                        goto = self.final[new_state[-1]][self.symbols[symbol]]
                        assert len(goto) == 1 and goto[0][0] == "Goto", goto
                        call_stack.append(
                            (position, new_state + [goto[0][1]], to_output,)
                        )
                    elif action[0] == "Accept":
                        return adapt(output[0], to_parse)
        except KeyError as e:
            raise ParseError(f"Unknown key {e}")
        raise ParseError(f"Didn't find parse for {' '.join(to_parse)}")

    def parse_logical(self, to_parse: List[str]):
        """ Converts a logical like '|- 1 > 2' to a wff expression before parsing """
        expr = to_parse[1:]
        expr.insert(0, "wff")
        return self.parse(expr)

    def has_valid_syntax(self, expression: List[str]):
        try:
            self.parse(expression)
        except ParseError:
            return False
        return True

    def parse_to_proof(self, node):
        if node.data in {"wff_var", "class_var", "setvar_var"}:
            return [self.f_table[node.string[0]][1]]
        if node.data == "*":
            return sum([self.parse_to_proof(c) for c in node.children], [])
        else:
            ordered_children = sorted(
                zip(self.axiom_hyp_order[node.data], node.children)
            )
            children = sum([self.parse_to_proof(c) for _i, c in ordered_children], [])
            return children + [node.data]


def get_parser(grammar: str):
    grammar_root = clusterify_path("YOUR_PATH/metamath/grammars/")
    grammars = {
        "holophrasm": f"{grammar_root}grammar_holophrasm.in",
        "new2": f"{grammar_root}grammar_new3.in",
        "new3": f"{grammar_root}grammar_new3.in",
        "inequalities": f"{grammar_root}grammar_inequalities.in",
    }
    if grammar not in grammars:
        logger = getLogger()
        logger.warning(f"Unknown grammar {grammar}. Interpretring as path.")

    path_to_grammar = grammars.get(grammar, grammar)
    pre_computed = None
    filename, file_extension = os.path.splitext(path_to_grammar)
    if Path(filename + ".pkl").exists():
        try:
            pre_computed = pickle.load(open(filename + ".pkl", "rb"))
        except Exception:
            pass  # we should just rebuild the pkl in that case (denotes a change in the Parser code)

    with open(path_to_grammar, "r") as f:
        current_hash = hashlib.sha256(f.read().encode()).hexdigest()

    if (
        pre_computed is None
        or not hasattr(pre_computed, "hash")
        or current_hash != pre_computed.hash
    ):
        print(
            "Pickle not found or deprecated. Building grammar. This might take time, but the result will be cached."
        )
        p = Parser(path_to_grammar)
        pickle.dump(p, open(filename + ".pkl", "wb"))
    else:
        p = pre_computed
    return p


if __name__ == "__main__":

    a = get_parser("inequalities")
    # a = get_parser(sys.argv[1])

    to_parse = "wff ( A ^ - 1 ) < ( B ^ - 1 )".split()
    # to_parse = "wff ( E. x ( x e. A /\ ph ) -> A. x E. x ( x e. A /\ ph ) )".split()
    # to_parse = "wff ( E. x e. A ph -> A. x E. x e. A ph )".split()
    to_parse_str = " ".join(to_parse)

    def pt(node, d):
        tabs = " " * d
        p = f"{tabs} {node.data} ({to_parse_str[node.start_pos:node.end_pos]}) [{(node.start_pos, node.end_pos)}]\n"
        c = "\n".join([pt(c, d + 1) for c in node.children])
        return p + c

    parse_tree = a.parse(to_parse)
    print(pt(parse_tree, 0))
    print(a.parse_to_proof(parse_tree))
    # print(a.axiom_hyp_order["wrex"])
