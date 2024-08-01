# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Set
import os
import pickle
from copy import deepcopy
from evariste.envs.mm.env import MetamathEnv
from evariste.envs.mm.utils import enumerate_nodes
from evariste.syntax.parser import get_parser, ParseError
import time

typemap = {"wff": "wff", "set": "setvar", "setvar": "setvar", "class": "class"}


class IntervalTree:
    def __init__(self, node, id, syntax):
        self.begin = node.start_pos
        self.end = node.end_pos
        self.id = id
        self.syntax = syntax
        self.next = None
        self.child = None

    def __repr__(self):
        return f"[{self.begin}, {self.end}]"

    def string(self, expression):
        return expression[self.begin : self.end]


class TargetError(Exception):
    pass


def build(tree, env):
    intervals = []

    def _build(node):
        if node.data in {
            "wff_var",
            "class_var",
            "setvar_var",
        }:
            intervals.append(
                IntervalTree(
                    node,
                    len(intervals),
                    {"wff_var": "wff", "class_var": "class", "setvar_var": "setvar"}[
                        node.data
                    ],
                )
            )
            intervals[-1].next = len(intervals)
            return
        label = env.labels.get(node.data, None)
        if label is not None:
            this_elem = IntervalTree(node, len(intervals), label[1].tokens[0])
            intervals.append(this_elem)
            child_id = len(intervals)
            for child in node.children:
                if this_elem.child is None:
                    this_elem.child = child_id
                _build(child)
            this_elem.next = len(intervals)
        else:
            _build(node.children[0])

    _build(tree)
    return intervals


def find(theorems, target, types_list, env, parser):
    assert len(theorems) == len(
        types_list
    ), f"Bad input. {len(theorems)} but {len(types_list)} types"
    try:
        parse_tree_target = parser.parse(target.split())
        ta_intervals = build(parse_tree_target, env)
    except ParseError:
        raise TargetError(target)
    results = []
    for theorem, types in zip(theorems, types_list):

        def find_internal(th_node_id, ta_node_id, cur_subs):
            if th_node_id == len(th_intervals) and ta_node_id == len(ta_intervals):
                return cur_subs
            elif th_node_id == len(th_intervals):
                return None
            elif ta_node_id == len(ta_intervals):
                return None

            th_node = th_intervals[th_node_id]
            ta_node = ta_intervals[ta_node_id]

            if th_node.syntax != ta_node.syntax:
                return None
            cur_ta_string = ta_node.string(target)
            cur_th_string = th_node.string(theorem)
            if cur_th_string in types:
                sub = cur_subs.get(cur_th_string)
                if sub is not None and sub == cur_ta_string:
                    return find_internal(th_node.next, ta_node.next, cur_subs,)
                if sub is None:  # and types[cur_th_string] == ta_node.syntax:
                    new_subs = deepcopy(cur_subs)
                    new_subs[cur_th_string] = cur_ta_string
                    return find_internal(th_node.next, ta_node.next, new_subs,)
                return None
            else:
                if ta_node.child is not None and th_node.child is not None:
                    return find_internal(th_node.child, ta_node.child, cur_subs,)
                return find_internal(th_node.next, ta_node.next, cur_subs,)

        try:
            parse_tree_theorem = parser.parse(theorem.split())
            th_intervals = build(parse_tree_theorem, env)

            results.append(find_internal(0, 0, {}))
        except ParseError as e:
            results.append(None)

    return results


class SubstFinder:
    def __init__(self, env, parser):
        self.env = env
        self.parser = parser

    def get_subs(self, goal: str, label: str):
        assertion = self.env.labels[label][1]
        tokens = assertion.tokens
        f_hyps = assertion.f_hyps
        types = {v: k for k, v in f_hyps if v in assertion.mand_vars}

        return find(
            theorems=[" ".join(["wff"] + tokens[1:])],
            target=" ".join(["wff"] + goal.split()[1:]),
            types_list=[types],
            env=self.env,
            parser=self.parser,
        )[0]

    def process(self, goal: str, labels: Set[str]):
        """

        @param goal: the string we want after substitutions in labels
        @param labels: a list of potential labels we want to apply
        @return: a mapping from label to substitution
        """
        theorems, f_hyps, types, valid_label = [], [], [], []
        results = {}
        for label in labels:
            try:
                assertion = self.env.labels[label][1]
                theorems.append(" ".join(["wff"] + assertion.tokens[1:]))
                f_hyps.append(assertion.f_hyps)
                types.append({v: k for k, v in f_hyps[-1] if v in assertion.mand_vars})
                valid_label.append(label)
            except KeyError:
                results[label] = None
        try:
            substitutions = find(
                theorems=theorems,
                target=" ".join(["wff"] + goal.split()[1:]),
                types_list=types,
                env=self.env,
                parser=self.parser,
            )
            for label, subs in zip(valid_label, substitutions):
                results[label] = subs
        except TargetError as e:
            print(f"Target error: {e}")
            return {x: None for x in labels}

        return results


if __name__ == "__main__":

    data_dir = "resources/metamath/DATASET_HOLOPHRASM/"
    database_path = "resources/metamath/DATASET_HOLOPHRASM/set.mm"

    # build env
    mm_env = MetamathEnv(
        filepath=database_path,
        decompress_proofs=False,
        verify_proofs=False,
        log_level="info",
    )
    mm_env.process()

    # exit()
    # reload proof trees
    print("Reloading proof trees...")
    with open(os.path.join(data_dir, "proof_trees.pkl"), "rb") as f:
        proof_trees = pickle.load(f)
    print(f"Reloaded {len(proof_trees)} proof trees.")

    count_tok, total_tok = 0, 0
    count_sub, total_sub = 0, 0
    count_seq, total_seq = 0, 0

    for tree in proof_trees.values():

        for node in enumerate_nodes(
            tree, ignore_f_e=True, ignore_empty=True, no_syntactic=True
        ):

            assert not node.is_syntactic
            tokens = mm_env.labels[node.label][1]["tokens"]
            subs = node.substitutions
            keys = set(subs.keys())

            # before
            total_tok += sum(len(subs[k]) for k in keys)
            total_sub += len(keys)
            total_seq += 1

            # after
            found = set(k for k in keys if k in tokens)
            count_tok += sum(len(subs[k]) for k in found)
            count_sub += len(found)
            count_seq += int(len(found) == len(subs))

    print(f"Skip {100 * count_tok / total_tok:.4}% tokens ({count_tok}/{total_tok})")
    print(f"Skip {100 * count_sub / total_sub:.4}% substs ({count_sub}/{total_sub})")
    print(f"Skip {100 * count_seq / total_seq:.4}% nodes ({count_seq}/{total_seq})")

    my_parser = get_parser("holophrasm")
    subst_finder = SubstFinder(mm_env, my_parser)

    time_mine = 0
    time_old = 0

    for i, (label, tree) in enumerate(proof_trees.items()):
        if (i + 1) % 10 == 0:
            print(i + 1, time_old / time_mine)
        for node in enumerate_nodes(
            tree, ignore_f_e=True, ignore_empty=True, no_syntactic=True
        ):
            statement = node.statement
            tokens = mm_env.labels[node.label][1].tokens
            f_hyps = mm_env.labels[node.label][1].f_hyps
            subs = node.substitutions
            types = {v: k for k, v in f_hyps if v in subs and v in tokens}
            # sanity check
            assert statement == sum([subs.get(k, [k]) for k in tokens], [])
            assert set(x[1] for x in f_hyps) == subs.keys()
            assert tokens[0] == statement[0] == "|-"
            found = None
            try:
                start = time.time()
                found = subst_finder.get_subs(" ".join(statement), node.label)
                time_mine += time.time() - start

                if found is None or not all(
                    found[k] == " ".join(subs[k]) for k in found.keys()
                ):
                    raise TargetError("wrong found")

                start = time.time()
                time_old += time.time() - start
            except TargetError as e:
                print(e)
                print("====")
                print(label)
                print(node)
                print(subs)
                print(" ".join(["wff"] + tokens[1:]))
                print(" ".join(["wff"] + statement[1:]))
                print(types)
                print(found)
                assert False
