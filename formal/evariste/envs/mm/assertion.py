# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pprint
import itertools
from collections import deque, OrderedDict


class Assertion:
    """
    Associated with each assertion is a set of hypotheses that must be satisfied in order
    for the assertion to be used in a proof. These are called the "mandatory hypotheses"
    of the assertion.

    The set of mandatory variables associated with an assertion is the set of (zero or more)
    variables in the assertion and in any active $e statements.

    The (possibly empty) set of mandatory hypotheses is the set of all active $f statements
    containing mandatory variables, together with all active $e statements.

    The set of mandatory $d statements associated with an assertion are those active $d
    statements whose variables are both among the assertion’s mandatory variables.

    Attributes
    ----------
        label: str (optional)
            If this assertion is in the metamath database, this is its label
        active_vars : List of str
            All variables in scope in the current scope. They are not necessarily
            used by in the assertion.
        mand_vars : List of str
            Variables occuring in either the hypotheses or the assertion statement.
        active_disj
            All disjoint variable hypotheses in the current scope.
        active_f_labels: Map of label to (var_type, var_name) tuple
             All f hypotheses (syntatic) in scope. They are not necessarily used by
             the assertion.
        active_e_labels: Map of label to list of tokens
            All e (logical) hypotheses in scope. They are all "mandatory" unlike the
            set of f hypotheses.
        mand_disj: set of tuples (str, str) of variable names
            Disjoint variable hypotheses. Each pair of variables in the set must
            be disjoint. Refer to the metamath manual for more details.
        e_hyps : list of expressions (expressions are lists of tokens)
            The logical hypotheses of the assertion in order.
            By convention each e hypothesis starts with the |- turnstile token.
        f_hyps : list of (var_type, var_name) tuples
            This list includes all syntatic hypotheses that include mandatory
            variables in order. The proof of the associated assertion may internally
            use additional hypotheses.
        e_hyps_labels: The labels of the e_hyps in order
        f_hyps_labels: The labels of the f_hyps in order
        f_hyps_map : OrderedDict mapping label -> (var_type, var_name)
        e_hyps_map : OrderedDict mapping label -> (var_type, var_name)
        tokens
    """

    def __init__(self, frame_stack, tokens, label=None):
        self.label = label
        self.tokens = tokens
        self.tokens_str = " ".join(self.tokens)
        # active $v variables, $d statements, $f and $e hypotheses
        self.active_vars = set.union(*(frame.v for frame in frame_stack.frames))
        self.active_disj = set.union(*(frame.d for frame in frame_stack.frames))
        self.active_f_labels = dict(
            kv for frame in frame_stack.frames for kv in frame.f_labels.items()
        )
        self.active_e_labels = dict(
            kv for frame in frame_stack.frames for kv in frame.e_labels.items()
        )

        # $e hypotheses
        self.e_hyps = [
            e_hyp for frame in frame_stack.frames for e_hyp in frame.e_labels.values()
        ]

        # mandatory $v variables
        self.mand_vars = {
            tok for hyp in itertools.chain(self.e_hyps, [tokens]) for tok in hyp
        } & self.active_vars

        # mandatory $d statements
        self.mand_disj = (
            set(itertools.product(self.mand_vars, self.mand_vars)) & self.active_disj
        )

        # $f hypotheses with mandatory variables
        remaining_mand_vars = self.mand_vars.copy()
        self.f_hyps = deque()
        for frame in reversed(frame_stack.frames):
            for var_type, var_name in reversed(frame.f_labels.values()):
                if var_name in remaining_mand_vars:
                    self.f_hyps.appendleft((var_type, var_name))
                    remaining_mand_vars.remove(var_name)
        assert len(remaining_mand_vars) == 0

        # retrieve labels of $e and $f hypotheses
        self.f_hyps_labels = [frame_stack.get_f_label(v) for _, v in self.f_hyps]
        self.e_hyps_labels = [frame_stack.get_e_label(s) for s in self.e_hyps]

        self.f_hyps_map = OrderedDict(
            [(k, v) for (k, v) in zip(self.f_hyps_labels, self.f_hyps)]
        )
        self.e_hyps_map = OrderedDict(
            [(k, v) for (k, v) in zip(self.e_hyps_labels, self.e_hyps)]
        )

    def get_var_type(self, var):
        vtype = None
        for vt, vf in self.f_hyps:
            if var == vf:
                vtype = vt
        return vtype

    def __contains__(self, item):
        return item in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, item):
        return self.__dict__.__setitem__(key, item)

    def __repr__(self):
        return pprint.pformat(self.__dict__, width=120)

    def __str__(self):
        max_len = 120

        def trun(ll):
            ls = ""
            if len(ll) == 0:
                return "(none)"
            for l in ll:
                if len(ls) + len(l) > max_len:
                    ls += " ..."
                    break
                ls += " " + l
            return ls

        s = ""
        s += f"Assertion {self.label if self.label is not None else ''}\n"
        s += f"  active vars   : {trun(self.active_vars)}\n"
        s += f"  mandatory vars: {trun(self.mand_vars)}\n"
        s += f"  active disj: {trun(self.active_disj)}\n"
        s += "  syntatic hypotheses: \n"
        for (label, (var_type, var_name)) in self.f_hyps_map.items():
            s += f"    {label:5}   {var_type} {var_name}\n"
        s += "  logical hypotheses: \n"
        if len(self.e_hyps_map) > 0:
            for (label, hyp) in self.e_hyps_map.items():
                s += f"    {label:12}   {' '.join(hyp)}\n"
        else:
            s += "(none)\n"
        s += f"  goal:\n"
        s += f"    {self.label if self.label is not None else '':12}   {self.tokens_str}\n"
        return s
