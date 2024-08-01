# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, List, Set, Dict, Union
from dataclasses import dataclass, field
from collections import defaultdict
from logging import getLogger
from pathlib import Path
import numpy as np
import sympy as sp
from sympy import false, true, CoercionFailed

from evariste import json
from evariste.logger import create_logger
from params.params import ConfStore, Params, cfg_from_cli
from evariste.backward.env.equations.graph import EQTactic, EQTheorem, EQRuleTactic
from evariste.backward.prover.utils import compute_usage_entropy
from evariste.envs.eq.env import EquationsEnvArgs, EquationEnv, EqGraphSamplerParams
from evariste.envs.eq.imo_generation import basic_sample, generate_graph_IMO
from evariste.envs.eq.graph import Node, ZERO, VNode, NodeParseError
from evariste.envs.eq.rules import TRule, ARule, eval_assert_with_rule
from evariste.envs.eq.rules_lib import ALL_RULES
from evariste.envs.eq.lean_utils import LEAN_HYP_NAMES, LEAN_THEOREM_TEMPLATE
from evariste.envs.eq.generation import (
    GraphAssertNode,
    GraphInternalNode,
    GraphNode,
    GraphRuleNode,
    GraphTransformNode,
    walk_to_graph,
    extract_graph_steps,
    EquationGraphGenerator,
    EquationGraphSampler,
)
from evariste.envs.eq.sympy_utils import sympy_to_prefix
from evariste.envs.lean.utils import EXPECTED_HEADER
from evariste.utils import timeout, MyTimeoutError


@dataclass
class RandomGenerationArgs(Params):

    n_generations: int  # number of generations

    # env params
    rule_envs: List[str]
    env: EquationsEnvArgs

    # rule properties
    use_implicit: bool = False
    overwrite_int_matching: bool = True

    # export
    path: Optional[str] = None
    remove_comments: bool = True

    # common params
    rwalk: bool = True
    imo_gen: bool = False
    max_hyps: int = 5

    # walk
    bidirectional: bool = False
    n_steps: int = 10
    n_init_ops: int = 3

    # graph params
    graph_nodes: int = 500
    max_trials: int = 1000
    graph_sampler: EqGraphSamplerParams = field(
        default_factory=lambda: EqGraphSamplerParams()
    )

    # generator args
    hyp_max_ops: int = 3
    tf_prob: float = 0.5
    bias_rules: float = 1.0
    bias_nodes: float = 0.0
    max_true_nodes: int = 30

    # remove generations with false hyps (according to sympy)
    # or where one hyp == statement once simplified
    filter_hard: bool = False

    # if simplified is True, output the sympy simplified statement
    simplified: bool = False

    seed: Optional[int] = None

    # for test data, only generate statements
    no_proof: bool = False

    def __post_init__(self):
        for rule_env in self.rule_envs:
            assert rule_env in ["lean_real", "lean_nat", "lean_int", "imo",], rule_env
            if rule_env == "lean_nat":
                assert self.env.positive
                assert self.env.pos_hyps


def is_a_eq_a(goal: Node, allow_leq_div: bool) -> bool:
    assert goal.is_comp()
    res = goal.value == "==" or allow_leq_div and goal.value in ["<=", "∣"]
    res = res and goal.children[0].eq(goal.children[1])
    return res


def set_subgoal_counts(root: GraphNode, vtype: str) -> None:

    """
    For each node, count the number of occurrences of each subgoal in the subtree.
    """

    seen_counts: Set[str] = set()

    def _counts_subgoals(node: GraphNode) -> None:
        prefix = node.node.prefix()
        if prefix in seen_counts:
            if node.subgoal_counts is None:
                node.subgoal_counts = defaultdict(int)
                assert node.ntype == "true"
            return
        seen_counts.add(prefix)
        assert node.subgoal_counts is None
        node.subgoal_counts = defaultdict(int)
        if isinstance(node, GraphInternalNode):
            for hyp in node.hyps:
                _counts_subgoals(hyp)
                node.subgoal_counts[hyp.node.infix_lean(vtype)] += 1
                assert hyp.subgoal_counts is not None
                for k, v in hyp.subgoal_counts.items():
                    node.subgoal_counts[k] += v
        else:
            return

    _counts_subgoals(root)


def lean_proof_from_graph(
    root: GraphInternalNode,
    init_hyps: List[Node],
    goals_with_tactics: List[Tuple[EQTheorem, EQTactic, List[Node]]],
    egg: EquationGraphGenerator,
    use_implicit: bool,
    overwrite_int_matching: bool,
    vtype: str,
    no_comments: bool = True,
) -> Tuple[List[str], Set[str]]:
    """
    Extract the steps required to prove a node in a graph.
    """
    assert isinstance(root, GraphNode)
    assert root.ntype in ["transform", "assert"]
    assert vtype in ["nat", "int", "real"]
    used_hyps: List[Node] = []

    # initial hypotheses
    hyps = [
        (LEAN_HYP_NAMES[i], hyp.infix_lean(vtype)) for i, hyp in enumerate(init_hyps)
    ]

    # all hypotheses
    existing_hyps: Dict[str, str] = {content: name for name, content in hyps}

    # retrieve all variables that appear in the theorem
    all_theorem_vars: Set[str] = set.union(
        *[goal.eq_node.get_vars() for goal, _, _ in goals_with_tactics],
        *[hyp.get_vars() for hyp in init_hyps],
    )
    assert len(all_theorem_vars) > 0

    # set subgoal counts (downward counts only)
    set_subgoal_counts(root, vtype)

    # proof steps
    proof_lines: List[str] = []

    def add_line(l: str):
        proof_lines.append(l)

    def add_comment(l: str):
        if no_comments:
            return
        if not l.startswith("--") and not l == "":
            l = "-- " + l
        proof_lines.append(l)

    def add_hyp(s: str) -> str:
        assert s not in existing_hyps
        hyp_name = [x for x in LEAN_HYP_NAMES if x not in existing_hyps.values()][0]
        existing_hyps[s] = hyp_name
        return hyp_name

    def _traverse(node: GraphNode, prove_hyps_first: bool = False):

        goal = node.node
        lean_pp = goal.infix_lean(vtype)

        add_comment(f"")
        add_comment(f"goal: {lean_pp}")
        add_comment(f"prefix: {goal.prefix()}")

        # add variables that may not appear in the original statement / hypotheses
        for vname in goal.get_vars():
            if vname not in all_theorem_vars:
                add_comment(f"adding missing variable: {vname}")
                all_theorem_vars.add(vname)

        # trivial statement
        if node.ntype == "true":
            add_comment(f"true node: {lean_pp}")
            res = eval_assert_with_rule(goal, egg.rules_a_no_hyps, vtype)
            if res == "__NUMERIC__":
                ps = "norm_num,"
            elif isinstance(res, ARule):
                ps = f"apply {res.name},"
            else:
                raise RuntimeError(f"{lean_pp} is not trivial (res={res})")
            add_line(ps)
            return

        # hypothesis
        elif node.ntype == "hyp":
            add_comment(f"hyp node: {lean_pp}")
            add_line(f"exact {existing_hyps[lean_pp]},")
            return

        # transform or assert node
        assert isinstance(node, GraphRuleNode)

        # if some hypotheses appear multiple times, prove them first
        # so we can re-use them. also useful to deal with nth_rewrite issues:
        # https://leanprover.zulipchat.com/#narrow/stream/113488-general/topic/nth_rewrite.20issue
        local_hyps: Set[str] = set()
        for hyp in node.hyps:
            assert node.subgoal_counts is not None
            hyp_lean = hyp.node.infix_lean(vtype)
            hyp_count = node.subgoal_counts[hyp_lean]
            if hyp_lean in existing_hyps:
                continue
            if is_a_eq_a(hyp.node, allow_leq_div=True):  # hack for ←mul_eq_zero_of_left
                add_comment(f"unproved a_eq_a hyp -> prove_hyps_first=True")
                prove_hyps_first = True
            if hyp_count > 1 or prove_hyps_first:
                hyp_name = add_hyp(hyp_lean)
                local_hyps.add(hyp_name)
                add_comment(f"creating {hyp_name} ({hyp_count}): {hyp_lean}")
                add_line(f"have {hyp_name} : {hyp_lean},")
                _traverse(hyp)

        # hypotheses / variables that appear in existing hypotheses.
        # variables fully determined by hypotheses do not need to be specified
        rule_hyps_lean: List[str] = []
        vars_in_hyps: Set[str] = set()
        for h in node.rule.hyps:
            rule_hyp = h.set_vars(node.substs, mandatory=False)
            rule_hyps_lean.append(rule_hyp.infix_lean(vtype))
            if rule_hyps_lean[-1] in existing_hyps:
                vars_in_hyps |= h.get_vars()

        # retrieve Lean rule
        rule = node.rule
        lean_rule = rule.lean_rule
        assert node.hyps is not None
        assert lean_rule is not None
        label = rule.name

        nonlocal use_implicit
        implicit = use_implicit

        def build_args(to_provide: Set[str], implicit_args: bool) -> Tuple[str, int]:

            all_args: List[Tuple[str, str]] = []
            hyp_id = 0

            assert lean_rule is not None and lean_rule.args is not None
            for arg in lean_rule.args:
                if arg[0] in ["instance", "type"]:
                    if implicit_args:
                        all_args.append((arg[0], "_"))
                    else:
                        continue
                elif arg[0] == "hyp":
                    h_name = existing_hyps.get(rule_hyps_lean[hyp_id], "_")
                    all_args.append((arg[0], h_name))
                    hyp_id += 1
                else:
                    assert arg[0] == "var"
                    _, _vname, _vimplicit, _vtype = arg
                    # if variable to provide is implicit, implicit must have already been set to True
                    if _vname in to_provide and _vimplicit:
                        assert implicit_args
                    if implicit_args or not _vimplicit:
                        if _vname in to_provide:
                            assert isinstance(node, GraphTransformNode) or isinstance(
                                node, GraphAssertNode
                            )
                            all_args.append(
                                (arg[0], node.substs[_vname].infix_lean(vtype))
                            )
                        else:
                            all_args.append((arg[0], "_"))
            assert hyp_id == len(rule.hyps) == len(rule_hyps_lean)

            # pop unnecessary "_"
            popped_hyps = 0
            while len(all_args) > 0 and all_args[-1][1] == "_":
                arg_type, _ = all_args.pop()
                if arg_type == "hyp":
                    popped_hyps += 1

            # merge args
            if len(all_args) == 0:
                return "", popped_hyps
            else:
                return " " + " ".join(v for _, v in all_args), popped_hyps

        # build tactic
        if isinstance(node, GraphTransformNode):
            assert isinstance(rule, TRule)
            assert node.substs is not None
            src_vars = rule.l_vars if node.fwd else rule.r_vars
            tactic = EQRuleTactic(
                label=rule.name,
                fwd=node.fwd,
                prefix_pos=node.prefix_pos,
                to_fill={k: v for k, v in node.substs.items() if k not in src_vars},
            )
        else:
            assert isinstance(node, GraphAssertNode)
            assert isinstance(rule, ARule)
            assert node.substs is not None
            tactic = EQRuleTactic(
                label=rule.name,
                fwd=None,
                prefix_pos=None,
                to_fill={k: v for k, v in node.substs.items() if k not in rule.n_vars},
            )
        add_comment(f"tactic: {rule.name}")
        add_comment(f"hyps {len(node.hyps)}:")
        for hyp in node.hyps:
            add_comment(f"   hyps {hyp.node.infix_lean(vtype)}:")

        popped_hyps: Optional[int] = None
        prove_only_rewrite = False  # to address issues with INT matching

        if node.ntype == "transform":

            assert isinstance(rule, TRule)
            if tactic.fwd:
                src_pattern, src_vars = rule.left, rule.l_vars
                tgt_pattern, tgt_vars = rule.right, rule.r_vars
            else:
                src_pattern, src_vars = rule.right, rule.r_vars
                tgt_pattern, tgt_vars = rule.left, rule.l_vars

            # list of variables that we will explicitly provide to reduce ambiguity.
            # special exception when the source pattern is a variable, because otherwise
            # we don't know what to match
            to_provide: Set[str] = set()
            src_pattern_is_var: Optional[str] = None
            if src_pattern.is_var():
                add_comment(f"adding explicit source pattern {src_pattern}")
                to_provide.add(str(src_pattern))
                src_pattern_is_var = str(src_pattern)
                implicit = True
            if lean_rule.mandatory_vars is not None:  # very rarely used
                for vname in lean_rule.mandatory_vars:
                    add_comment(f"adding mandatory var {vname}")
                    to_provide.add(vname)
            not_provided = sorted(rule.all_vars - vars_in_hyps - to_provide)

            # some variables appear exclusively in the target, so we need to specify them,
            # e.g. variable `c` in `real.add_lt_add_iff_left`
            # or exclusively in the hypotheses, e.g. eq_zero_of_mul_eq_self_right
            for vname in sorted((tgt_vars | rule.h_vars) - (src_vars | vars_in_hyps)):
                if vname in tgt_vars:
                    add_comment(f"providing var in target only {vname}")
                else:
                    add_comment(f"providing var in hypothesis only {vname}")
                assert vname in not_provided and vname not in to_provide
                not_provided.remove(vname)
                to_provide.add(vname)
                assert vname in lean_rule.all_vars

            if not implicit and len(set(lean_rule.implicit_vars) & to_provide) > 0:
                add_comment("some variables to provide are implicit")
                implicit = True
            add_comment(f"src_vars: {src_vars}")
            add_comment(f"tgt_vars: {tgt_vars}")
            add_comment(f"vars_in_hyps: {vars_in_hyps}")
            add_comment(f"rule.h_vars: {rule.h_vars}")
            add_comment(f"implicit vars: {lean_rule.implicit_vars}")

            # try to not provide variables unless there are several matching locations
            occ: Optional[int] = None
            while True:

                to_fill = {
                    k: v
                    for k, v in node.substs.items()
                    if k in vars_in_hyps or k in to_provide
                }
                final_src_pattern = src_pattern.set_vars(to_fill, mandatory=False)
                add_comment(f"src_vars: {src_vars}")
                add_comment(f"tgt_vars: {tgt_vars}")
                add_comment(f"to_provide: {to_provide}")
                add_comment(f"not_provided: {not_provided}")
                add_comment(f"final_src_pattern: {final_src_pattern}")

                # look where we can apply the rule
                eligible_pos = rule.eligible(goal, tactic.fwd, final_src_pattern)
                assert len(eligible_pos) > 0
                add_comment(f"eligible_pos ({len(eligible_pos)}): {eligible_pos}")

                # look for matching with negative numbers
                for v_name in sorted(final_src_pattern.get_vars()):
                    v = VNode(v_name)
                    assert v in final_src_pattern
                    if -v not in final_src_pattern:
                        continue
                    neg_int_pattern = final_src_pattern.replace(v, -v).replace(--v, v)
                    neg_eligible_pos = rule.eligible(goal, tactic.fwd, neg_int_pattern)
                    add_comment(
                        f"final_src_pattern={final_src_pattern}, neg_int_pattern={neg_int_pattern}"
                    )
                    if len(neg_eligible_pos) > 0:
                        add_comment(
                            f"neg_eligible_pos ({len(neg_eligible_pos)}): {neg_eligible_pos}"
                        )
                        for x in neg_eligible_pos:
                            sub_pos, sub_goal = x
                            match = neg_int_pattern.match(sub_goal)
                            assert match is not None
                            if match[v_name].is_int() and match[v_name].value < 0:
                                add_comment(f"adding possible negative int match: {x}")
                                eligible_pos.append(x)

                # to avoid nth_rewrite, try to specify more variables
                if len(not_provided) > 0 and len(eligible_pos) > 1:
                    var_index = egg.rng.randint(len(not_provided))
                    new_var = not_provided.pop(var_index)
                    assert new_var not in to_provide
                    if not implicit and new_var in lean_rule.implicit_vars:
                        add_comment(f"{new_var} in implicit, will not add")
                        continue
                    to_provide.add(new_var)
                    add_comment(f"try to add variable: {new_var}")
                    continue

                # look for valid occurrences
                eligible_occ = [
                    (i, x)
                    for i, x in enumerate(eligible_pos)
                    if x[0] == tactic.prefix_pos
                ]
                assert len(eligible_occ) == 1, eligible_occ
                occ, eq = eligible_occ[0]
                break

            # rewrite direction
            direction = "" if tactic.fwd else "← "

            assert len(eligible_pos) >= 1
            if len(eligible_pos) == 1:
                rw = "rw"
            else:
                # as there are issues with nth_rewrite handling hypotheses,
                # prove the unproved hypotheses first
                if not prove_hyps_first and any(
                    hyp not in existing_hyps for hyp in rule_hyps_lean
                ):
                    add_comment(
                        f"setting prove_hyps_first=True as {len(eligible_pos)} "
                        f"occurrences and at least one unproved hypothesis"
                    )
                    _traverse(node, prove_hyps_first=True)
                    return
                assert occ is not None
                rw = f"nth_rewrite {occ}"

            # special cases
            IDENTITY_LABELS = [
                "THEOREM__TRANSFORM__T_A_B_A_eq_B__02366162",
                "LEAN__IDENTITY",
            ]
            if rule.name in IDENTITY_LABELS:
                hyp_node = node.substs["A"] == node.substs["B"]
                hyp_lean = hyp_node.infix_lean(vtype)
                if hyp_lean in existing_hyps:
                    hyp_name = existing_hyps.get(hyp_lean, "_")
                else:
                    add_comment(f"WARNING: {hyp_lean} not found {tactic.fwd}")
                    hyp_name = add_hyp(hyp_lean)
                    local_hyps.add(hyp_name)
                    add_line(f"have {hyp_name} : {hyp_lean}, sorry,")
                ps = f"{rw} {direction}{hyp_name},"
            else:
                assert src_pattern_is_var is None or src_pattern_is_var in to_provide
                if src_pattern_is_var is not None:
                    assert implicit
                all_args, _ = build_args(to_provide, implicit_args=implicit)
                add_comment(
                    f"label={label} " f"implicit={implicit} " f"all_args={all_args}"
                )
                impl_str = "@" if implicit else ""
                ps = f"{rw} {direction}{impl_str}{label}{all_args},"

            # need to overwrite the tactic because of int matching
            if (
                final_src_pattern.is_int()
                and final_src_pattern.ne(ZERO)
                and overwrite_int_matching
            ):
                add_comment(f"overwriting original tactic: {ps}")
                ps = f"suffices : {node.hyps[0].node.infix_lean(vtype)}, from sorry,"
                prove_only_rewrite = True

            add_line(ps)

        else:
            assert node.ntype == "assert"
            assert isinstance(rule, ARule)
            to_provide = rule.all_vars - rule.n_vars - vars_in_hyps
            if not implicit and len(set(lean_rule.implicit_vars) & to_provide) > 0:
                add_comment("some variables to provide are implicit")
                implicit = True
            all_args, popped_hyps = build_args(to_provide, implicit_args=implicit)
            impl_str = "@" if implicit else ""
            ps = f"apply {impl_str}{label}{all_args},"
            add_line(ps)

        # hypotheses. transformation rules that have unproved hypotheses
        # have the rewritten statement as first subgoal
        hyps_to_prove: List[GraphNode] = []

        if isinstance(node, GraphInternalNode):
            for i, hyp in enumerate(node.hyps):
                hn = hyp.node

                # if this hypothesis was provided as an argument, it will not
                # appear in the goalstack
                was_hyp = node.ntype == "transform" and i > 0 or node.ntype == "assert"
                was_hyp_arg = was_hyp and hn.infix_lean(vtype) in existing_hyps
                if was_hyp_arg:
                    continue

                # if we applied a rewrite, the goal will automatically be removed if
                # it is of the form A == A or A <= A or A ∣ A
                if (
                    node.ntype == "transform"
                    and i == 0
                    and is_a_eq_a(hn, allow_leq_div=True)
                ):
                    continue

                # if we had issues with INT matching, we only have to prove
                # the rewrite statement and not the associated hypotheses
                if prove_only_rewrite and node.ntype == "transform" and i > 0:
                    continue

                hyps_to_prove.append(hyp)

        # for assertion rules, if the goal is D and we apply theorem LABEL `A -> B -> C -> D` with
        # "apply LABEL"       then we have to prove A, B, C
        # "apply LABEL A"     then we have to prove B, C
        # "apply LABEL _"     then we have to prove B, C, A
        # "apply LABEL A _"   then we have to prove C, B
        # "apply LABEL A B"   then we have to prove C
        # "apply LABEL A B _" then we have to prove C
        # "apply LABEL A B C" then we have nothing to prove
        if node.ntype == "transform":
            for hyp in hyps_to_prove:
                _traverse(hyp)
        else:
            assert popped_hyps is not None
            for hyp in hyps_to_prove[-popped_hyps:]:
                _traverse(hyp)
            for hyp in hyps_to_prove[:-popped_hyps]:
                _traverse(hyp)

        # remove local hypotheses
        for hn_ in local_hyps:
            temp = [hc for hc, hn in existing_hyps.items() if hn == hn_]
            assert len(temp) == 1
            hc_ = temp[0]
            existing_hyps.pop(hc_)
            add_comment(f"remove local hyp {hn_} : {hc_}")

    _traverse(root)

    # sanity check
    assert all(hyp in init_hyps for hyp in used_hyps)

    return proof_lines, all_theorem_vars


@timeout(seconds=10)
def generate_lean_theorem(
    name: str, egg: EquationGraphGenerator, args: RandomGenerationArgs
) -> Optional[Tuple[str, List[str]]]:

    # generate random walk and convert it to a graph
    if args.rwalk:
        walk = egg.random_walk(
            bidirectional=args.bidirectional,
            n_steps=args.n_steps,
            n_init_ops=args.n_init_ops,
            max_created_hyps=egg.rng.randint(args.max_hyps + 1),
            prob_add_hyp=0.5,
        )
        graph_nodes, graph_hyps = walk_to_graph(walk=walk, egg=egg)

    # generate random graph
    elif not args.imo_gen:
        graph_nodes, graph_hyps = egg.generate_graph(
            n_nodes=args.graph_nodes,
            max_trials=args.max_trials,
            n_init_hyps=egg.rng.randint(0, args.max_hyps + 1),
        )

    # use the imo_generator instead
    else:
        try:
            res = generate_graph_IMO(
                egg,
                1,
                n_nodes=args.graph_nodes,
                max_trials=args.max_trials,
                rwalk=True,
                complexity=4,
            )
            if not res:
                print("No IMO statement generated")
                return None
            else:
                graph_nodes, graph_hyps = res
        except Exception as e:
            print("Error of type", e)
            return None

    # statement to prove
    # sample a node which is not an hypothesis or trivially true

    # sample theorems to prove from the graph
    if args.rwalk:
        available = [
            i for i, node in enumerate(graph_nodes) if node.ntype not in ["hyp", "true"]
        ]
        assert len(available) > 0
        node_id = available[-1]
        to_prove = graph_nodes[node_id]
    elif not args.imo_gen:
        graph_sampler = EquationGraphSampler(egg.rng, args.graph_sampler)
        sampled_ids, _ = graph_sampler.sample(graph=egg, n_samples=1)
        assert len(sampled_ids) == 1
        node_id = sampled_ids[0]
        to_prove = graph_nodes[node_id]
    else:
        to_prove = basic_sample(egg, graph_nodes)
        if to_prove.ntype not in ["transform", "assert"]:
            return None
    # sanity check
    assert isinstance(to_prove, GraphInternalNode)

    # if no variable, skip the theorem
    if len(to_prove.node.get_vars()) == 0:
        logger.info(to_prove.node)
        return None

    # used theorems
    used_theorems: List[str] = []
    for node in graph_nodes:
        if isinstance(node, GraphRuleNode):
            assert node.rule.lean_rule is not None
            used_theorems.append(node.rule.lean_rule.label)

    if args.filter_hard:
        _, init_hyps, _ = extract_graph_steps(to_prove)
        try:
            r = sp.simplify(str(to_prove.node))
            if r is true or r is false:
                return None
            simp_hyps = [sp.simplify(str(hyp)) for hyp in init_hyps]
            if any([sh is false or r.equals(sh) for sh in simp_hyps]):
                return None
        # Errors in sympy parsing or simplification of expressions
        except (ValueError, TypeError, CoercionFailed, OverflowError):
            return None
        # error when comparing if r.equals(sh)...
        except RecursionError:
            return None

    # standard traversal
    goals_with_tactics, init_hyps, _ = extract_graph_steps(to_prove)

    # extract lean proof
    proof_lines, all_theorem_vars = lean_proof_from_graph(
        root=to_prove,
        init_hyps=init_hyps,
        goals_with_tactics=goals_with_tactics,
        egg=egg,
        use_implicit=args.use_implicit,
        overwrite_int_matching=args.overwrite_int_matching,
        vtype=egg.env.vtype,
    )

    # define variables
    var_type = {"nat": "ℕ", "int": "ℤ", "real": "ℝ"}[egg.env.vtype]
    var_names = " ".join(sorted(all_theorem_vars))
    if len(all_theorem_vars) > 0:
        var_names = f"\n  ({var_names} : {var_type})"

    # create hyps string
    hyps = [
        (LEAN_HYP_NAMES[i], hyp.infix_lean(egg.env.vtype))
        for i, hyp in enumerate(init_hyps)
    ]
    if len(init_hyps) == 0:
        hyps_str = ""
    else:
        hyps_str = "\n  " + "\n  ".join(f"({hn} : {hc})" for hn, hc in hyps)

    if args.simplified:
        simplified = sp.simplify(str(to_prove.node))
        try:
            statement = Node.from_prefix_tokens(sympy_to_prefix(simplified)).infix_lean(
                egg.env.vtype
            )
        except NodeParseError as e:
            print("===============")
            print(str(to_prove.node))
            print("----Simplified to ---")
            print(simplified)
            print(f"Error {e} -- skipping")
            return None  # skip
    else:
        statement = to_prove.node.infix_lean(egg.env.vtype)

    # build theorem
    theorem = LEAN_THEOREM_TEMPLATE.format(
        name=name, var_names=var_names, hyps=hyps_str, statement=statement,
    )
    if not args.no_proof:
        proof = "  " + "\n  ".join(proof_lines)
        theorem = theorem.replace("  sorry", proof)
    return theorem, used_theorems


def export_random_theorems(
    prefix: Optional[str], args: RandomGenerationArgs
) -> Dict[str, int]:

    logger = getLogger()

    print(f"Generate and export random Lean theorems with: {args}")

    # export path
    if args.path is None:
        args.path = str(Path.home() / "generations.lean")
    logger.info(f"Exporting theorems to {args.path} ...")

    if args.seed is None:
        args.seed = np.random.randint(1_000_000_000)
    logger.info(f"Random generator seed: {args.seed}")

    # build env
    env = EquationEnv.build(args.env, seed=args.seed)

    # build rules
    rules = []
    for rule_env in args.rule_envs:
        rules += [
            rule
            for rule in ALL_RULES[rule_env]
            if rule.get_unary_ops().issubset(env.unary_ops)
            and rule.get_binary_ops().issubset(env.binary_ops)
        ]
    logger.info(f"Found {len(rules)} rules.")

    # build generator
    egg = EquationGraphGenerator(
        env=env,
        rules=rules,
        hyp_max_ops=args.hyp_max_ops,
        tf_prob=args.tf_prob,
        bias_nodes=args.bias_nodes,
        bias_rules=args.bias_rules,
        max_true_nodes=args.max_true_nodes,
    )

    # keep track of used Lean theorems
    all_used_theorems: Dict[str, int] = defaultdict(int)

    with open(args.path, "w") as f:
        f.write(EXPECTED_HEADER + "\n")
        i = 0
        while i < args.n_generations:
            try:
                name = f"THEOREM_{i}"
                if prefix is not None:
                    name = f"{prefix}_THEOREM_{i}"
                generation = generate_lean_theorem(name=name, egg=egg, args=args)
            except MyTimeoutError:
                logger.info("Timeout !")
                continue
            if generation is None:
                logger.info(f"Skipping ...")
                continue
            theorem, used_theorems = generation
            for label in used_theorems:
                all_used_theorems[label] += 1
            if args.remove_comments:
                theorem = (
                    "\n".join(
                        line
                        for line in theorem.split("\n")
                        if not (line.strip() == "" or "--" in line)
                    )
                    + "\n"
                )
            f.write(theorem + "\n")
            f.flush()
            logger.info(f"THEOREM {i}")
            i += 1
            if i % 20 == 0:
                stats = compute_generation_stats(args.rule_envs, all_used_theorems)
                logger.info(
                    f"Usage: {stats['usage']:.5f} -- Entropy: {stats['entropy']:.5f}"
                )
        logger.info(f"Exported to {args.path}")

    return all_used_theorems


ConfStore["lean_gen_rwalk_real"] = RandomGenerationArgs(
    n_generations=10,
    rule_envs=["lean_real"],
    env=EquationsEnvArgs(
        vtype="real",
        positive=False,
        pos_hyps=False,
        unary_ops_str="neg,exp,ln,sqrt,cos,sin,tan,abs",
        binary_ops_str="add,sub,mul,div",
        comp_ops_str="==,!=,<=,<",
    ),
)

ConfStore["lean_gen_rwalk_imo"] = RandomGenerationArgs(
    n_generations=10,
    rule_envs=["imo", "lean_real"],
    imo_gen=True,
    rwalk=False,
    env=EquationsEnvArgs(
        vtype="real",
        positive=False,
        pos_hyps=False,
        unary_ops_str="neg,exp,ln,sqrt,cos,sin,tan,abs",
        binary_ops_str="add,sub,mul,div",
        comp_ops_str="==,!=,<=,<",
    ),
)


ConfStore["lean_gen_rwalk_nat"] = RandomGenerationArgs(
    n_generations=10,
    rule_envs=["lean_nat"],
    env=EquationsEnvArgs(
        vtype="nat",
        positive=True,
        pos_hyps=True,
        unary_ops_str="sqrt",
        binary_ops_str="add,sub,mul,div,%,min,max,**,gcd,lcm",
        comp_ops_str="==,!=,<=,<,∣",
    ),
)

ConfStore["lean_gen_rwalk_int"] = RandomGenerationArgs(
    n_generations=10,
    rule_envs=["lean_int"],
    env=EquationsEnvArgs(
        vtype="int",
        positive=False,
        pos_hyps=False,
        unary_ops_str="neg,abs",
        binary_ops_str="add,sub,mul,div,%,min,max",
        comp_ops_str="==,!=,<=,<,∣",
    ),
)


def compute_generation_stats(rule_envs: List[str], used_theorems: Dict[str, int]):
    rule_names: List[str] = []
    for rule_env in rule_envs:
        rule_names += sorted([rule.name for rule in ALL_RULES[rule_env]])
    assert len(rule_names) == len(set(rule_names))
    all_rules: Dict[str, int] = {name: i for i, name in enumerate(rule_names)}
    assert all(k in all_rules for k in used_theorems.keys())
    counts = np.zeros((len(all_rules),), dtype=np.int64)
    for name, count in used_theorems.items():
        counts[all_rules[name]] = count
    usage, entropy = compute_usage_entropy(counts)
    stats = {
        "n_rules": len(counts),
        "n_used_rules": (counts != 0).sum().item(),
        "usage": usage,
        "entropy": entropy,
        "rule_names": rule_names,
        "counts": counts.tolist(),
    }
    return stats


if __name__ == "__main__":

    logger = create_logger(None)

    cfg: RandomGenerationArgs = cfg_from_cli(ConfStore["lean_gen_rwalk_real"])
    used_theorems = export_random_theorems(None, cfg)

    stats = compute_generation_stats(cfg.rule_envs, used_theorems)
    logger.info(f"Usage: {stats['usage']:.5f} -- Entropy: {stats['entropy']:.5f}")
    with open("gen_stats.json", "w") as f:
        json.dump(stats, f, sort_keys=True, indent=4)
