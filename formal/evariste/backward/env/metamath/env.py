# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Set, List, Dict, Optional, Tuple
from pathlib import Path
import itertools
from logging import getLogger

from evariste.backward.env.core import BackwardEnv, EnvGen
from evariste.backward.env.worker import SyncEnv
from evariste.backward.graph import (
    Token,
    Proof,
    NonPicklableProof,
    UnMaterializedTheorem,
)
from evariste.backward.env.worker import EnvWorker, TacticJobResult
from evariste.backward.env.core import InvalidTactic

from evariste.datasets.metamath import MetamathDatasetConf
from evariste.backward.graph import Theorem, Tactic
from evariste.backward.env.metamath.graph import (
    MMTheorem,
    MMTactic,
    MMSubstitutions,
    MM_UNK_LABEL,
    MM_HYPS_MISMATCH,
    MM_NEW_DISJ,
    MM_DISJ_V,
    MM_WRONG,
    MM_SUB_SYNTAX,
)
from evariste.envs.mm.env import MetamathEnv
from evariste.envs.mm.utils import MMProof
from evariste.syntax.parser import Parser, get_parser
from evariste.syntax.subst_finder import SubstFinder
from evariste.backward.dag_factory import get_dag
from evariste.utils import find_descendents


logger = getLogger()


class MMEnvWorker(EnvWorker):
    def __init__(
        self, dataset: MetamathDatasetConf, mm_env: Optional[MetamathEnv] = None
    ):
        self.dataset = dataset
        self.parser: Optional[Parser] = None
        self.subst_finder: Optional[SubstFinder] = None
        self.proven: Optional[Dict[MMTheorem, str]] = None
        self.mm_env: Optional[MetamathEnv] = mm_env
        self.active_vars: Optional[Set[str]] = None

    def init(self, rank: Optional[int] = None) -> None:
        train_labels = set()
        if self.dataset.data_dir is not None:
            train_split_file = Path(self.dataset.data_dir) / "split.train"
            assert train_split_file.exists(), f"{train_split_file} missing"

            with open(train_split_file, "r") as train_split:
                for line in train_split.readlines():
                    train_labels.add(line.strip())

        if self.mm_env is None:
            self.mm_env = MetamathEnv(
                filepath=self.dataset.database_path,
                rename_e_hyps=True,
                decompress_proofs=False,
                verify_proofs=False,
                log_level="info",
            )
        assert self.mm_env.database_path == self.dataset.database_path

        self.mm_env.process()
        self.proven = {}
        self.dag = get_dag(self.dataset)
        self.forbidden: Dict[str, List[Token]] = {}

        # record axioms
        for label, (type, assertion) in self.mm_env.labels.items():
            if type == "$a" or (type == "$p" and label in train_labels):
                if len(assertion.e_hyps) == 0:
                    self.proven[MMTheorem(assertion.tokens_str, [])] = label

        self.active_vars = set.union(*(frame.v for frame in self.mm_env.fs.frames))
        self.parser = get_parser(self.dataset.parser)
        self.subst_finder = SubstFinder(self.mm_env, self.parser)

    def is_true(self, theorem: MMTheorem, ignore_hyps: bool = False) -> bool:
        """
        Raises ParseError if syntactic and wrong.
        @param theorem:
        @return: true if node is proven. False if not proven and not syntactic.
        """
        if not ignore_hyps and theorem.conc_in_hyp():
            return True
        tokens = theorem.conclusion.split()
        if len(tokens) > 0 and tokens[0] in {"class", "wff", "setvar", "set"}:
            assert self.parser is not None
            self.parser.parse(tokens)
            return True
        assert self.proven is not None
        proving_label = self.proven.get(MMTheorem(theorem.conclusion, []), None)
        if theorem.train_label not in self.forbidden:
            assert theorem.train_label is not None
            self.forbidden[theorem.train_label] = find_descendents(
                self.dag, theorem.train_label
            )
        if (
            proving_label is not None
            and proving_label not in self.forbidden[theorem.train_label]
        ):
            try:
                assert self.mm_env is not None
                return self._valid_sub(
                    theorem,
                    label=proving_label,
                    sub={
                        x: x for x in self.mm_env.labels[proving_label][1]["mand_vars"]
                    },
                )[0]
            except InvalidTactic:
                return False

        return False

    def check_syntax(self, theorem: MMTheorem) -> Tuple[bool, bool]:
        tokens = theorem.conclusion.split()
        syntax_prefixes = {"class", "wff", "setvar", "set"}
        is_syntactic = len(tokens) > 0 and tokens[0] in syntax_prefixes
        if tokens[0] == "|-":
            tokens[0] = "wff"
        assert self.parser is not None
        is_valid = self.parser.has_valid_syntax(tokens)
        return is_syntactic, is_valid

    @staticmethod
    def _substr(tokens: List[Token], sub: MMSubstitutions) -> str:
        """
            Applies the substitutions, then flatten the resulting list of list of token
        """
        return " ".join(sub.get(x, x) for x in tokens)

    def _valid_sub(
        self,
        theorem: MMTheorem,
        label: str,
        sub: MMSubstitutions,
        ignore_not_initialized: bool = False,
    ) -> Tuple[bool, Set[Tuple[Token, Token]]]:
        """
        Checks that a (label, subs) can be applied to a goal.
            - label needs to be a valid self.mm_env label
            - substitutions can only happen in f hypothesis of the theorem
            - substituting into label must yield goal.goal
        If a disjoint statement is missing, verify that it would not add un-necessary
        constraints to the theorem statement.
        """
        if not ignore_not_initialized:
            assert None not in [
                theorem.mand_disj,
                theorem.mand_vars,
            ], f"Theorem has no disj/vars {theorem.mand_vars} {theorem.mand_disj}"
        assert self.mm_env is not None
        if label not in self.mm_env.labels:
            raise InvalidTactic(f"{MM_UNK_LABEL} {label}")
        theo = self.mm_env.labels[label][1]

        # substitutions must only concerns free variables (in f_hyps)
        f_hyps_names = {name for _, name in theo["f_hyps"]}
        if sub.keys() != f_hyps_names:
            raise InvalidTactic(
                f"{MM_HYPS_MISMATCH} {sorted(sub.keys())} != {sorted(f_hyps_names)}"
            )

        # same as code in metamath.env  TODO: factor
        to_add_disj = set()
        assert self.active_vars is not None
        for x, y in theo["mand_disj"]:
            x_vars = set(sub[x].split()) & self.active_vars
            y_vars = set(sub[y].split()) & self.active_vars
            for sub_x, sub_y in itertools.product(x_vars, y_vars):
                if sub_x == sub_y:
                    raise InvalidTactic(f"{MM_DISJ_V} {x} -> {sub[x]} {y} -> {sub[y]}")
                new_disj = (min(sub_x, sub_y), max(sub_x, sub_y))
                assert theorem.mand_disj is not None and theorem.mand_vars is not None
                if new_disj not in theorem.mand_disj:
                    if sub_x in theorem.mand_vars and sub_y in theorem.mand_vars:
                        raise InvalidTactic(f"{MM_NEW_DISJ} $d {sub_x}, {sub_y}")
                    to_add_disj.add(new_disj)
        new_conclusion = self._substr(theo["tokens"], sub)
        if new_conclusion != theorem.conclusion:
            raise InvalidTactic(
                f"{MM_WRONG} {new_conclusion} is not {theorem.conclusion}."
            )
        return True, to_add_disj

    def _apply_tactic(
        self, theorem: MMTheorem, tactic: MMTactic
    ) -> Tuple[Optional[List[MMTheorem]], Set[Tuple[Token, Token]]]:
        """
        If substitution is invalid, return None.
        Otherwise, apply substitution and return a list of new MMGoals to solve.
        Also update the validity of the tactic.
        """
        assert tactic.is_valid, tactic
        try:
            valid, disj = self._valid_sub(theorem, tactic.label, tactic.subs)
        except InvalidTactic as e:
            tactic.is_valid = False
            tactic.error_msg = f"{e} -- label: {tactic._label}, substs: {tactic._subs}"
            return None, set()
        assert self.mm_env is not None
        assertion = self.mm_env.labels[tactic.label][1]
        subgoals = [
            MMTheorem(
                self._substr(hyp, tactic.subs),
                theorem.hyps,
                mand_disj=theorem.mand_disj,
                mand_vars=theorem.mand_vars,
                train_label=theorem.train_label,
            )
            for hyp in list(assertion["f_hyps"]) + assertion["e_hyps"]
        ]
        return subgoals, disj

    def apply_tactic(
        self,
        theorem: Theorem,
        tactic_tokens: Optional[List[Token]],
        tactic: Optional[Tactic] = None,
        keep_if_hyp: bool = False,
    ) -> TacticJobResult:

        # tactic
        assert (tactic_tokens is None) != (tactic is None)
        if tactic is None:
            assert tactic_tokens is not None
            tactic = MMTactic.from_tokens(tactic_tokens)
        else:
            assert isinstance(tactic, MMTactic)

        # Apply SubstFinder
        if tactic.is_valid:
            labels = {tactic.label}
            assert self.subst_finder is not None
            label_to_subs = self.subst_finder.process(theorem.conclusion, labels)
            if label_to_subs[tactic.label] is not None:
                tactic.update_subs(label_to_subs[tactic.label])

        if not tactic.is_valid:
            return TacticJobResult(tactic, children=[])
        assert isinstance(theorem, MMTheorem) and isinstance(tactic, MMTactic)
        children, new_disj = self._apply_tactic(theorem, tactic)
        if children is None:
            return TacticJobResult(tactic, children=[])

        # check syntax / remove syntactic subgoals
        all_valid = True
        new_children = []
        for child in children:
            # if one is invalid, the tactic is invalid
            is_syntactic, is_valid = self.check_syntax(child)
            if not is_valid:
                tactic.is_valid = False
                tactic.error_msg = f"{MM_SUB_SYNTAX} {child}"
                all_valid = False
                break
            # only add subgoals that are not true / not syntactic
            if not is_syntactic and not self.is_true(child, ignore_hyps=keep_if_hyp):
                new_children.append(child)

        if not all_valid:
            return TacticJobResult(tactic, children=[])
        children = new_children
        return TacticJobResult(tactic, children=children)

    def to_mm_proof(self, proof: Proof) -> MMProof:
        def visit(prover_proof):
            theorem, tactic, children = prover_proof
            th_toks = theorem.conclusion.split()
            if th_toks[0] in {"class", "wff", "setvar", "set"}:
                parse_tree = self.parser.parse(th_toks)
                return self.parser.parse_to_proof(parse_tree), set()
            elif tactic is None:
                if theorem.conc_in_hyp():
                    return [theorem.get_proving_hyp()], set()
                proving_label = self.proven.get(MMTheorem(theorem.conclusion, []), None)
                assert proving_label is not None
                tactic = MMTactic(
                    proving_label,
                    {x: x for x in self.mm_env.labels[proving_label][1]["mand_vars"]},
                )

            subgoal_to_child = {}
            for child in children:
                subgoal_to_child[child[0]] = child

            children, new_disj = self._apply_tactic(theorem, tactic)
            proof = []
            if children is not None:
                for child in children:
                    if child in subgoal_to_child:
                        extra_proof, extra_disj = visit(subgoal_to_child[child])
                    else:
                        extra_proof, extra_disj = visit([child, None, []])
                    proof += extra_proof
                    new_disj = new_disj.union(extra_disj)
            proof += [tactic.label]
            return proof, new_disj

        final_proof, extra_disj = visit(proof)

        assert not isinstance(proof, NonPicklableProof) and isinstance(
            proof[0], MMTheorem
        )
        assert proof[0].mand_disj is not None
        mm_proof = MMProof(
            proof=final_proof,
            statement=proof[0].conclusion.split(),
            e_hyps={f"E_HYP_{i}": h.split() for i, (_, h) in enumerate(proof[0].hyps)},
            disjoints=proof[0].mand_disj.union(extra_disj),
        )
        mand_disj = mm_proof.get_mand_disj()
        assert mand_disj.issubset(proof[0].mand_disj), "Too many disjoint variables"
        return mm_proof

    def materialize_theorem(self, th: UnMaterializedTheorem) -> Theorem:
        assert self.mm_env is not None
        theo = self.mm_env.labels[th.label][1]
        hyps = [(None, " ".join(x)) for x in theo["active_e_labels"].values()]
        return MMTheorem(
            conclusion=theo.tokens_str,
            hyps=hyps,
            mand_vars=theo["mand_vars"],
            mand_disj=theo["mand_disj"],
            train_label=th.label,
        )


class MMEnvGenerator(EnvGen):
    def __init__(self, dataset: MetamathDatasetConf):
        self.dataset = dataset

    def __call__(self):
        return BackwardEnv(expander_env=SyncEnv(MMEnvWorker(self.dataset)))

    def close(self):
        logger.info("Closed MMEnvGenerator")
