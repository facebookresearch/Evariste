# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pathlib
import logging
import timeit
from lark.exceptions import LarkError

from lark import Lark

from evariste.syntax.deprecated.grammars import recent_earley, holophrasm_earley
from evariste.syntax.deprecated.syntax_engine import SyntaxEngine
from evariste.envs.mm.env import get_parser


syntatic_vars = {"wff_var", "class_var", "setvar_var"}
syntatic = set(["setvar", "wff", "class", "set"])


from evariste.syntax.deprecated.earley.earley_parser import PyGrammar


class EarleyParser:
    def __init__(self, parser_type, grammar_module=recent_earley):
        """
        Parameters
        ----------
        grammar_module : module, optional
            One of the autogenerated modules in the grammar folder corresponding
            to the set.mm database you want to parse.
            Current either recent_earley or holophrasm_earley
        """
        dir = pathlib.Path(__file__).parent.absolute()
        if parser_type == "lark":
            self.module = grammar_module
            logging.debug(f"Loading Lark Earley grammar: {self.module.grammar_file}")
            with open(str(dir / "grammars" / self.module.grammar_file), "r") as file:
                grammar = file.read()
            self.lark = Lark(grammar, propagate_positions=True)
            logging.debug("Lark Earley grammar loaded")
        elif parser_type == "cpp":
            path = str(dir / "earley" / f"grammar_{grammar_module}.in").encode("utf-8")
            logging.info(f"Loading cpp grammar from {path}")
            self.grammar = PyGrammar(path)
        else:
            raise RuntimeError(f"Unexpected parser type {parser_type}")

    def parse(self, expression):
        """Parse a string or list of tokens into a Lark parse tree.

        Parameters
        ----------
        expression : str or list of str
            Metamath expression
        Returns
        -------
        Tree
            A Lark parse tree
        """
        if isinstance(expression, list):
            expression = " ".join(expression)
        if hasattr(self, "lark"):
            tree = self.lark.parse(expression)
            self._label_nodes(tree)
            return tree
        else:
            return self.grammar.parse(expression.encode("utf-8"))

    def parse_to_proof(self, tree, assertion=None):
        """Extract a syntax proof from a parse tree
        Parameters
        ----------
        tree : Tree
            Metamath proof tree as output from .parse
        assertion : dict, optional
            This object corresponds to a assertion dictionary as returned
            by the metamath proof engine. It is only needed if the tree uses
            f hypotheses that are scoped locally rather than globally.

        Returns
        -------
        list of str
            A proof in list-of-labels format as used in the Metamath engine
        """
        if tree.data in syntatic:
            return self.parse_to_proof(tree.children[0], assertion)

        elif tree.data in syntatic_vars:
            var = tree.children[0].value
            # Check local f hyps before checking global ftable
            if assertion is not None:
                for flbl, fhyp in zip(assertion["f_hyps_labels"], assertion["f_hyps"]):
                    ftype, fvar = fhyp
                    if fvar == var:
                        return [flbl]

            ftype, flbl = self.module.ftable[var]
            return [flbl]
        else:
            label = tree.data
            # Sub proofs have to be reordered to match the order of the hypotheses
            order = self.module.axiom_hyp_order[label]
            proof_lists = [None] * len(tree.children)
            for cidx, child in enumerate(tree.children):
                subproof = self.parse_to_proof(child, assertion)
                proof_lists[order[cidx]] = subproof

            proof = []
            for subproof in proof_lists:
                for step in subproof:
                    proof.append(step)
            proof.append(label)
            return proof

    def _label_nodes(self, tree):
        if tree.data in syntatic:  # Top level
            self._label_nodes(tree.children[0])
        elif tree.data in syntatic_vars:  # Leaf
            pass
        else:  # Relabel then recurse on children
            tree.data = self.module.label_lookup[tree.data]
            for child in tree.children:
                self._label_nodes(child)


###################################

if __name__ == "__main__":
    ##### Parses a small subset of syntax from metamath proofs for testing purposes
    logging.basicConfig(level=logging.INFO)

    # parse arguments
    arg_parser = get_parser()
    args = arg_parser.parse_args()

    args.stop_label = "axltadd"  #'axltadd' #'nfmod' # For testing
    args.database_path = "resources/metamath/holophrasm/set.mm"

    parser = EarleyParser(holophrasm_earley)
    parse_tree = parser.parse(" ".join(["wff", "-.", "ph"]))

    # create Metamath instance
    mm_env = SyntaxEngine(filepath=args.database_path, args=args)

    logging.info("Extracting WFF by running proof engine in validate mode ...")
    mm_env.process()

    nwffs = len(mm_env.wffs)
    nonmatch = 0
    failed_parses = 0
    start_time = timeit.default_timer()
    last_timed = 0

    logging.info("Parsing expressions ...")

    for i, wff in enumerate(mm_env.wffs):
        if wff["tokens"][0] not in syntatic:
            continue
        ftbl = {}
        for lbl, expr in wff["active_f_labels"].items():
            ftbl[" ".join(expr)] = lbl
        existing_proof = wff["proof_tree"].proof(ftbl)

        try:
            expr = " ".join(wff["tokens"])
            parse_tree = parser.parse(expr)
            # parse_tree = parser.parse(wff['tokens'])

            proof = parser.parse_to_proof(parse_tree, wff)

            if proof != existing_proof:

                # There are a few cases where the proofs won't match
                # but the parse is still correct. We just skip those for now
                if ("weq" in existing_proof and "wceq" in proof) or (
                    "wceq" in existing_proof and "weq" in proof
                ):
                    continue
                if ("wel" in existing_proof and "wcel" in proof) or (
                    "wcel" in existing_proof and "wel" in proof
                ):
                    continue
                if ("wsb" in existing_proof and "wsbc" in proof) or (
                    "wsbc" in existing_proof and "wsb" in proof
                ):
                    continue

                nonmatch += 1
                existing_str = " ".join(existing_proof)
                generated_str = " ".join(proof)
                skip = []

                if existing_str not in skip:
                    logging.info(f"Expression: {expr}")
                    logging.info(f"existing: {existing_str}")
                    logging.info(f"generated: {generated_str}")
                    logging.info("Doesn't match base proof")
        except LarkError:
            failed_parses += 1
            proof = []
        if i % min(50, nwffs - 1) == 0:
            # code you want to evaluate
            elapsed = timeit.default_timer() - start_time
            logging.info(
                f"({i}) processed of {nwffs}, nonmatches: {nonmatch}, failed_parses: {failed_parses} time: {elapsed}, per expression: {elapsed/(1+i-last_timed)}"
            )
            start_time = timeit.default_timer()
            last_timed = i

    logging.info("Finished")
