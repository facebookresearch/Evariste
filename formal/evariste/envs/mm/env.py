# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import Counter, OrderedDict, defaultdict
from typing import Tuple, List, Dict
import os
import re
import logging
import argparse
import itertools
import gzip

from params.params import bool_flag
from evariste import json as json
from evariste.envs.mm.utils import (
    Node_f,
    Node_e,
    Node_a_p,
    utils_fix_latex_map,
    decompress_all_proofs,
)
from evariste.logger import create_logger, LOGGING_LEVELS
from evariste.envs.mm.assertion import Assertion


E_HYP_NAMES = "E_HYP_%i"


logger = logging.getLogger()


def decompress_ints(s):
    """
    Uncompress proof integers.
    """
    assert type(s) is str
    ints = []
    i = 0
    for c in s:
        if c == "Z":
            ints.append(-1)
        elif "A" <= c <= "T":
            i = 20 * i + ord(c) - ord("A") + 1
            ints.append(i - 1)
            i = 0
        elif "U" <= c <= "Y":
            i = 5 * i + ord(c) - ord("U") + 1
        else:
            raise Exception(f"Invalid compressed ints: {s}")
    return ints


class Frame:
    def __init__(self):

        self.c = set()
        self.v = set()
        self.d = set()
        self.f_labels = OrderedDict()
        self.e_labels = OrderedDict()


class FrameStack:
    def __init__(self):
        self.frames = []

    def __len__(self):
        return len(self.frames)

    def push(self):
        self.frames.append(Frame())

    def pop(self):
        self.frames.pop()
        assert len(self.frames) >= 1

    def add_c(self, x):
        frame = self.frames[-1]
        if len(self.frames) != 1:
            raise Exception(f"Constant {x} was not added in the outermost block!")
        if self.has_c(x) or self.has_v(x):
            raise Exception(f"Constant {x} is already declared in scope!")
        frame.c.add(x)

    def add_v(self, x):
        frame = self.frames[-1]
        if self.has_c(x) or self.has_v(x):
            raise Exception(f"Variable {x} is already declared in scope!")
        frame.v.add(x)

    def add_d(self, tokens):
        assert len(tokens) >= 2
        self.frames[-1].d.update(
            (
                (min(x, y), max(x, y))
                for x, y in itertools.product(tokens, tokens)
                if x != y
            )
        )

    def add_f(self, label, var_type, var_name):
        if not self.has_v(var_name):
            raise Exception(f"Variable name in $f not defined: {var_name}")
        if not self.has_c(var_type):
            raise Exception(f"Constant type in $f not defined: {var_type}")
        if any(
            label in frame.f_labels or label in frame.e_labels for frame in self.frames
        ):
            raise Exception(f"$f label {label} already defined in scope")
        if any(
            var_name == name
            for frame in self.frames
            for _, name in frame.f_labels.values()
        ):
            raise Exception(f"Variable {var_name} in $f already defined in scope")
        frame = self.frames[-1]
        frame.f_labels[label] = [var_type, var_name]

    def add_e(self, label, statement):
        if any(
            label in frame.f_labels or label in frame.e_labels for frame in self.frames
        ):
            raise Exception(f"$e label {label} already defined in scope")
        frame = self.frames[-1]
        frame.e_labels[label] = statement

    def has_c(self, name):
        return any(name in f.c for f in self.frames)

    def has_v(self, name):
        return any(name in f.v for f in self.frames)

    def has_d(self, x, y):
        return any((min(x, y), max(x, y)) in f.d for f in self.frames)

    def get_f_label(self, _var_name):
        assert type(_var_name) is str
        for frame in reversed(self.frames):
            for label, (_, var_name) in frame.f_labels.items():
                if var_name == _var_name:
                    return label
        raise Exception(f"{_var_name} not in floating hypotheses")

    def get_e_label(self, _statement):
        assert type(_statement) is list
        for frame in reversed(self.frames):
            for label, statement in frame.e_labels.items():
                if statement == _statement:
                    return label
        raise Exception(f"{_statement} not in essential hypotheses")

    def make_assertion(self, tokens, label=None):
        return Assertion(self, tokens, label=label)


class MetamathEnv:
    def __init__(
        self,
        filepath=None,
        buffer=None,
        start_label=None,
        stop_label=None,
        rename_e_hyps=True,
        decompress_proofs=True,
        verify_proofs=True,
        log_level="debug",
    ):
        # database path
        self.database_path = filepath

        # set log level
        self.log_level = LOGGING_LEVELS[log_level]

        # either a file is provided, or a buffer of tokens
        assert (filepath is None) != (buffer is None)
        if filepath is None:
            self.buffer = buffer
            self.info(f"Created environment. {len(buffer)} tokens in buffer.")
        else:
            self.buffer = self.load_file(filepath)

        # start / stop at specific labels
        self.start_label = None if start_label == "" else start_label
        self.stop_label = None if stop_label == "" else stop_label

        # optionally rename $e hypotheses / skip proof decompression / verification
        self.rename_e_hyps = rename_e_hyps
        self.decompress_proofs = decompress_proofs
        self.verify_proofs = verify_proofs
        assert decompress_proofs or not verify_proofs

        # initialize frame stack, labels and proofs
        self.fs = FrameStack()
        self.labels: Dict[str, Tuple[str, Assertion]] = OrderedDict()
        self.axioms: Dict[str, Tuple[str, Assertion]] = OrderedDict()
        self.comments: Dict[str, List[str]] = OrderedDict()
        self.latex_map: Dict[str, str] = OrderedDict()
        self.compressed_proofs: Dict[str, List[str]] = OrderedDict()
        self.decompressed_proofs = OrderedDict()

    def parse_label_dag(self):
        """
        Compute DAG.
        Export DAG into a file. Reload file if available.
        """
        dag_path = os.path.join(os.path.dirname(self.database_path), "dag.gz")

        # if the DAG has been dumped, reload it
        if os.path.isfile(dag_path):
            logger.info(f"Reloading DAG from {dag_path} ...")
            with gzip.open(dag_path, "rt", encoding="ascii") as f:
                dag = json.load(f)
            dag = {k: set(v) for k, v in dag.items()}
            logger.info(f"Reloaded DAG for {len(dag)} proofs.")
            return dag

        # otherwise, build it and export it
        logger.info(f"File {dag_path} does not exist. Building DAG ...")

        # decompress proofs
        if len(self.decompressed_proofs) == 0:
            logger.info(f"DAG requires decompressed proofs. Reloading ...")
            decompress_all_proofs(self)

        # build DAG
        logger.info("Building DAG ...")
        dag = defaultdict(set)
        for theorem, proof in self.decompressed_proofs.items():
            for label in proof:
                if "E_HYP_" not in label:
                    dag[label].add(theorem)
            if theorem not in dag:
                dag[theorem] = set()

        # export DAG
        logger.info(f"Exporting DAG to {dag_path} ...")
        with gzip.open(dag_path, "wt", encoding="ascii") as f:
            json.dump({k: list(v) for k, v in dag.items()}, f)
        logger.info(f"Exported DAG for {len(dag)} proofs to {dag_path}")

        return dag

    def debug(self, msg):
        if self.log_level <= logging.DEBUG:
            logger.debug(msg)

    def info(self, msg):
        if self.log_level <= logging.INFO:
            logger.info(msg)

    def warning(self, msg):
        if self.log_level <= logging.WARNING:
            logger.warning(msg)

    def error(self, msg):
        if self.log_level <= logging.ERROR:
            logger.error(msg)

    def log(self, msg, level):
        if self.log_level <= level:
            logger.log(level, msg)

    def load_file(self, filepath):

        assert os.path.isfile(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            r = f.read()

        self.info(f"Read {len(r.split())} tokens from {filepath}")

        r = re.sub(r"/\*(.*?)\*/", r"", r, flags=re.DOTALL)  # remove C style comments

        # split into tokens
        tokens = r.split()

        # self.loaded_files.add(filepath)
        self.info(f"Read {len(tokens)} tokens from {filepath}")

        return tokens[::-1]

    def read_tokens_until(self, stop="$."):
        """
        Read tokens from buffer until the end of statement "$.".
        """
        tokens = []
        while True:
            if len(self.buffer) == 0:
                raise Exception(f"Unexpected end of buffer")
            tok = self.buffer.pop()
            if tok == stop:
                break
            tokens.append(tok)
        return tokens

    def rename_essential_hyps(self, compressed_proof, assertion):
        """
        Rename $e hypotheses.
        """
        # create remapping
        remapping = {}
        for i, label in enumerate(assertion["active_e_labels"].keys()):
            assert label not in self.labels, label
            assert label not in self.fs.frames[0].e_labels, label
            remapping[label] = E_HYP_NAMES % len(remapping)

        # apply remapping
        assertion["active_e_labels"] = {
            remapping.get(k, k): v for k, v in assertion["active_e_labels"].items()
        }
        assertion["e_hyps_labels"] = [
            remapping.get(k, k) for k in assertion["e_hyps_labels"]
        ]
        compressed_proof = [remapping.get(k, k) for k in compressed_proof]

        return compressed_proof, assertion

    def parse_typesetting(self, definition):
        cur_tok = 0
        while cur_tok < len(definition):
            if definition[cur_tok] == "latexdef":
                end_tok = cur_tok
                while definition[end_tok][-1] != ";":
                    end_tok += 1
                this_def = " ".join(definition[cur_tok + 1 : end_tok + 1])
                cur_char, cur_parse, in_parse, last_delim = 0, 0, False, None
                parsed = [""]
                delimiters = {'"', "'"}
                while this_def[cur_char] != ";" or this_def[cur_char - 1] == "\\":
                    this_char = this_def[cur_char]
                    if this_char in delimiters:
                        if not in_parse:
                            in_parse = True
                            last_delim = this_char
                        elif in_parse and this_char == last_delim:
                            in_parse = False
                            last_delim = None
                            cur_parse += 1
                            parsed.append("")
                        elif in_parse and this_char != last_delim:
                            parsed[cur_parse] += this_char
                    elif in_parse:
                        parsed[cur_parse] += this_char
                    cur_char += 1
                src, target = parsed[0], "".join(parsed[1:])
                self.latex_map[src] = target
                # print(f"{src} -> {target} // {this_def} // {parsed}")
                cur_tok = end_tok + 1
            else:
                cur_tok += 1

    @staticmethod
    def remove_comments(tokens):
        this_str = " ".join(tokens)
        this_str = re.sub(r"\$\((.*?)\$\)", r"", this_str, flags=re.DOTALL)
        return this_str.split()

    def _process_start(self):
        pass

    def _process_token_start(self):
        pass

    def process(self):
        """
        Process buffer.
        """
        logger.info(
            f"========== Processing buffer ({len(self.buffer)} tokens) =========="
        )
        self.fs.push()
        label = None
        last_comment = None
        started_process = self.start_label is None
        self._process_start()
        while len(self.buffer) > 0:
            self._process_token_start()

            tok = self.buffer.pop()

            # comments are saved into last_comment to be pushed with corresponding later next
            if tok == "$(":
                last_comment = self.read_tokens_until("$)")
                last_comment_str = " ".join(last_comment)
                # remove HTML and <A> here to deal with unclosed <HTML> tags
                last_comment_str = re.sub(
                    r"<A(.*?)>(.*?)</A>", r"", last_comment_str, flags=re.DOTALL
                )
                last_comment_str = re.sub(
                    r"<HTML>(.*?)</HTML>", r"", last_comment_str, flags=re.DOTALL
                )  # remove HTML (which appears in comments and is annoying to parse)
                last_comment = last_comment_str.split()
                if "$t" in last_comment:
                    # parse this comment further to get metamath <-> latex mappings
                    self.parse_typesetting(last_comment)
                continue

            # open block
            if tok == "${":
                self.fs.push()
                self.log(f"========== New block (level {len(self.fs)})", 19)
                continue

            # close block
            if tok == "$}":
                self.fs.pop()
                self.log(f"========== End block (level {len(self.fs) + 1})", 19)
                continue

            # $c
            if tok == "$c":
                tokens = self.read_tokens_until()
                tokens = self.remove_comments(tokens)
                for tok in tokens:
                    self.fs.add_c(tok)
                self.log(f"$c: {', '.join(tokens)}", 18)
                continue

            # $v
            if tok == "$v":
                tokens = self.read_tokens_until()
                tokens = self.remove_comments(tokens)
                for tok in tokens:
                    self.fs.add_v(tok)
                self.log(f"$v: {', '.join(tokens)}", 18)
                continue

            # $d
            if tok == "$d":
                tokens = self.read_tokens_until()
                tokens = self.remove_comments(tokens)
                self.fs.add_d(tokens)
                self.log(f"$d: {', '.join(tokens)}", 18)
                continue

            # check label exists
            if tok in ["$f", "$e", "$a", "$p"]:
                if label is None:
                    raise Exception(f"{tok} must have a label")
                assert label not in self.labels

            # $f
            if tok == "$f":
                tokens = self.read_tokens_until()
                tokens = self.remove_comments(tokens)
                if len(tokens) != 2:
                    raise Exception("$f must contain a typecode and a variable")
                self.fs.add_f(label, tokens[0], tokens[1])
                self.log(f"$f ({label}): name={tokens[1]} type={tokens[0]}", 18)
                label = None
                continue

            # $e
            if tok == "$e":
                tokens = self.read_tokens_until()
                tokens = self.remove_comments(tokens)
                self.fs.add_e(label, tokens)
                self.log(f"$e ({label}): {' '.join(tokens)}", 18)
                label = None
                continue

            # $a
            if tok == "$a":
                assert label is not None and label not in self.labels
                tokens = self.read_tokens_until()
                tokens = self.remove_comments(tokens)
                assertion = self.fs.make_assertion(tokens, label)
                self.labels[label] = ("$a", assertion)
                self.axioms[label] = ("$a", assertion)
                if last_comment is not None:
                    self.comments[label] = last_comment
                    self.log(
                        f"$a ({label}): {' '.join(tokens)} // {' '.join(last_comment)}",
                        19,
                    )
                else:
                    self.log(f"$a ({label}): {' '.join(tokens)}", 19)
                label = None
                last_comment = None
                continue

            # $p
            if tok == "$p":
                tokens = self.read_tokens_until()
                tokens = self.remove_comments(tokens)
                if "$=" not in tokens:
                    raise Exception("$p must contain proof after $=")
                i = tokens.index("$=")
                statement = tokens[:i]
                comp_proof = tokens[i + 1 :]
                assertion = self.fs.make_assertion(statement, label)
                if self.rename_e_hyps:
                    comp_proof, assertion = self.rename_essential_hyps(
                        comp_proof, assertion
                    )
                self.log(f"$p ({label}): {' '.join(statement)}", 19)
                self.compressed_proofs[label] = comp_proof
                self.labels[label] = ("$p", assertion)
                if last_comment is not None:
                    self.comments[label] = last_comment
                if started_process and self.decompress_proofs:
                    if comp_proof[0] == "(":
                        self.log(f'===== Decompressing "{label}" proof', 19)
                        decomp_proof = self.decompress_proof(assertion, comp_proof)
                    else:
                        decomp_proof = comp_proof  # Already in decompressed form
                    self.decompressed_proofs[label] = decomp_proof
                if started_process and self.verify_proofs:
                    self.log(
                        f"===== Verifying \"{label}\" statement: {' '.join(statement)}",
                        19,
                    )
                    self.verify(decomp_proof, assertion)
                label = None
                last_comment = None
                continue

            # label
            assert tok[0] != "$", f"unexpected token {tok}"
            assert label is None, label
            assert tok not in self.labels
            label = tok

            # start label (optional)
            if self.start_label is not None and label == self.start_label:
                logger.info(f"Start label: {self.start_label} - Starting processing...")
                started_process = True

            # stop label (optional)
            if self.stop_label is not None and label == self.stop_label:
                logger.info(f"Stop label: {self.stop_label} - Stopping processing...")
                break

        # label stats
        logger.info(f"========== Read {len(self.labels)} labels ==========")
        label_types = Counter(k for (k, _) in self.labels.values())
        for k, v in label_types.items():
            logger.info(f"{k} labels: {v}")

        # fix LaTeX
        utils_fix_latex_map(self)

    def apply_subst(self, statement: List[str], subst: Dict[str, List[str]]):
        """
        Substitute variables in a statement.
        """
        assert type(statement) is list and all(type(x) is str for x in statement)
        assert all(
            type(k) is str and type(v) is list and all(type(t) is str for t in v)
            for k, v in subst.items()
        )
        result = []
        for tok in statement:
            if tok in subst:
                result.extend(subst[tok])
            else:
                result.append(tok)
        self.log(
            f"Applying substitutions:\n"
            f"{os.linesep.join((f'    {k} -> ' + ' '.join(v)) for k, v in subst.items())}\n"
            f"Old statement: {' '.join(statement)}\n"
            f"New statement: {' '.join(result)}",
            13,
        )
        return result

    def decompress_proof(self, assertion, proof, return_subproofs=False):
        """
        The integers that the upper-case letters correspond to are mapped to labels as follows:

        - If the statement being proved has m mandatory hypotheses, integers 1 through m correspond
          to the labels of these hypotheses in the order shown by the show statement ... / full
          command, i.e., the RPN order of the mandatory hypotheses.

        - Integers m + 1 through m + n correspond to the labels enclosed in the parentheses of the
          compressed proof, in the order that they appear, where n is the number of those labels.

        - Integers m + n + 1 on up don’t directly correspond to statement labels but point to proof
          steps identified with the letter Z, so that these proof steps can be referenced later in
          the proof. Integer m + n + 1 corresponds to the first step tagged with a Z, m + n + 2 to
          the second step tagged with a Z, etc. When the compressed proof is converted to a normal
          proof, the entire subproof of a step tagged with Z replaces the reference to that step.
        """
        # retrieve labels
        labels = assertion["f_hyps_labels"] + assertion["e_hyps_labels"]
        m = len(labels)

        active_f_labels = assertion["active_f_labels"]
        active_e_labels = assertion["active_e_labels"]

        def get_label(label):
            if label in self.labels:
                return self.labels[label]
            if label in active_f_labels:
                return ("$f", active_f_labels[label])
            if label in active_e_labels:
                return ("$e", active_e_labels[label])
            raise Exception(f"Label {label} not found!")

        # split labels / compressed proof, uncompress proof ints
        assert proof[0] == "("
        ep = proof.index(")")
        labels += proof[1:ep]
        n = len(proof[1:ep])
        compressed_ints = "".join(proof[ep + 1 :])
        decompressed_ints = decompress_ints(compressed_ints)

        self.log(
            f"m: {m}\nn: {n}\nlabels: {labels}\n"
            f"Compressed ints: {compressed_ints}\n"
            f"Uncompressed ints: {str(decompressed_ints).replace(' ', '')}",
            15,
        )

        # - `proof_ints` (List[int]) is the decompressed proof
        # - `subproofs` (List[List[int]]) is the list of subtrees that have been indexed
        # - `prev_proofs` (List[List[int]]) is the list of subtrees that are being constructed, and
        #    will have its last element corresponing to the currently substree being processed.
        #    At the end, it will be a list with a single element corresponding to `proof_ints`
        proof_ints = []
        subproofs = []
        prev_proofs = []

        for i in decompressed_ints:
            if i == -1:
                assert prev_proofs[-1] not in subproofs
                subproofs.append(prev_proofs[-1])
            elif 0 <= i < m:
                prev_proofs.append([i])
                proof_ints.append(i)
            elif m <= i < m + n:
                proof_ints.append(i)
                step_type, step_data = get_label(labels[i])
                if step_type in ("$a", "$p"):
                    n_mand_hyps = len(step_data["f_hyps"]) + len(step_data["e_hyps"])
                    if n_mand_hyps > 0:
                        new_prev_proof = [
                            j for p in prev_proofs[-n_mand_hyps:] for j in p
                        ] + [i]
                        prev_proofs = prev_proofs[:-n_mand_hyps]
                    else:
                        new_prev_proof = [i]
                    prev_proofs.append(new_prev_proof)
                else:
                    prev_proofs.append([i])
            elif i >= m + n:
                subproof = subproofs[i - (m + n)]
                proof_ints += subproof
                prev_proofs.append(subproof)
            else:
                raise Exception(f"Unexpected int value: {i} (n = {n}, m = {m})")

        self.log(
            f"Subproofs       : {str(subproofs).replace(' ', '')}\n"
            f"Previous proofs : {str(prev_proofs).replace(' ', '')}\n"
            f"Proof ints      : {str(proof_ints).replace(' ', '')}",
            16,
        )

        # sanity check
        assert len(prev_proofs) == 1 and prev_proofs[0] == proof_ints

        # optionally return subproofs
        proof_labels = [labels[i] for i in proof_ints]
        if return_subproofs:
            return proof_labels, [[labels[i] for i in sp] for sp in subproofs]
        else:
            return proof_labels

    def _verify_start(self, assertion, stack):
        pass

    def _verify_hypothesis_after(self, label, step_type, step_data):
        pass

    def _verify_proposition_before(self, label, step_type, step_data):
        pass

    def _verify_proposition_after(self, label, formula, n_pop):
        pass

    def verify(self, proof, assertion):
        """
        An expression is a sequence of math symbols. A substitution map associates a set of
        variables with a set of expressions. It is acceptable for a variable to be mapped
        to an expression containing it. A substitution is the simultaneous replacement of all
        variables in one or more expressions with the expressions that the variables map to.

        A proof is scanned in order of its label sequence.
        - If the label refers to an active hypothesis ($e or $f), the expression in the
        hypothesis is pushed onto a stack.
        - If the label refers to an assertion ($a or $p), a (unique) substitution must exist that,
        when made to the mandatory hypotheses of the referenced assertion, causes them to match
        the topmost (i.e. most recent) entries of the stack, in order of occurrence of the mandatory
        hypotheses, with the topmost stack entry matching the last mandatory hypothesis of the
        referenced assertion. As many stack entries as there are mandatory hypotheses are then
        popped from the stack.

        The same substitution is made to the referenced assertion, and the result is pushed
        onto the stack.

        After the last label in the proof is processed, the stack must have a single entry
        that matches the expression in the $p statement containing the proof.

        If two variables replaced by a substitution exist in a mandatory $d statement of the
        assertion referenced, the two expressions resulting from the substitution must satisfy
        the following conditions. First, the two expressions must have no variables in common.
        Second, each possible pair of variables, one from each expression, must exist in an
        active $d statement of the $p statement containing the proof.
        """
        assert proof[0] != "("
        stack = []
        statement = assertion["tokens"]

        active_vars = assertion["active_vars"]
        active_disj = assertion["active_disj"]
        active_f_labels = assertion["active_f_labels"]
        active_e_labels = assertion["active_e_labels"]

        def get_label(label):
            if label in self.labels:
                return self.labels[label]
            if label in active_f_labels:
                return ("$f", active_f_labels[label])
            if label in active_e_labels:
                return ("$e", active_e_labels[label])
            raise Exception(f"Label {label} not found!")

        self._verify_start(assertion, stack)

        for label in proof:

            step_type, step_data = get_label(label)
            assert step_type in {"$a", "$p", "$e", "$f"}

            if step_type in ("$a", "$p"):
                self._verify_proposition_before(label, step_type, step_data)
                mand_disj = step_data["mand_disj"]
                f_hyps = step_data["f_hyps"]
                e_hyps = step_data["e_hyps"]
                tokens = step_data["tokens"]

                self.log(
                    f"$p label \"{label}\": {' '.join(tokens)}\n"
                    f"\tmand_disj: {mand_disj}\n"
                    f"\t$f hyps: {f_hyps}\n"
                    f"\t$e hyps: {e_hyps}",
                    14,
                )

                n_pop = len(f_hyps) + len(e_hyps)
                sp = len(stack) - n_pop
                if sp < 0:
                    raise Exception("Stack underflow")

                # substitutions
                subst = {}

                # add floating hypotheses
                for var_type, var_name in f_hyps:
                    entry = stack[sp]
                    if entry[0] != var_type:
                        raise Exception(
                            f"Stack entry ({entry}) does not match variable "
                            f'"{var_name}" of type "{var_type}".'
                        )
                    subst[var_name] = entry[1:]
                    sp += 1

                # check there is no disjoint variable violation
                for x, y in mand_disj:
                    x_vars = set(subst[x]) & active_vars
                    y_vars = set(subst[y]) & active_vars
                    for x, y in itertools.product(x_vars, y_vars):
                        if x == y or (min(x, y), max(x, y)) not in active_disj:
                            raise Exception(f"Disjoint violation: {x}, {y}")

                # add essential hypotheses
                for h in e_hyps:
                    entry = stack[sp]
                    subst_h = self.apply_subst(h, subst)
                    if entry != subst_h:
                        raise Exception(
                            f"Stack entry {entry} does not match hypothesis {subst_h}"
                        )
                    sp += 1

                # remove hypotheses from the stack
                del stack[len(stack) - n_pop :]

                # perform substitution
                formula = self.apply_subst(tokens, subst)
                stack.append(formula)

                self._verify_proposition_after(label, formula, n_pop)

            elif step_type in ("$e", "$f"):
                self.log(f"{step_type} label \"{label}\" {' '.join(step_data)}", 14)
                stack.append(step_data)
                self._verify_hypothesis_after(label, step_type, step_data)

        if len(stack) != 1:
            raise Exception(f"Stack does not have 1 entry at the end: {stack}")

        if stack[0] != statement:
            raise Exception(
                f"The proven assertion does not match the statement. "
                f'Expected: "{statement}", proved "{stack[0]}"'
            )

    def build_proof_tree(
        self,
        proof,
        assertion=None,
        return_all_stacks=False,
        stat_history=None,
        verify_final=True,
    ):
        """
        Does the same thing as `verify`, but also builds the proof tree.
        `assertion` is optional, i.e. the context will be the outermost frame of
        the environment if no assertion is provided.
        Optionally return all intermediary stacks.
        """
        assert proof[0] != "("
        stack = []
        if return_all_stacks:
            all_stacks = []

        # build proof context
        if assertion is None:
            active_vars = set.union(*(frame.v for frame in self.fs.frames))
            active_disj = set.union(*(frame.d for frame in self.fs.frames))
            active_f_labels = dict(
                kv for frame in self.fs.frames for kv in frame.f_labels.items()
            )
            active_e_labels = dict(
                kv for frame in self.fs.frames for kv in frame.e_labels.items()
            )
        else:
            active_vars = assertion["active_vars"]
            active_disj = assertion["active_disj"]
            active_f_labels = assertion["active_f_labels"]
            active_e_labels = assertion["active_e_labels"]

        # check no label overlap
        assert not any(label in self.labels for label in active_f_labels.keys())
        assert not any(label in self.labels for label in active_e_labels.keys())
        assert not any(label in active_f_labels for label in active_e_labels.keys())

        def get_label(label):
            if label in self.labels:
                return self.labels[label]
            if label in active_f_labels:
                return ("$f", active_f_labels[label])
            if label in active_e_labels:
                return ("$e", active_e_labels[label])
            raise Exception(f"Label {label} not found!")

        if stat_history is not None:
            short_proof = []
            subproof_len = 0

        for i, label in enumerate(proof):

            step_type, step_data = get_label(label)
            assert step_type in {"$a", "$p", "$e", "$f"}

            if return_all_stacks and stat_history is None:
                all_stacks.append([" ".join(x.statement) for x in stack])

            if stat_history is not None:

                # check if the next labels belong to statement history
                just_finished = False
                if subproof_len == 0:
                    subproof, available = stat_history.get_subproof(proof[i:])
                    if subproof is not None:
                        subproof_len = subproof["proof_len"]
                        assert subproof_len >= 1
                        assert subproof["proof_tok"] == proof[i : i + subproof_len]
                        subproof_len -= 1
                        short_proof.append(
                            {
                                "is_subproof": True,
                                "available": available,
                                "statement": subproof["statement"],
                                "tokens": subproof["proof_tok"],
                            }
                        )
                    else:
                        short_proof.append(
                            {
                                "is_subproof": False,
                                "available": available,
                                "token": label,
                            }
                        )
                    if return_all_stacks:
                        all_stacks.append([" ".join(x.statement) for x in stack])
                else:
                    subproof_len -= 1
                    just_finished = subproof_len == 0
                assert subproof_len >= 0

                if just_finished:
                    assert step_type in ("$a", "$p")

            if step_type in ("$a", "$p"):

                mand_disj = step_data["mand_disj"]
                f_hyps = step_data["f_hyps"]
                e_hyps = step_data["e_hyps"]
                f_hyps_labels = step_data["f_hyps_labels"]
                e_hyps_labels = step_data["e_hyps_labels"]
                tokens = step_data["tokens"]

                self.log(
                    f"$p label \"{label}\": {' '.join(tokens)}\n"
                    f"\tmand_disj: {mand_disj}\n"
                    f"\t$f hyps: {f_hyps}\n"
                    f"\t$e hyps: {e_hyps}",
                    14,
                )

                n_pop = len(f_hyps) + len(e_hyps)
                sp = len(stack) - n_pop
                if sp < 0:
                    raise Exception("Stack underflow")

                # substitutions
                subst = {}

                # add floating hypotheses
                for var_type, var_name in f_hyps:
                    node = stack[sp]
                    # retrieve node type / data
                    if type(node) is Node_a_p:
                        node_type = node.statement[0]
                        node_data = node.statement[1:]
                    elif type(node) is Node_f:
                        node_type = node.var_type
                        node_data = [node.var_name]
                    else:
                        raise Exception("Invalid node type!")
                    # verify type
                    if node_type != var_type:
                        raise Exception(
                            f"Stack entry ({node}) does not match variable "
                            f'"{var_name}" of type "{var_type}".'
                        )
                    subst[var_name] = node_data
                    sp += 1

                # check there is no disjoint variable violation
                for x, y in mand_disj:
                    x_vars = set(subst[x]) & active_vars
                    y_vars = set(subst[y]) & active_vars
                    for x, y in itertools.product(x_vars, y_vars):
                        if x == y or (min(x, y), max(x, y)) not in active_disj:
                            raise Exception(f"Disjoint violation: {x}, {y}")

                # add essential hypotheses
                for h in e_hyps:
                    node = stack[sp]
                    assert type(node) is Node_a_p or type(node) is Node_e
                    subst_h = self.apply_subst(h, subst)
                    if node.statement != subst_h:
                        raise Exception(
                            f"Stack entry {node.statement} does not match hypothesis {subst_h}"
                        )
                    sp += 1

                # remove hypotheses from the stack
                children = stack[len(stack) - n_pop :]
                del stack[len(stack) - n_pop :]

                # perform substitution
                stack.append(
                    Node_a_p(
                        ltype=step_type,
                        label=label,
                        disjoint=mand_disj,
                        substitutions={k: " ".join(v) for k, v in subst.items()},
                        statement_str=" ".join(self.apply_subst(tokens, subst)),
                        children=children,
                    )
                )

                if stat_history is not None:
                    stat_history.add_node(stack[-1])

            elif step_type == "$e":
                self.log(f"$e label \"{label}\" {' '.join(step_data)}", 14)
                stack.append(Node_e(label, " ".join(step_data)))

            elif step_type == "$f":
                assert len(step_data) == 2
                self.log(f"$f label \"{label}\" {' '.join(step_data)}", 14)
                stack.append(Node_f(label, step_data[0], step_data[1]))

        if len(stack) != 1:
            raise Exception(f"Stack does not have 1 entry at the end: {stack}")

        if (
            verify_final
            and assertion is not None
            and "tokens" in assertion
            and stack[0].statement != assertion["tokens"]
        ):
            raise Exception(
                f"The proven assertion does not match the statement. "
                f"Expected: \"{assertion['tokens']}\", "
                f'proved "{stack[0].statement}"'
            )

        result = {"root_node": stack[0]}
        if return_all_stacks:
            assert len(all_stacks) == len(
                proof if stat_history is None else short_proof
            )
            result["all_stacks"] = all_stacks
        if stat_history is not None:
            result["compressed_proof"] = short_proof

        return result


def get_parser():

    parser = argparse.ArgumentParser(description="Metamath")
    parser.add_argument(
        "--database_path",
        type=str,
        default="resources/metamath/DATASET_3/set.mm",
        help="Metamath database path",
    )
    parser.add_argument(
        "--start_label",
        type=str,
        default="",
        help="Decompress and verify from a given label",
    )
    parser.add_argument("--stop_label", type=str, default="", help="Stop label")
    parser.add_argument(
        "--rename_e_hyps", type=bool_flag, default=True, help="Rename $e hypotheses"
    )
    parser.add_argument(
        "--decompress_proofs", type=bool_flag, default=True, help="Decompress proofs"
    )
    parser.add_argument(
        "--verify_proofs", type=bool_flag, default=True, help="Verify proofs"
    )
    parser.add_argument(
        "--api_log_level", type=str, default="debug", help="API log level"
    )
    parser.add_argument(
        "--console_log_level", type=str, default="debug", help="Console log level"
    )
    parser.add_argument("--debug", type=bool_flag, default=False, help="Debug")
    return parser


if __name__ == "__main__":

    # parse arguments
    parser = get_parser()
    args = parser.parse_args()
    assert os.path.isfile(args.database_path)

    # create logger
    logger = create_logger(None, console_level=args.console_log_level)

    # create Metamath instance
    mm_env = MetamathEnv(
        filepath=args.database_path,
        start_label=args.start_label,
        stop_label=args.stop_label,
        rename_e_hyps=args.rename_e_hyps,
        decompress_proofs=args.decompress_proofs,
        verify_proofs=args.verify_proofs,
        log_level=args.api_log_level,
    )
    mm_env.process()

    # debugger
    if args.debug:
        import ipdb

        ipdb.set_trace()
        for label, (label_type, assertion) in mm_env.labels.items():
            if label_type != "$p":
                continue
            proof = mm_env.proofs[label]
            if not args.decompress_proofs:
                proof = mm_env.decompress_proof(assertion, proof)
            mm_env.verify(proof, assertion)
