# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Union, List, Optional, Tuple, Set, Dict, Sequence
from dataclasses import dataclass
from logging import getLogger
import subprocess
import os
import copy
import gzip
import tempfile
import itertools


logger = getLogger()


class Node_f:
    def __init__(self, label: str, var_type: str, var_name: str):
        self.ltype = "$f"
        self.label = label
        self.var_type = var_type
        self.var_name = var_name
        self.is_syntactic = True

    @property
    def statement(self):
        return [self.var_type, self.var_name]

    @property
    def statement_str(self):
        return f"{self.var_type} {self.var_name}"

    @property
    def e_hyps(self):
        return {}

    @property
    def proof(self):
        return self.label

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"$f ({self.label}): {self.var_type} {self.var_name}"

    def set_nodes_and_depth(self):
        self.nodes = {cat: 0 for cat in ["syntactic", "no_syntactic"]}
        self.depth = {cat: 0 for cat in ["syntactic", "no_syntactic"]}


class Node_e:
    def __init__(self, label: str, statement_str: str):
        assert type(statement_str) is str
        self.ltype = "$e"
        self.label = label
        self.statement_str = statement_str
        self.is_syntactic = False
        self.var_type = "|-"

    @property
    def statement(self) -> List[str]:
        return self.statement_str.split()

    @property
    def e_hyps(self) -> Dict[str, str]:
        return {self.label: self.statement_str}

    @property
    def proof(self):
        return self.label

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"$e ({self.label}): {self.statement_str}"

    def set_nodes_and_depth(self):
        self.nodes = {cat: 0 for cat in ["syntactic", "no_syntactic"]}
        self.depth = {cat: 0 for cat in ["syntactic", "no_syntactic"]}


class Node_a_p:
    def __init__(
        self,
        ltype: str,
        label: str,
        disjoint: Set[Tuple[str, str]],
        substitutions: Dict[str, str],
        statement_str: str,
        children: Sequence["Node"],
    ):
        assert ltype in {"$a", "$p"}
        assert all(type(k) is str and type(v) is str for k, v in substitutions.items())
        assert type(statement_str) is str
        self.ltype = ltype
        self.label = label
        self.disjoint = disjoint
        self.substitutions = substitutions
        self.statement_str = statement_str
        self.children = children
        self.var_type = self.statement[0]
        self.is_syntactic = not self.var_type == "|-"
        assert self.var_type in ["|-", "wff", "class", "set", "setvar"]

    @property
    def statement(self) -> List[str]:
        return self.statement_str.split()

    @property
    def e_hyps(self) -> Dict[str, str]:
        if not hasattr(self, "_e_hyps"):
            self._e_hyps = dict(kv for c in self.children for kv in c.e_hyps.items())
        return self._e_hyps

    @property
    def proof(self):
        """
        WARNING: The forward proof is only valid if
        the syntactic nodes have not been removed.
        """
        if not hasattr(self, "_proof"):
            self._proof = " ".join([c.proof for c in self.children] + [self.label])
        return self._proof

    def __repr__(self):
        s = [f"{self.ltype} ({self.label}): {self.statement_str}"]
        # disjoint variables
        if len(self.disjoint) > 0:
            s.append("Disjoint variables:")
            for v1, v2 in self.disjoint:
                s.append(f"    {v1} {v2}")
        # substitutions
        if len(self.substitutions) > 0:
            s.append("Substitutions:")
            for k, v in self.substitutions.items():
                s.append(f"    {k}: {v}")
        # $e hypotheses
        if len(self.e_hyps) > 0:
            s.append("$e hypotheses:")
            for name, hyp in self.e_hyps.items():
                s.append(f"    {name}: {hyp}")
        # $f children
        s.append(f"Children: {len(self.children)}")
        return "\n".join(s)

    def __str__(self):
        return f"{self.ltype} ({self.label}): {self.statement_str}"

    def set_nodes_and_depth(self):
        """
        Compute the number of nodes in the node, and its maximal depth. Should be
        applied on an acyclic graph, otherwise raises a RecursionError. Cycles can
        be detected with the `has_cycle` function.

        The depth is always valid, however, the number of nodes is not accurate
        when the graph is not a tree. For instance, the following proof results
        in an incorrect number of nodes:
            mm_env.build_proof_tree(['wph', 'wph', 'wi', 'wph', 'wph', 'wi', 'wi'])
            wff ( ( ph -> ph ) -> ( ph -> ph ) )

        In the following acyclic graph, the nodes in D will be counted twice:
                A
               /|
              B C
              |/
              D
        """
        if not hasattr(self, "children") or len(self.children) == 0:
            self.nodes = {cat: 0 for cat in ["syntactic", "no_syntactic"]}
            self.depth = {cat: 0 for cat in ["syntactic", "no_syntactic"]}
            return
        assert not any(c is self for c in self.children)

        # compute nodes / depth for children
        for c in self.children:
            c.set_nodes_and_depth()

        # nodes / depth including syntactic nodes
        self.nodes = {}
        self.depth = {}
        self.nodes["syntactic"] = sum([c.nodes["syntactic"] for c in self.children], 1)
        self.depth["syntactic"] = 1 + max([c.depth["syntactic"] for c in self.children])

        # nodes / depth excluding syntactic nodes
        if self.statement[0] in ["class", "setvar", "set", "wff"]:
            self.nodes["no_syntactic"] = 0
            self.depth["no_syntactic"] = 0
        else:
            self.nodes["no_syntactic"] = sum(
                [c.nodes["no_syntactic"] for c in self.children], 1
            )
            self.depth["no_syntactic"] = 1 + max(
                [c.depth["no_syntactic"] for c in self.children]
            )


Node = Union[Node_a_p, Node_f, Node_e]


def decompress_all_proofs(mm_env):
    """
    Decompress all proofs.
    Export decompressed proofs into a file. Reload file if available.
    """
    database_path = mm_env.database_path
    assert len(mm_env.decompressed_proofs) == 0
    assert os.path.isfile(database_path)
    decompressed_path = os.path.join(
        os.path.dirname(database_path), "decompressed_proofs.gz"
    )

    if os.path.isfile(decompressed_path):

        logger.info(f"Reloading decompressed proofs from {decompressed_path} ...")
        with gzip.open(decompressed_path, "r") as f:
            lines = f.read().decode("utf-8").rstrip("\n").split("\n")
            logger.info(
                f"Found {len(lines)} decompressed proofs in {decompressed_path}"
            )
            for line in lines:
                label, proof = line.rstrip().split("\t")
                proof = proof.split()
                assert len(proof) >= 1 and label not in mm_env.decompressed_proofs
                if label in mm_env.compressed_proofs:
                    mm_env.decompressed_proofs[label] = proof

        assert len(mm_env.compressed_proofs) == len(mm_env.decompressed_proofs)
        logger.info(
            f"Reloaded {len(mm_env.decompressed_proofs)} "
            f"decompressed proofs from {decompressed_path}"
        )

    else:
        logger.info(f"Decompressing {len(mm_env.compressed_proofs)} proofs ...")
        for label, comp_proof in mm_env.compressed_proofs.items():
            label_type, assertion = mm_env.labels[label]
            assert label_type == "$p"
            if comp_proof[0] == "(":
                proof = mm_env.decompress_proof(assertion, comp_proof)
            else:
                proof = comp_proof  # Already in decompressed form
            mm_env.decompressed_proofs[label] = proof
        logger.info(
            f"Decompressed {len(mm_env.compressed_proofs)} proofs. "
            f"Exporting to {decompressed_path} ..."
        )

        with gzip.open(decompressed_path, "w") as f:
            for label, proof in mm_env.decompressed_proofs.items():
                f.write(f"{label}\t{' '.join(proof)}\n".encode("utf-8"))
        logger.info(
            f"Exported {len(mm_env.decompressed_proofs)} "
            f"decompressed proofs into {decompressed_path}"
        )


def count_unique_nodes(node, ignore_e_hyps: bool = False):
    """
    Count the number of unique nodes in a tree.
    """

    def traverse(node, seen):
        if ignore_e_hyps and node.ltype == "$e":
            return 0
        if id(node) in seen:
            return 0
        seen.add(id(node))
        if not hasattr(node, "children"):
            return 1
        return 1 + sum([traverse(c, seen) for c in node.children])

    return traverse(node, set())


def simplify_proof_tree(proof_tree):
    """
    Transforms proof trees into proof DAGs, according to node statements.
    Storing ~20x fewer nodes on average.
    TODO: this sometimes creates cycles, e.g. theorem "onfrALTlem2"
    """
    statement_to_node = {}

    def simplify(node):
        statement = " ".join(node.statement)
        to_ret = statement_to_node.get(statement, None)
        if to_ret is not None:
            return to_ret
        statement_to_node[statement] = node
        if not hasattr(node, "children") or len(node.children) == 0:
            return node

        new_children = []
        for child in node.children:
            new_children.append(simplify(child))
        node.children = new_children
        return node

    return simplify(proof_tree)


def has_cycle(root_node):
    """
    Detect whether a graph has a cycle.
    """

    def traverse(node, parents):
        node_id = id(node)
        if node_id in parents:
            return True
        if not hasattr(node, "children"):
            return False
        parents.add(node_id)
        for c in node.children:
            if traverse(c, parents):
                return True
        parents.remove(node_id)
        return False

    return traverse(root_node, parents=set())


class SimpleNode:
    """
    Use SimpleNode to store the number of childrens, the parents, and keep the graph
    structure intact. If a theorem uses the same children multiple times (e.g. in
    theorem "id", to show that "wff ph -> ph" from "wff ph" and "wff ph"), include
    the duplicates in the count.
    """

    def __init__(self, node: Node):
        self.node = node
        self.n_children = len(node.children) if isinstance(node, Node_a_p) else 0
        self.parents: List[
            Tuple[Optional["SimpleNode"], int]
        ] = []  # List[Tuple[parent_id, id_in_parent]]

        # Each child is (hypothesis_id, id_in_graph)
        # hypothesis_id is used to put hypothesis onto the stack in the right order.
        self.children: List[Tuple[int, int]] = []

    def __str__(self):
        return str(self.node)

    def __repr__(self):
        return str(self)


def remove_syntactic(root: Node):
    """
    Remove syntactic nodes from a graph.
    """
    root = copy.deepcopy(root)

    def traverse(node: Node):
        if node.is_syntactic:
            return None
        if node.ltype == "$e":
            return node
        assert isinstance(node, Node_a_p)
        new_children = [traverse(c) for c in node.children]
        node.children = [c for c in new_children if c is not None]
        return node

    return traverse(root)


def enumerate_nodes(
    root: Node, ignore_f_e: bool, ignore_empty: bool, no_syntactic: bool = False
) -> List[Node]:
    """
    Enumerate all nodes in a proof tree. Only select $a and $p nodes. Optionally
    ignore $f and $e nodes, and "empty" nodes that can be directly resolved.
    """
    assert not ignore_empty or ignore_f_e
    nodes_seen = {}

    def traverse(node: Node):
        statement = node.statement_str
        if statement in nodes_seen:
            return
        nodes_seen[statement] = node
        if not isinstance(node, Node_a_p):
            return
        for child in node.children:
            traverse(child)

    traverse(root)
    nodes = list(nodes_seen.values())

    # only select $a and $p nodes
    if ignore_f_e:
        nodes = [x for x in nodes if x.ltype == "$a" or x.ltype == "$p"]

    # optionally ignore syntactic nodes
    try:
        if no_syntactic:
            nodes = [x for x in nodes if not x.is_syntactic]
    except AttributeError as e:
        for node in nodes:
            print(dir(node))
        raise e

    # Is this dead code ? No Node_f, Node_e or Node_a_p has a f_hyps attribute
    # if ignore_empty:
    #     nodes = [
    #         node
    #         for node in nodes
    #         if not (
    #             len(node.f_hyps)
    #             == len(node.e_hyps)
    #             == len(node.substitutions)
    #             == len(node.children)
    #             == 0
    #         )
    #     ]

    return nodes


def random_topological_sort(root: Node, e_hyps_first: bool, rng) -> List[SimpleNode]:
    """
    Enumerate all nodes in a proof tree, in a random topological order.
    """
    candidates = list()

    # Since the graph is a DAG and not a tree we need to keep track of where we went
    all_nodes: Dict[str, SimpleNode] = dict()

    def traverse(node: SimpleNode, id_in_parent: int, parent: Optional[SimpleNode]):
        statement = node.node.statement_str
        if statement in all_nodes:
            all_nodes[statement].parents.append((parent, id_in_parent))
            return
        node.parents.append((parent, id_in_parent))
        all_nodes[statement] = node
        if node.n_children == 0:
            candidates.append(node)
            return
        assert isinstance(node.node, Node_a_p)
        for id_in_parent, child in enumerate(node.node.children):
            traverse(SimpleNode(child), id_in_parent, node)

    simple_root = SimpleNode(root)
    traverse(simple_root, 0, None)

    def _update_candidates(order_, candidates_):
        to_add = []
        for parent, id_in_parent in order_[-1].parents:
            if parent is None:
                continue
            parent.children.append(
                (id_in_parent, len(order) - 1)
            )  # this points to a node in the topological ordering
            parent.n_children -= 1
            assert parent.n_children >= 0
            if parent.n_children == 0:
                to_add.append(parent)
        if to_add:
            # add more randomness for order in which we append to candidates
            rng.shuffle(to_add)
        for node in to_add:
            candidates_.append(node)

    order = []
    if e_hyps_first:
        e_hyps = [c for c in candidates if c.node.ltype == "$e"]
        candidates = [c for c in candidates if c.node.ltype != "$e"]
        rng.shuffle(e_hyps)
        for hyp in e_hyps:
            order.append(hyp)
            _update_candidates(order, candidates)

    # Then pop random candidates until list is empty
    while candidates:
        # for O(1) random pop : select index, swap with end, pop end
        to_pop = rng.randint(0, len(candidates))
        candidates[to_pop], candidates[-1] = candidates[-1], candidates[to_pop]
        order.append(candidates[-1])
        assert candidates[-1].n_children == 0
        candidates.pop()
        _update_candidates(order, candidates)

    # Check that all nodes are generated, that the topological ordering is correct,
    # i.e. each parent appears after all its childrens, and that the root node is
    # the last generated node.
    assert all(v.n_children == 0 for v in all_nodes.values())
    assert len(order) == len(all_nodes)
    assert order[-1].node == root
    for i, node in enumerate(order):
        for c in node.children:
            assert c[1] < i, f"child {c} after parent {i}"

    return order


def to_latex(mm_env, tokens: List[str], all_math_mode: bool) -> str:
    """
    Convert a comment with math-mode (``) into valid latex.
    """
    # first make sure ` is always it's own token
    new_tokens = []
    for token in tokens:
        if "`" in token and len(token) > 1:
            other_parts = token.split("`")
            for i, x in enumerate(other_parts):
                if i > 0:
                    new_tokens.append("`")
                if len(x) > 0:
                    new_tokens.append(x)
        else:
            new_tokens.append(token)
    tokens = new_tokens
    math_mode = False
    for i, x in enumerate(tokens):
        if not all_math_mode:
            if tokens[i] == "`" and not math_mode:
                math_mode = True
                tokens[i] = "$"
                continue
            elif tokens[i] == "`" and math_mode:
                math_mode = False
                tokens[i] = "$"
                continue

        if math_mode or all_math_mode:
            tokens[i] = mm_env.latex_map.get(x, tokens[i])
        else:
            new_tok = tokens[i]
            for char in {"$", "_", "^", "&", "#"}:
                new_tok = new_tok.replace(char, f"\\{char}")
            tokens[i] = new_tok
    return " ".join(tokens)


def utils_fix_latex_map(mm_env):
    """
    Fixes obtained after a painful day of debugging...
    This should lead to a buildable tex file.
    """
    subst = {
        "#": r"\#",
        "&": r"\&",
        "..^": r"..\textasciicircum",
        "XX.": r"\times\times",
        "FF/": r"\Finv\Finv",
        "RRVec": r"\mathbb{R}-{\rm Vec}",
        "CCfld": r"\mathbb{C}_{\rm fld}",
        "ZZring": r"\mathbb{Z}_{\rm ring}",
        "RR^": r"\mathbb{R}\textasciicircum",
        "EEhil": r"\mathbb{E}_{\rm hil}",
        "'''": r"'''",
        "_Ind": r"\mathbbm{1}",
        "(|": r"\llparenthesis",
        "|)": r"\rrparenthesis",
        ".": r".",
        "iota": r"\iota",
        "iota_": r"\underline{\iota}",
        "log_": r"\log",
        "->..": r"\implies",
    }
    for k, v in subst.items():
        mm_env.latex_map[k] = v

    for k in mm_env.latex_map.keys():
        if r"\bb" in mm_env.latex_map[k]:
            mm_env.latex_map[k] = mm_env.latex_map[k].replace(r"\bb", r"\rm")


def get_mm_informal_data(mm_env, label):
    """
    Map Metamath theorem formal statements to formal and informal LaTeX.
    """
    if label in {"mathbox", "conventions", "natded", "stoweidlem59"}:
        return None

    # extract informal theorem description
    informal_latex = to_latex(mm_env, mm_env.comments[label], all_math_mode=False)

    # remove end notes
    for s in {
        "Contributed by",
        "New usage is discouraged.",
        "Proof modification is discouraged.",
    }:
        if informal_latex.find(s) > 0:
            informal_latex = informal_latex[: informal_latex.find(s) - 1].strip()

    # formal latex
    goal = mm_env.labels[label][1]["tokens"]
    hyps = mm_env.labels[label][1]["e_hyps"]
    hyps = sorted(hyps)  # same order as formal tokenization
    formal_latex = {
        "goal": to_latex(mm_env, goal, all_math_mode=True),
        "hyps": [to_latex(mm_env, hyp, all_math_mode=True) for hyp in hyps],
    }

    return {
        "label": label,
        "informal_latex": informal_latex,
        "formal_latex": formal_latex,
    }


def get_canonical_order(node: Node) -> List[Node]:
    r"""
    Postorder traversal of the dag. Respect topological order
    dag:
            0
          / / \
        1  /   4
       /\ /
      2 3

    should return [2, 3, 1, 4, 0]
    """
    order: List[Node] = []
    _postorder_traversal(node, order, handled=set([]))
    return order


def _postorder_traversal(node: Node, order: List[Node], handled: Set[str]):
    stat = node.statement_str
    if stat in handled:
        return
    children: List[Node] = getattr(node, "children", [])
    for child in children:
        _postorder_traversal(child, order, handled=handled)
    assert stat not in handled
    handled.add(stat)
    order.append(node)


def get_canonical_order_with_hyps_first(proof_tree: Node):
    ordered = get_canonical_order(proof_tree)
    return [n for n in ordered if n.ltype == "$e"] + [
        n for n in ordered if n.ltype != "$e"
    ]


def select_potential_goals(candidates: List[Node]) -> List[Node]:
    """
    Returns potential goals that are ascendant of the target node + the target node.
    Candidates needs to be in topological order
    The target node needs to be the first candidate
    """
    assert len(candidates) > 0
    if len(candidates) == 1:
        return candidates
    assert len(set(n.statement_str for n in candidates)) == len(candidates)
    potential_goals = [candidates[0]]
    ascendants_set = {candidates[0]}
    for i, node in enumerate(candidates):
        for children in getattr(node, "children", []):
            if children in ascendants_set:
                potential_goals.append(node)
                ascendants_set.add(node)
                break
    return potential_goals


def find_not_needed_nodes(order: List[Node], goal_idx: int) -> List[Node]:
    """
    All the nodes that are not descendants or ascendants of goal can be used
    for data augmentation of the proof. Indeed these nodes are noise for
    reaching the goal. They are possibly harder noise (since they use same variables)

    :param order: List[Node] topological ordering of graph
    :param goal_idx: index of the goal
    :return: noisy_nodes: List[Node]
    """
    goal = order[goal_idx]
    noise: List[Node] = []

    descendants = {goal}
    for node in reversed(order[: goal_idx + 1]):
        if node in descendants:
            for child in getattr(node, "children", []):
                descendants.add(child)
        else:
            # to respect top ordering
            noise.insert(0, node)

    ascendants = {goal}
    for node in order[goal_idx + 1 :]:
        for child in getattr(node, "children", []):
            if child in ascendants:
                ascendants.add(node)
                break
        else:
            noise.append(node)

    return noise


@dataclass
class MMProof:
    proof: List[str]
    statement: List[str]
    e_hyps: Dict[str, List[str]]
    disjoints: Set[Tuple[str, str]]

    def get_mand_disj(self) -> Set[Tuple[str, str]]:
        # we don't need mand_vars, we can gather all tokens in statement and hyps
        statement_and_hyps_toks = set(
            self.statement + [t for h in self.e_hyps.values() for t in h]
        )
        mand_disj = {
            (x, y)
            for x, y in self.disjoints
            if x in statement_and_hyps_toks and y in statement_and_hyps_toks
        }
        return mand_disj


def check_proof(mm_proof: MMProof, set_mm: str) -> bool:
    disj = "\n".join([f"$d {a} {b} $." for a, b in mm_proof.disjoints])
    hyps = "\n".join(
        [f'{key} $e {" ".join(hyp)} $.' for key, hyp in mm_proof.e_hyps.items()]
    )
    with tempfile.NamedTemporaryFile() as f:
        final = f"""$[ {set_mm} $]
        {disj}
        {hyps}
        computed_proof_mine $p {' '.join(mm_proof.statement)} $= {' '.join(mm_proof.proof)} $.\n"""
        f.write(final.encode("utf-8"))
        f.flush()
        filename = os.path.split(f.name)[-1]
        output = subprocess.check_output(
            [
                "YOUR_PATH/metamath/metamath",
                f"read {filename}",
                f"verify proof computed_proof_mine",
                "exit",
            ],
            cwd=tempfile.gettempdir(),
        ).decode("utf-8")
    error = "?Error" in output
    if not error:
        return True
    else:
        raise RuntimeError(output)


def get_false_missing_hyps(node: Node, max_neg: int) -> List[List[str]]:
    """
    Return false statements (i.e. goals with missing $e hypotheses).
    These are most likely to be wrong, and are used to train the critic.
    """
    # TODO: fix import
    from evariste.backward.env.metamath.graph import MMTheorem

    # if the node has no $e hypotheses, the statement is always true
    if len(node.e_hyps) == 0:
        return []

    # check that all hypotheses are effectively used in the proof
    assert all(hyp_name in node.proof for hyp_name in node.e_hyps.keys())
    e_hyps = list(node.e_hyps.values())

    negatives = []

    # for each number of hypotheses `n < len(e_hyps)`
    # start from len(e_hyps) - 1 (should be the biggest offenders)
    for n in range(len(e_hyps))[::-1]:

        # for each subset of `n` hypotheses
        for sub_e_hyps in itertools.combinations(e_hyps, n):

            hyps = [(None, h) for h in sub_e_hyps]
            neg = MMTheorem(conclusion=node.statement_str, hyps=hyps).tokenize()
            negatives.append(neg)

            if len(negatives) >= max_neg:
                break
        if len(negatives) >= max_neg:
            break

    assert len(negatives) <= max_neg
    return negatives


def node_tok(i: int):
    return f"<NODE_{i}>"


def reward_quantile_tok(i: int):
    return f"<REWARD_QUANTILE_{i}>"
