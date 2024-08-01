# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple, List, Dict, NamedTuple, Optional, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
import os
import pickle
import itertools
import numpy as np

from evariste.envs.mm.env import MetamathEnv, logger
from evariste.envs.mm.assertion import Assertion
from evariste.envs.mm.utils import (
    count_unique_nodes,
    Node,
    MMProof,
    Node_e,
    Node_a_p,
    get_canonical_order,
    enumerate_nodes,
)
from evariste.forward.common import (
    ForwardGoal,
    GenerationHistory,
    ForwardTactic,
    EnvInfo,
)

from evariste.utils import find_descendents
from evariste.backward.env.metamath import MMTheorem
from evariste.metrics import Avg
from evariste.syntax.parser import Parser, ParseError

TRMS_TO_FILTER = {"dummylink", "iin2", "iin3"}


#################
# MM Forward Goal
#################

# TODO: create subclass?


def forward_goal_from_assertion(
    assertion: Assertion, forbidden: Set[str]
) -> ForwardGoal:
    e_hyps = [" ".join(h) for h in assertion.e_hyps]
    goal = " ".join(assertion.tokens)
    return ForwardGoal(
        statement=goal,
        e_hyps=e_hyps,
        forbidden=forbidden,
        mand_disj=assertion.mand_disj,
        label=assertion.label,
    )


def get_mand_vars(goal: ForwardGoal, active_vars: Set[str]) -> Set[str]:
    return (
        set(goal.statement.split() + [tok for h in goal.e_hyps for tok in h.split()])
        & active_vars
    )


def forward_goal_from_mm_proof(
    mm_proof: MMProof, forbidden: Optional[Set[str]]
) -> "ForwardGoal":
    e_hyps = [" ".join(h) for h in mm_proof.e_hyps.values()]
    goal = " ".join(mm_proof.statement)
    return ForwardGoal(
        statement=goal,
        e_hyps=e_hyps,
        forbidden=forbidden,
        mand_disj=mm_proof.get_mand_disj(),
    )


def build_forward_goals(
    data_dir: str, split: str, mm_env: MetamathEnv, debug: bool
) -> List[ForwardGoal]:

    if debug and not data_dir.endswith("/100"):
        new_data_dir = data_dir + "/100"
        if Path(new_data_dir).exists():
            data_dir = new_data_dir

    if split.startswith("minif2f_"):
        # don't like it but "_minif2f" is in another dataset
        from evariste.datasets.metamath import NEW3_MINIF2F
        from evariste.model.data.envs.metamath import MetamathDataEnvironment

        dataset = NEW3_MINIF2F
        mm_env = MetamathDataEnvironment.build_mm_env(dataset.database_path)
        labels = load_split(dataset.data_dir, split)
        trm_names = sorted(labels)
        dag = None
    else:
        splits = load_splits(data_dir)
        dag = mm_env.parse_label_dag()
        trm_names = sorted([name for name in splits[split]])

    prev_len = len(trm_names)
    trm_names = [name for name in trm_names if name not in TRMS_TO_FILTER]
    if len(trm_names) < prev_len:
        print(
            f"WARNING: removed {prev_len - len(trm_names)} trms "
            f"(among {TRMS_TO_FILTER})"
        )

    goals = []
    for name in trm_names:
        forbidden = set(find_descendents(dag, name))
        assert forbidden is not None  # and len(forbidden) > 0
        goal = forward_goal_from_assertion(mm_env.labels[name][1], forbidden=forbidden)
        goals.append(goal)
    return goals


def build_forward_goals_from_samples(data_path: str, debug: bool) -> List[ForwardGoal]:
    assert os.path.isfile(data_path)

    goals = []
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            goal = MMTheorem.from_tokens(line.rstrip().split())
            goal = ForwardGoal(
                statement=goal.conclusion,
                e_hyps=[hyp for _, hyp in goal.hyps],
                forbidden=set(),
                mand_disj=None,
                label=f"theorem_{i}",
            )
            goals.append(goal)
            if debug and len(goals) >= 100:
                break

    print(f"Loaded {len(goals)} goals from {data_path}")
    return goals


def build_forward_goals_from_proof_steps(
    mm_data_dir: str,
    database_path: str,
    split: str,
    use_forbidden: bool = True,
    use_subgoals: bool = False,
) -> List[ForwardGoal]:
    data = load_splitted_proof_trees(mm_data_dir)[split]

    goals = []

    forbiddens = {}
    if use_forbidden:
        logger.info("Loading mm_env")
        mm_env = MetamathEnv(
            filepath=database_path,
            start_label="",
            stop_label="",
            rename_e_hyps=True,
            decompress_proofs=True,
            verify_proofs=False,
            log_level="info",
        )
        mm_env.process()
        logger.info("Computing forbidden")
        dag = mm_env.parse_label_dag()
        for name, _ in data:
            forbidden = set(find_descendents(dag, name))
            assert len(forbidden) > 0, name
            forbiddens[name] = forbidden

    logger.info("Extracting goals from proofs")
    for trm_name, root in data:
        if use_subgoals:
            nodes = get_canonical_order(root)
        else:
            nodes = [root]
        for node in nodes:
            if node.ltype == "$e":
                continue
            if use_forbidden:
                forbidden = forbiddens[trm_name]
            else:
                forbidden = None
            e_hyps = list(node.e_hyps.values())
            new_goal = ForwardGoal(
                statement=node.statement_str,
                e_hyps=e_hyps,
                forbidden=forbidden,
                label=trm_name,
                mand_disj=node.disjoint,
            )
            goals.append(new_goal)
    return goals


def extract_nodes(
    gen: GenerationHistory, run_uuid: str, gen_id: int
) -> Tuple[List[str], List[Dict], List[Node]]:
    nodes = history_to_mm_head_nodes(gen)
    node_uuids = [f"{run_uuid}_{gen_id}_{i}" for i in range(len(nodes))]
    node_infos = []
    for node in nodes:
        node.set_nodes_and_depth()
        size = count_unique_nodes(node, ignore_e_hyps=True)
        node_infos.append({"size": size, "depth": node.depth["no_syntactic"]})
    return node_uuids, node_infos, nodes


def load_splits(mm_data_dir) -> Dict[str, List[str]]:
    splits = {}
    for split in ["train", "valid", "test"]:
        splits[split] = load_split(mm_data_dir, split)
    return splits


def load_split(mm_data_dir: str, split: str) -> List[str]:
    path = os.path.join(mm_data_dir, f"split.{split}")
    assert os.path.isfile(path), path
    with open(path, "r", encoding="utf-8") as f:
        labels = [x.rstrip() for x in f]
    assert len(labels) == len(set(labels))
    return labels


def load_proof_trees(mm_data_dir) -> Dict[str, Node]:
    path = os.path.join(mm_data_dir, "proof_trees.pkl")
    assert os.path.isfile(path)
    with open(path, "rb") as f:
        all_proof_trees = pickle.load(f)
    return all_proof_trees


def load_splitted_proof_trees(mm_data_dir) -> Dict[str, List[Tuple[str, Node]]]:
    splits = load_splits(mm_data_dir)
    print("Beginning to load all proof trees")
    trees = load_proof_trees(mm_data_dir)
    print(f"Finished to load all proof trees: {len(trees)}")
    splitted = {}
    for split, names in splits.items():
        names = set(names)
        splitted[split] = [(k, v) for k, v in trees.items() if k in names]
    return splitted


#######################
# MM Generation History
#######################

# TODO: create subclass?


def history_to_mm_nodes(gen: GenerationHistory) -> List[Node]:
    e_hyps = sorted(gen.goal.e_hyps)
    e_nodes: List[Node] = [
        Node_e(label=f"E_HYP_{i}", statement_str=stat) for i, stat in enumerate(e_hyps)
    ]
    e_hyps_dict = {n.label: n.statement_str for n in e_nodes}
    stat2node = {n.statement_str: n for n in e_nodes}

    fwd_graph = gen.forward_graph()

    nodes = e_nodes
    for step in gen.forward_steps():
        assert isinstance(step.tactic, MMForwardTactic)
        assert isinstance(step.env_info, MMEnvInfo)
        children = [stat2node[fwd_graph.nodes[cix]] for cix in step.children]
        disjoints = set.union(
            step.env_info.new_disjoints,
            *[c.disjoint for c in children if c.ltype != "$e"],
        )
        node = Node_a_p(
            ltype="$p",  # by default
            label=step.tactic.label,
            disjoint=get_mand_disj(
                step.statement, e_hyps=e_hyps_dict, disjoints=disjoints
            ),
            substitutions=step.tactic.substitutions,
            statement_str=step.statement,
            children=children,
        )
        stat2node[node.statement_str] = node
        nodes.append(node)
    return nodes


def history_to_mm_head_nodes(gen: GenerationHistory) -> List[Node]:
    heads = set([])
    for node in history_to_mm_nodes(gen):
        if node.ltype == "$e":
            continue
        children = node.children
        for children in children:
            if children in heads:
                heads.remove(children)
        heads.add(node)
    return list(heads)


def get_mand_disj(
    statement: str, e_hyps: Dict[str, str], disjoints: Set[Tuple[str, str]]
) -> Set[Tuple[str, str]]:
    # we don't need mand_vars, we can gather all tokens in statement and hyps
    statement_and_hyps_toks = set(
        statement.split() + [t for h in e_hyps.values() for t in h.split()]
    )
    mand_disj = {
        (x, y)
        for x, y in disjoints
        if x in statement_and_hyps_toks and y in statement_and_hyps_toks
    }
    return mand_disj


def flatten_proof(node: Node) -> List[Node]:
    """
    Flatten a proof into a 1d array of nodes respecting the topological order,
    with the leftmost children first.
    """
    order = []
    statement2id = {}
    _flatten_proof(node, order, statement2id)
    return order


def _flatten_proof(node: Node, order: List[Node], statement2id: Dict[str, int]):
    stat = node.statement_str
    if stat in statement2id:
        return
    children: List[Node] = getattr(node, "children", [])
    for child in children:
        _flatten_proof(child, order, statement2id=statement2id)
    assert stat not in statement2id
    statement2id[stat] = len(order)
    order.append(node)


class ProofError(Exception):
    pass


class DisjointError(ProofError):
    pass


class SyntacticError(ProofError):
    pass


class SyntacticNodeError(ProofError):
    pass


class ParserProofError(Exception):
    pass


class MaybeMMProof(NamedTuple):
    proof: Optional[MMProof]
    err: Optional[ProofError]

    def success(self) -> bool:
        return self.proof is not None


def get_mm_proof(
    root_node: Node,
    env: MetamathEnv,
    parser: Optional[Parser],
    syntactic_proof: bool = True,
) -> MMProof:
    if parser is None:
        assert not syntactic_proof
    active_vars = set.union(*(frame.v for frame in env.fs.frames))
    proof_tokens = []
    syntactic_proofs: Dict[str, List[str]] = {}
    disjoints = set()
    e_hyps = {}

    def _build_proof(node_):
        if node_.ltype == "$e":
            proof_tokens.append(node_.label)
            e_hyps[node_.label] = node_.statement
            return
        elif node_.ltype == "$f":
            raise NotImplementedError
        if node_.is_syntactic:
            raise SyntacticNodeError(
                f"Generated a syntactic node: {node_.statement_str}"
            )
        assertion: Assertion = env.labels[node_.label][1]
        subs = node_.substitutions
        for x, y in assertion.mand_disj:
            assert isinstance(subs[x], str)
            assert isinstance(subs[y], str)
            x_vars = set(subs[x].split()) & active_vars
            y_vars = set(subs[y].split()) & active_vars
            for sub_x, sub_y in itertools.product(x_vars, y_vars):
                if sub_x == sub_y:
                    raise DisjointError(
                        f"Disjoint not respected for label {node_.label}"
                    )
                new_disj = (min(sub_x, sub_y), max(sub_x, sub_y))
                disjoints.add(new_disj)

        if syntactic_proof:
            f_hyps = assertion.f_hyps
            for var_type, var_name in f_hyps:
                expression = [var_type] + node_.substitutions[var_name].split()
                key = " ".join(expression)
                if key not in syntactic_proofs:
                    try:
                        tree = parser.parse(expression)
                    except ParseError:
                        raise SyntacticError(expression)
                    try:
                        proof = parser.parse_to_proof(tree)
                    except KeyError as err:
                        raise ParserProofError(" ".join(expression), err)
                    syntactic_proofs[key] = proof
                synt_proof = syntactic_proofs[key]
                proof_tokens.extend(synt_proof)

        children = node_.children
        assert len(children) == len(assertion.e_hyps)
        for child in children:
            _build_proof(child)
        proof_tokens.append(node_.label)

    _build_proof(root_node)

    return MMProof(
        statement=root_node.statement,
        proof=proof_tokens,
        disjoints=disjoints,
        e_hyps=e_hyps,
    )


@dataclass
class MMForwardTactic(ForwardTactic):
    label: str
    substitutions: Dict[str, str]
    children_ids: Optional[List[int]]


@dataclass
class MMEnvInfo(EnvInfo):
    new_disjoints: Set[Tuple[str, str]]


@dataclass
class GenerationEvalStats:
    n_statements: int = 0
    n_discarded: int = 0
    n_cond_label_in_proof: int = 0
    n_cond_label_in_generation: int = 0
    n_label_in_proof: int = 0
    n_label_in_generation: int = 0
    label_coverage: float = 0
    label_entropy: float = 0
    n_tok_used: int = 0
    tok_coverage: float = 0
    tok_entropy: float = 0
    n_statements_in_train: int = 0
    n_statements_distinct: int = 0
    avg_proof_size: Avg = field(default_factory=lambda: Avg())
    avg_statement_len: Avg = field(default_factory=lambda: Avg())

    def to_dict(self) -> Dict:
        stats = asdict(self)
        for k, v in stats.items():
            if isinstance(v, Avg):
                stats[k] = v.stats_and_reset()
        return stats


def evaluate_mm_generations(
    histories: List[GenerationHistory],
    vocab_size: int,
    n_labels: int,
    train_set: Set[str],
) -> Dict:

    labels: List[List[str]] = []
    labels_in_proof: Set[str] = set()
    labels_in_generation: Set[str] = set()
    statements: List[str] = []
    stats = GenerationEvalStats()

    for gen in histories:
        if len(gen.forward_steps()) == 0:
            stats.n_discarded += 1
            continue
        stats.n_statements += 1
        statement = gen.forward_steps()[-1].statement
        statements.append(statement)
        nodes = history_to_mm_nodes(gen)
        # select labels used for goal
        proof_labels = [
            n.label
            for n in enumerate_nodes(nodes[-1], ignore_f_e=True, ignore_empty=False)
        ]
        labels_in_proof.update(set(proof_labels))
        generation_labels = [n.label for n in nodes if n.ltype != "$e"]
        labels_in_generation.update(set(generation_labels))
        labels.append(proof_labels)
        stats.avg_proof_size.act(count_unique_nodes(nodes[-1], ignore_e_hyps=True))
        stats.avg_statement_len.act(len(nodes[-1].statement))

        if gen.goal.label_conditioning is not None:
            label_cond = gen.goal.label_conditioning
            gen_labels = set(n.label for n in nodes if n.ltype != "$e")
            proof_labels_ = set(proof_labels)
            stats.n_cond_label_in_proof += label_cond in proof_labels_
            stats.n_cond_label_in_generation += label_cond in gen_labels

    stats.n_statements_distinct = len(set(statements))

    stats.tok_coverage, stats.n_tok_used, stats.tok_entropy, _ = evaluate_diversity(
        [s.split() for s in statements], vocab_size
    )
    (stats.label_coverage, _, stats.label_entropy, _,) = evaluate_diversity(
        labels, n_labels
    )

    stats.n_label_in_proof = len(labels_in_proof)
    stats.n_label_in_generation = len(labels_in_generation)
    stats.n_statements_distinct = len(set(statements))
    stats.n_statements_in_train = len([s for s in set(statements) if s in train_set])
    return stats.to_dict()


def evaluate_diversity(sentences: List[List[str]], dico_size: int):
    tok2index = {}
    counts = np.zeros((dico_size,), dtype=np.float64)
    for sent in sentences:
        for tok in sent:
            if tok not in tok2index:
                tok2index[tok] = len(tok2index)
            counts[tok2index[tok]] += 1

    # usage
    usage = (counts != 0).sum() / len(counts)
    n_different = (counts != 0).sum()

    # compute entropy
    p = counts / counts.sum()
    p[p == 0] = 1
    entropy = -(np.log(p) * p).sum()

    # average length
    avg_len = np.mean([len(sent) for sent in sentences])

    return float(usage), int(n_different), float(entropy), float(avg_len)
