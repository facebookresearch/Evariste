# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
from collections import defaultdict
from dataclasses import dataclass, field, fields, MISSING, asdict
from functools import partial
from logging import getLogger
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple

from numpy.random.mtrand import RandomState

from evariste.backward.env.lean.graph import LeanTheorem
from evariste.datasets import LeanDatasetConf
from evariste.forward.fwd_lean.lean_fwd_thm_tokenizer import (
    LeanFwdThmTokenizer,
    CompressCfg,
)
from evariste.forward.fwd_lean.training.common import LeanProofNode
from evariste.forward.fwd_lean.training.lean_graph_dataset import (
    LeanGraphTrainingDataset,
)
from evariste.forward.training.graph_sampler import GraphTrainingDataset, GraphSampler

from evariste.forward.training.helpers import (
    postorder_traversal,
    tokenize_fwd_graph,
    tokenize_command,
)
from evariste.metrics import Avg
from evariste.model.data.dictionary import Dictionary, TARGET_IS_GOAL_WORD
from params import Params

logger = getLogger()

LEAN_FWD_TASK = "lean_fwd_seq2seq"
LEAN_FWD_ERROR_COND_TASK = "lean_fwd_error_cond_seq2seq"
LEAN_FWD_TASKS = {LEAN_FWD_TASK, LEAN_FWD_ERROR_COND_TASK}


@dataclass
class LeanGraphConfig(Params):
    insert_noise_prob: float = 0.0
    compress_cfg: CompressCfg = field(default_factory=lambda: CompressCfg())

    # If true, forward model output "{next_node} {bwd_tactic}" instead of
    # "{bwd_tactic} {next_node}"
    next_node_first: bool = True
    # At training time, condition the encoder by a "target_is_goal" token
    # when the generated node outputed by the model is the final goal.
    # We hope that by removing this token at 'proving' time the model will not try
    # to solve everything in one step
    target_is_goal_conditioning: bool = False
    allow_uncompleted_proofs: bool = False
    # instead of sampling from meta proof, take the longest proof
    choose_longest_proof: bool = False
    # auxiliary task, predict children
    predict_children: bool = False
    # use old dataset create with apply_tactic (and not parse_goal_and_apply_tactic)
    use_deprecated_apply_tactic_dataset: bool = False

    generation_prob: float = 0.0
    use_mcts_subproof: bool = False
    mcts_min_proof_dir: str = ""

    use_solved_mcts_subproof: bool = False
    solved_mcts_subproof_min_size: int = -1

    def _check_and_mutate_args(self):
        assert not (self.use_solved_mcts_subproof and self.use_mcts_subproof)


class LeanBaseGraphSampler(GraphSampler):
    """
    This class could be easily transformed in an env generic class and merged with HL

    However for the moment there is something specific to Lean in this class
    When we sample noise, we change the LeanContext to match the one of the current
    proof. This could be changed by changing tokenization of LeanTheorem.
    """

    def __init__(
        self,
        dataset: GraphTrainingDataset[LeanProofNode],
        dataset_cfg: LeanDatasetConf,
        max_len: int,
        dico: Dictionary,
        cfg: LeanGraphConfig,
        allow_global_hyps: bool,
    ):

        self.dico = dico
        self.max_len = max_len

        self.dataset = dataset
        self.dataset_cfg = dataset_cfg

        self.stats: Dict[str, GraphSamplerStats] = defaultdict(
            lambda: GraphSamplerStats()
        )
        self.cfg = cfg
        self.insert_noise_prob = cfg.insert_noise_prob
        self.next_node_first = cfg.next_node_first
        self.allow_global_hyps = allow_global_hyps
        if self.cfg.compress_cfg.compress_hyps:
            from evariste.backward.env.lean.tokenizer import tokenizer as lean_tok

            assert lean_tok is not None

            thm_tokenizer = LeanFwdThmTokenizer(
                dataset=self.dataset_cfg,
                lean_tok=lean_tok,
                compress_cfg=self.cfg.compress_cfg,
            )
        else:
            thm_tokenizer = None
        self.thm_tokenizer = thm_tokenizer
        assert (self.thm_tokenizer is None) == (not self.cfg.compress_cfg.compress_hyps)

    def get_stats(self) -> Dict[str, Number]:
        stats = self.stats["train"].stats_and_reset()
        return stats

    def get_graph_sample(
        self, split: str, task: str, rng: RandomState
    ) -> Optional[Dict[str, Any]]:

        assert task == LEAN_FWD_TASK
        stats = self.stats[split]

        training_sample = self.dataset.get_graph_training_sample(
            split=split, task=task, rng=rng
        )
        root = training_sample.root
        label = training_sample.label

        order = postorder_traversal(
            root, rng=rng, allow_global_hyps=self.allow_global_hyps
        )
        if self.allow_global_hyps:
            hyps = [n for n in order if n.is_hyp]
            n_hyps = len(hyps)
            potential_tgt = [n for n in order if not n.is_hyp]
            order = hyps + potential_tgt
            assert len(potential_tgt) > 0
        else:
            assert all([not n.is_hyp for n in order])
            n_hyps = 0

        # don't select hyp as tgt
        tgt_idx = rng.randint(n_hyps, len(order))

        graph = [n.theorem for n in order[:tgt_idx]]
        graph, noise = self.insert_noise(graph, order, split, rng)

        goal = order[-1].theorem

        assert all(thm.context == goal.context for thm in graph)

        tgt = order[tgt_idx]
        assert not tgt.is_hyp

        conditioning = []
        if self.cfg.target_is_goal_conditioning and tgt.theorem == goal:
            conditioning.append(TARGET_IS_GOAL_WORD)

        tokenize_thm_fn = (
            partial(self.thm_tokenizer.encode_thm, goal=goal)
            if self.thm_tokenizer
            else None
        )

        tokenize_goal_fn = (
            self.thm_tokenizer.encode_goal
            if self.thm_tokenizer and self.cfg.compress_cfg.hyp_tok_in_goal
            else None
        )

        enc_inp = tokenize_fwd_graph(
            goal=goal,
            graph=graph,
            include_goal=True,
            conditioning=conditioning,
            tokenize_thm_fn=tokenize_thm_fn,
            tokenize_goal_fn=tokenize_goal_fn,
        )

        dec_out = tokenize_command(
            target=tgt,
            next_node_first=self.next_node_first,
            predict_children=self.cfg.predict_children,
            tokenize_thm_fn=tokenize_thm_fn,
        )

        try:
            sample = build_sample_dict(
                enc_inp, dec_out, label=label, max_len=self.max_len, dico=self.dico
            )
        except TooLong:
            stats.n_too_long += 1
            return None
        except DicoError:
            stats.n_dico_error += 1
            return None

        stats.n_sampled += 1
        stats.n_children.act(len(tgt.children))
        stats.ratio_no_child.act(len(tgt.children) == 0)
        stats.ratio_one_child.act(len(tgt.children) == 1)
        stats.n_noise_nodes.act(len(noise))
        stats.graph_size.act(len(graph))
        stats.proof_size.act(len(order))
        stats.inp_toks.act(len(enc_inp))
        stats.cmd_toks.act(len(dec_out))
        stats.ratio_tgt_is_final_goal.act(tgt.theorem == goal)

        return sample

    def insert_noise(
        self,
        graph: List[LeanTheorem],
        all_nodes: List[LeanProofNode],
        split: str,
        rng: RandomState,
    ) -> Tuple[List[LeanTheorem], List[LeanTheorem]]:
        if self.insert_noise_prob == 0:
            return graph, []

        noises = []
        old_graph = graph
        graph = []
        present = set([n.theorem for n in all_nodes])

        def _try_insert_noisy_node():
            data = self.dataset.nodes_for_noise(split)
            idx = rng.randint(len(data))
            noisy_node = data[idx].theorem

            if noisy_node not in present:
                # Lean specific, we replace context via proof context
                noisy_node = copy.deepcopy(noisy_node)
                noisy_node.context = all_nodes[0].theorem.context

                present.add(noisy_node)
                graph.append(noisy_node)
                noises.append(noisy_node)

        while old_graph:
            if rng.random() < self.insert_noise_prob:
                _try_insert_noisy_node()
            else:
                graph.append(old_graph.pop(0))

        # insert noise at the end
        while rng.random() < self.insert_noise_prob:
            _try_insert_noisy_node()
        return graph, noises

    @classmethod
    def from_human_dataset(
        cls, params, dico: Dictionary, parsed_rows: List[Dict],
    ) -> "LeanBaseGraphSampler":
        from evariste.trainer.args import TrainerArgs

        assert isinstance(params, TrainerArgs)
        return cls.from_dataset(
            dataset=LeanGraphTrainingDataset.from_trainer_args(params, parsed_rows),
            dico=dico,
            params=params,
            allow_global_hyps=params.lean.graph.allow_uncompleted_proofs,
        )

    @classmethod
    def from_dataset(
        cls,
        params,
        dico: Dictionary,
        dataset: GraphTrainingDataset[LeanProofNode],
        allow_global_hyps: bool,
    ) -> "LeanBaseGraphSampler":
        from evariste.trainer.args import TrainerArgs

        assert isinstance(params, TrainerArgs)
        graph_cfg = params.lean.graph

        return cls(
            dataset=dataset,
            dico=dico,
            max_len=params.batch.max_len,
            cfg=graph_cfg,
            dataset_cfg=params.lean.dataset,
            allow_global_hyps=allow_global_hyps,
        )

    def close(self):
        self.dataset.close()


@dataclass
class GraphSamplerStats:
    n_sampled: int = 0
    n_dico_error: int = 0
    n_too_long: int = 0
    graph_size: Avg = field(default_factory=lambda: Avg())
    proof_size: Avg = field(default_factory=lambda: Avg())
    inp_toks: Avg = field(default_factory=lambda: Avg())
    cmd_toks: Avg = field(default_factory=lambda: Avg())
    n_children: Avg = field(default_factory=lambda: Avg())
    ratio_no_child: Avg = field(default_factory=lambda: Avg())
    ratio_one_child: Avg = field(default_factory=lambda: Avg())
    n_noise_nodes: Avg = field(default_factory=lambda: Avg())
    ratio_tgt_is_final_goal: Avg = field(default_factory=lambda: Avg())

    def stats_and_reset(self) -> Dict[str, Number]:
        dict_ = asdict(self)
        name_to_field = {field.name: field for field in fields(self)}
        stats = {}
        for key, value in dict_.items():
            try:
                stats[key] = value.stats_and_reset()
            except AttributeError:
                stats[key] = value
                field = name_to_field[key]
                assert field.default != MISSING
                setattr(self, field.name, field.default)

        return stats


class TooLong(Exception):
    pass


class DicoError(Exception):
    pass


def build_sample_dict(
    enc_inp: List[str], dec_out: List[str], label: str, max_len: int, dico: Dictionary
):
    # skip too long sequences
    if max(len(enc_inp), len(dec_out)) > max_len:
        raise TooLong

    # index sequences
    try:
        enc_inp = [dico.index(t) for t in enc_inp]
    except KeyError as err:
        logger.warning(f"Detected KeyError in GraphSampler {err}, {label} {enc_inp}")
        raise DicoError

    try:
        dec_out = [dico.index(t) for t in dec_out]
    except KeyError as err:
        logger.warning(f"Detected KeyError in GraphSampler {err}, {label} {dec_out}")
        raise DicoError

    return {"x": enc_inp, "y": dec_out, "name": label}
