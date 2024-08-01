# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import functools
from typing import List, Optional, Any

from evariste.backward.env.lean.graph import LeanTheorem, LeanTactic
from evariste.backward.env.lean.tokenizer import LeanTokenizer
from evariste.forward.common import ForwardGraph
from evariste.forward.env_specifics.prover_env_specifics import FwdTokenizer
from evariste.forward.fwd_lean.common import LeanForwardTactic
from evariste.forward.fwd_lean.lean_fwd_thm_tokenizer import LeanFwdThmTokenizer
from evariste.forward.fwd_lean.training.lean_graph_sampler import LeanGraphConfig
from evariste.forward.training.helpers import tokenize_fwd_graph, detokenize_command
from evariste.model.data.dictionary import EOS_WORD, EOU_WORD

LeanForwardGraph = ForwardGraph[LeanTheorem]


class LeanFwdTokenizer(FwdTokenizer):
    def __init__(
        self,
        graph_cfg: LeanGraphConfig,
        last_token: str,
        thm_tokenizer: Optional[LeanFwdThmTokenizer] = None,
    ):
        self.graph_cfg = graph_cfg
        self.last_token = last_token
        self.thm_tokenizer = thm_tokenizer
        assert (self.thm_tokenizer is not None) == graph_cfg.compress_cfg.compress_hyps
        if self.graph_cfg.compress_cfg.hyp_tok_in_goal:
            assert self.thm_tokenizer is not None

    def tokenize_graph(self, graph: LeanForwardGraph) -> List[str]:

        tokenize_thm_fn = (
            functools.partial(self.thm_tokenizer.encode_thm, goal=graph.fwd_goal.thm)
            if self.thm_tokenizer
            else None
        )

        tokenize_goal_fn = (
            self.thm_tokenizer.encode_goal
            if self.thm_tokenizer and self.graph_cfg.compress_cfg.hyp_tok_in_goal
            else None
        )

        return tokenize_fwd_graph(
            goal=graph.fwd_goal.thm,
            graph=graph.generated_thms,
            include_goal=True,
            tokenize_thm_fn=tokenize_thm_fn,
            tokenize_goal_fn=tokenize_goal_fn,
        )

    def detokenize_command(
        self, command: List[str], graph: LeanForwardGraph
    ) -> LeanForwardTactic:
        """
        Retrieve the next node and the associated tactic.
        """

        # note: we might have some issues with next_node, since it might miss
        # some information to create a real LeanTheorem (like context
        # that should not be predicted by the transformer)

        parse_thm_fn = (
            functools.partial(self.thm_tokenizer.decode_thm, goal=graph.fwd_goal.thm)
            if self.thm_tokenizer
            else None
        )

        next_node, tactic = detokenize_command(
            command,
            tactic_cls=LeanTactic,
            theorem_cls=LeanTheorem,
            next_node_first=self.graph_cfg.next_node_first,
            last_token=self.last_token,
            parse_thm_fn=parse_thm_fn,
        )
        return LeanForwardTactic(next_node=next_node, bwd_tactic=tactic)

    @classmethod
    def from_trainer_args(cls, params: Any) -> "LeanFwdTokenizer":
        from evariste.trainer.args import TrainerArgs

        assert isinstance(params, TrainerArgs)
        import evariste.backward.env.lean.tokenizer as lean_tok_module

        lean_tok = (
            lean_tok_module.tokenizer
            if lean_tok_module.tokenizer is not None
            else LeanTokenizer.build(params.lean.dataset.tokenizer)
        )

        aux_tasks = params.lean.graph.predict_children
        stop_symbol = EOS_WORD if not aux_tasks else EOU_WORD

        thm_tokenizer = (
            LeanFwdThmTokenizer(
                lean_tok,
                dataset=params.lean.dataset,
                compress_cfg=params.lean.graph.compress_cfg,
            )
            if params.lean.graph.compress_cfg.compress_hyps
            else None
        )

        assert (thm_tokenizer is None) == (
            not params.lean.graph.compress_cfg.compress_hyps
        )

        return cls(
            graph_cfg=params.lean.graph,
            last_token=stop_symbol,
            thm_tokenizer=thm_tokenizer,
        )
