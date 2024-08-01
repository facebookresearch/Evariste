# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from evariste.datasets.equations import EquationsDatasetConf

from evariste.envs.eq.env import EquationEnv

from evariste.forward.common import (
    ForwardGraph,
    ForwardTactic,
    FwdEnvOutput,
)
from evariste.forward.fwd_eq.eq_fwd_env import EqForwardEnv
from evariste.forward.fwd_eq.gen.tactics import EqGenForwardTactic
from evariste.forward.fwd_eq.gen.graph import EqGenForwardGraph, EqGenRuleEnv


@dataclass
class EqGenFwdEnvOutput(FwdEnvOutput):
    graph: EqGenForwardGraph
    tactic: EqGenForwardTactic


class EqGenForwardEnv(EqForwardEnv):
    def __init__(self, eq_env: EquationEnv, params: EquationsDatasetConf):
        super().__init__(eq_env, params.rule_env)
        self.params = params
        self.eq_gen_rule_env = EqGenRuleEnv(self.params.rule_env, self.eq_env)
        self.proof_id = -1

    def get_init(self) -> EqGenForwardGraph:
        """Returns the initial graph, potentially generating random elements."""
        # TODO: need to know if rwalk / graph or complex here to know how to initialize !
        # for now, rwalk
        self.proof_id += 1
        if self.params.gen_type == "walk":
            return EqGenForwardGraph(
                self.proof_id,
                self.eq_gen_rule_env,
                self.eq_env,
                max_true_nodes=self.params.max_true_nodes,
                max_created_hyps=self.params.max_created_hyps,
                prob_add_hyp=self.params.prob_add_hyp,
            ).init_rwalk(max_ops=self.params.hyp_max_ops)
        elif self.params.gen_type in {"graph", "complex", "example"}:
            return EqGenForwardGraph(
                self.proof_id,
                self.eq_gen_rule_env,
                self.eq_env,
                max_true_nodes=self.params.max_true_nodes,
            ).init_graph(
                max_ops=self.params.hyp_max_ops, n_init_hyps=self.params.max_init_hyps
            )
        raise RuntimeError(f"unreachable {repr(self.params.gen_type)}")

    def _apply_tactic(
        self, graph: ForwardGraph, fwd_tactic: ForwardTactic
    ) -> FwdEnvOutput:
        # Either the tactic was created synthetically, and this is just a sanity check
        # Or it came from a model and we need to actually run it in the env to check for validity

        assert isinstance(fwd_tactic, EqGenForwardTactic)
        assert isinstance(graph, EqGenForwardGraph)
        new_graph = fwd_tactic.apply(graph, self.params)
        return EqGenFwdEnvOutput(graph=new_graph, tactic=fwd_tactic)
