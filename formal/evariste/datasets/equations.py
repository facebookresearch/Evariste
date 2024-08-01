# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Optional
from evariste.model.data.envs.node_sampling_strategy import NodeSamplingStrategy

from params import Params, ConfStore
from evariste.envs.eq.env import EquationsEnvArgs, EqGraphSamplerParams


@dataclass
class EquationsDatasetConf(Params):

    env: EquationsEnvArgs
    gen_type: Optional[str] = None

    rule_env: str = field(
        default="default",
        metadata={"help": "Rule type: default / lean_real / lean_nat / lean_int"},
    )

    rule_types_str: str = field(
        default="basic,exp,trigo,hyper",
        metadata={
            "help": "Rule types to include, comma separated (basic, exp, trigo, hyper)"
        },
    )

    # random walk generation parameters
    n_walk_steps: int = field(
        default=20, metadata={"help": "Number of random walk steps"},
    )
    bidirectional_walk: bool = field(
        default=False,
        metadata={
            "help": (
                "Perform two random walks from an initial equation eq0 (to reach "
                "eq and eq'), and map eq to eq'."
            )
        },
    )
    max_init_ops: int = field(
        default=10,
        metadata={
            "help": "Maximum number of operators in the random walk initial equation."
        },
    )
    max_created_hyps: int = field(
        default=10, metadata={"help": "Maximum number of created hypotheses."},
    )
    prob_add_hyp: float = field(
        default=0.5,
        metadata={
            "help": "Probability of adding an hyp missing but required to apply the chosen rule during a random walk."
        },
    )
    # graph generation parameters
    n_nodes: int = field(default=500, metadata={"help": "Number of nodes to generate"})
    max_trials: int = field(
        default=1000,
        metadata={"help": "Maximum number of trials to generate a new node."},
    )
    max_init_hyps: int = field(
        default=10,
        metadata={"help": "Maximum number of initial hypotheses in the graph."},
    )
    max_true_nodes: int = field(
        default=30, metadata={"help": "Maximum number of true nodes in the graph."}
    )
    n_graph_samples: int = field(
        default=30, metadata={"help": "Number of nodes to sample from a random graph."}
    )
    sampler_params: EqGraphSamplerParams = field(
        default_factory=lambda: EqGraphSamplerParams()
    )
    hyp_max_ops: int = field(
        default=3,
        metadata={
            "help": (
                "Maximum number of operators in generated hypotheses (not in the "
                "overall expression, but both in the left and the right hand sides)."
            )
        },
    )
    tf_prob: float = field(
        default=0.5,
        metadata={
            "help": (
                "Probability of sampling a transformation rule (as opposed to an "
                "assertion rule when generating a random graph of expressions)."
            )
        },
    )
    bias_rules: float = field(
        default=1,
        metadata={
            "help": (
                "Bias generation towards least frequently used rules. "
                "Used both for walk and graph genration. 0 to disable."
            )
        },
    )
    bias_nodes: float = field(
        default=0,
        metadata={
            "help": (
                "Bias generation towards shallower or deeper nodes. "
                "Graph generation only, 0 to disable."
            )
        },
    )
    bias_small_simp: float = field(
        default=0,
        metadata={
            "help": (
                "Bias generation on which node to do a small_simp (cf. EqGenForwardSubstTactic)"
                "Graph generation only, 0 to disable."
            )
        },
    )
    bias_node_simp: float = field(
        default=0,
        metadata={
            "help": (
                "Bias generation on which node to do a simplify (cf. EqGenForwardSimpTactic)"
                "Graph generation only, 0 to disable."
            )
        },
    )
    bias_hyp_simp: float = field(
        default=0,
        metadata={
            "help": (
                "Bias generation on which hyp to allow for a simplify (cf. EqGenForwardSimpTactic)"
                "Graph generation only, 0 to disable."
            )
        },
    )

    bias_norm_num: float = field(
        default=0,
        metadata={
            "help": (
                "Bias generation on which node to sample for norm_num(cf. EqGenForwardNormNumTactic)"
                "Graph generation only, 0 to disable."
            )
        },
    )

    bias_hyp_match: float = field(
        default=0,
        metadata={
            "help": (
                "Bias generation on which hyp to match and make true using a substitution (cf. EqGenForwardHypMatchTactic)"
                "Graph generation only, 0 to disable."
            )
        },
    )

    bias_hyp_match_rule: float = field(
        default=0,
        metadata={
            "help": (
                "Bias generation on which rule to match with the considered hyp (cf. EqGenForwardHypMatchTactic)"
                "Graph generation only, 0 to disable."
            )
        },
    )

    insert_noise_prob: float = field(
        default=0, metadata={"help": "Probability to add a random node."}
    )

    insert_external_noise_prob: float = field(
        default=0,
        metadata={
            "help": "Probability to add a random node from a random graph in 'eq_gen_graph_offline_seq2seq' or 'eq_bwd_graph_offline_seq2seq'."
        },
    )

    n_async_envs: int = field(
        default=4, metadata={"help": "Number of async Equations envs (0 for sync)"}
    )

    offline_dataset_path: str = field(
        default="",
        metadata={
            "help": (
                "Paths from which to retrieve offline training data for task 'eq_gen_graph_offline_seq2seq' and 'eq_bwd_graph_offline_seq2seq'. "
                "May be a comma-separated list and may contain globs."
            )
        },
    )

    offline_dataset_splits: str = field(
        default="0.8,0.15,0.05",
        metadata={
            "help": "Proportions for the dataset split for train / valid / test, respectively."
        },
    )

    fwd_offline_sampling_strategy: NodeSamplingStrategy = field(
        default=NodeSamplingStrategy.SamplingMinimal,
        metadata={"help": "Which proof steps to include during retraining."},
    )

    @cached_property
    def rule_types(self) -> List[str]:
        rules = [x for x in self.rule_types_str.split(",") if len(x) > 0]
        assert len(set(rules)) == len(rules) > 0
        return rules

    def __post_init__(self):

        # rules
        assert self.rule_env in ["default", "lean_real", "lean_nat", "lean_int"]
        _ = self.rule_types

        # random walk parameters
        assert self.n_walk_steps > 0
        assert self.max_init_ops >= 0
        assert self.max_created_hyps >= 0

        # graph parameters
        assert self.n_nodes > 0
        assert self.max_trials > 0
        assert self.max_init_hyps >= 0
        assert self.max_true_nodes >= 1
        assert self.hyp_max_ops >= 0
        assert 0 <= self.tf_prob <= 1
        assert 0 <= self.insert_noise_prob <= 1
        assert 0 <= self.insert_external_noise_prob <= 1


def register_equations_datasets():
    ConfStore["eq_env_basic"] = EquationsEnvArgs(
        unary_ops_str="neg,inv", binary_ops_str="add,sub,mul,div"
    )
    ConfStore["eq_env_exp_trigo"] = EquationsEnvArgs(
        unary_ops_str="neg,inv,exp,ln,pow2,sqrt,abs,cos,sin,tan",
        binary_ops_str="add,sub,mul,div",
    )
    ConfStore["eq_env_exp_trigo_hyper"] = EquationsEnvArgs(
        unary_ops_str="neg,inv,exp,ln,pow2,sqrt,abs,cos,sin,tan,cosh,sinh,tanh",
        binary_ops_str="add,sub,mul,div",
    )
    ConfStore["eq_env_all"] = EquationsEnvArgs(
        unary_ops_str="", binary_ops_str="add,sub,mul,div"
    )
    ConfStore["eq_dataset_basic"] = EquationsDatasetConf(
        env=ConfStore["eq_env_basic"], rule_types_str="basic",
    )
    ConfStore["eq_dataset_exp_trigo"] = EquationsDatasetConf(
        env=ConfStore["eq_env_exp_trigo"], rule_types_str="basic,exp,trigo",
    )
    ConfStore["eq_dataset_exp_trigo_hyper"] = EquationsDatasetConf(
        env=ConfStore["eq_env_exp_trigo_hyper"], rule_types_str="basic,exp,trigo,hyper",
    )
    ConfStore["eq_dataset_exp_trigo_hyper_50_nobi"] = EquationsDatasetConf(
        env=ConfStore["eq_env_exp_trigo_hyper"],
        rule_types_str="basic,exp,trigo,hyper",
        n_walk_steps=50,
        bidirectional_walk=False,
    )
    ConfStore["eq_dataset_all"] = EquationsDatasetConf(
        env=ConfStore["eq_env_all"], rule_types_str="basic,exp,trigo,hyper",
    )

    # lean dataset
    ConfStore["eq_env_lean"] = EquationsEnvArgs(
        vtype="real",
        positive=False,
        pos_hyps=False,
        unary_ops_str="neg,exp,ln,sqrt,cos,sin,tan,abs",
        binary_ops_str="add,sub,mul,div",
        comp_ops_str="==,!=,<=,<",
    )
    ConfStore["eq_dataset_lean"] = EquationsDatasetConf(
        env=ConfStore["eq_env_lean"], rule_types_str="lean", rule_env="lean_real",
    )

    ConfStore["eq_graph_sampler_default"] = EqGraphSamplerParams()
    ConfStore["eq_graph_sampler_depth"] = EqGraphSamplerParams(depth_weight=1.0)
