# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple, Iterator, Optional, Dict
from collections import defaultdict
import fire

from evariste.forward import forward_model_factory
from evariste.datasets.equations import ConfStore, EquationsDatasetConf
from evariste.forward.common import ForwardGoal
from evariste.forward.core.forward_policy import BatchConfig
from evariste.forward.core.generation_errors import GenerationError
from evariste.forward.fwd_eq.gen.proof_search import EqGenProofSearch
from evariste.forward.fwd_eq.gen.specifics import eq_gen_env_specifics_for_gen
from evariste.forward.fwd_eq.gen.graph import GraphHypNode
from evariste.forward.forward_prover import ForwardProver, ProverConfig, SearchConfig
from evariste.model.transformer_args import DecodingParams

from evariste.adversarial_offline.replay_backward import BackwardReplayer
from evariste.envs.eq.env import EquationEnv

from evariste.backward.graph import BackwardGoal
from evariste.backward.env.equations.graph import EQTheorem


def gen(
    seed: int = 0,
    n: int = 100,
    max_size: int = 20,
    bias_rules: float = 1,
    from_ckpt: Optional[str] = None,
):
    eq_conf: EquationsDatasetConf = ConfStore["eq_dataset_lean"]
    eq_conf.max_init_hyps = 5
    eq_conf.bias_rules = bias_rules
    eq_conf.gen_type = "complex"
    eq_conf.max_true_nodes = 1 << 32
    eq_conf.prob_add_hyp = 0.5
    prover_config = ProverConfig(
        # relevant
        # max_nodes = n_nodes, max_generations = max_trials, max_cons_inv does not exist in old generate
        SearchConfig(
            max_nodes=max_size,
            max_generations=max_size,
            max_cons_inv_allowed=1 << 32,  # unused
            n_simultaneous_proofs=1,
        ),
        DecodingParams(),  # unused
        BatchConfig(max_batch_mem=1 << 32),  # unused
        name="model_free_gen",
    )

    if from_ckpt is None:
        forward_prover = ForwardProver.from_random_args(
            prover_config, eq_gen_env_specifics_for_gen(eq_conf, seed=seed), eq_conf
        )
    else:
        prover_config.decoding_params = DecodingParams(
            n_samples=1, use_beam=False, use_sampling=True
        )
        prover_config.search_cfg.n_simultaneous_proofs = 100
        (forward_prover, _, _, _,) = forward_model_factory.from_checkpoint(
            ckpt_path=from_ckpt,
            device_str="cuda:0",
            cfg=prover_config,
            overwrite_gen_type="graph",
            overwrite_prefix=None,
        )

    def dummy_goal_stream(n_goals: int) -> Iterator[Tuple[int, ForwardGoal]]:
        for i in range(n_goals):
            yield i, ForwardGoal(thm=None, label=f"unused_{i}")

    bwd_replayer = BackwardReplayer(
        dataset=eq_conf, eq_env=EquationEnv.build(eq_conf.env)
    )
    goal_stream = dummy_goal_stream(n)

    proof_stream = forward_prover.generate_proofs(goal_stream)
    # with open("new_gen", "w") as f:
    total_tacs: Dict[str, int] = defaultdict(int)
    for i, res in proof_stream:
        if isinstance(res, GenerationError):
            print(res, "generror")
            continue
        assert isinstance(res, EqGenProofSearch), res
        g = res.next_graph
        print(f"---- {len(g.nodes)} ---")
        for node in g.nodes:
            if isinstance(node, GraphHypNode):
                print(node.node.infix())
        print("###")
        print(g.nodes[-1].node.infix())
        # f.write(g.nodes[-1].node.infix() + "\n")
        print(res.stats())
        for x, y in res.stats().items():
            total_tacs[x] += y
        print(total_tacs)
        print(f"============== {i} ============")
        for i, node in enumerate(res.next_graph.nodes):
            if isinstance(node, GraphHypNode) or node.is_true:
                continue
            hyps = res.next_graph.get_hyps_for_node(node)
            goal_stmt = EQTheorem(node=node.node, hyps=[h.node for h in hyps])
            bwd_goal = BackwardGoal(goal_stmt, "meh")
            try:
                bwd_replayer.replay_proof(bwd_goal, res)
            except Exception as e:
                print("FAILED", goal_stmt.eq_node)
                for x in res.next_graph.nodes[: i + 1]:
                    print(type(x), x.node.infix())
                raise e


if __name__ == "__main__":
    fire.Fire(gen)
