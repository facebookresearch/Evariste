# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from contextlib import closing
from dataclasses import dataclass, field
from typing import List, Tuple, Iterator, Dict, Set
from params import Params

import torch
from tqdm import tqdm

from evariste.forward.forward_model_configs import ModelConfig, MODELS, PROVER_CFGS
from evariste.forward.forward_prover import ForwardProver, ProverConfig, AsyncProver
from evariste.forward.common import ForwardGoal, GenerationHistory, GenerationInfos
from evariste.forward.proof_search import StandardProofSearch
from evariste.backward.env.metamath import MMTheorem

MODEL_NAME = "new3_v2_0_ckpt217"
prover_name = "sampling"


@dataclass
class FwdConfig(Params):
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(ckpt=MODELS[MODEL_NAME], name=MODEL_NAME)
    )
    prover: ProverConfig = field(default_factory=lambda: PROVER_CFGS[prover_name])
    async_model: bool = True
    seed: int = 42
    n_trials_by_goal: int = 128


def prove_fwd(
    cfg: FwdConfig, goals: Dict[str, List[ForwardGoal]]
) -> Dict[str, Tuple[List[bool], Set[MMTheorem]]]:
    torch.manual_seed(cfg.seed)
    prover, dico, params = ForwardProver.from_checkpoint(
        ckpt_path=cfg.model.ckpt, device="cuda", cfg=cfg.prover
    )

    local_id2goal_uuid = {}
    is_proved: Dict[str, List[bool]] = {}
    generated: Dict[str, Set[MMTheorem]] = {}

    for name, these_goals in goals.items():
        is_proved[name] = [False] * len(these_goals)
        generated[name] = set()

    def inputs() -> Iterator[Tuple[int, ForwardGoal]]:
        for _ in range(cfg.n_trials_by_goal):
            for proof_name, these_goals in goals.items():
                for goal_idx, goal in enumerate(these_goals):
                    if is_proved[proof_name][goal_idx]:
                        # we don't reschedule it
                        continue
                    goal_id = (proof_name, goal_idx)
                    local_id = len(local_id2goal_uuid)
                    local_id2goal_uuid[local_id] = goal_id
                    yield local_id, goal

    if cfg.async_model:
        prover = AsyncProver(prover)
    with closing(prover):
        proof_iterator = prover.generate_proofs_from_goals(inputs())
        for local_id, proof in tqdm(
            proof_iterator,
            total=sum([len(subgoals) for subgoals in goals.values()])
            * cfg.n_trials_by_goal,
        ):
            assert isinstance(proof, StandardProofSearch)

            gen = proof.generation
            gen_info = proof.info
            assert isinstance(gen, GenerationHistory)
            assert isinstance(gen_info, GenerationInfos)
            print("gen", gen)

            solved = gen_info.solved
            goal_uuid = local_id2goal_uuid[local_id]
            proof_name, goal_idx = goal_uuid

            is_proved[proof_name][goal_idx] |= solved
            for node in history_to_mm_nodes(gen):
                if node.ltype == "$e":
                    continue
                trm = MMTheorem(
                    conclusion=node.statement_str,
                    hyps=[(None, h) for h in node.e_hyps.values()],
                )
                generated[proof_name].add(trm)

        results = {}
        for name in generated:
            proved = is_proved[name]
            assert len(proved) == len(goals[name])
            results[name] = (is_proved[name], generated[name])

    return results


if __name__ == "__main__":
    cfg = FwdConfig()
    goals = {
        "foo": [
            ForwardGoal(
                statement="|- ( 2 + 2 ) = 4", e_hyps=[], forbidden=None, mand_disj=set()
            ),
            ForwardGoal(
                statement="|- ( 2 + 3 ) = 5", e_hyps=[], forbidden=None, mand_disj=set()
            ),
        ],
        "bar": [
            ForwardGoal(
                statement="|- ( 2 + 3 ) = 6", e_hyps=[], forbidden=None, mand_disj=set()
            ),
        ],
    }
    print(prove_fwd(cfg, goals))
