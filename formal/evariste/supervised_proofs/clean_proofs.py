# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import tempfile
from pathlib import Path

from evariste.backward.env.lean.cleaner import AsyncProofCleaner
from evariste.backward.env.lean.env import LeanExpanderEnv
from evariste.backward.graph import BackwardGoal
from evariste.backward.prover.mcts import MCTSResult
from evariste.backward.prover.prover_args import (
    ProverParams,
    ProverKind,
    ProofCleaningParams,
    CleaningLevel,
)
from evariste.datasets import LeanDatasetConf
from evariste.model.data.envs.minproof_replay_buffer import MCTSSampleProof
from evariste.model.data.subproof_args import MCTSSubProofArgs
from evariste.supervised_proofs.common import get_repo, CleaningMetadata
from params import ConfStore, MISSING

"""
python -m evariste.supervised_proofs.clean_proofs --split oai_curriculum --model YOUR_PATH --dump_name proofs_dump0 --dataset_name lean_v1.0
python -m evariste.supervised_proofs.clean_proofs --split annotations_v1 --model YOUR_PATH --dump_name proofs_dump0
"""


def clean_proofs(
    split: str,
    model: str,
    dump_name: str,
    fifo: str = "",
    dataset_name: str = "",
    debug: bool = False,
):
    repo = get_repo()
    metadata, proofs = repo.load_dump(split, model, dump_name)
    new_dump_name = f"{dump_name}_cleaned0"
    if not debug:
        repo.check_not_dump_exists(split, model_name=model, dump_name=new_dump_name)
    print(f"Going to clean {len(proofs)}")
    with tempfile.TemporaryDirectory() as tmp_dir:
        dump_path = Path(tmp_dir)
        dataset_name = dataset_name if dataset_name else metadata.dataset_name
        dataset = ConfStore[dataset_name]
        assert isinstance(dataset, LeanDatasetConf)
        print("dataset", dataset)
        if fifo:
            fifo_root = Path(fifo)
            dataset.path_to_fifos = (fifo_root / "stdin", fifo_root / "stdout")
        waiting = {}
        cleaner = AsyncProofCleaner(
            env=LeanExpanderEnv(dataset=dataset, dump_path=dump_path),
            prover_params=ProverParams(
                n_simultaneous_proofs=5,
                mcts=ConfStore["mcts_slow"],
                beam_path=MISSING,
                dump_path=dump_path,
                prover_kind=ProverKind.BackwardMCTS,
                beam_kind=MISSING,
                mcts_subproof_params=MCTSSubProofArgs(),
                proof_cleaning_params=ProofCleaningParams(
                    level=CleaningLevel.TacRemoval_TacClean_SimpOnly,
                    timeout_useless_tactics_removal=600,
                    timeout_individual_tactic_cleaning=600,
                ),
            ),
        )
        for pid, proof in enumerate(proofs):
            waiting[pid] = proof
            assert isinstance(proof.proof, tuple)
            cleaner.process(
                pid=pid,
                to_process=MCTSResult(
                    proof=proof.proof,
                    goal=BackwardGoal(label=proof.label, theorem=proof.proof[0]),
                    exception=None,
                    mcts_samples_critic=[],
                    mcts_samples_tactic=[],
                    mcts_samples_effect=[],
                    sample_proof=MCTSSampleProof(
                        label=proof.label, size=-1, stype="none", samples=[]
                    ),
                    simplified_state=None,
                    stats={},
                    hist_stats={},
                ),
            )

        results = {}
        fail = 0
        success = 0
        while waiting:
            for pid, (result, errors) in cleaner.get_ready():
                old_proof = waiting.pop(pid)

                if len(errors) > 0:
                    print(pid, [type(e) for e in errors], errors)
                    results[pid] = old_proof
                    fail += 1
                else:
                    new_proof = copy.deepcopy(old_proof)
                    assert result.proof is not None
                    new_proof.proof = result.proof
                    results[pid] = new_proof
                    success += 1
        print(len(results))
        print("fail", fail)
        print("success", success)
        if debug:
            return

        new_proofs = []
        for i in range(len(proofs)):
            new_proofs.append(results[i])

        new_metadata = copy.deepcopy(metadata)

        new_metadata.cleaning_metadata = CleaningMetadata(
            cleaner_version=f"cleaner_of_{metadata.dataset_name}",
            succeed=success,
            failed=fail,
        )

        repo.make_dump(
            split=split,
            model_name=model,
            dump_name=new_dump_name,
            proofs=new_proofs,
            metadata=metadata,
        )

        tgt_dir = repo.model_dir(split, model)

        print("tgt_dir", tgt_dir)

        print("To commit")
        print(f"cd {repo.root}")
        print(
            f'git add -A && git commit -m "proofs from {model} {split} {new_dump_name}"'
        )


if __name__ == "__main__":
    import fire

    fire.Fire(clean_proofs)
