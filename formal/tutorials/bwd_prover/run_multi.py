# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import random
from pathlib import Path
from contextlib import closing

from evariste.logger import create_logger
from evariste.backward.prover.prover_args import ProverParams
from evariste.backward.prover.mcts_prover import MCTSProver
from evariste.backward.prover.expander import MPExpander
from evariste.backward.prover.utils import GPUMonitor
from evariste.backward.model.beam_search import load_beam_search_from_checkpoint
from evariste.backward.env.metamath.env import MMEnvGenerator

from evariste.backward.env.core import BackwardGoal
from params import ConfStore


logger = create_logger(None)
if __name__ == "__main__":
    # This will run 10 MCTS process in parallel to feed the GPU
    n_simultaneous_proofs = 10
    mcts_params = ConfStore["mcts_fast"]
    mcts_params.early_stop = True

    # MM
    dataset = ConfStore["new3_100"]  # faster start up
    ckp_path = "YOUR_PATH"
    env_gen = MMEnvGenerator(dataset)

    # Load labels to prove
    with open(Path(dataset.data_dir) / "split.valid", "r") as f:
        labels = []
        for label in f:
            labels.append(label.strip())
    random.seed(43)
    random.shuffle(labels)

    # Instantiate the BackwardProver class with an env_generator, an expander and some params.
    prover = MCTSProver(
        env_generator=env_gen,
        expander=MPExpander(
            load_beam_search_from_checkpoint(
                ckp_path, decoding_params=ConfStore["decoding_fast"], device="cuda",
            ),
            dag=env_gen.dag,
            tokens_per_batch=5000,
            max_clients=n_simultaneous_proofs,
            profile=False,
        ),
        prover_params=ProverParams(
            n_simultaneous_proofs=n_simultaneous_proofs,
            mcts=mcts_params,
            beam_search_path=None,
        ),
    )
    # monitor gpu usage
    gpu_mon = GPUMonitor(delay=1.0)
    with closing(gpu_mon), closing(prover):
        for label in labels:
            prover.prove_one(BackwardGoal(label=label))  # this is asynchronous

        proved, total = 0, 0
        for res in prover.get_results(
            wait_for_all=True
        ):  # wait_for_all will block until all proofs are solved
            proved += 1 if res.proof is not None else 0
            total += 1
            print(proved / total, total)
            print([x.stats for x in gpu_mon.stats])
        print("All done")
