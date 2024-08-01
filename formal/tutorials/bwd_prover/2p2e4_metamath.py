# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from contextlib import closing

from evariste.logger import create_logger
from evariste.model.transformer_args import DecodingParams
from evariste.backward.prover.expander import MPExpander
from evariste.backward.prover.mcts import MCTS
from evariste.backward.prover.args import MCTSParams

from evariste.model.data.dictionary import EOU_WORD
from evariste.backward.model.beam_search import load_beam_search_from_checkpoint
from evariste.backward.env.metamath import MMEnvGenerator
from evariste.backward.env.core import BackwardGoal

from params import ConfStore

# this is required for multiprocess
if __name__ == "__main__":
    logger = create_logger(None)

    ckp_path = "YOUR_PATH"
    env_gen = MMEnvGenerator(ConfStore["new3"])

    mcts_params: MCTSParams = ConfStore["mcts_fast"]
    mcts_params.early_stop = True  # stop proving as soon as a proof is found
    mcts_params.succ_expansions = (
        50  # how many sub-tree to expand at once (uses a virtual loss)
    )

    decoding_params: DecodingParams = ConfStore[
        "decoding_fast"
    ]  # beam search of width 8
    decoding_params.stop_symbol = EOU_WORD  # Stop decoding after <END_OF_USEFUL> token

    # Create the Expander. This launches a background process
    expander = MPExpander(
        load_beam_search_from_checkpoint(
            ckp_path, decoding_params=decoding_params, device="cuda",
        ),
        dag=env_gen.dag,
        max_clients=10,  # Maximum number of concurrent client
        tokens_per_batch=5000,
        profile=False,
        quiet=True,
    )

    # Ensure everything dies nicely
    with closing(expander), closing(env_gen):

        # Create VecBackwardEnv and set the goal with a BackwardGoal.
        # Either set it with a label (then the label has to be available to the env), or with a given conclusion/hyps.
        # See the BackwardGoal class and VecBackwardEnv.reset for more details

        env = env_gen(expander.dico)
        env.reset(goal=BackwardGoal(label="2p2e4"))

        # Finally create the MCTS with our env, our expander client and our parameters.
        mcts = MCTS(
            env,
            expander.get_client(),  # this let's the MCTS communicate with the Expander on its own mp.Queues
            mcts_params,
        )
        res = mcts.simulate(dump_to=f"{os.environ.get('USER')}")
        print("Done", print(res.proof))
