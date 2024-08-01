# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
from typing import Tuple, Iterator, Dict, Generator, Callable, Any

from evariste.datasets import DatasetConf


from evariste.backward.prover.mcts_prover import MCTSProofHandlerGenerator

from evariste.model.transformer_args import DecodingParams
from evariste.backward.prover.prover import (
    ProverParams,
    BackwardGoal,
    backward_runner,
    ProofHandler,
)

from evariste.backward.prover_factory import backward_init


class BackwardClient:
    """
    Store everything needed to make backward proofs and wraps up the closing process
    """

    def __init__(
        self,
        dataset: DatasetConf,
        decoding: DecodingParams,
        prover: ProverParams,
        decoder_type: str,
        proof_handler,
    ):
        self.env_gen, self.env_gen_inst, self.expander = backward_init(
            dataset, decoding, prover, decoder_type
        )

        self.dataset = dataset
        self.decoding = decoding
        self.decoder_type = decoder_type
        self.proofhandler = proof_handler
        self.prover_params = prover

    def generate_proofs_from_goals(
        self, input_it: Callable[[], Iterator[BackwardGoal]]
    ):
        """
        Make a call and yield from backward runner using the parts build by backward_init.
        Can be called multiple times without restarting everything.
        @param input_it:
        @return:
        """
        raise NotImplementedError
        # yield from backward_run_on_goals(
        #     self.env_gen_inst,
        #     self.expander,
        #     self.proofhandler,
        #     input_it(),
        #     self.prover_params.n_simultaneous_proofs,
        #     self.prover_params.dump_path,
        #     self.prover_params.is_master,
        # )

    def generate_proofs(self, input_it: Callable[[], Iterator[ProofHandler]]):
        """
        Make a call and yield from backward runner using the parts build by backward_init.
        Can be called multiple times without restarting everything.
        @param input_it:
        @return:
        """
        yield from backward_runner(
            self.env_gen_inst,
            self.expander,
            input_it(),
            self.prover_params.n_simultaneous_proofs,
            self.prover_params.dump_path,
            self.prover_params.is_master,
        )

    def close(self):
        self.env_gen.close()
        self.expander.close()
        if isinstance(self.proofhandler, MCTSProofHandlerGenerator):
            self.proofhandler.close()
