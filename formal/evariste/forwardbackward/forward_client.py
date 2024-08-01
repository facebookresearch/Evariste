# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from evariste.forward import forward_model_factory
from evariste.forward.cli.prove import Config
from evariste.backward.prover.prover_args import ProverParams


ForwardProverParams = Config
BackwardProverParams = ProverParams


class ForwardClient:
    """
    Simpler than the backward client, just to be consistent but not that necessary.
    """

    def __init__(
        self, prover: ForwardProverParams,
    ):

        (
            self.forwardprover,
            self.forward_dico,
            self.forward_param,
            self.forward_env_helper,
        ) = forward_model_factory.from_checkpoint(
            ckpt_path=prover.model.ckpt, device_str="cuda", cfg=prover.prover,
        )
