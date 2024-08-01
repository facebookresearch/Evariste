# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple, Iterator, Union

import numpy as np

from evariste.datasets.equations import EquationsDatasetConf
from evariste.envs.eq.env import EquationEnv
from evariste.forward.core.forward_policy import BatchConfig
from evariste.forward.core.generation_errors import GenerationError
from evariste.forward.fwd_eq.gen.env import EqGenForwardEnv
from evariste.forward.proof_search import ForwardProofSearch
from evariste.forward.forward_prover import ForwardProver, ProverConfig, SearchConfig
from evariste.forward.fwd_eq.gen.proof_search import EqGenProofSearch
from evariste.forward.fwd_eq.gen.specifics import eq_gen_env_specifics_for_gen
from evariste.model.transformer_args import DecodingParams
from evariste.forward.common import ForwardGoal


def get_gen(
    conf: EquationsDatasetConf, eq_env: EquationEnv
) -> Tuple[ForwardProver, Iterator[EqGenProofSearch]]:
    specifics = eq_gen_env_specifics_for_gen(conf, eq_env=eq_env)
    print("NEWGEN CONF", conf.n_nodes, conf.max_trials)
    fp = ForwardProver.from_random_args(
        ProverConfig(
            SearchConfig(
                max_nodes=conf.n_nodes,
                max_generations=conf.max_trials,
                max_cons_inv_allowed=1 << 32,  # unused
                n_simultaneous_proofs=1,
            ),
            DecodingParams(),  # unused
            BatchConfig(max_batch_mem=1 << 32),  # unused
            name="model_free_gen",
        ),
        specifics,
        conf,
    )

    def dummy_goal_stream() -> Iterator[Tuple[int, ForwardGoal]]:
        i = 0
        while True:
            yield i, ForwardGoal(thm=None, label=f"unused_{i}")
            i += 1

    ps = fp.generate_proofs(dummy_goal_stream())

    def final_iterator() -> Iterator[EqGenProofSearch]:
        for _i, res in ps:
            if isinstance(res, GenerationError):
                continue
            assert isinstance(res, EqGenProofSearch)
            yield res

    return fp, final_iterator()
