# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from params import ConfStore
from pathlib import Path

import pickle
import random

from evariste.backward.prover.prover import init_and_run_prover
from evariste.backward.prover.args import MCTSParams
from evariste.backward.prover.prover_args import (
    ProverParams,
    ProverKind,
    BeamSearchKind,
)
from evariste.model.utils import reload_ckpt


def one_eval(
    ckpt: str,
    gen_root: Path,
    exp_id: str,
    subsampling: float,
    dump_path: Path,
    only_hard: bool,
):
    def input_iter():
        sent_for_proving = 0
        for p in Path(gen_root).glob(f"{exp_id}_*"):
            for g in Path(p).glob("generations/*.pkl"):
                loaded = pickle.load(open(g, "rb"))
                for x in loaded:
                    if x[3] and only_hard:  # do not reprove easy
                        continue
                    if random.random() > subsampling:
                        continue
                    yield x[1]  # backwardgoal
                    sent_for_proving += 1
        return

    params, _, _ = reload_ckpt(Path(ckpt), only_model_params=False)
    print(params.eq.dataset)
    print("====================")
    print(ckpt)
    print(gen_root, exp_id)

    pf_iter = init_and_run_prover(
        dataset=params.eq.dataset,
        decoding=ConfStore["decoding_greedy"],
        prover_params=ProverParams(
            prover_kind=ProverKind.BackwardGreedy,
            mcts=MCTSParams(),
            beam_path=Path(ckpt),
            dump_path=dump_path,
            n_simultaneous_proofs=100,
            beam_kind=BeamSearchKind.Fixed,
        ),
        input_it=input_iter(),
        decoder_type="decoder",
    )
    total, proved = 0, 0
    for pr, _ in pf_iter:
        total += 1
        proved += pr.proof is not None
    return proved, total
