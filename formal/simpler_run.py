# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import os
import random
import getpass

from evariste.datasets import EquationsDatasetConf
from params import ConfStore, Params, MISSING, cfg_from_cli
from evariste.model_zoo import ZOO
from evariste.datasets.lean import LeanDatasetConf, TacticFilter
from evariste.model.transformer_args import DecodingParams
from evariste.clusters.utils import clusterify_path, clusterify_partitions
from evariste.backward.prover.prover_args import (
    BeamSearchKind,
    ProverParams,
    ProverKind,
)
from evariste.backward.prover.zmq_prover import ZMQProverParams
from evariste.backward.goal_factory import get_goals_to_prove
from evariste.backward.prover.bwd_prove import bwd_prove


ROOT_DIR = clusterify_path(f"")
assert os.path.isdir(ROOT_DIR)


ConfStore["default_zmq_simpler"] = ZMQProverParams(
    prover=ProverParams(
        n_simultaneous_proofs=20,
        mcts=ConfStore["mcts_fast"],
        beam_path=Path("later"),  # will be provided by check_and_mutate
        beam_kind=BeamSearchKind.Manual,
        prover_kind=ProverKind.BackwardMCTS,
        dump_mcts=False,
        dump_path=Path("later"),  # will be provided by check_and_mutate
    ),
    decoding=DecodingParams(n_samples=8, use_beam=True, use_sampling=True,),
    n_machines=1,
    max_attempts=1,
    partition=clusterify_partitions("Theorem_Proving"),
    root_dir=Path("later"),  # will be provided by check_and_mutate
    dump_proofs=True,
    n_th_to_prove=500,
    shuffle_seed=43,
    max_restarts=0,  # account for OOM / segfaults in lean :|
    copy_model=False,
    timeout_min=3 * 24 * 60,
)


@dataclass
class SimplerRunParams(Params):

    zmq: ZMQProverParams = field(
        default_factory=lambda: ConfStore["default_zmq_simpler"]
    )

    lang: str = MISSING
    exp_name: str = MISSING
    split: str = MISSING

    n_samples: int = 8

    lean_timeout: int = 2000
    lean_filter: TacticFilter = field(
        default_factory=lambda: ConfStore["tac_filter_no_split"]
    )
    no_cluster: bool = True
    model_name: str = ""

    def _check_and_mutate_args(self):
        model = None
        if self.model_name:
            model = ZOO.get_model(self.model_name)
        if self.lang == "lean":
            dataset: LeanDatasetConf = ConfStore["lean_v33"]
            if not self.no_cluster:
                dataset.lean_cluster = True
                dataset.partition = clusterify_partitions(
                    "Theorem_Proving"
                )  # local if commented out
                dataset.num_instances = 4
                dataset.num_threads = 10
                dataset.dump_tactic_times = True
            dataset.proof_cleaning = self.proof_cleaning
            dataset.timeout = self.lean_timeout
            dataset.filter_tactics = self.lean_filter
            # set dataset
            assert model.lang == self.lang
            self.zmq.set_dataset(self.lang, dataset)

            dataset.check_and_mutate_args()
        elif self.lang == "eq":
            assert model is not None
            dataset = ConfStore[model.dataset]
            self.zmq.set_dataset(lang=self.lang, dataset=dataset)
            assert isinstance(dataset, EquationsDatasetConf)
        else:
            raise RuntimeError(f"unknown language {self.lang}")

        beam_path = model.path
        if str(self.zmq.prover.beam_path) == "later":
            print(
                f"WARNING: BEAM PATH LEFT UNSPECIFIED. "
                f"USING DEFAULT FOR LANGUAGE {self.lang}: {beam_path}"
            )
            self.zmq.prover.beam_path = Path(beam_path)
        self.zmq.prover.beam_path = Path(clusterify_path(self.zmq.prover.beam_path))
        assert self.zmq.prover.beam_path.exists()

        now = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        dirname = f"{now}__{random.randint(1, 1_000_000)}"
        dump_path = Path(clusterify_path(f"{ROOT_DIR}/{self.exp_name}/{dirname}/"))
        assert not dump_path.is_dir(), dump_path
        os.makedirs(dump_path)
        self.zmq.prover.dump_path = dump_path


if __name__ == "__main__":

    params: SimplerRunParams = cfg_from_cli(base_config=SimplerRunParams())
    params.check_and_mutate_args()
    params.zmq.decoding.__post_init__()

    to_prove = get_goals_to_prove(params.zmq.dataset, params.split, n_to_prove=500)

    proved = bwd_prove(
        dataset=params.zmq.dataset,
        decoding=params.zmq.decoding,
        prover_params=params.zmq.prover,
        n_attempts=1,
        to_prove=to_prove,
        dump_proofs=True,
    )
    print(len(proved) / len(to_prove))
