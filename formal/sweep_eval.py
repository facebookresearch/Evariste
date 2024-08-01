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
import time
import random
import getpass

from evariste.utils import OurSignalHandler
from params import ConfStore, Params, MISSING, cfg_from_cli
from evariste.model_zoo import ZOO, ZOOModel
from evariste.datasets import LeanDatasetConf
from evariste.clusters.utils import clusterify_path, clusterify_partitions
from evariste.backward.prover.args import MCTSParams
from evariste.backward.prover.prover_args import (
    BeamSearchKind,
    ProverParams,
    ProverKind,
)
from evariste.backward.prover.zmq_prover import launch_async, ZMQProverParams


ROOT_DIR = clusterify_path(f"")
assert os.path.isdir(ROOT_DIR)


mcts_params: MCTSParams = ConfStore["mcts_fast"]
mcts_params.backup_one_for_solved = True

ConfStore["default_zmq_eval_one"] = ZMQProverParams(
    prover=ProverParams(
        n_simultaneous_proofs=5,
        mcts=mcts_params,
        beam_path=Path("later"),  # will be provided by check_and_mutate
        beam_kind=BeamSearchKind.Manual,
        prover_kind=ProverKind.BackwardMCTS,
        dump_mcts=False,
        dump_path=Path("later"),  # will be provided by check_and_mutate
    ),
    decoding=ConfStore["decoding_bwd_eval"],
    n_machines=1,
    max_attempts=1,
    partition=clusterify_partitions("Theorem_Proving"),
    root_dir=Path("later"),  # will be provided by check_and_mutate
    dump_proofs=True,
    n_th_to_prove=None,
    shuffle_seed=43,
    max_restarts=0,  # account for OOM / segfaults in lean :|
    copy_model=False,
    timeout_min=3 * 24 * 60,
)


@dataclass
class EvalOneParams(Params):

    zmq: ZMQProverParams = field(
        default_factory=lambda: ConfStore["default_zmq_eval_one"]
    )
    lang: str = MISSING
    exp_name: str = MISSING
    split: str = MISSING
    statement_splits_path: str = MISSING
    lean_timeout: int = 2000
    nosplit: bool = False
    model_name: str = ""
    checkpoint_path: str = ""
    local: bool = False

    # to override model dataset, for legacy only
    override_dataset: str = ""
    lean_cluster: bool = False
    lean_partition: str = ""
    lean_cluster_num_instances: int = -1

    def _check_and_mutate_args(self):

        # use specified model, or use default one
        if self.model_name != "":
            print(f"Using model {self.model_name}")
            assert self.checkpoint_path == ""
            model = ZOO.get_model(self.model_name)
        elif self.checkpoint_path != "":
            assert Path(self.checkpoint_path).exists()
            assert self.lang != ""
            assert self.override_dataset != ""
            model = ZOOModel(
                path=self.checkpoint_path, dataset=self.override_dataset, lang=self.lang
            )

        # dataset conf
        if params.override_dataset:
            print(f"Using dataset {params.override_dataset}")
            dataset = ConfStore[params.override_dataset]
            if self.lang in ["lean", "lean_parsable"]:
                assert isinstance(dataset, LeanDatasetConf)
                if self.split is not MISSING:
                    dataset.splits_str = ",".join(set([*dataset.splits, self.split]))
                if self.statement_splits_path is not MISSING:
                    dataset.statement_splits_path = clusterify_path(
                        self.statement_splits_path
                    )
                dataset.timeout = self.lean_timeout
                dataset.nosplit = self.nosplit
                if self.lang == "lean_parsable":
                    dataset.parsable_bwd = True
                dataset.lean_cluster = self.lean_cluster
                if self.lean_partition:
                    dataset.partition = self.lean_partition
                if self.lean_cluster_num_instances > 0:
                    dataset.num_instances = self.lean_cluster_num_instances

            # set dataset
            assert model.lang == self.lang
            self.zmq.set_dataset(self.lang, dataset)

        # beam path
        beam_path = model.path
        if str(self.zmq.prover.beam_path) == "later":
            print(
                f"WARNING: BEAM PATH LEFT UNSPECIFIED. "
                f"USING DEFAULT FOR LANGUAGE {self.lang}: {beam_path}"
            )
            self.zmq.prover.beam_path = Path(beam_path)
        self.zmq.prover.beam_path = Path(clusterify_path(self.zmq.prover.beam_path))
        assert self.zmq.prover.beam_path.exists(), self.zmq.prover.beam_path

        # dump path
        now = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        dirname = f"{now}__{random.randint(1, 1_000_000)}"
        dump_path = Path(clusterify_path(f"{ROOT_DIR}/{self.exp_name}/{dirname}/"))
        assert not dump_path.is_dir(), dump_path
        os.makedirs(dump_path)
        self.zmq.root_dir = dump_path
        self.zmq.prover.dump_path = dump_path

        if self.local:
            self.zmq.n_machines = 1
            self.zmq.prover.mcts = ConfStore["mcts_very_fast"]
            self.zmq.prover.mcts.early_stop = False
            self.zmq.prover.mcts.n_expansions = 50
            self.zmq.prover.mcts.expander.tokens_per_batch = 3000
            self.zmq.max_restarts = 0
            self.zmq.local = self.local


if __name__ == "__main__":

    params: EvalOneParams = cfg_from_cli(base_config=EvalOneParams())
    params.check_and_mutate_args()
    params.zmq.decoding.__post_init__()

    print(f"SWEEP_EVAL dump path: {params.zmq.root_dir}")

    with (params.zmq.root_dir / "eval_one_params.json").open("w") as fp:
        fp.write(params.to_json())

    OurSignalHandler.start()

    start = time.time()
    job = launch_async(
        params.zmq,
        split=params.split,
        name=f"sweep_eval_{params.exp_name}__{params.lang}__{params.split}",
        timeout_min=3 * 24 * 60,
    )
    job.cancel_at_deletion()
    print(f"Job ID: {job.job_id}")
    print(f"Accuracy: {job.result()}%")
    print(f"Time: {time.time() - start}")
