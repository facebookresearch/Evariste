# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field
from typing import Tuple, Dict
from logging import getLogger
from pathlib import Path
import os
import copy
import getpass

from params.params import ConfStore, Params
from evariste import slurm_conf_factory
from evariste.forward.env_specifics.generation_annotator import NodeSelectionConfig

from evariste.model_zoo import ZOO
from evariste.adversarial.prover import AdvProverKind
from configs.bwd_configs import _bwd_config
from configs.fwd_configs import fwd_mm_config, fwd_eq_config
from evariste.forward.online_generation.worker_type import WorkerType
from evariste.clusters.utils import clusterify_path
from evariste.comms.rl_distributed_config import DistributedSetup, RLDistributedConfig
from evariste.trainer.args import TrainerArgs
from evariste.utils import get_dump_path
from evariste.slurm import SlurmConf


logger = getLogger()


@dataclass
class AdversarialConfig(Params):
    env_name: str
    prover_trainer_args: TrainerArgs
    generator_trainer_args: TrainerArgs
    num_prover_trainers: int
    num_generator_trainers: int
    num_prover_actors: int
    num_generator_actors: int

    distributed_setup: DistributedSetup = DistributedSetup.ADVERSARIAL
    fixed_prover: str = ""
    fixed_generator: str = ""
    prover_kind: AdvProverKind = AdvProverKind.BackwardGreedy
    use_zmq: bool = True

    # todo: we can probably generate this master port with slurm_job_id
    master_port: int = -1

    root_dir: str = "PATH/USERNAME/dumped"
    exp_name: str = ""  # must be specified
    exp_path: str = ""  # root_dir/exp_name/exp_id (+/worker_type)
    exp_id: str = ""  # optionally specified

    # {weight,name}:... name=generator or dataset_{lang}_{name}
    generator_mix_str: str = "1,generator"

    # flags below allow to send only failed proofs, or all proofs to the relevant trainer
    generator_send_only_failed: bool = True
    prover_send_only_failed: bool = True

    # for local debug, you can set the slurm_rank or slurm_world_size
    debug: bool = False
    overwrite_slurm_rank: int = -1
    overwrite_slurm_world_size: int = -1

    node_select_cfg: NodeSelectionConfig = field(
        default_factory=lambda: NodeSelectionConfig()
    )

    def get_slurm_task_id_and_n_tasks(self) -> Tuple[int, int]:
        """
        This gives the task id and the number of tasks for the adversarial job
        Therefore n_tasks is the sum of world_size for all kind of workers:
        (prover actors, trainers etc...)
        """
        if self.overwrite_slurm_rank >= 0:
            assert self.debug
            assert self.overwrite_slurm_world_size >= 1
            n_tasks = self.overwrite_slurm_world_size
            task_id = self.overwrite_slurm_rank
        else:
            is_slurm_job = "SLURM_JOB_ID" in os.environ
            assert is_slurm_job
            n_tasks = int(os.environ["SLURM_NTASKS"])
            task_id = int(os.environ["SLURM_PROCID"])
            assert self.overwrite_slurm_rank == -1
            assert self.overwrite_slurm_world_size == -1
        assert 0 <= task_id < n_tasks
        return task_id, n_tasks

    def is_master(self) -> bool:
        """
        We check if this worker is the job master, i.e. the master for all job, not
        for a particular group (group are prover actors for instance). We check that
        the slurm task if is 0.
        """
        slurm_task_id, _ = self.get_slurm_task_id_and_n_tasks()
        return slurm_task_id == 0

    def __post_init__(self):

        # root dir
        if "USERNAME" in self.root_dir:
            self.root_dir = self.root_dir.replace("USERNAME", getpass.getuser())
        self.root_dir = clusterify_path(self.root_dir)
        assert os.path.isdir(self.root_dir), self.root_dir
        assert self.exp_path == ""

        assert self.prover_trainer_args.slurm_conf.torch_world_size == -1
        assert self.generator_trainer_args.slurm_conf.torch_world_size == -1

        if self.fixed_prover != "":
            assert self.distributed_setup == DistributedSetup.ADV_FIXED_PROVER
            # fixed prover setup, no prover training
            assert self.num_prover_trainers == 0
            assert self.num_prover_actors > 0
        elif self.num_prover_actors == 0:
            # generation only setup
            assert self.num_prover_trainers == 0
            assert self.fixed_prover == ""
        else:
            # std adv setup
            assert self.num_prover_trainers > 0
            assert self.num_prover_actors > 0

        if self.fixed_generator != "":
            assert self.distributed_setup == DistributedSetup.ADV_FIXED_GENERATOR
            # fixed generator setup, no generator training
            assert self.num_generator_trainers == 0
            assert self.num_generator_actors > 0
        else:
            # std adv setup
            assert self.num_generator_trainers > 0
            assert self.num_generator_actors > 0

    def get_distributed_cfg(self) -> RLDistributedConfig:
        # Heavy but convenient
        return RLDistributedConfig.from_adv_cfg(adv_cfg=self)

    def get_worker_type(self) -> WorkerType:
        """
        Returns the worker type of the worker given the current
        slurm_task_id
        """
        worker_type, _, _ = self._get_worker_type()
        return worker_type

    def make_trainer_args(self) -> TrainerArgs:
        """
        Depending on the current slurm_task_id, returns the
        generator TrainerArgs or prover TrainerArgs.

        It also populate the fields:
        - rl_distributed
        - slurm_conf
        - exp_id, override_dump_path etc...
        of this TrainerArgs
        """
        assert len(self.exp_name) > 0
        slurm_task_id, _ = self.get_slurm_task_id_and_n_tasks()

        distributed_cfg = self.get_distributed_cfg()
        worker_type, group_cfg = self.get_slurm_group_cfg()
        is_generator = worker_type.is_generator()

        base = self.generator_trainer_args if is_generator else self.prover_trainer_args
        trainer_cfg = copy.deepcopy(base)
        trainer_cfg.slurm_conf = group_cfg
        trainer_cfg.override_dump_path = os.path.join(self.exp_path, worker_type)
        trainer_cfg.exp_id = self.exp_id
        trainer_cfg.exp_name = self.exp_name

        logger.warning("Overriding distributed_rl config")
        trainer_cfg.rl_distributed = distributed_cfg
        if self.debug:
            logger.warning("Setting TrainerArgs debug.debug to True")
            trainer_cfg.debug.debug = True

        return trainer_cfg

    def get_slurm_group_cfg(self) -> Tuple[WorkerType, SlurmConf]:
        """
        Depending on the current slurm_task_id, it will be build a slurm
        config for the group.

        The group could be the group of the prover actors, prover generators.

        WARNING: the global rank and the world size in the SlurmConf
        is therefore the rank IN group and the group size.
        """
        rank_in_slurm_job, n_tasks = self.get_slurm_task_id_and_n_tasks()
        (
            worker_type,
            rank_of_first_task_in_group,
            group_world_size,
        ) = self._get_worker_type()

        is_generator = worker_type.is_generator()
        rank_in_group = rank_in_slurm_job - rank_of_first_task_in_group
        assert 0 <= rank_in_group < group_world_size

        if is_generator:
            group_master_port = self.master_port
        else:
            group_master_port = self.master_port + 1

        print(
            f"=== slurm_group_cfg for {worker_type}: "
            f"-- is_generator: {is_generator} "
            f"-- global rank: {rank_in_slurm_job}/{n_tasks} "
            f"-- group rank: {rank_in_group}/{group_world_size} "
            f"-- rank_of_first_task_in_group: {rank_of_first_task_in_group} "
            f"-- group_master_port: {group_master_port}"
        )

        is_slurm_job = "SLURM_JOB_ID" in os.environ
        if is_slurm_job:
            n_tasks_per_node = int(os.environ["SLURM_NTASKS"]) // int(
                os.environ["SLURM_JOB_NUM_NODES"]
            )
            group_master_node_id = rank_of_first_task_in_group // n_tasks_per_node
            group_cfg = slurm_conf_factory.from_slurm_env(
                master_port=group_master_port,
                global_rank=rank_in_group,
                world_size=group_world_size,
                master_node_id=group_master_node_id,
            )
        else:
            assert self.debug
            # fake config to launch locally workers
            assert group_world_size == 1  # multi gpu not supported for debug
            group_cfg = slurm_conf_factory.from_cli(
                local_rank=0, global_rank=rank_in_group, world_size=group_world_size,
            )
        print(f"=== slurm conf for {worker_type}: {group_cfg}")

        return worker_type, group_cfg

    def _get_worker_type(self) -> Tuple[WorkerType, int, int]:
        slurm_task_id, _ = self.get_slurm_task_id_and_n_tasks()
        cfg = self.get_distributed_cfg()
        worker_types = [
            WorkerType.GENERATOR_TRAINER,
            WorkerType.GENERATOR_ACTOR,
            WorkerType.PROVER_TRAINER,
            WorkerType.PROVER_ACTOR,
        ]
        world_sizes = [
            cfg.n_generator_trainers,
            cfg.n_generators,
            cfg.n_prover_trainers,
            cfg.n_provers,
        ]
        offset = 0
        for worker_type, world_size in zip(worker_types, world_sizes):
            if offset <= slurm_task_id < offset + world_size:
                return worker_type, offset, world_size
            offset += world_size
        raise ValueError(f"Invalid rank {slurm_task_id} (>= {sum(world_sizes)})")

    def check_adversarial_config(self):
        """
        Not called in post init since we need to be in a slurm job to check this
        """
        # check if created correctly
        _ = self.get_distributed_cfg()

        # set experiment path and ID
        assert self.exp_path == ""
        self.exp_path, self.exp_id = get_dump_path(
            root_dump_path=self.root_dir,
            exp_name=self.exp_name,
            given_exp_id=self.exp_id,
            overwrite_dump_path="",
        )

        # expected number of tasks
        task_id, n_tasks = self.get_slurm_task_id_and_n_tasks()
        assert n_tasks == (
            self.num_generator_trainers
            + self.num_prover_actors
            + self.num_prover_trainers
            + self.num_generator_actors
        ), (
            n_tasks,
            self.num_generator_trainers,
            self.num_prover_actors,
            self.num_prover_trainers,
            self.num_generator_actors,
        )
        assert 0 <= task_id < n_tasks


def default_mm_adv_fwd_cfg(debug: bool, local: bool) -> AdversarialConfig:
    prover_trainer_cfg = fwd_mm_config(
        task="mm_fwd_seq2seq",
        debug=debug,
        is_generation=False,
        other_tasks="mm_fwd_rl",
        stop_action=True,
    )
    generator_trainer_cfg = fwd_mm_config(
        task="mm_gen_seq2seq",
        debug=debug,
        is_generation=True,
        other_tasks="mm_gen_rl",
        stop_action=True,
        reload_model=PATHS["mm_gen_stop"],
    )
    if debug:
        prover_trainer_cfg.rl_params.replay_buffer.min_len = 10
        generator_trainer_cfg.rl_params.replay_buffer.min_len = 10

    return AdversarialConfig(
        env_name="mm",
        prover_trainer_args=prover_trainer_cfg,
        generator_trainer_args=generator_trainer_cfg,
        num_prover_trainers=1,
        num_generator_trainers=1,
        num_prover_actors=1,
        num_generator_actors=1,
        prover_kind=AdvProverKind.ForwardGreedy,
        debug=debug,
        overwrite_slurm_world_size=4 if (debug and local) else -1,
        exp_name="debug_adversarial" if (debug and local) else "",
        prover_send_only_failed=True,
        generator_send_only_failed=True,
    )


def mm_adv_gen_only_cfg(debug: bool, local: bool) -> AdversarialConfig:
    generator_trainer_cfg = fwd_mm_config(
        task="mm_gen_seq2seq",
        debug=debug,
        is_generation=True,
        other_tasks="mm_gen_rl",
        stop_action=True,
        reload_model=PATHS["mm_gen_stop"],
    )
    if debug:
        generator_trainer_cfg.rl_params.replay_buffer.min_len = 10

    return AdversarialConfig(
        distributed_setup=DistributedSetup.GENERATOR_ONLY,
        env_name="mm",
        prover_trainer_args=ConfStore["default_cfg"],  # not used
        generator_trainer_args=generator_trainer_cfg,
        num_prover_trainers=0,
        num_generator_trainers=1,
        num_prover_actors=0,
        num_generator_actors=1,
        prover_kind=AdvProverKind.ForwardGreedy,  # not used
        debug=debug,
        overwrite_slurm_world_size=2 if (debug and local) else -1,
        exp_name="debug_adversarial" if (debug and local) else "",
    )


def mm_adv_no_prover_reward_cfg(debug: bool, local: bool) -> AdversarialConfig:
    generator_trainer_cfg = fwd_mm_config(
        task="mm_gen_seq2seq",
        debug=debug,
        is_generation=True,
        other_tasks="mm_gen_rl",
        stop_action=True,
        reload_model=PATHS["mm_gen_stop"],
    )

    prover_trainer_cfg = fwd_mm_config(
        task="mm_fwd_seq2seq",
        debug=debug,
        is_generation=False,
        other_tasks="mm_fwd_rl",
        stop_action=True,
    )

    if debug:
        generator_trainer_cfg.rl_params.replay_buffer.min_len = 10
        prover_trainer_cfg.rl_params.replay_buffer.min_len = 10

    return AdversarialConfig(
        distributed_setup=DistributedSetup.ADV_GENERATOR_REWARD,
        env_name="mm",
        prover_trainer_args=prover_trainer_cfg,
        generator_trainer_args=generator_trainer_cfg,
        num_prover_trainers=1,
        num_generator_trainers=1,
        num_prover_actors=1,
        num_generator_actors=1,
        prover_kind=AdvProverKind.ForwardGreedy,
        debug=debug,
        overwrite_slurm_world_size=4 if (debug and local) else -1,
        exp_name="debug_adversarial" if (debug and local) else "",
    )


def default_eq_adv_cfg(
    prover_kind: AdvProverKind,
    debug: bool,
    local: bool,
    reload_fwd: str = "",
    reload_bwd: str = "",
    reload_gen: str = "",
) -> AdversarialConfig:

    assert reload_fwd == "" or prover_kind.is_forward and Path(reload_fwd).is_file()
    assert reload_bwd == "" or prover_kind.is_backward and Path(reload_bwd).is_file()
    assert reload_gen == "" or Path(reload_gen).is_file()

    # prover trainer
    if prover_kind.is_forward:
        prover_trainer_cfg = fwd_eq_config(
            task="eq_fwd_graph_seq2seq",
            debug=debug,
            is_generation=False,
            other_tasks="eq_fwd_rl",
            stop_action=True,
            reload_model=reload_fwd,
        )
    else:
        prover_trainer_cfg = _bwd_config(
            task="eq_bwd_rwalk_seq2seq",
            debug=debug,
            other_tasks="eq_bwd_rl",
            reload_model=reload_bwd,
        )

    # generator trainer
    generator_trainer_cfg = fwd_eq_config(
        task="eq_gen_graph_seq2seq",
        debug=debug,
        is_generation=True,
        other_tasks="eq_gen_rl",
        stop_action=True,
        reload_model=reload_gen,
    )

    if debug:
        prover_trainer_cfg.rl_params.replay_buffer.min_len = 10
        generator_trainer_cfg.rl_params.replay_buffer.min_len = 10

    return AdversarialConfig(
        env_name="eq",
        prover_trainer_args=prover_trainer_cfg,
        generator_trainer_args=generator_trainer_cfg,
        num_prover_trainers=1,
        num_generator_trainers=1,
        num_prover_actors=1,
        num_generator_actors=1,
        prover_kind=prover_kind,
        debug=debug,
        overwrite_slurm_world_size=4 if (debug and local) else -1,
        exp_name="debug_adversarial" if (debug and local) else "",
        prover_send_only_failed=True,
        generator_send_only_failed=True,
        generator_mix_str=f"1,generator",
        distributed_setup=DistributedSetup.ADVERSARIAL,
        # generator_mix_str=f"1,generator:1,dataset_eq_eq_dataset_exp_trigo_hyper",
    )


PATHS = {
    "eq_graph_gen": "YOUR_PATH",
    "eq_graph_bwd_sup": "YOUR_PATH",
    "mm_gen_stop": "YOUR_PATH",
}


def register_adversarial_configs():

    # MM
    ConfStore["mm_adv_fwd_debug_local"] = lambda: default_mm_adv_fwd_cfg(
        debug=True, local=True
    )
    ConfStore["mm_gen_only_debug_local"] = lambda: mm_adv_gen_only_cfg(
        debug=True, local=True
    )
    ConfStore[
        "mm_adv_fwd_no_prover_reward_debug_local"
    ] = lambda: mm_adv_no_prover_reward_cfg(debug=True, local=True)

    # EQ
    ConfStore["eq_adv_fwd_debug_local"] = lambda: default_eq_adv_cfg(
        debug=True, local=True, prover_kind=AdvProverKind.ForwardGreedy
    )
    ConfStore["eq_adv_bwd_debug_local"] = lambda: default_eq_adv_cfg(
        debug=True,
        local=True,
        prover_kind=AdvProverKind.BackwardGreedy,
        reload_bwd=PATHS["eq_graph_bwd_sup"],
        reload_gen=PATHS["eq_graph_gen"],
    )
    ConfStore["eq_adv_bwd_debug"] = lambda: default_eq_adv_cfg(
        debug=True,
        local=False,
        prover_kind=AdvProverKind.BackwardGreedy,
        reload_bwd=PATHS["eq_graph_bwd_sup"],
        reload_gen=PATHS["eq_graph_gen"],
    )
    ConfStore["eq_adv_bwd"] = lambda: default_eq_adv_cfg(
        debug=False,
        local=False,
        prover_kind=AdvProverKind.BackwardGreedy,
        reload_bwd=PATHS["eq_graph_bwd_sup"],
        reload_gen=PATHS["eq_graph_gen"],
    )
    ConfStore["eq_adv_bwd_mcts"] = lambda: default_eq_adv_cfg(
        debug=False,
        local=False,
        prover_kind=AdvProverKind.BackwardMCTS,
        reload_bwd=PATHS["eq_graph_bwd_sup"],
        reload_gen=PATHS["eq_graph_gen"],
    )
