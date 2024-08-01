# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from enum import Enum, unique
from typing import List, Dict

from evariste.forward.online_generation.worker_type import WorkerType
from params import Params


@unique
class DistributedSetup(str, Enum):
    NONE = "none"
    ADVERSARIAL = "adversarial"
    ADV_FIXED_PROVER = "adv_fixed_prover"
    ADV_FIXED_GENERATOR = "adv_fixed_generator"

    # generator trained on generation reward, but still prover trained on hard mined
    # generations from this generator
    ADV_GENERATOR_SL = "adv_generator_sl"

    # generator trained with supervised learning, but still prover trained on hard mined
    # generations from this generator
    ADV_GENERATOR_REWARD = "adv_generator_reward"

    # no provers
    GENERATOR_ONLY = "generator_only"

    # no generators
    PROVER_ONLY = "prover_only"


@dataclass
class RLDistributedConfig(Params):
    """
    Config to parametrise the communications across workers.
    Used to initialise senders and receivers.

    WARNING: As a user you should not create one directly via cli.
    It should be created from AdversarialConfig for adversarial setup,
    or from TrainerArgs for fwd_rl setup. See the two factory methods.
    """

    exp_root_path: str = ""

    is_adversarial_training: bool = False
    is_fwd_rl_training: bool = False

    distributed_setup: DistributedSetup = DistributedSetup.NONE

    use_zmq: bool = True
    zip_chunk_size: int = 2048  # used only if zip

    n_provers: int = -1
    n_generators: int = -1
    n_prover_trainers: int = -1
    n_generator_trainers: int = -1

    @property
    def generator_actor_send_to(self) -> List[WorkerType]:
        return _SETUP_TO_CFG[self.distributed_setup].generator_actor_send_to

    @property
    def prover_actor_send_to(self) -> List[WorkerType]:
        return _SETUP_TO_CFG[self.distributed_setup].prover_actor_send_to

    @property
    def is_rl_distributed_training(self):
        return self.is_fwd_rl_training or self.is_adversarial_training

    def __post_init__(self):
        assert not (self.is_fwd_rl_training and self.is_adversarial_training)

        if self.is_adversarial_training:
            assert self.distributed_setup != DistributedSetup.NONE

        if self.is_fwd_rl_training:
            assert self.distributed_setup == DistributedSetup.PROVER_ONLY

        if self.is_rl_distributed_training:
            # we check that everything is as expected
            setup_cfg = _SETUP_TO_CFG[self.distributed_setup]
            assert self.n_generators >= 0
            assert self.n_generator_trainers >= 0
            assert self.n_provers >= 0
            assert self.n_prover_trainers >= 0
            assert (self.n_provers == 0) == setup_cfg.no_prover_actor
            assert (self.n_prover_trainers == 0) == setup_cfg.no_prover_trainer
            assert (self.n_generator_trainers == 0) == setup_cfg.no_generator_trainer
            assert (self.n_generators == 0) == setup_cfg.no_generator_actor

        # prover actor only sends to trainers
        assert all(wt.is_trainer() for wt in self.prover_actor_send_to)

    def get_receiver_types(self, sender_type: WorkerType) -> List[WorkerType]:
        if sender_type == WorkerType.GENERATOR_ACTOR:
            return self.generator_actor_send_to
        elif sender_type == WorkerType.PROVER_ACTOR:
            return self.prover_actor_send_to
        else:
            raise NotImplementedError(sender_type)

    def get_sender_types(self, receiver_type: WorkerType) -> List[WorkerType]:
        sender_to_recs = {
            WorkerType.GENERATOR_ACTOR: set(self.generator_actor_send_to),
            WorkerType.PROVER_ACTOR: set(self.prover_actor_send_to),
        }
        return [s for s, rs in sender_to_recs.items() if receiver_type in rs]

    @staticmethod
    def from_adv_cfg(adv_cfg) -> "RLDistributedConfig":
        from configs.adv_configs import AdversarialConfig

        assert isinstance(adv_cfg, AdversarialConfig)

        # TODO: instead of registering setups here,
        #  let the user the possibility to enter directly the
        #  RLDistributedConfig in the AdversarialConfig
        #  using a prover_actor_send_to_str field
        #  using a generator_actor_send_to_str field

        return RLDistributedConfig(
            is_adversarial_training=True,
            n_provers=adv_cfg.num_prover_actors,
            n_generators=adv_cfg.num_generator_actors,
            n_prover_trainers=adv_cfg.num_prover_trainers,
            n_generator_trainers=adv_cfg.num_generator_trainers,
            use_zmq=adv_cfg.use_zmq,
            exp_root_path=adv_cfg.exp_path,
            distributed_setup=adv_cfg.distributed_setup,
        )

    @staticmethod
    def new_fwd_distributed_rl_cfg(params) -> "RLDistributedConfig":
        """
        Create a RLDistributedConfig from TrainerArgs. This is needed when doing
        fwd online generation (NOT adversarial training).
        Indeed the cfg in params.rl_distributed is created only in
        adversarial training by adversarial_train script, so we need
        to create a new one for fwd distributed rl.
        """
        from evariste.trainer.args import TrainerArgs  # circular

        assert isinstance(params, TrainerArgs)
        trainer_ws = params.slurm_conf.torch_world_size
        actors_ws = params.slurm_conf.world_size - trainer_ws

        return RLDistributedConfig(
            n_provers=actors_ws,
            n_prover_trainers=trainer_ws,
            exp_root_path=params.dump_path,
            use_zmq=False,
            zip_chunk_size=params.online_gen_cfg.zip_chunk_size,
            n_generators=0,
            n_generator_trainers=0,
            is_fwd_rl_training=True,
            distributed_setup=DistributedSetup.PROVER_ONLY,
        )


@dataclass
class _DistributedSetupCfg:
    """Not exposed to user, dataclass to store different setups"""

    generator_actor_send_to: List[WorkerType]
    prover_actor_send_to: List[WorkerType]
    no_generator_actor: bool = False
    no_prover_actor: bool = False
    no_generator_trainer: bool = False
    no_prover_trainer: bool = False

    def __post_init__(self):

        assert self.no_prover_actor == (len(self.prover_actor_send_to) == 0)
        assert self.no_generator_actor == (len(self.generator_actor_send_to) == 0)

        if self.no_prover_actor:
            assert self._is_not_receiving(WorkerType.PROVER_ACTOR)
        if self.no_generator_actor:  # not used?
            assert self._is_not_receiving(WorkerType.GENERATOR_ACTOR)
        if self.no_prover_trainer:
            assert self._is_not_receiving(WorkerType.PROVER_TRAINER)
        if self.no_generator_trainer:
            assert self._is_not_receiving(WorkerType.GENERATOR_TRAINER)

        for wt in self.prover_actor_send_to:
            assert wt in {WorkerType.GENERATOR_TRAINER, WorkerType.PROVER_TRAINER}
        for wt in self.generator_actor_send_to:
            assert wt in {WorkerType.GENERATOR_TRAINER, WorkerType.PROVER_ACTOR}

    def _is_not_receiving(self, wt: WorkerType):
        return not (wt in self.prover_actor_send_to + self.generator_actor_send_to)


_SETUP_TO_CFG: Dict[DistributedSetup, _DistributedSetupCfg] = {
    DistributedSetup.NONE: _DistributedSetupCfg(
        generator_actor_send_to=[],
        prover_actor_send_to=[],
        no_prover_actor=True,
        no_generator_trainer=True,
        no_prover_trainer=True,
        no_generator_actor=True,
    ),
    DistributedSetup.ADVERSARIAL: _DistributedSetupCfg(
        generator_actor_send_to=[WorkerType.PROVER_ACTOR],
        prover_actor_send_to=[WorkerType.GENERATOR_TRAINER, WorkerType.PROVER_TRAINER],
    ),
    DistributedSetup.GENERATOR_ONLY: _DistributedSetupCfg(
        generator_actor_send_to=[WorkerType.GENERATOR_TRAINER],
        prover_actor_send_to=[],
        no_prover_actor=True,
        no_prover_trainer=True,
    ),
    DistributedSetup.PROVER_ONLY: _DistributedSetupCfg(
        generator_actor_send_to=[],
        prover_actor_send_to=[WorkerType.PROVER_TRAINER],
        no_generator_actor=True,
        no_generator_trainer=True,
    ),
    DistributedSetup.ADV_FIXED_PROVER: _DistributedSetupCfg(
        generator_actor_send_to=[WorkerType.PROVER_ACTOR],
        prover_actor_send_to=[WorkerType.GENERATOR_TRAINER],
        no_prover_trainer=True,
    ),
    DistributedSetup.ADV_FIXED_GENERATOR: _DistributedSetupCfg(
        generator_actor_send_to=[WorkerType.PROVER_ACTOR],
        prover_actor_send_to=[WorkerType.PROVER_TRAINER],
        no_generator_trainer=True,
    ),
    DistributedSetup.ADV_GENERATOR_REWARD: _DistributedSetupCfg(
        generator_actor_send_to=[WorkerType.PROVER_ACTOR, WorkerType.GENERATOR_TRAINER],
        prover_actor_send_to=[WorkerType.PROVER_TRAINER],
    ),
    DistributedSetup.ADV_GENERATOR_SL: _DistributedSetupCfg(
        generator_actor_send_to=[WorkerType.PROVER_ACTOR],
        prover_actor_send_to=[WorkerType.PROVER_TRAINER],
    ),
}
