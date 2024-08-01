# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import os
import time
from collections import deque, Counter
from logging import getLogger
from numbers import Number
from typing import Dict, Optional, Any, List, Tuple, Type, Set

from numpy.random import RandomState


from evariste.backward.graph import Theorem
from evariste.comms.store import Receiver
from evariste.comms.zip import ZipReceiver
from evariste.forward.common import (
    GenerationHistory,
    GenericForwardTactic,
    ProofNode,
    GenerationError,
)
from evariste.forward.core.generation_errors import (
    InvalidTactic,
    ParseGoalError,
    NodeInGraph,
    MissingHyp,
)
from evariste.forward.fwd_lean.training.lean_graph_sampler import (
    build_sample_dict,
    TooLong,
    DicoError,
)
from evariste.forward.online_generation.online_generation_common import FwdTrajectory
from evariste.forward.online_generation.worker_type import WorkerType
from evariste.forward.training.graph_sampler import GraphSampler
from evariste.forward.training.helpers import tokenize_fwd_graph, tokenize_command
from evariste.model.data.dictionary import Dictionary

SUPPORTED_ERRORS: Set[Type[GenerationError]] = {
    InvalidTactic,
    ParseGoalError,
    NodeInGraph,
    MissingHyp,
}

logger = getLogger()


class OnlineErrorSampler(GraphSampler):
    """Class that is responsible for storing and sampling
    the error-conditioned samples"""

    def __init__(
        self,
        receiver: Receiver[FwdTrajectory],
        dico: Dictionary,
        max_len: int,
        max_samples: int,
        refresh_every: int,
        env_name: str,
        next_node_first: bool,  # TODO: create cfg with tokenizer options?
    ):
        self.dico = dico
        self.max_len = max_len
        self.receiver = receiver
        self.iteration = 0
        self.max_samples = max_samples
        self.refresh_every = refresh_every
        self.samples = []
        self.sample_deque = deque(maxlen=self.max_samples)
        self.next_node_first = next_node_first
        self.env_name = env_name

    def _refresh_data(self, reload: bool = False):
        name = f"{self.__class__.__name__}"
        logger.info(f"(PID: {os.getpid()}) - {name}: start refreshing data")
        if reload and isinstance(self.receiver, ZipReceiver):
            # in case of checkpointing we reload last data stored in zip
            trajs = self.receiver.reload_last_chunks(n_sequences=10_000)
        else:
            trajs = self.receiver.receive_batch()
        for traj in trajs:
            history = traj.history
            samples = make_error_conditioning_samples(
                history, next_node_first=self.next_node_first, env_name=self.env_name
            )
            built = []
            for (enc_inp, dec_out, label) in samples:
                try:
                    sample = build_sample_dict(
                        enc_inp, dec_out, label, max_len=self.max_len, dico=self.dico
                    )
                except (TooLong, DicoError):
                    continue
                built.append(sample)

            self.sample_deque.extend(built)

        present = set([])
        for sample in self.sample_deque:
            key = (tuple(sample["x"]), tuple(sample["y"]))
            if key not in present:
                self.samples.append(sample)
                present.add(key)

        if self.samples:
            n_labels = len({x["name"] for x in self.samples})
            n_samples = len(self.samples)
            error_codes = Counter(self.dico.id2word[s["x"][1]] for s in self.samples)
            logger.info(
                f"(PID: {os.getpid()}) - {name}:"
                f"refreshed data"
                f" loaded {len(trajs)} new trajs,"
                f" n_samples: {n_samples},"
                f" n_labels: {n_labels}"
                f" error_codes: {error_codes.most_common()}"
            )
        else:
            logger.info(f"(PID: {os.getpid()}) - {name}: no data yet")

    def get_graph_sample(
        self, split: str, task: str, rng: RandomState
    ) -> Optional[Dict[str, Any]]:
        assert split == "train"
        wait = 2
        reload = True
        while len(self.samples) == 0:
            logger.info(
                f"(PID: {os.getpid()}) Waiting {wait}s to have samples in "
                f"{self.__class__.__name__}"
            )
            time.sleep(wait)
            wait = min(60, 2 * wait)
            self._refresh_data(reload=reload)
            reload = False

        if self.iteration % self.refresh_every == 0:
            self._refresh_data()

        assert len(self.samples) > 0
        index = rng.randint(len(self.samples))
        self.iteration += 1
        return self.samples[index]

    def get_stats(self) -> Dict[str, Number]:
        # TODO: plug refresh stats
        pass

    def close(self):
        pass

    def error_codes(self) -> List[str]:
        codes = [
            make_error_token(err_type("dummy").type) for err_type in SUPPORTED_ERRORS
        ]
        if self.env_name == "lean":
            from evariste.backward.env.lean.graph import lean_error_str

            codes.extend(
                [make_error_token(error_str[0]) for error_str in lean_error_str]
            )
            codes.append(make_error_token("UNK_ERROR"))
        else:
            raise NotImplementedError(self.env_name)
        return codes

    @staticmethod
    def from_trainer_args(
        params: Any, env_name: str, dico: Dictionary
    ) -> "OnlineErrorSampler":
        from evariste.trainer.args import TrainerArgs
        from evariste.comms.comms import make_receiver
        from evariste.comms.rl_distributed_config import RLDistributedConfig

        # for circular imports
        assert isinstance(params, TrainerArgs)
        rl_dist_cfg = RLDistributedConfig.new_fwd_distributed_rl_cfg(params)
        logger.info(
            f"Creating new RLDistributedConfig for this fwd rl setup: {rl_dist_cfg}"
        )

        receiver = make_receiver(
            rank=params.slurm_conf.global_rank,
            receiver_type=WorkerType.PROVER_TRAINER,
            cfg=rl_dist_cfg,
        )
        max_samples = params.online_gen_cfg.n_max_samples
        if env_name == "lean":
            next_node_first = params.lean.graph.next_node_first
        else:
            raise NotImplementedError(env_name)
        return OnlineErrorSampler(
            receiver=receiver,
            dico=dico,
            max_len=params.batch.max_len,
            next_node_first=next_node_first,
            max_samples=max_samples,
            refresh_every=params.online_gen_cfg.refresh_every,
            env_name=env_name,
        )


def make_error_conditioning_samples(
    history: GenerationHistory, next_node_first: bool, env_name: str
) -> List[Tuple[List[str], List[str], str]]:
    assert history.goal.is_new_fmt()
    graph: List[Theorem] = []
    goal = history.goal.thm
    label = history.goal.label
    samples = []
    for maybe_step in history.stack:
        if maybe_step.step:
            graph.append(maybe_step.step.generated)
            continue
        error = maybe_step.err
        assert error is not None
        if type(error) not in SUPPORTED_ERRORS:
            continue
        if isinstance(error, InvalidTactic):
            error_code = get_error_code(error.msg, env_name)
        else:
            error_code = error.type

        error_token = make_error_token(error_code)

        # maybe make specific subclass ?
        assert hasattr(error, "fwd_tactic")
        fwd_tactic = error.fwd_tactic
        assert isinstance(fwd_tactic, GenericForwardTactic)

        # to be allowed to tokenize
        tactic = copy.deepcopy(fwd_tactic.bwd_tactic)
        tactic.is_valid = True

        dec_out = tokenize_command(
            ProofNode(theorem=fwd_tactic.next_node, tactic=tactic, children=[]),
            next_node_first=next_node_first,
        )

        enc_inp = tokenize_fwd_graph(
            goal=goal, graph=graph, include_goal=True, conditioning=[error_token]
        )
        sample = (enc_inp, dec_out, label)
        samples.append(sample)
    return samples


def make_error_token(error_type: str) -> str:
    return f"<ERROR:{error_type}>"


def get_error_code(error_msg: str, env_name: str):
    if env_name == "lean":
        from evariste.backward.env.lean.graph import get_error

        return get_error(error_msg)
    else:
        raise NotImplementedError(env_name)
