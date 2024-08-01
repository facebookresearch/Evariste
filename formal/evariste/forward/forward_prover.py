# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import (
    List,
    Tuple,
    Iterator,
    Dict,
    Optional,
    Any,
    Generator,
    Type,
    TypeVar,
    Union,
)
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
import copy
import math
import time
from evariste.datasets.equations import EquationsDatasetConf
from evariste.forward.fwd_eq.gen.env import EqGenForwardEnv
from evariste.forward.fwd_eq.gen.proof_search import EqGenProofSearch
from evariste.forward.fwd_eq.gen.random_forward_policy import RandomForwardPolicy
import torch

from params import Params
from evariste.forward.core.forward_policy import ForwardPolicy, SyncForwardPolicy
from evariste.forward.env_specifics.prover_env_specifics import (
    FwdTrainParams,
    FwdTokenizer,
    ProverEnvSpecifics,
    AsyncForwardEnv,
)
from evariste.forward.model.seq2seq_model import Seq2SeqModel
from evariste.forward.proof_search import (
    StandardProofSearch,
    ForwardProofSearch,
    StopConfig,
    TreeProofSearch,
    SearchType,
)
from evariste.model.data.dictionary import Dictionary
from evariste.model.transformer import DecodingParams, TransformerModel
from evariste.forward.core.forward_stepper import ForwardStepper
from evariste.forward.core.forward_policy import BatchConfig
from evariste.forward.common import ForwardGoal, GenerationError

N_SIMULTANEOUS_PROOFS = 1024


logger = getLogger(__name__)


@dataclass
class SearchConfig(Params):
    max_nodes: int
    max_generations: int
    # If not None, the prover stop in n consecutive generation candidates are invalid
    max_cons_inv_allowed: int
    n_simultaneous_proofs: int = N_SIMULTANEOUS_PROOFS

    proof_search_type: SearchType = SearchType.STD
    seed: int = 43


@dataclass
class ProverConfig(Params):
    search_cfg: SearchConfig
    decoding_params: DecodingParams
    batch_config: BatchConfig
    fp16: bool = False
    async_model: bool = False
    name: str = "unk_prover_name"
    is_generator: bool = False


class ForwardProver:
    """
    IMPORTANT: this class is used to prove but also to generate. The behaviour
    will depend on the task that what used for training.
    To extend forward prover to new environment you will need to create env specifics:
    - FwdTokenizer
    - ForwardEnv
    - FwdTrainParams (small subset of TrainerArgs used by ForwardProver, how to
    create it depends on the env)
    Corresponding abstract classes are defined in forward.env_specifics.py
    To see how they are created from TrainerArgs for existing supported env, please
    look at `ForwardProver.from_trainer_args(...)` factory method.
    """

    def __init__(
        self,
        cfg: ProverConfig,
        # policy
        sync_policy: ForwardPolicy,
        # env specifics
        fwd_env: AsyncForwardEnv,
        train_params: Optional[FwdTrainParams] = None,
    ):
        use_async_model = cfg.async_model
        self.stepper = ForwardStepper(
            sync_policy=sync_policy, fwd_env=fwd_env, use_async_model=use_async_model,
        )
        self.search_cfg = cfg.search_cfg

        self.max_generations = (
            self.search_cfg.max_generations
            if self.search_cfg.max_generations is not None
            else math.inf
        )

        self.max_cons_inv_allowed = (
            self.search_cfg.max_cons_inv_allowed
            if self.search_cfg.max_cons_inv_allowed is not None
            else math.inf
        )

        # for batch proving
        self.n_simultaneous_proofs = self.search_cfg.n_simultaneous_proofs

        self._stats: Dict[str, Any] = defaultdict(float)

        self.verbose = False
        self.cfg = cfg

        self._proof_iterator: Optional[Generator] = None

        self.is_generator = cfg.is_generator
        self._train_params = train_params

    def generate_proofs(
        self, goals: Iterator[Optional[Tuple[int, ForwardGoal]]]
    ) -> Iterator[Tuple[int, Union[ForwardProofSearch, GenerationError]]]:
        """
        Main function of the ForwardProver.

        Note when the ForwardProver is used as a generator, this method allows
        also to generate. In this case the ForwardGoals are excpected to be without
        statements.
        """
        # note: this is a little bit hacky:
        # we create an intermediate proof_iterator and
        # store it as attribute of the prover to be able to
        # close this generator (and therefore the associated processes
        # spawned inside) when calling prover.close()
        if self._proof_iterator is not None:
            raise RuntimeError("Previous proof_iterator was not exhausted!")
        self._proof_iterator = self._make_proof_iterator(goals)
        yield from self._proof_iterator
        self._proof_iterator.close()
        self._proof_iterator = None
        return

    def _make_proof_iterator(
        self, goals: Iterator[Optional[Tuple[int, ForwardGoal]]]
    ) -> Generator[Tuple[int, Union[ForwardProofSearch, GenerationError]], None, None]:
        start = time.time()

        # Dict with input_idx as key
        proofs: Dict[int, ForwardProofSearch] = {}
        # list of search trees ids that need to be fed into the step generator
        to_process_proof_ids: List[int] = []

        n_steps = 0
        # mapping between the global graph id and the id of the tree that generated it +
        # the graph_id of this graph for this search tree
        graph_id2proof_id: Dict[int, Tuple[int, int]] = {}
        cur_global_id = 0
        should_stop = False

        while True:
            # 1. if not enough open proofs, open some proofs

            to_open_goals: List[Tuple[int, ForwardGoal]] = []
            for _ in range(self.n_simultaneous_proofs - len(proofs)):
                try:
                    goal_with_id = next(goals)
                except StopIteration:
                    should_stop = True
                    break
                if goal_with_id is None:  # hack for the adversarial
                    break
                assert goal_with_id is not None
                to_open_goals.append(goal_with_id)

            open_goals = self.stepper.start_goals(to_open_goals)
            for proof_id, maybe_goal in open_goals:
                if not maybe_goal.ok():
                    err = maybe_goal.err()
                    assert isinstance(err, GenerationError), type(err)
                    yield proof_id, err
                    continue
                goal = maybe_goal.unwrap()
                assert isinstance(goal, ForwardGoal)
                assert proof_id not in proofs, proof_id
                proofs[proof_id] = self.create_proof(goal)
                to_process_proof_ids.append(proof_id)

            if should_stop and len(proofs) == 0:
                break

            assert len(to_process_proof_ids) <= self.n_simultaneous_proofs

            if self.verbose:
                logger.info(
                    f"n proofs: {len(proofs)}, "
                    f"n_steps {n_steps}, "
                    f"to_p {len(to_process_proof_ids)}, "
                    f"valid {self._stats['valid_steps']}, "
                    f"invalid {self._stats['invalid_steps']}"
                )

            # 2. gather graphs to send
            to_send = []
            for proof_id in to_process_proof_ids:
                proof = proofs[proof_id]
                for id_in_proof, graph in proof.next_graphs():
                    graph_id = cur_global_id
                    cur_global_id += 1
                    assert graph_id not in graph_id2proof_id, graph_id
                    graph_id2proof_id[graph_id] = (proof_id, id_in_proof)
                    to_send.append((graph_id, graph))

            # 3. send graphs and receive ready steps
            gen_start = time.time()
            for graph_id, graph in to_send:
                self.stepper.send_graph(graph_id, graph)
            received = self.stepper.ready_steps()
            self._stats["time_in_stepper"] += time.time() - gen_start

            # 4. update proofs with received steps, remove finished proofs
            to_process_proof_ids = []
            to_finish_proof_ids = []
            for graph_id, step_beam in received:
                proof_id, id_in_proof = graph_id2proof_id.pop(graph_id)
                n_steps += 1
                for step_ in step_beam:
                    if step_.step:
                        self._stats["valid_steps"] += 1
                    else:
                        self._stats["invalid_steps"] += 1
                    self._stats["steps"] += 1
                proofs[proof_id].update_with_beam(
                    id_in_search=id_in_proof, step_beam=step_beam
                )
                if proofs[proof_id].should_stop():
                    to_finish_proof_ids.append(proof_id)
                else:
                    to_process_proof_ids.append(proof_id)

            self.stepper.end_goals(
                [proofs[proof_id].goal for proof_id in to_finish_proof_ids]
            )

            for proof_id in to_finish_proof_ids:
                # no need anymore in the buffer
                proof = proofs.pop(proof_id)
                proof.finish()
                yield proof_id, proof

        assert len(proofs) == len(to_process_proof_ids) == 0
        # stats
        self._stats["duration"] = time.time() - start
        self._stats["steps_per_sec"] = self._stats["steps"] / self._stats["duration"]

    @property
    def stats(self):
        stats = dict(self._stats)
        stats.update(self.stepper.stats)
        return stats

    def create_proof(self, goal: ForwardGoal) -> ForwardProofSearch:
        if self.search_cfg.proof_search_type == "std":
            stop_cfg = StopConfig(
                max_nodes=self.search_cfg.max_nodes,
                max_generations=self.search_cfg.max_generations,
                max_cons_inv_allowed=self.search_cfg.max_cons_inv_allowed,
            )
            if isinstance(self.stepper.fwd_env, EqGenForwardEnv):
                env = self.stepper.fwd_env
                assert isinstance(env, EqGenForwardEnv)
                return EqGenProofSearch(stop_cfg, env=env, init_graph=env.get_init())
            return StandardProofSearch(
                stop_config=stop_cfg, goal=goal, is_generation=self.is_generator
            )
        elif self.search_cfg.proof_search_type in {
            SearchType.DFS,
            SearchType.BFS,
            SearchType.OPEN_AI,
        }:
            return TreeProofSearch(
                goal=goal, search_type=self.search_cfg.proof_search_type
            )
        else:
            raise NotImplementedError(self.search_cfg.proof_search_type)

    @property
    def fwd_env(self) -> AsyncForwardEnv:
        return self.stepper.fwd_env

    @property
    def dico(self) -> Dictionary:
        return self.stepper.sync_policy.dico

    @property
    def tokenizer(self) -> FwdTokenizer:
        return self.stepper.sync_policy.tokenizer

    @property
    def train_params(self) -> FwdTrainParams:
        assert self._train_params is not None
        return self._train_params

    @property
    def encoder(self) -> TransformerModel:
        return self.stepper.sync_policy.model.encoder

    @property
    def decoder(self) -> TransformerModel:
        return self.stepper.sync_policy.model.decoder

    @property
    def critic(self) -> Optional[torch.nn.Module]:
        return self.stepper.sync_policy.model.critic

    def close(self):
        self.stepper.close()
        if self._proof_iterator:
            # we close the proof loop and therefore associated resources
            logger.info("Closing proof loop")
            self._proof_iterator.close()
            self._proof_iterator = None

    def __del__(self):
        self.close()

    def reload_model_weights(self):
        self.stepper.reload_model_weights()

    # factory method
    @staticmethod
    def from_trainer_args(
        cfg: ProverConfig,
        prover_env_specifics: ProverEnvSpecifics,
        dico: Dictionary,
        params: Any,
        # models
        encoder: TransformerModel,
        decoder: TransformerModel,
        critic: Optional[torch.nn.Module] = None,
        overwrite_max_inp_len: Optional[int] = None,
    ) -> "ForwardProver":
        from evariste.trainer.args import TrainerArgs

        assert isinstance(params, TrainerArgs)

        train_params = prover_env_specifics.fwd_params
        if overwrite_max_inp_len:
            train_params.max_inp_len = overwrite_max_inp_len

        logger.info(f"Initializing forward with train_params: {train_params}")
        new_decoding_params = copy.deepcopy(cfg.decoding_params)
        new_decoding_params.stop_symbol = train_params.stop_symbol
        new_decoding_params.prefix = train_params.command_prefix

        seq2seq = Seq2SeqModel(
            encoder=encoder,
            decoder=decoder,
            critic=critic,
            decoding_params=new_decoding_params,
            use_ptrs=False,
            max_len=train_params.max_inp_len,
            train_dir=train_params.train_dir,
            discr_conditioning=train_params.discr_conditioning,
            fp16=cfg.fp16,
        )

        sync_policy = SyncForwardPolicy(
            fwd_tokenizer=prover_env_specifics.tokenizer,
            model=seq2seq,
            batch_cfg=cfg.batch_config,
            dico=dico,
            max_len=train_params.max_inp_len,
            allow_stop=train_params.use_stop_action,
            stop_command=train_params.stop_command,
        )

        cfg.is_generator = train_params.is_generator
        forward_prover = ForwardProver(
            cfg=cfg,
            sync_policy=sync_policy,
            fwd_env=prover_env_specifics.env,
            train_params=train_params,
        )
        return forward_prover

    # factory method
    @staticmethod
    def from_random_args(
        cfg: ProverConfig,
        prover_env_specifics: ProverEnvSpecifics,
        params: EquationsDatasetConf,
    ) -> "ForwardProver":
        assert isinstance(prover_env_specifics.env, EqGenForwardEnv)
        sync_policy: RandomForwardPolicy = RandomForwardPolicy.get_policy(
            fwd_env=prover_env_specifics.env, params=params
        )
        forward_prover = ForwardProver(
            cfg=cfg, sync_policy=sync_policy, fwd_env=prover_env_specifics.env,
        )

        forward_prover.is_generator = True
        return forward_prover
