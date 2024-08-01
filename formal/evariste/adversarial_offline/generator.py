# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import itertools
import time
from typing import Any, Tuple, Iterator, Union, Iterable, Dict, List, Optional
from pathlib import Path
from glob import glob
import logging
from dataclasses import asdict
from evariste.envs.eq.env import EqGraphSamplerParams
from evariste.envs.eq.generation import (
    EquationGraphSampler,
    EquationGraphStats,
    GraphNode,
)
import gc
from evariste.forward.fwd_eq.eq_fwd_env import EqForwardEnv

from evariste.forward.fwd_eq.history_to_eq_nodes import history_to_eq_nodes
from evariste.model.data.dictionary import UNPROVED_WORD

import torch.cuda

from evariste.adversarial_offline.args import GeneratorArgs
from evariste.adversarial_offline.replay_backward import BackwardReplayer
from evariste.backward.prover.utils import GPUMonitor
from evariste.forward import forward_model_factory
from evariste.adversarial.generator import env_goal_stream
from numpy.random import RandomState

from evariste.backward.prover.prover import ProofResult, init_and_run_prover
from evariste.backward.env.equations import EQTheorem
from evariste.backward.env.core import BackwardGoal
from evariste.forward.core.generation_errors import GenerationError
from evariste.forward.fwd_eq.eq_env_helper import EQFwdEnvHelper
from evariste.forward.fwd_eq.eq_helpers import evaluate_eq_generations
from evariste.forward.proof_search import SearchType, StandardProofSearch
from evariste.forward.prover_loss_search import ProverLossForwardProver
from evariste.forward.proof_search import ForwardProofSearch
from evariste.forward.fwd_eq.gen.proof_search import EqGenProofSearch
from evariste.forward.common import ForwardGoal
from evariste.metrics import Logger, ActionCounter
from evariste.utils import load_stream, stream_saver
from evariste import json


class OfflineGenerator:
    """
    Class for running forward generators with filters in an offline (file stream) setup.
    Use sequential=True to load generator and prover to the same GPU. In this case, specify the finite number of
    generations to be retrieved before loading the prover.

    :param cfg.sequential: If true, use a single GPU and first generate then prove. Otherwise, generate en prove in parallel on two GPUS.
      Note, the non sequential mode is GPU wasteful for the prover which doesn't have enough work to do.
    :param cfg.load_path_globs: Instead of generating goals, load from a globby (ie, with *) path
    """

    def __init__(self, cfg: GeneratorArgs):
        self.env_helper = None
        self.generator = None
        self.rng = RandomState(cfg.seed)
        self.goal_stream = None
        self.generations = ActionCounter(name="generations", is_rate=True)
        self.hard_generations = ActionCounter(name="hard_generations", is_rate=False)
        self.invalid_generations = ActionCounter(
            name="invalid_generations", is_rate=False
        )
        self.steps = ActionCounter(name="steps", is_rate=False)
        self.valid_steps = ActionCounter(name="valid_steps", is_rate=False)

        logging.info(f"Creating dump folder: {cfg.dump_path}")
        cfg.dump_path.mkdir(parents=True, exist_ok=True)
        with open(cfg.dump_path / "params.json", "w") as f:
            json.dump(asdict(cfg), f, sort_keys=True, indent=4)

        self.generations_path = cfg.dump_path / "generations"
        self.generations_path.mkdir(exist_ok=True)
        self.metrics = Logger(outdir=cfg.dump_path, tag="gen")
        self.metrics.log_config(cfg)
        self.gpu_mon = GPUMonitor(delay=1.0)
        self.sampler = EquationGraphSampler(rng=self.rng, params=cfg.sample_cfg)

        # If only a finite number of generations is expected, do not open more proofs than necessary.
        if cfg.n_gens:
            cfg.forward_cfg.search_cfg.n_simultaneous_proofs = min(
                cfg.forward_cfg.search_cfg.n_simultaneous_proofs, cfg.n_gens
            )

        self.cfg = cfg
        self.goal_stream = self.make_goal_stream()
        if cfg.n_gens:
            logging.info(f"Retrieving {cfg.n_gens} generations.")
            self.goal_stream = itertools.islice(self.goal_stream, cfg.n_gens)

    def make_goal_stream(self):
        """
        Creates the BackwardGoal stream, either by launching a forward model, or loading from directories
        """
        if self.cfg.load_path_globs:
            # load proof goals from directories
            dirs = [
                Path(d)
                for pattern in self.cfg.load_path_globs.split(sep=",")
                for d in glob(pattern)
            ]
            goal_stream = load_stream(dirs)
            logging.info(f"Opened goal stream from directories {dirs}")
        else:
            # generate proof goals

            # TODO: make robust
            if self.cfg.sequential:
                device_str = f"cuda:{torch.cuda.current_device()}"
            else:
                # init_and_run_prover chooses cuda.current_device, so we pick cuda:1 for the generator
                device_str = "cuda:1"

            if self.cfg.gen_checkpoint_path != "random":
                (
                    self.generator,
                    _,
                    params,
                    self.env_helper,
                ) = forward_model_factory.from_checkpoint(
                    ckpt_path=str(self.cfg.gen_checkpoint_path),
                    device_str=device_str,
                    cfg=self.cfg.forward_cfg,
                    # overwrite_tasks="eq_gen_notask",
                    overwrite_gen_type="graph",
                    overwrite_prefix=None
                    if not self.cfg.prefix_with_unproved
                    else [UNPROVED_WORD],
                )
                self.cfg.dataset_conf = params.eq.dataset
            else:
                from evariste.datasets.equations import ConfStore

                # from evariste.forward.common import ForwardGoal
                # from evariste.forward.core.generation_errors import GenerationError
                # from evariste.forward.fwd_eq.gen.proof_search import EqGenProofSearch
                from evariste.forward.forward_prover import ForwardProver

                (
                    _,
                    _,
                    params,
                    self.env_helper,
                ) = forward_model_factory.from_checkpoint(
                    ckpt_path="",
                    device_str=device_str,
                    cfg=self.cfg.forward_cfg,
                    overwrite_gen_type="graph",
                    overwrite_prefix=None,
                )
                self.cfg.dataset_conf = ConfStore["eq_dataset_lean"]
                self.cfg.dataset_conf.gen_type = "graph"
                self.generator = ForwardProver.from_random_args(
                    self.cfg.forward_cfg,
                    self.env_helper.get_prover_env_specifics(),
                    self.cfg.dataset_conf,
                )
            logging.info(
                f"Loaded generator to {device_str} from checkpoint: {self.cfg.gen_checkpoint_path}"
            )
            if (
                self.cfg.forward_cfg.search_cfg.proof_search_type
                == SearchType.PROVER_LOSS
            ):
                if not isinstance(self.generator, ProverLossForwardProver):
                    logging.info(f"Making the generator use prover losses...")
                    self.generator = ProverLossForwardProver.from_forward_prover(
                        self.generator,
                        bwd_prover_params=self.cfg.prover_params,
                        bwd_decoding_params=self.cfg.decoding_params,
                        max_tokens=self.cfg.max_tokens,
                        use_critic=self.cfg.use_bwd_critic,
                    )
            self.generator.verbose = True

            def dummy_goal_stream() -> Iterator[Tuple[int, ForwardGoal]]:
                i = 0
                while True:
                    yield i, ForwardGoal(thm=None, label=f"unused_{i}")
                    i += 1

            # hyp_str = env_goal_stream(self.rng, self.env_helper, split="train")
            proof_str = self.generator.generate_proofs(dummy_goal_stream())
            if not self.cfg.check_backward_proof:
                goal_stream = self.to_goal_stream(proof_str)
            else:
                fwd_env = self.generator.fwd_env
                assert isinstance(fwd_env, EqForwardEnv)
                self.bwd_replayer = BackwardReplayer(
                    dataset=self.cfg.dataset_conf, eq_env=fwd_env.eq_env
                )

                def checked_stream(
                    input_stream,
                ) -> Iterator[
                    Tuple[
                        Optional[BackwardGoal],
                        Union[EqGenProofSearch, GenerationError],
                    ]
                ]:
                    for goal, ground_truth_proof in input_stream:
                        if isinstance(ground_truth_proof, EqGenProofSearch):
                            try:
                                self.bwd_replayer.replay_proof(goal, ground_truth_proof)
                            except RuntimeError as e:
                                raise e
                        yield goal, ground_truth_proof

                goal_stream = checked_stream(self.to_goal_stream(proof_str))

        return goal_stream

    def filter_generations(
        self,
    ) -> Iterator[
        Tuple[
            str, Optional[BackwardGoal], Union[EqGenProofSearch, GenerationError], bool,
        ]
    ]:
        """
        Runs the prover over the goal stream, returning the proved status and saving to
        """
        if self.cfg.verbose:
            logging.info("Entered filter_generations")
        if self.cfg.sequential:
            assert self.goal_stream is not None
            self.goal_stream = list(self.goal_stream)
            assert self.generator is not None
            self.generator.close()
            gc.collect()
            torch.cuda.empty_cache()
        for x in stream_saver(
            self.filter_unproved_goals(self.prove_backward_goals()),
            path=self.generations_path,
            chunk_length=self.cfg.save_chunk_length,
        ):
            if self.cfg.verbose:
                logging.info(x)
            yield x

    def do_filter_generations(
        self,
    ) -> List[
        Tuple[
            str, Optional[BackwardGoal], Union[EqGenProofSearch, GenerationError], bool,
        ]
    ]:
        return list(self.filter_generations())

    def prove_backward_goals(
        self,
    ) -> Iterator[
        Tuple[str, Optional[ProofResult], Union[EqGenProofSearch, GenerationError]]
    ]:
        if self.cfg.verbose:
            logging.info("Entered prove_backward_goals")
        assert isinstance(self.goal_stream, Iterable)
        local_goal_stream, self.goal_stream = itertools.tee(self.goal_stream)

        buffer = {}

        def input_it() -> Iterator[BackwardGoal]:
            assert isinstance(local_goal_stream, Iterable)
            for goal, ground_truth_proof in local_goal_stream:
                if isinstance(ground_truth_proof, EqGenProofSearch):
                    buffer[goal.name] = ground_truth_proof
                    yield goal

        pf_iter = init_and_run_prover(
            dataset=self.cfg.dataset_conf,
            decoding=self.cfg.decoding_params,
            prover_params=self.cfg.prover_params,
            input_it=input_it(),
            decoder_type="decoder",
        )
        for goal, ground_truth_proof in self.goal_stream:
            if isinstance(ground_truth_proof, EqGenProofSearch):
                # retrieve one proof (unrelated to `goal`!) - but `goal` will be worked on eventually by the prover as well
                x = next(pf_iter)
                result = x[0]
                assert isinstance(result, ProofResult)
                if self.cfg.verbose:
                    logging.info(f"Received proof result for {result.goal.name}.")
                yield result.goal.name, result, buffer[result.goal.name]
                del buffer[result.goal.name]
            else:
                assert isinstance(ground_truth_proof, GenerationError)
                yield goal.name, None, ground_truth_proof

    def to_goal_stream(
        self,
        proof_stream: Iterator[Tuple[int, Union[ForwardProofSearch, GenerationError]]],
    ) -> Iterator[
        Tuple[Optional[BackwardGoal], Union[EqGenProofSearch, GenerationError]]
    ]:
        """Generate ForwardGoals according to the sample_cfg."""

        for _, proof in proof_stream:
            if isinstance(proof, GenerationError):
                self.invalid_generations.act(1)
                if self.cfg.include_errors:
                    yield None, proof
                continue
            assert isinstance(proof, EqGenProofSearch), type(proof)
            g = proof.next_graph

            assert isinstance(self.env_helper, EQFwdEnvHelper)
            env = self.env_helper.eq_data_env
            graph = EquationGraphStats(nodes=g.nodes, rules=env.rules)

            # retrieve best node
            ids, _ = self.sampler.sample(graph=graph, n_samples=1, greedy=True)
            if not ids:
                self.invalid_generations.act(1)
                if self.cfg.include_errors:
                    yield None, proof
                continue
            node = g.nodes[ids[0]]

            hyps = g.get_hyps_for_node(node)

            # construct backward goal based on the node and all hyps (use only needed ones?)
            goal_stmt = EQTheorem(node=node.node, hyps=[h.node for h in hyps])
            bwd_goal = BackwardGoal(goal_stmt, self.cfg.label)
            self.invalid_generations.act(0)
            yield bwd_goal, proof

    def compare_selected_nodes(
        self, graph: EquationGraphStats
    ) -> Dict[str, Optional[int]]:
        res: Dict[str, Optional[int]] = {}
        for strat in ["depth", "size", "sd_ratio", "rule", "prefix_len"]:
            params = EqGraphSamplerParams()
            setattr(params, f"{strat}_weight", 1.0)
            ids, _ = EquationGraphSampler(self.rng, params).sample(
                graph, n_samples=1, greedy=True
            )
            if not ids:
                res[strat] = None
            else:
                res[strat] = ids[0]
        res["last"] = len(graph.nodes) - 1
        return res

    def filter_unproved_goals(
        self,
        proof_stream: Iterator[
            Tuple[str, Optional[ProofResult], Union[EqGenProofSearch, GenerationError]]
        ],
    ) -> Iterator[
        Tuple[
            str, Optional[BackwardGoal], Union[EqGenProofSearch, GenerationError], bool,
        ]
    ]:
        last_log = time.time()
        hard = 0
        for proof_id, pf_res, ground_truth_pf in proof_stream:
            self.generations.act()
            if isinstance(ground_truth_pf, EqGenProofSearch):
                self.steps.act(len(ground_truth_pf.all_steps))
                self.valid_steps.act(len(ground_truth_pf.steps))
                assert isinstance(pf_res, ProofResult)
                hard = 0 if pf_res.proof else 1
                self.hard_generations.act(hard)

                if hard or self.cfg.return_all:
                    proved = pf_res is not None and pf_res.proof is not None
                    yield proof_id, pf_res.goal if pf_res else None, ground_truth_pf, proved

            if time.time() - last_log > self.cfg.log_interval:
                last_log = time.time()
                gens_rate = self.generations.rate()
                hard_prop = self.hard_generations.rate()
                invalid_prop = self.invalid_generations.rate()
                log_data = {
                    "gens/s": gens_rate,
                    "hard_proportion": hard_prop,
                    "hard_gens/s": gens_rate * hard_prop,
                    "invalid_proportion": invalid_prop,
                    "invalid_gens/s": gens_rate * invalid_prop,
                    "avg_steps": self.steps.rate(),
                    "avg_valid_steps": self.valid_steps.rate(),
                }
                for i, stat in enumerate(self.gpu_mon.stats):
                    log_data[f"gpu_util_{i}"] = stat.stats.get("gpu", -1)
                    log_data[f"gpu_mem_{i}"] = stat.stats.get("mem", -1)
                if not self.cfg.all_time_averages:
                    self.generations.reset()
                    self.hard_generations.reset()
                    self.invalid_generations.reset()
                    self.steps.reset()
                    self.valid_steps.reset()

                self.metrics.log_metrics(log_data)
                for k, v in log_data.items():
                    logging.info(f"Stats: {k}: {v}")
                logging.info(log_data)
                for stat in self.gpu_mon.stats:
                    stat.reset()

    def close(self):
        if self.generator:
            self.generator.close()
        if self.env_helper:
            self.env_helper.close()
        if self.gpu_mon:
            self.gpu_mon.close()
        if self.metrics:
            self.metrics.close()

    def finite_generation_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Performs a finite generation and returns the number of GenerationErrors produced as well as the average
        number of steps and of valid steps in the generations, respectively.
        """
        assert self.cfg.n_gens is not None and self.cfg.n_gens > 0
        assert self.cfg.return_all
        gens = self.do_filter_generations()
        first_gen = (
            gens[0][2].generation.stack
            if isinstance(gens[0][2], StandardProofSearch)
            else type(gens[0][2])
        )
        logging.info(f"First generation: {first_gen}")
        assert isinstance(self.env_helper, EQFwdEnvHelper)
        return self.generation_statistics(gens, self.env_helper, self.cfg.n_gens)

    @staticmethod
    def generation_statistics(
        gens: List[Any], env_helper: EQFwdEnvHelper, n_total_gens: Optional[int] = None
    ) -> Dict[str, Union[int, float]]:
        # gens either of type
        # Tuple[
        #     str,
        #     BackwardGoal,
        #     Union[StandardProofSearch, GenerationError],
        #     bool
        # ]
        # or
        # Tuple[
        #     str,
        #     BackwardGoal,
        #     Union[StandardProofSearch, GenerationError]
        # ]

        if n_total_gens is None:
            # calculate hard_prop without the (unknown) number of errors
            n_total_gens = len(gens)

        # find the selected nodes from the BackwardGoals
        selected_nodes: List[Optional[GraphNode]] = []
        for _, goal, fwd_pf, _ in gens:
            if not isinstance(fwd_pf, StandardProofSearch):
                # generation error
                continue
            assert isinstance(fwd_pf, StandardProofSearch)
            nodes, _ = history_to_eq_nodes(fwd_pf.generation)
            for node in nodes:
                assert isinstance(goal.theorem, EQTheorem)
                if goal.theorem.eq_node.prefix() == node.node.prefix():
                    selected_nodes.append(node)
                    break

        stats = evaluate_eq_generations(
            [
                gen[2].generation
                for gen in gens
                if isinstance(gen[2], StandardProofSearch)
            ],
            env_helper,
            selected_nodes=selected_nodes,
        )
        stats["hard_prop"] = 1.0
        stats["error_prop"] = 0.0
        if len(gens) >= 1 and len(gens[0]) >= 4:
            hard = sum(not gen[3] for gen in gens)
            stats["hard_prop"] = hard / n_total_gens
        errors = sum(isinstance(gen[2], GenerationError) for gen in gens)
        stats["error_prop"] = errors / n_total_gens

        return stats
