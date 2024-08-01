# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from evariste import json as json
from dataclasses import fields, MISSING, dataclass, asdict
from logging import getLogger

import psutil

from evariste.forward.fwd_mm.training.curriculum import (
    CurriculumDataset,
    MaxLenSchedule,
)
from evariste.forward.fwd_mm.training.mm_training_helpers import (
    log_sizes,
    sample_from_cumulative_mm,
)
from evariste.forward.fwd_mm.training.common import MMFwdTrainingProof
from evariste.forward.fwd_mm.training.mm_online_dataset import MMOnlineDataset
from evariste.forward.fwd_mm.training.rb_utils import sample_node_sequence_from_rb
from evariste.model.data.envs.replay_buffer_loader import ReplayBuffer

from typing import List, Dict, Optional, Hashable, Callable, Any, NamedTuple
import os
import time
import numpy as np
import pickle

from numpy.random.mtrand import RandomState
from evariste.forward.fwd_mm import mm_fwd_tasks
from evariste.forward.fwd_mm.training import mm_training_helpers
from evariste.forward.fwd_mm.mm_fwd_tokenizer import tokenize_mm_graph
from evariste.forward.fwd_mm.mm_fwd_tasks import (
    LEGACY_FWD_INP_FMT,
    FWD_INP_FMT,
    GEN_INP_FMT,
    MMFwdFormat,
)

from evariste.comms.store import EmptyStore
from evariste.model.utils import create_subseq_pos
from evariste.envs.mm.utils import (
    count_unique_nodes,
    enumerate_nodes,
    get_canonical_order,
    Node,
    node_tok,
    Node_a_p,
    select_potential_goals,
    random_topological_sort,
    reward_quantile_tok,
)
from evariste.model.data.dictionary import (
    B_CMD_WORD,
    E_CMD_WORD,
    B_SUBST_WORD,
    M_SUBST_WORD,
    E_SUBST_WORD,
    B_STACK_WORD,
    E_STACK_WORD,
    E_NODE_WORD,
    EOU_WORD,
    B_NODE_WORD,
    EMPTY_GOAL_WORD,
    STOP_WORD,
    UNPROVED_WORD,
    PROVED_WORD,
    Dictionary,
)

logger = getLogger()


LOG_INTERVAL = 100_000

Sample = Dict[str, Any]


class MMGraphSampler:
    def __init__(
        self,
        trainer_args,
        proof_trees: Dict,
        dico: Dictionary,
        label_remap: Dict[str, str],
        replay_buffer: Optional[ReplayBuffer],
    ):
        """

        Class to hide all the details to sample a forward graph and to tokenize it.

        Check MetamathGraphArgs for help on parameters

        Note all majority pf methods are prefixed by a _ to specify to the reader
        these fonctions are only used internally (and are not called by
        the MetamathDataEnv)


        TODO: we should further clean this task
         removing experimental features for instance, simplify sampling paths,
         clean x2y stuff
        """
        from evariste.trainer.args import TrainerArgs  # local because of cyclic deps

        # TODO
        assert isinstance(trainer_args, TrainerArgs)
        self.dico = dico
        self.params = trainer_args
        self.proof_trees = proof_trees
        self.cumulative = self._make_cumulative_proof_weights(proof_trees)
        self.label_remap = label_remap

        for task in self.params.parsed_tasks("mm"):
            if mm_fwd_tasks.use_graph_sampler(task):
                logger.info(
                    f"Fwd/gen task: {task} using fmt: "
                    f"{MMFwdFormat.from_task(task, self.params)}"
                )

        if self.params.mm.graph.generated_proof_path:
            self.generated_proofs = self._load_generated_proofs()
            self.generated_cumulative = self._make_cumulative_proof_weights(proof_trees)
        else:
            self.generated_proofs = None
            self.generated_cumulative = None

        if self.params.mm.graph.insert_noise_prob > 0:
            self.noise_nodes = self._make_noise_nodes()
        else:
            self.noise_nodes = None

        self.train_online_dataset: Optional[MMOnlineDataset] = None
        if self.params.online_fwd_generation:
            # sampling from online generation dataset (not adversarial training)
            self.train_online_dataset = MMOnlineDataset.from_trainer_args(self.params)

        self.replay_buffer = replay_buffer

        self.curriculum: Optional[CurriculumDataset] = None
        if self.params.mm.graph.curriculum_str:
            self.curriculum = CurriculumDataset(
                proof_trees=self.proof_trees,
                dump_path=self.params.dump_path,
                refresh_every=5000 if not self.params.debug.train else 10,
                curriculum_str=self.params.mm.graph.curriculum_str,
            )
        self.max_len_schedule: Optional[MaxLenSchedule] = None
        if self.params.mm.graph.max_len_schedule_str:
            self.max_len_schedule = MaxLenSchedule(
                dump_path=self.params.dump_path,
                refresh_every=5000 if not self.params.debug.train else 10,
                max_len_schedule_str=self.params.mm.graph.max_len_schedule_str,
                max_len=self.params.batch.max_len,
            )
        self._cache = {}
        self._check()

    def _check(self):
        if (
            self.params.rl_distributed.is_adversarial_training
            and self.params.mm.graph.generated_prob > 0
        ):
            assert self.replay_buffer is not None

    def get_graph_sample(
        self, split: str, task: str, rng: np.random.RandomState
    ) -> Optional[Sample]:
        start = time.time()
        dropout = self.params.mm.graph.dropout
        sample_goal = self.params.mm.graph.sample_goal
        insert_noise_prob = self.params.mm.graph.insert_noise_prob
        topo_sort_version = self.params.mm.graph.topo_sort_version
        drop_if_too_long = self.params.mm.graph.drop_if_too_long

        is_gen_task = mm_fwd_tasks.is_gen_task(task)
        use_stop_action = self.params.mm.stop_action and is_gen_task
        if is_gen_task:
            assert not sample_goal

        key = ("fwd_sample_stats", split, task)
        stats: GraphSamplingStats = self._cached(key, lambda: GraphSamplingStats())

        try:
            proof_root = self.sample_proof_root(rng, split)
        except EmptyStore:
            stats.empty_store += 1
            return None

        name = proof_root.name
        proof_tree = proof_root.root
        generated = proof_root.generated

        stats.n_sample += 1
        if proof_root.proved:
            stats.n_proved += 1

        if generated:
            stats.n_sample_generated += 1
        else:
            stats.n_sample_std += 1
        stats.time_in_proof_sampling += time.time() - start

        # Step 1: Choose goal node (and output_node)

        # note: sampling output_node and then goal_node (and not the contrary) allows to
        # be sure that every node is view the same number of time as target.
        # But we could choose:
        #     - to sample goal uniformly instead (or with a given distribution
        #       that oversample easy goal or hard goal)
        #     - sample target node given a distribution that oversample certain nodes

        # tmp_order only for choosing target and goal
        start_sample_goal = time.time()
        tmp_order = get_canonical_order(proof_tree)
        output_cand_idx = [i for i, node in enumerate(tmp_order) if node.ltype != "$e"]
        if not output_cand_idx:
            raise ValueError(f"Only e_hyps: {tmp_order}")

        if use_stop_action:
            assert not sample_goal
            output_cand_idx.append(len(tmp_order))  # we can sample stop_action

        output_idx = rng.choice(output_cand_idx)

        if output_idx == len(tmp_order):
            assert use_stop_action
            output_node = fake_stop_node(
                goal=tmp_order[-1]
            )  # fake node to be compatible with graph augmentations
            is_stop = True
        else:
            is_stop = False
            output_node = tmp_order[output_idx]

        if sample_goal:
            candidates = tmp_order[output_idx:]
            potential_goals = select_potential_goals(candidates)
            assert potential_goals[0] == output_node
            assert potential_goals[-1] == proof_tree
            goal_node = rng.choice(potential_goals)
        else:
            goal_node = proof_tree
        stats.time_in_sampling_goal += time.time() - start_sample_goal

        # Step 2: now that we know the goal we can build the order that we will use
        start_topo_sort = time.time()
        if topo_sort_version == 0:
            order = random_topological_sort(goal_node, e_hyps_first=True, rng=rng)
            order = [n.node for n in order]
        elif topo_sort_version == 1:
            order = mm_training_helpers.random_order_v1(goal_node, rng=rng)
        elif topo_sort_version == 2:
            order = mm_training_helpers.random_order_v2(goal_node, rng=rng)
        elif topo_sort_version == 3:
            order = mm_training_helpers.random_order_v3(goal_node, rng=rng)
        elif topo_sort_version == 4:
            order = mm_training_helpers.random_order_v4(
                goal_node, rng=rng, output_node=output_node
            )
        else:
            raise NotImplementedError(topo_sort_version)

        if is_stop:
            gsize = len(order)
        else:
            gsize = order.index(output_node)
        graph = order[:gsize]
        stats.time_in_topo_sort += time.time() - start_topo_sort

        # Step 3: insert noisy nodes in graph
        start_noise = time.time()
        noises = set()
        len_before = len(graph)
        if insert_noise_prob > 0.0:
            old_graph = graph
            graph = []
            present = set([n.statement_str for n in tmp_order if n.statement_str])

            def _try_insert_noisy_node(graph_, present_, noises_):
                noisy_node: Node = self._sample_node_for_noise(split, rng)
                if noisy_node.statement_str not in present:
                    present_.add(noisy_node.statement_str)
                    graph_.append(noisy_node)
                    noises_.add(noisy_node.statement_str)

            while old_graph:
                if rng.random() < insert_noise_prob:
                    _try_insert_noisy_node(graph, present, noises)
                else:
                    graph.append(old_graph.pop(0))

            # insert noise at the end
            while rng.random() < insert_noise_prob:
                if is_stop:  # no noise between last generated node and stop_action
                    break
                _try_insert_noisy_node(graph, present, noises)
        assert len_before + len(noises) == len(graph)
        stats.time_in_adding_noise += time.time() - start_noise

        # Step 4: drop some nodes
        start_noise_drop = time.time()
        children_set = set([c for c in getattr(output_node, "children", [])])
        if dropout > 0:
            old_graph = graph
            graph = []
            # noinspection PyArgumentList
            drops = rng.rand(len(old_graph))
            for drop, node in zip(drops, old_graph):
                is_a_parent = node in children_set
                if is_a_parent or drop > dropout:
                    graph.append(node)

        if drop_if_too_long:
            graph = mm_training_helpers.drop_oldest_nodes(
                goal_node, graph, output_node, max_len=self.params.batch.max_len,
            )
        else:
            if (
                sum(len(n.statement) for n in [goal_node] + graph)
                >= self.params.batch.max_len
            ):
                graph = None
        stats.time_in_node_drop += time.time() - start_noise_drop
        if graph is None:
            stats.n_too_long += 1
            stats.time_in_too_long += time.time() - start
            if generated:
                stats.n_too_long_generated += 1
            else:
                stats.n_too_long_std += 1
            return None

        # Step 5: make sample
        start_make = time.time()
        node2id = {n: i for i, n in enumerate(graph)}
        children_ids = [node2id[c] for c in getattr(output_node, "children", [])]
        nodes = [n.statement for n in graph]

        # check that children are in the graph exactly match expected hypotheses
        assert (not hasattr(output_node, "children") and len(children_set) == 0) or (
            [child.statement for child in output_node.children]
            == [nodes[i] for i in children_ids]
        )
        # print(f"duration: {time.time()  - start:.03f}s")

        if is_stop:
            assert graph[-1] == goal_node
            assert children_ids == [len(graph) - 1]

        if proof_root.proved is None:
            # we consider for now that supervised dataset is not proved
            was_proved = False
        else:
            was_proved = proof_root.proved

        max_len = self.params.batch.max_len
        curriculum_max_len = None
        if split == "train" and self.max_len_schedule is not None:
            curriculum_max_len = self.max_len_schedule.get_max_len()
            max_len = curriculum_max_len

        # don't respect curriculum for generated data
        if generated and self.params.mm.graph.no_curriculum_for_generated_data:
            max_len = self.params.batch.max_len

        graph_data = MMFwdGraphData(
            name=name,
            graph=graph,
            goal=goal_node,
            target=output_node,
            children_ids=children_ids,
            order=order,
            target_id=gsize,
            is_stop=is_stop,
            was_proved=was_proved,
            is_generated=proof_root.generated,
            reward_quantile=proof_root.reward_quantile,
        )

        sample = fwd_x2y_sample(
            fmt=MMFwdFormat.from_task(task, params=self.params),
            max_len=max_len,
            n_reward_quantiles=self.params.rl_params.replay_buffer.n_reward_quantiles,
            dico=self.dico,
            label_remap=self.label_remap if self.params.mm.graph.remap_label else None,
            rng=rng,
            graph_data=graph_data,
        )
        stats.time_in_make_sample += time.time() - start_make

        stats.total_duration += time.time() - start
        if sample is None:
            stats.n_too_long += 1
            stats.time_in_too_long += time.time() - start
            if generated:
                stats.n_too_long_generated += 1
            else:
                stats.n_too_long_std += 1
        else:
            xlen = len(sample["x"])
            if curriculum_max_len is not None and xlen > curriculum_max_len:
                assert generated
                assert self.params.mm.graph.no_curriculum_for_generated_data
                stats.n_gen_longer_than_curriculum += 1
            stats.update_avg_size(len(sample["x"]))
            stats.update_avg_size_y(len(sample["y"]))

        log_int = 1000 if self.params.debug.train else LOG_INTERVAL
        if stats.n_sample >= log_int:
            logger.info(f"{key}:{json.dumps(asdict(stats))}")
            stats.reset()

        return sample

    def _cached(self, key: Hashable, create: Callable):
        if key not in self._cache:
            self._cache[key] = create()
        return self._cache[key]

    def _load_generated_proofs(self):
        proof_path_or_paths = self.params.mm.graph.generated_proof_path
        world_size = self.params.slurm_conf.world_size
        assert proof_path_or_paths
        proof_paths = proof_path_or_paths.split(",")
        shards = []
        for proof_path in proof_paths:
            if os.path.isdir(proof_path):
                these_shards = []
                for p in os.listdir(proof_path):
                    if not p.endswith(".pkl"):
                        continue
                    these_shards.append(os.path.join(proof_path, p))

                if len(these_shards) < world_size:
                    logger.info(
                        f"Detected {len(these_shards)} with world_size {world_size}"
                    )
                    assert (
                        world_size % len(these_shards) == 0
                    ), f"{len(these_shards)} not a multiple of {world_size}"
                    these_shards = these_shards * int(world_size / len(these_shards))

                these_shards = [
                    s
                    for i, s in enumerate(sorted(these_shards))
                    if i % world_size == self.params.slurm_conf.global_rank
                ]
                shards.extend(these_shards)
            else:
                shards.append(proof_path)
        logger.info(f"Generated data: going to load shards: {shards}")
        gen_proofs = {}
        for shard in shards:
            with open(shard, "rb") as f:
                gen_proofs.update(pickle.load(f))
        logger.info(f"Loaded {len(gen_proofs)} generated proofs")
        proof_trees = {}
        for split in ["train", "valid", "test"]:
            if split == "train":
                proof_trees[split] = list(gen_proofs.items())
            else:
                proof_trees[split] = []
        return proof_trees

    @staticmethod
    def _make_cumulative_proof_weights(proof_trees) -> Dict[str, np.array]:
        cumulative_proof_weight = {}
        for split in proof_trees.keys():
            sizes = np.array(
                [
                    count_unique_nodes(proof_tree, ignore_e_hyps=True)
                    for _, proof_tree in proof_trees[split]
                ]
            )
            cumulative_proof_weight[split] = np.cumsum(sizes)
            log_sizes(logger, sizes, split=split)
        return cumulative_proof_weight

    def _make_noise_nodes(self) -> Dict[str, List[Node]]:
        split2nodes = {}
        proof_trees = self.proof_trees
        for split in proof_trees.keys():
            if split == "train" and self.params.mm.graph.generated_proof_path:
                logger.info("Using generated for noise nodes")
                assert self.generated_proofs
                gen_proofs = self.generated_proofs
                proofs = proof_trees[split] + gen_proofs[split]
            else:
                proofs = proof_trees[split]
            split2nodes[split] = []
            for _, proof_tree in proofs:
                for node in enumerate_nodes(
                    proof_tree, ignore_f_e=False, ignore_empty=False, no_syntactic=True,
                ):
                    split2nodes[split].append(node)
        return split2nodes

    def _sample_node_for_noise(self, split: str, rng: RandomState) -> Node:
        nodes = self.noise_nodes[split]
        if split == "train" and self.params.online_fwd_generation:
            dataset = self.train_online_dataset
            generated_nodes = dataset.nodes()
            n1 = len(nodes)
            n2 = len(generated_nodes)
            # faster than rng.choice
            idx = rng.randint(0, n1 + n2)
            if idx < n1:
                return nodes[idx]
            return generated_nodes[idx - n1]
        else:
            return nodes[rng.randint(0, len(nodes))]

    def sample_proof_root(self, rng: RandomState, split: str) -> "MMFwdTrainingProof":
        sample_generated = (
            split == "train" and rng.random() < self.params.mm.graph.generated_prob
        )
        if sample_generated:
            return self._sample_generated_proof_root(rng=rng, split=split)
        elif self.params.mm.graph.curriculum_str and split == "train":
            assert self.curriculum is not None
            return self.curriculum.sample_training_graph(rng, split)
        else:
            data = self.proof_trees[split]
            cumulative = self.cumulative[split]
            name, root = sample_from_cumulative_mm(cumulative, data, rng=rng)
            return MMFwdTrainingProof(
                root=root, name=name, generated=False, proved=None
            )

    def _sample_generated_proof_root(
        self, rng: RandomState, split: str
    ) -> "MMFwdTrainingProof":
        if self.params.rl_distributed.is_adversarial_training:
            # TODO: we sample here uniformly on replay buffer, we should
            #  sample with a weight == proof_size
            assert self.replay_buffer is not None
            assert not self.params.online_fwd_generation
            block = False
            if (
                self.params.mm.graph.generated_prob == 1.0
                or self.params.mm.graph.wait_generated
            ):
                block = True
            sample, _, _ = sample_node_sequence_from_rb(
                params=self.params,
                replay_buffer=self.replay_buffer,
                block=block,
                rng=rng,
                split=split,
            )
            assert isinstance(sample, MMFwdTrainingProof)
            assert sample.generated
            return sample
        elif self.params.online_fwd_generation:
            # sampling from online generation dataset (not adversarial training)
            return self.train_online_dataset.sample_training_graph(rng=rng, split=split)
        else:
            # sampling a generated proof fixed dataset
            data = self.generated_proofs[split]
            cumulative = self.generated_cumulative[split]
            name, root = sample_from_cumulative_mm(cumulative, data, rng=rng)
            return MMFwdTrainingProof(name=name, generated=True, proved=None, root=root)

    def close(self):
        if self.train_online_dataset:
            self.train_online_dataset.close()

    def __del__(self):
        self.close()


class MMFwdGraphData(NamedTuple):
    """
    All data necessary to build (input, target, aux_preds) for fwd task.
    Is used by the function fwd_x2y_sample()

    It is obtained from a MMFwdTrainingProof by applying different data augmentations
    goal and target sampling. These transformations are done in the MMGraphSampler
    """

    name: str
    graph: List[Node]
    goal: Node
    target: Node
    target_id: int
    is_stop: bool
    was_proved: bool
    order: List[Node]
    children_ids: List[int]
    is_generated: bool
    reward_quantile: Optional[int] = None


def fwd_x2y_sample(
    fmt: MMFwdFormat,
    graph_data: MMFwdGraphData,
    max_len: int,
    dico: Dictionary,
    label_remap: Optional[Dict[str, str]],
    rng: Optional[RandomState] = None,
    n_reward_quantiles: Optional[int] = None,
) -> Optional[Sample]:
    # TODO: simplify this mess
    available_subsequences = {
        "goal",
        "nogoal",
        "stack",
        "graph",
        "graph:v0",
        "graph:v1",
        "graph:v2",
        "label",
        "theorem",
        "subst",
        "generated",
        "EOU",
        "children",
        "bcmd",
        "ecmd",
        "proved",
    }

    assert fmt.inp_fmt in [
        LEGACY_FWD_INP_FMT,
        FWD_INP_FMT,
        GEN_INP_FMT,
    ], f"{fmt.inp_fmt} is not implemented"
    legacy_fmt = fmt.inp_fmt == LEGACY_FWD_INP_FMT
    is_generation = fmt.inp_fmt == GEN_INP_FMT

    conditioning = []
    if fmt.label_conditioning:
        conditioning.append(_sample_label_from_proof(graph_data.goal, rng))
    if fmt.proved_conditioning:
        conditioning.append(PROVED_WORD if graph_data.was_proved else UNPROVED_WORD)
    if fmt.reward_quantile_conditioning:
        assert n_reward_quantiles is not None
        assert n_reward_quantiles > 0
        if graph_data.is_generated:
            assert graph_data.reward_quantile is not None
            quantile = graph_data.reward_quantile
        else:
            # human data: we consider that it is the best reward for now
            assert graph_data.reward_quantile is None
            quantile = n_reward_quantiles - 1

        assert quantile < n_reward_quantiles
        conditioning.append(reward_quantile_tok(quantile))

    # input
    inp = tokenize_mm_graph(
        goal_statement=graph_data.goal.statement_str,
        node_statements=[n.statement_str for n in graph_data.graph],
        is_generation=is_generation,
        legacy_fmt=legacy_fmt,
        conditioning=conditioning,
    )

    cmd_fmt = list(fmt.cmd_fmt)
    target = graph_data.target
    if graph_data.is_stop:
        assert target.statement_str == STOP_ACTION_STATEMENT
        assert is_generation
        assert fmt.use_stop_action
        cmd_fmt = ["stop"]
    else:
        assert target.statement_str != STOP_ACTION_STATEMENT

    s_x = ["input"]
    if fmt.aux_predictions:
        s_y = cmd_fmt + ["EOU"] + fmt.aux_predictions
    else:
        s_y = cmd_fmt

    # label
    label = None
    if "label" in s_y:
        label = target.label
        label = label_remap[label] if label_remap else label

    # substitutions
    subst = None
    if "subst" in s_y:
        subst = []
        for key, value in getattr(target, "substitutions", {}).items():
            value = value.split()
            assert type(key) is str and len(key.split()) == 1
            assert type(value) is list and len(value) >= 1
            subst.extend([B_SUBST_WORD, key, M_SUBST_WORD] + value + [E_SUBST_WORD])

    # plan prediction
    plan = None
    plan_specs = [x for x in s_y if x.startswith("plan:")]
    if plan_specs:
        plan = _make_fwd_plan(
            graph_data.goal, graph_data.order, plan_specs, target, graph_data.target_id
        )
        assert len(plan_specs) == 1
        s_y = tuple((x if not x.startswith("plan") else "plan") for x in s_y)

    # generated statement
    generated = None
    if "generated" in s_y:
        generated = [B_STACK_WORD] + target.statement + [E_STACK_WORD]

    # placeholder for pointers to children
    # Q: do we need special token like <CHILDREN> </CHILDREN> to wrap children?
    children = None
    if "children" in s_y:
        children = [node_tok(cid) for cid in graph_data.children_ids]

    theorem = None
    if "theorem" in s_y:
        raise NotImplementedError

    # create input / output sequences
    item = {
        "input": inp,
        "nogoal": [EMPTY_GOAL_WORD],
        "bcmd": [B_CMD_WORD],
        "ecmd": [E_CMD_WORD],
        "generated": generated,
        "label": [label],
        "stop": [STOP_WORD],
        "theorem": theorem,
        "subst": subst,
        "EOU": [EOU_WORD],
        "children": children,
        "plan": plan,
    }
    item: Dict[str, List[str]] = {k: v for k, v in item.items() if v is not None}

    eos = dico.eos_word
    # build x and y sequences
    x = [item[s] for s in s_x]
    y = [item[s] for s in s_y]

    # sub-sequences positions
    if not fmt.use_stop_action:
        x_subseq_pos = create_subseq_pos(x, labels=s_x)
        y_subseq_pos = create_subseq_pos(y, labels=s_y)
    else:
        # with stop action we don't have the same subseqs everywhere
        x_subseq_pos = {}
        y_subseq_pos = {}

    # add sequence delimiters
    x = [eos, *sum(x, []), eos]
    y = [eos, *sum(y, []), eos]

    # skip too long sequences
    # before applying dictionary, if not it can crash since NODE_{i} is not here
    if max(len(x), len(y)) > max_len:
        return None

    # index sequences
    x = [dico.index(t) for t in x]
    y = [dico.index(t) for t in y]

    sample = {
        "name": graph_data.name,
        "x": x,
        "y": y,
        "x_subseq_pos": x_subseq_pos,
        "y_subseq_pos": y_subseq_pos,
    }
    if fmt.hum_vs_gen_disc:
        sample.update(
            _human_vs_generated_disc(graph_data, cmd_fmt, items=item, dico=dico)
        )
    return sample


def _human_vs_generated_disc(
    graph_data: MMFwdGraphData,
    cmd_fmt: List[str],
    items: Dict[str, List[str]],
    dico: Dictionary,
) -> Dict:
    """
    We don't take y directly to remove need for parse for EOU
    """
    cmd = [
        dico.eos_word,
        *[tok for name in cmd_fmt for tok in items[name]],
        dico.eos_word,
    ]
    cmd = [dico.index(t) for t in cmd]
    tgt = 0 if graph_data.is_generated else 1
    return {"disc_inp": cmd, "disc_tgt": tgt}


def _sample_label_from_proof(goal: Node, rng: RandomState) -> str:
    # conditioning of the generation with a label that will be used in the proof
    # to favor diversity
    assert rng is not None
    proof_labels = list({n.label for n in get_canonical_order(goal) if n.ltype != "$e"})
    cond_label = proof_labels[rng.randint(len(proof_labels))]
    return cond_label


def _make_fwd_plan(goal, order, plan_specs, target, target_id):
    assert len(plan_specs) == 1
    (plan_spec,) = plan_specs
    plan = []
    assert order[target_id] == target
    needed_order = order[target_id:]
    path = mm_training_helpers.find_path(needed_order)
    assert path[0] == target
    assert path[-1] == goal
    if len(path) > 2:
        path = path[1:-1]
    else:
        path = []
    if plan_spec == "plan:labels":
        for node in path:
            plan.append(node.label)
    elif plan_spec == "plan:all":
        for node in path:
            plan.extend([B_NODE_WORD] + node.statement + [E_NODE_WORD])
    elif plan_spec == "plan:1":
        for node in path[:1]:
            plan.extend([B_NODE_WORD] + node.statement + [E_NODE_WORD])
    elif plan_spec == "plan:3":
        for node in path[:3]:
            plan.extend([B_NODE_WORD] + node.statement + [E_NODE_WORD])
    elif plan_spec == "plan:last":
        for node in path[-1:]:
            plan.extend([B_NODE_WORD] + node.statement + [E_NODE_WORD])
    else:
        raise NotImplementedError(plan_spec)
    return plan


def _mem() -> str:
    return f"mem_usage: {psutil.virtual_memory().used / 1024 ** 3:.01f}GB"


STOP_ACTION_STATEMENT = f"wff {STOP_WORD}"


@dataclass
class GraphSamplingStats:
    n_sample: int = 0
    n_too_long: int = 0
    avg_size: float = 0
    n_sample_std: int = 0
    empty_store: int = 0
    n_too_long_std: int = 0
    n_sample_generated: int = 0
    n_too_long_generated: int = 0
    time_in_adding_noise: float = 0
    time_in_topo_sort: float = 0
    time_in_too_long: float = 0
    total_duration: float = 0
    size_total: int = 0
    size_count: int = 0
    time_in_proof_sampling: float = 0
    time_in_sampling_goal: float = 0
    time_in_node_drop: float = 0
    time_in_make_sample: float = 0
    avg_size_y: float = 0
    size_total_y: int = 0
    size_count_y: int = 0
    n_proved: int = 0

    # n_generations longer than curriculum max_len
    n_gen_longer_than_curriculum: int = 0

    def update_avg_size(self, size: int):
        self.size_total += size
        self.size_count += 1
        self.avg_size = self.size_total / self.size_count

    def update_avg_size_y(self, size: int):
        self.size_total_y += size
        self.size_count_y += 1
        self.avg_size_y = self.size_total_y / self.size_count_y

    def reset(self):
        for field in fields(self):
            if field.default != MISSING:
                setattr(self, field.name, field.default)


def fake_stop_node(goal: Node):
    return Node_a_p(
        ltype="$p",
        label="",
        substitutions={},
        statement_str=STOP_ACTION_STATEMENT,
        children=[goal],
        disjoint=set(),
    )
