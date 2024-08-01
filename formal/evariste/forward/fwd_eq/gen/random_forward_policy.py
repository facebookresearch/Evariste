# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, Iterator, Union
from collections import defaultdict
import numpy as np

from evariste.datasets.equations import EquationsDatasetConf
from evariste.forward.common import (
    ForwardGraph,
    ForwardTactic,
    GenerationError,
)
from evariste.forward.common import PolicyOutput, PolicyOutputBeam
from evariste.forward.core.forward_policy import ForwardPolicy

from evariste.forward.core.maybe import Fail, Ok
from evariste.forward.fwd_eq.gen.env import EqGenForwardEnv
from evariste.forward.fwd_eq.gen.graph import EqGenForwardGraph
from evariste.forward.fwd_eq.gen.tactics import (
    EqGenForwardATactic,
    EqGenForwardHypMatchTactic,
    EqGenForwardHypSimpTactic,
    EqGenForwardNormNumTactic,
    EqGenForwardSimpTactic,
    EqGenForwardSubstTactic,
    EqGenForwardTTactic,
    EqGenForwardTactic,
)


logger = getLogger()


class RandomForwardPolicy(ForwardPolicy):
    def __init__(self, fwd_env: EqGenForwardEnv, params: EquationsDatasetConf):
        self.params = params
        self.fwd_env = fwd_env
        self.waiting: List[Tuple[int, EqGenForwardGraph]] = []
        self.cur_stats: Dict[str, Any] = defaultdict(float)

    @staticmethod
    def get_policy(
        fwd_env: EqGenForwardEnv, params: EquationsDatasetConf
    ) -> "RandomForwardPolicy":
        assert params.gen_type is not None
        return {
            "graph": RandomForwardGraphPolicy,
            "walk": RandomForwardWalkPolicy,
            "complex": RandomForwardComplexPolicy,
            "example": ExampleStatefulPolicy,
        }[params.gen_type](fwd_env=fwd_env, params=params)

    def submit_graph(self, graph_id: int, graph: ForwardGraph):
        assert isinstance(graph, EqGenForwardGraph)
        self.waiting.append((graph_id, graph))

    def ready_beams(self) -> List[Tuple[int, PolicyOutputBeam]]:
        graphs, self.waiting = self.waiting, []
        return self._get_beams(graphs)

    def _get_beams(
        self, graphs: List[Tuple[int, EqGenForwardGraph]]
    ) -> List[Tuple[int, PolicyOutputBeam]]:
        steps: List[Tuple[int, PolicyOutputBeam]] = []
        for graph_id, graph in graphs:
            beam: PolicyOutputBeam = []
            start = time.time()
            try:
                fwd_tactic = self._get_tactic(graph)
                beam.append(
                    Ok(
                        PolicyOutput(
                            # TODO: common parent for ForwardGraph, SimpleForwardGraph and EqGenForwardGraph
                            graph=graph,  # type: ignore
                            command=[],
                            command_str="",
                            fwd_tactic=fwd_tactic,
                            score=-1,
                            normalized_score=-1,
                            critic=None,
                        )
                    )
                )
            except GenerationError as err:
                beam.append(Fail(err))
            steps.append((graph_id, beam))
            self.cur_stats["time_in_policy.random"] += time.time() - start
        return steps

    def _get_tactic(self, graph: EqGenForwardGraph) -> ForwardTactic:
        raise NotImplementedError

    def stats(self) -> Dict[str, Any]:
        return dict(self.cur_stats)

    def reload_model_weights(self):
        pass

    def close(self):
        pass

    def sample_tactic(
        self,
        graph: EqGenForwardGraph,
        allowed_type: List[str],
        type_probas: Optional[List[float]] = None,
    ) -> EqGenForwardTactic:
        """
        Sample a random forward tactic among allowed type of tactics
        """
        eq = graph.nodes[-1].node
        rng = self.fwd_env.eq_env.rng
        if type_probas is not None:
            p = np.array(type_probas)
        else:
            p = np.ones(len(allowed_type))
        rule_type = rng.choice(allowed_type, p=p / p.sum())
        try:
            if rule_type == "t":
                tac: EqGenForwardTactic = EqGenForwardTTactic.sample(graph, self.params)
            elif rule_type == "rt":
                tac = EqGenForwardTTactic.sample(graph, self.params, src_node=eq)
            elif rule_type == "a":
                tac = EqGenForwardATactic.sample(graph, self.params)
            elif rule_type == "sub":
                tac = EqGenForwardSubstTactic.sample(graph, self.params)
            elif rule_type == "simp":
                tac = EqGenForwardSimpTactic.sample(graph, self.params)
            elif rule_type == "nn":
                tac = EqGenForwardNormNumTactic.sample(graph, self.params)
            elif rule_type == "hm":
                tac = EqGenForwardHypMatchTactic.sample(graph, self.params)
            elif rule_type == "hs":
                tac = EqGenForwardHypSimpTactic.sample(graph, self.params)
            else:
                raise RuntimeError("unreachable")
        except Exception as e:
            raise e
        else:
            pass
        return tac


class RandomForwardGraphPolicy(RandomForwardPolicy):
    def __init__(
        self, fwd_env: EqGenForwardEnv, params: EquationsDatasetConf,
    ):
        super().__init__(fwd_env, params)

    def _get_tactic(self, graph: EqGenForwardGraph) -> ForwardTactic:
        return self.sample_tactic(
            graph, ["t", "a"], [1 - self.params.tf_prob, self.params.tf_prob]
        )


class RandomForwardComplexPolicy(RandomForwardPolicy):
    def __init__(
        self, fwd_env: EqGenForwardEnv, params: EquationsDatasetConf,
    ):
        super().__init__(fwd_env, params)

    def _get_tactic(self, graph: EqGenForwardGraph) -> ForwardTactic:
        return self.sample_tactic(
            graph,
            ["t", "a", "sub", "simp", "nn",],
            type_probas=[0.26, 0.26, 0.26, 0.05, 0.17],
        )


class RandomForwardWalkPolicy(RandomForwardPolicy):
    def __init__(
        self, fwd_env: EqGenForwardEnv, params: EquationsDatasetConf,
    ):
        super().__init__(fwd_env, params)

    def _get_tactic(self, graph: EqGenForwardGraph) -> ForwardTactic:
        return self.sample_tactic(graph, ["rt"])


class ExampleStatefulPolicy(RandomForwardPolicy):
    def __init__(
        self, fwd_env: EqGenForwardEnv, params: EquationsDatasetConf,
    ):
        super().__init__(fwd_env, params)
        self.graph_inputs: Dict[int, EqGenForwardGraph] = {}
        self.tactic_iterators: Dict[int, Iterator[Union[ForwardTactic, Exception]]] = {}

    def the_policy(self, pid: int) -> Iterator[Union[ForwardTactic, Exception]]:
        def my_graph():
            return self.graph_inputs[pid]

        for tac_type in ["t", "a", "sub", "simp", "nn"]:
            while True:
                try:
                    t = self.sample_tactic(my_graph(), [tac_type], [1])
                    yield t
                    break
                except Exception as e:
                    yield e

    def _get_tactic(self, graph: EqGenForwardGraph) -> ForwardTactic:
        pid = graph.proof_id
        self.graph_inputs[pid] = graph
        if pid not in self.tactic_iterators:
            self.tactic_iterators[pid] = self.the_policy(pid)
        try:
            maybe_res = next(self.tactic_iterators[pid])
            if isinstance(maybe_res, Exception):
                raise maybe_res
            return maybe_res
        except StopIteration:
            raise GenerationError("too lazy to handle this")
