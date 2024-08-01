# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import unique, Enum
from copy import deepcopy
from pathlib import Path
import time
from collections import defaultdict


from params import Params
from evariste.backward.graph import GoalParams
from evariste.backward.model.beam_search_kind import BeamSearchKind
from evariste.backward.prover.args import MCTSParams
from evariste.backward.prover.mcts_samples import ONLINE_MCTS_SUBTASKS
from evariste.model.data.subproof_args import MCTSSubProofArgs
from evariste.logger import create_logger
from evariste.utils import DocStrEnum

logger = create_logger(None)


@unique
class ProverKind(str, Enum):
    BackwardMCTS = "backward_mcts"
    BackwardSimpleMCTS = "backward_simple_mcts"
    BackwardFluidMCTS = "backward_fluid_mcts"
    BackwardGreedy = "backward_greedy"
    BackwardIGreedy = "backward_improved_greedy"
    BackwardBreadthFS = "backward_breadth_fs"
    BackwardBestFS = "backward_best_fs"
    PriorSearch = "prior_search"


@unique
class ProofStatus(str, DocStrEnum):
    TO_EXPAND = "to_expand", "`ProofHandler` selecting nodes to expand."
    WAITING_FOR_EXPAND = (
        "waiting for expansion",
        "`MPExpander` or `BackwardEnv` computing expansion",
    )
    WAITING_FOR_STATUS = (
        "waiting for status",
        "Waiting for `ProofHandler` done or not status",
    )
    STITCHING = "stitching", "In `AsyncProofStitcher`"
    CLEANING = "cleaning", "In `AsyncProofCleaner`"


class ProofStatusBackwardRunner:
    def __init__(self):
        self._status = ProofStatus.TO_EXPAND
        self._time_per_status: Dict[str, float] = defaultdict(float)
        self._last_status_change = time.time()
        self._creation_time = time.time()

    @property
    def time_per_status(self) -> Dict[str, float]:
        return self._time_per_status

    @property
    def last_status_change(self) -> float:
        return self._last_status_change

    @property
    def creation_time(self) -> float:
        return self._creation_time

    @property
    def status(self) -> ProofStatus:
        return self._status

    @status.setter
    def status(self, new_status: ProofStatus):
        """
        TO_EXPAND -> WAITING_FOR_EXPAND -> WAITING_FOR_APPLY -> TO_EXPAND -> ...
        ->STITCHING->CLEANING->FINISHING
        """
        if new_status == ProofStatus.TO_EXPAND:
            assert self.status in {None, ProofStatus.WAITING_FOR_STATUS}, self.status
        elif new_status == ProofStatus.WAITING_FOR_EXPAND:
            assert self.status == ProofStatus.TO_EXPAND, self.status
        elif new_status == ProofStatus.WAITING_FOR_STATUS:
            assert self.status == ProofStatus.WAITING_FOR_EXPAND, self.status
        elif new_status == ProofStatus.STITCHING:
            assert self.status == ProofStatus.WAITING_FOR_STATUS, self.status
        elif new_status == ProofStatus.CLEANING:
            assert self.status == ProofStatus.STITCHING, self.status
        else:
            raise RuntimeError(f"Unexpected status: {new_status}")
        self._time_per_status[self.status] += time.time() - self._last_status_change
        self._status = new_status
        self._last_status_change = time.time()


@unique
class ConditioningKind(str, Enum):
    GoodOrRandom = "good_or_random"
    Random = "random"
    No = "no"


@unique
class CleaningLevel(str, DocStrEnum):
    No = "no", "No proof cleaning"
    TacRemoval = "removal", "Only unecessary tactic removal"
    TacRemoval_TacClean = (
        "removal+cleaning",
        "Only unecessary tactic removal + individual tactic cleaning",
    )
    TacRemoval_TacClean_SimpOnly = (
        "removal+cleaning+simponly",
        "Only unecessary tactic removal + individual tactic cleaning + "
        "`simp only` with min args",
    )


@dataclass
class ProofCleaningParams(Params):
    """
    Params for proof cleaner

    :param version: version of proof cleaner `runtime_loaded_lean_files/cleaning_utils/<version>`, defaults to v1
    :type version: str
    :param level: cleaning level, defaults to CleaningLevel.No
    :type level: CleaningLevel
    :param quiet: indicates if the cleaning steps are not logged, defaults to True
    :type quiet: bool
    :param dump: indicates if the proofs are pickled and dumped, defaults to False
    :type dump: bool
    :param timeout_useless_tactics_removal: timeout in seconds for useless tactics removal, defaults to 120
    :type timeout_useless_tactics_removal: int
    :param timeout_individual_tactic_cleaning: timeout in seconds for individual tactic cleaning, defaults to 120
    :type timeout_individual_tactic_cleaning: int
    """

    version: str = "v1"  # cleaning_utils/<version>
    level: CleaningLevel = CleaningLevel.No
    quiet: bool = True
    dump: bool = False
    timeout_useless_tactics_removal: int = 120  # in seconds
    timeout_individual_tactic_cleaning: int = 120  # in seconds


@dataclass
class ProverParams(Params):
    mcts: MCTSParams
    beam_path: Path
    dump_path: Path
    n_simultaneous_proofs: int
    beam_kind: BeamSearchKind
    prover_kind: ProverKind = ProverKind.BackwardMCTS
    mcts_subproof_params: MCTSSubProofArgs = field(
        default_factory=lambda: MCTSSubProofArgs()
    )
    eval_split: str = "valid"
    dump_mcts: bool = False
    mcts_subtasks_online_training: List[str] = field(default_factory=lambda: [])
    mcts_subproofs_online_training: bool = False

    no_proof_check: bool = False
    debug: bool = False
    n_gpus: int = 1

    quiet: bool = False  # if true, suppresses all file dumps
    only_jsonl: bool = (
        False  # if true, only output stats to jsonl file, not to tensorboard
    )
    print_status: bool = (
        True  # print prover status (disabled in greedy evals to not pollute train.log)
    )

    heartbeats_freq: int = -1

    conditioning_kind: ConditioningKind = ConditioningKind.No

    only_keep_one_solving: bool = (
        True  # if False, activates #877 (keep all solving tactics)
    )
    add_tactic_fill: bool = False
    add_tactic_errors: bool = False

    try_stitch: bool = False

    # minimum time in seconds between the reloading of two models in AutomaticallyReloadingBeamSearch
    min_reload_time: int = 60

    proof_cleaning_params: ProofCleaningParams = field(
        default_factory=lambda: ProofCleaningParams()
    )

    def __post_init__(self):
        subtasks = self.mcts_subtasks_online_training
        assert len(subtasks) == len(set(subtasks))

    def _check_and_mutate_args(self):
        if self.dump_mcts:
            # assert os.path.isdir(self.dump_path), self.dump_path
            assert self.prover_kind in {
                ProverKind.BackwardMCTS,
                ProverKind.BackwardFluidMCTS,
            }
        # assert (
        #     len(self.mcts_subtasks_online_training) == 0
        # ) == self.mcts_subproofs_online_training
        for subtask in self.mcts_subtasks_online_training:
            assert subtask in ONLINE_MCTS_SUBTASKS

    def from_goal_params(self, gp: Optional[GoalParams]) -> "ProverParams":
        new_params = deepcopy(self)
        if gp is not None:

            # prover params
            if gp.tactic_fill is not None:
                if gp.tactic_fill == "all":
                    assert gp.length_penalty == 0.0
                    new_params.add_tactic_fill = True
                    new_params.add_tactic_errors = True
                elif gp.tactic_fill == "errors":
                    new_params.add_tactic_fill = False
                    new_params.add_tactic_errors = True
                else:
                    assert gp.tactic_fill == "none"
                    new_params.add_tactic_fill = False
                    new_params.add_tactic_errors = False

            # MCTS params
            if gp.exploration is not None:
                new_params.mcts.exploration = gp.exploration
            if gp.depth_penalty is not None:
                new_params.mcts.depth_penalty = gp.depth_penalty
            if gp.n_expansions is not None:
                new_params.mcts.n_expansions = gp.n_expansions
            if gp.succ_expansions:
                new_params.mcts.succ_expansions = gp.succ_expansions
            if gp.policy:
                new_params.mcts.policy = gp.policy
            if gp.policy_temp_level is not None:
                new_params.mcts.policy_temp_level = gp.policy_temp_level
            if gp.policy_temperature is not None:
                new_params.mcts.policy_temperature = gp.policy_temperature
            if gp.q_value_solved is not None:
                new_params.mcts.q_value_solved = gp.q_value_solved

        return new_params
