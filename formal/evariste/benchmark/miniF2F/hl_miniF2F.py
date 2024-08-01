# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Union, List, Dict
import os
import os.path
from collections import defaultdict
import re
import functools

from evariste.backward.env.hol_light.graph import HLTheorem
from evariste.backward.env.core import BackwardGoal
from evariste.forward.fwd_hl.common import HLForwardGoal
from evariste.envs.ocaml.api import OCamlAPI, OCamlError
from evariste.envs.hl.api import _ocaml_to_hol_tokens
from evariste.datasets.hol import ConfStore
from evariste.logger import create_logger

logger = create_logger(None)

MINIF2F_DIRPATH = ""
DEFAULT_CHECKPOINT = ConfStore["hl_plus_default_dataset"].checkpoint_path


def get_miniF2F_benchmark(
    miniF2F_dirpath: str = MINIF2F_DIRPATH, checkpoint_path: str = DEFAULT_CHECKPOINT,
):
    pattern_cmd = r"`(?P<term>[^`]+)`\s*;;\s*"
    pattern_term = r"val it : term =\s*`(?P<conclusion>[^`]+)`\s*"

    env = OCamlAPI(checkpoint_path=checkpoint_path, timeout=5.0)

    benchmark = {"success": defaultdict(dict), "error": defaultdict(dict)}
    src_dirpath = os.path.join(miniF2F_dirpath, "src")

    for dirname in os.listdir(src_dirpath):
        for filename in os.listdir(os.path.join(src_dirpath, dirname)):
            with open(os.path.join(src_dirpath, dirname, filename)) as f:
                cmd = f.read()
            match = re.match(pattern_cmd, cmd)
            raw_statement = " ".join(re.sub(r"\n", " ", match.group("term")).split())
            try:
                raw_reply = env.send(
                    f'parse_term_EVARISTE_original "{raw_statement}";;'
                )
                match = re.match(pattern_term, raw_reply)
                conclusion = match.group("conclusion")
                conclusion_ocaml_tokens = conclusion.split()
                conclusion_hl_tokens = _ocaml_to_hol_tokens(conclusion_ocaml_tokens)
                hl_conclusion = " ".join(conclusion_hl_tokens)
                benchmark["success"][dirname][filename] = {
                    "conclusion": hl_conclusion,
                    "hyps": [],
                }
            except OCamlError as e:
                benchmark["error"][dirname][filename] = {
                    "raw_statement": raw_statement,
                    "error": str(e),
                }
    del env
    logger.info(
        f"Extracted {len([filename for filenames in benchmark['success'].values() for filename in filenames])} "
        "HOL Light miniF2F statements: "
        f"{', '.join([str(len(benchmark['success'][dirname])) + ' ' + dirname for dirname in benchmark['success']])}"
    )
    if benchmark["error"]:
        logger.warning(
            f"Failed to extract {len([filename for filenames in benchmark['error'].values() for filename in filenames])} "
            "HOL Light miniF2F statements: "
            f"{', '.join([str(len(benchmark['error'][dirname])) + ' ' + dirname for dirname in benchmark['error']])}"
        )
    return benchmark


def get_miniF2F_goals(
    proving_mode: str, benchmark: Dict[str, Dict], splitted: bool = False,
) -> Union[Dict[str, List[BackwardGoal]], List[HLForwardGoal]]:
    goals = {} if splitted else []
    for split in benchmark["success"]:
        split_goals = []
        for name, theorem_dict in benchmark["success"][split].items():
            thm = HLTheorem(**theorem_dict)
            if proving_mode == "bwd":
                goal = BackwardGoal(name=name, theorem=thm)
            elif proving_mode == "fwd":
                goal = HLForwardGoal(label=name, thm=thm, statement="", e_hyps=[])
            else:
                raise NotImplementedError(
                    f'The proving mode "{proving_mode}" is not supported'
                )
            split_goals.append(goal)
        if splitted:
            goals[split] = split_goals
        else:
            goals += split_goals
    return goals


@functools.lru_cache
def get_miniF2F_goals_from_repo(
    proving_mode: str,
    miniF2F_dirpath: str = MINIF2F_DIRPATH,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    splitted: bool = False,
) -> Union[Dict[str, List[BackwardGoal]], List[HLForwardGoal]]:
    benchmark = get_miniF2F_benchmark(
        miniF2F_dirpath=miniF2F_dirpath, checkpoint_path=checkpoint_path
    )
    return get_miniF2F_goals(
        proving_mode=proving_mode, benchmark=benchmark, splitted=splitted
    )
