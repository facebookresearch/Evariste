# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from evariste import json as json
import logging
import shutil
from pathlib import Path
from typing import Optional, cast, Dict

import pandas
from leanml import get_api
from collections import defaultdict
from dataclasses import dataclass, asdict

from evariste.backward.env.lean.graph import LeanTheorem, LeanContext
from evariste.datasets.lean import LeanDatasetConf, LEAN_DATASET_LATEST_FWD
from evariste.forward.fwd_lean.lean_fwd_goal_factory import get_parsed_goals_path

"""
Stats:

21.08.27

Success
        test: 1054 1240 0
                 errors: session: 9,parse: 186, n_subgoals: 0
        train: 23418 27721      0
                 errors: session: 223,parse: 4303, n_subgoals: 0
        valid: 1066 1235        0
                 errors: session: 6,parse: 169, n_subgoals: 0


Success
        minif2f_valid: 220 244  0
                 errors: session: 0,parse: 24, n_subgoals: 0
        minif2f_test: 227 244   0
                 errors: session: 0,parse: 17, n_subgoals: 0

"""


@dataclass
class Session:
    decl_name: str
    filename: str
    split: str
    open_namespaces: str


@dataclass
class ParseGoal:
    session: Session
    goal_pp: str
    fingerprint: str
    session_name: str


@dataclass
class ParsedGoal:
    label: str
    succeed: bool
    goal_pp: Optional[str] = None
    fingerprint: Optional[str] = None
    parsed_goal: Optional[Dict] = None
    error_msg: Optional[str] = None


def parse_goals(dataset: LeanDatasetConf, mini_f2f: bool = False):
    print("Creating api")
    api = get_api(Path(dataset.checkpoint_path), quiet=False, fast=False)
    print("Loading dataset")
    data_dir = Path(dataset.data_dir)
    if not mini_f2f:
        df = pandas.read_csv(data_dir / "data.csv")
    else:
        df = pandas.read_csv(data_dir / "minif2f.csv")

    dst = get_parsed_goals_path(data_dir=data_dir, minif2f=mini_f2f)
    print("dst", dst)
    if dst.exists():
        print("WARNING\n" * 10)
        print(f"Going to replace {dst}")

    # i = 0
    rows_per_decl = defaultdict(list)
    for row in df.iloc:
        rows_per_decl[row.decl_name].append(row)
        # i += 1
        # if i > 100:
        #     break

    results = []

    req_id_to_request = {}

    per_split_ok, per_split_total = defaultdict(int), defaultdict(int)
    per_split_n_subgoal_error = defaultdict(int)
    per_split_parse_error = defaultdict(int)
    per_split_session_error = defaultdict(int)

    waiting_for_session_creation, waiting_for_parse_goal = 0, 0

    for i, (decl, rows) in enumerate(rows_per_decl.items()):
        filename = rows[0].filename
        split = rows[0].split
        req_id = api.new_session(
            filename, decl, pp_opts={"pp.full_names": dataset.pp_full_names}
        )
        try:
            open_namespaces = rows[0].open_namespaces
        except AttributeError:
            open_namespaces = ""
        if isinstance(open_namespaces, float):
            open_namespaces = ""
        assert isinstance(open_namespaces, str), type(open_namespaces)
        req_id_to_request[req_id] = Session(
            decl_name=decl,
            filename=filename,
            open_namespaces=open_namespaces,
            split=split,
        )
        waiting_for_session_creation += 1

    while waiting_for_session_creation > 0 or waiting_for_parse_goal > 0:
        res = api.recv()
        req = req_id_to_request[res["req_id"]]
        if isinstance(req, Session):
            if "error" not in res:
                goal_pp = res["initial_goal"]["full_pp"]
                fingerprint = res["initial_goal"]["fingerprint"]
                req_id = api.parse_goal(
                    goal_pp=goal_pp,
                    session_name=res["name"],
                    max_repeated_hyps=dataset.max_repeated_hyps,
                )
                req_id_to_request[req_id] = ParseGoal(
                    session=req,
                    goal_pp=goal_pp,
                    session_name=res["name"],
                    fingerprint=fingerprint,
                )
                per_split_total[req.split] += 1
                waiting_for_parse_goal += 1
            else:
                per_split_session_error[req.split] += 1
                results.append(
                    ParsedGoal(
                        label=req.decl_name, succeed=False, error_msg=res["error"]
                    )
                )
            waiting_for_session_creation -= 1
        elif isinstance(req, ParseGoal):
            if "error" in res:
                per_split_parse_error[req.session.split] += 1
                results.append(
                    ParsedGoal(
                        label=req.session.decl_name,
                        goal_pp=req.goal_pp,
                        fingerprint=req.fingerprint,
                        succeed=False,
                        error_msg=res["error"],
                    )
                )
            elif (
                res["n_subgoals"] != 1
                or len(res["nodes"]) != 1
                or any(node["n_subgoals"] != 1 for node in res["nodes"])
                or any(" goals" in node["full_pp"] for node in res["nodes"])
            ):
                results.append(
                    ParsedGoal(
                        label=req.session.decl_name,
                        goal_pp=req.goal_pp,
                        succeed=False,
                        error_msg=f"incorrect n subgoals: {res}",
                    )
                )
                per_split_n_subgoal_error[req.session.split] += 1
                print(f"incorrect n subgoals: {res}")
            else:
                per_split_ok[req.session.split] += 1
                thm = LeanTheorem(
                    conclusion=res["nodes"][0]["full_pp"],
                    fingerprint=res["nodes"][0]["fingerprint"],
                    context=LeanContext(
                        namespaces=set(req.session.open_namespaces.split())
                    ),
                    state=None,
                )
                results.append(
                    ParsedGoal(
                        label=req.session.decl_name,
                        goal_pp=req.goal_pp,
                        fingerprint=req.fingerprint,
                        succeed=True,
                        parsed_goal=thm.to_dict(light=True),
                    )
                )

            waiting_for_parse_goal -= 1

        print("Waiting")
        print(waiting_for_session_creation, waiting_for_parse_goal)
        print("Success")
        for x in per_split_total:
            print(
                f"\t{x}: {per_split_ok[x]} {per_split_total[x]}\t{per_split_n_subgoal_error[x]}"
            )
            print(
                f"\t\t errors: session: {per_split_session_error[x]},"
                f"parse: {per_split_parse_error[x]}, "
                f"n_subgoals: {per_split_n_subgoal_error[x]}"
            )
    serialized_results = []
    for result in results:
        try:
            result = asdict(result)
        except TypeError:
            print("Error", result)
        else:
            serialized_results.append(result)

    tmp = Path(dst.parent) / f"{dst.name}.tmp"
    with tmp.open("w") as fp:
        for result in serialized_results:
            fp.write(json.dumps(result) + "\n")
    print(f"Moving results to {dst}")
    shutil.move(str(tmp), str(dst))
    print(f"Moved results to {dst}")


if __name__ == "__main__":
    MINI_F2F = True
    dataset = LEAN_DATASET_LATEST_FWD.get_materialized()
    print(f"dataset conf: {dataset}")
    parse_goals(dataset=dataset, mini_f2f=MINI_F2F)
