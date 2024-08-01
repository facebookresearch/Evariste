# Copyright (c) Facebook, Inc. and its affiliates.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from pathlib import Path
import pandas
import numpy as np
from leanml import get_api
from evariste import json as json
from collections import defaultdict
from dataclasses import dataclass, asdict
import pickle

from evariste.datasets.lean import LEAN_DATASET_LATEST, LEAN_DATASET_LATEST_FWD

# - parse_goal(dataset.csv) -> [dataset_goal_1, dataset_goal_2, ...]
# - parse_goal_and_apply_tactic(dataset_goal_1, tactic, strip_tag=True) -> [
#       reparsed_dataset_goal1, (children_1_1, ...)
# ]
# on dump (dataset_goal_1, reparsed_dataset_goal1, (children_1_1 ...) a chaque fois avec les fingerprints

dataset = LEAN_DATASET_LATEST_FWD.get_materialized()

df = pandas.read_csv(f"{dataset.data_dir}/data.csv")

api = get_api(
    Path(dataset.checkpoint_path),
    fast=False,
    quiet=True,
    path_to_fifos=None,
    num_threads=80,
    dump_comms=True,
)

dst_dir = f"{dataset.data_dir}/fwd_dataset_creation_logs"
Path(dst_dir).mkdir(exist_ok=True)
print("dst_dir", dst_dir)

rows_per_decl = defaultdict(list)
for row in df.iloc:
    rows_per_decl[row.decl_name].append(row)


@dataclass
class Session:
    decl_name: str
    filename: str


@dataclass
class ParseGoal:
    session: Session
    goal_pp: str
    tactic: str
    session_name: str


@dataclass
class ApplyTactic:
    goal: ParseGoal
    reparsed: Any


req_id_to_request = {}

success, total = 0, 0
(waiting_for_session_creation, waiting_for_parse_goal, waiting_for_tactic,) = (0, 0, 0)
next_session = 0
for i, (decl, rows) in enumerate(rows_per_decl.items()):
    filename = rows[0].filename
    req_id = api.new_session(
        filename, decl, pp_opts={"pp.full_names": dataset.pp_full_names}
    )
    req_id_to_request[req_id] = Session(decl_name=decl, filename=filename)
    waiting_for_session_creation += 1

to_graph = {}
node_stats = {
    "size": [],
    "n_subgoals": [],
    "n_metavars": [],
    "n_repeated_hyps": [],
}
eval_times = {"tactic": [], "parse_goal": [], "parse_command": []}

parse_goal_timeouts = 0
added_with_parse_command = 0

apply_tactic_total, apply_tactic_success = 0, 0
(
    parse_goal_total,
    parse_goal_success,
    parse_goal_command_success,
    diff_reparsing,
    diff_children,
) = (0, 0, 0, 0, 0)

with open(f"{dst_dir}/parse_goal_errors.json", "a") as parse_goal_error:
    with open(f"{dst_dir}/apply_tactic_errors.json", "a") as output_tac_error:
        while (
            waiting_for_session_creation > 0
            or waiting_for_parse_goal > 0
            or waiting_for_tactic > 0
        ):
            res = api.recv()
            req = req_id_to_request[res["req_id"]]
            if isinstance(req, Session):
                if "error" not in res:
                    to_graph[(req.decl_name, req.filename)] = []
                    for row in rows_per_decl[req.decl_name]:
                        if isinstance(row.goal_pp, str):
                            req_id = api.parse_goal(
                                goal_pp=row.goal_pp,
                                session_name=res["name"],
                                timeout=100_000,
                                max_repeated_hyps=dataset.max_repeated_hyps,
                            )
                            waiting_for_parse_goal += 1
                            total += 1
                            req_id_to_request[req_id] = ParseGoal(
                                session=req,
                                goal_pp=row.goal_pp,
                                tactic=row.human_tactic_code,
                                session_name=res["name"],
                            )
                waiting_for_session_creation -= 1
            elif isinstance(req, ParseGoal):
                parse_goal_total += 1
                if "error" in res:
                    if "timeout" in res["error"]:
                        parse_goal_timeouts += 1
                    parse_goal_error.write(
                        json.dumps(
                            dict(
                                error=res["error"],
                                decl_name=req.session.decl_name,
                                filename=req.session.filename,
                                goal_pp=req.goal_pp,
                            )
                        )
                        + "\n"
                    )
                else:
                    eval_times["parse_goal"].append(res["eval_time"])
                    parse_goal_success += 1

                    nodes = res["nodes"]
                    for node in nodes:
                        req_id = api.parse_goal_and_apply_tactic(
                            node["full_pp"],
                            req.session_name,
                            req.tactic,
                            timeout=100_000,
                            max_size=1000 ** 3,
                            max_subgoals=10_000,
                            max_metavars=10_000,
                            strip_tags=True,
                            max_repeated_hyps=dataset.max_repeated_hyps,
                        )
                        waiting_for_tactic += 1
                        req_id_to_request[req_id] = ApplyTactic(goal=req, reparsed=node)
                    success += 1
                waiting_for_parse_goal -= 1
            elif isinstance(req, ApplyTactic):
                apply_tactic_total += 1
                if "error" in res:
                    output_tac_error.write(
                        json.dumps(
                            dict(
                                error=res["error"],
                                decl_name=req.goal.session.decl_name,
                                filename=req.goal.session.filename,
                                goal_pp=req.goal.goal_pp,
                                tactic=req.goal.tactic,
                                node=req.reparsed,
                            )
                        )
                        + "\n"
                    )
                else:
                    apply_tactic_success += 1
                    # node_stats["n_repeated_hyps"].append(res["repeated_hyps"])
                    eval_times["tactic"].append(res["eval_time"])
                    res["nodes"] = res.pop("subgoals")

                    to_graph[
                        (req.goal.session.decl_name, req.goal.session.filename)
                    ].append({"req": asdict(req), "res": res})
                    for x in res["nodes"]:
                        node_stats["size"].append(x["size"])
                        node_stats["n_subgoals"].append(x["n_subgoals"])
                        node_stats["n_metavars"].append(x["n_metavars"])
                waiting_for_tactic -= 1
            print("Waiting")
            print(
                waiting_for_session_creation,
                waiting_for_parse_goal,
                waiting_for_tactic,
            )
            print(
                f"Success {success} / Total {total} / added with parse command {added_with_parse_command}"
            )
            for k, v in node_stats.items():
                if len(v) == 0:
                    continue
                print(
                    f"{k}: avg={np.mean(v):.3f}, std={np.std(v):.3f}, min={np.min(v)}, max={np.max(v)}"
                )
            for k, v in eval_times.items():
                if len(v) == 0:
                    continue
                print(
                    f"eval_time : {k}: avg={np.mean(v):.3f}, std={np.std(v):.3f}, min={np.min(v)}, max={np.max(v)}"
                )
            print(
                f"Parse attempts {parse_goal_total}, success {parse_goal_success} with command {parse_goal_command_success}"
            )
            print(
                f"Apply tactic attempts {apply_tactic_total}, success {apply_tactic_success}"
            )
        print(total, success, flush=True)
        print("OVER AND OUT", flush=True)

pickle.dump(to_graph, open(f"{dst_dir}/to_graph_single_node.pkl", "wb"))
pickle.dump(node_stats, open(f"{dst_dir}/node_stats.pkl", "wb"))
pickle.dump(eval_times, open(f"{dst_dir}/eval_times.pkl", "wb"))
print("dst_dir", dst_dir)
