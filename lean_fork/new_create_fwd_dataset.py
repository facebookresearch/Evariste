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

from leanml.parse_goal import parse_goal
from typing import Optional, Any
from pathlib import Path
import os
import pandas
import numpy as np
from leanml import get_api
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
import pickle

cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
api = get_api(cur_dir / "build" / "release" / "ml_server", num_threads=80, quiet=False, fast=False, dump_comms=True)
# api = get_api(Path("/checkpoint/tlacroix/datasets/lean_3.30/lean_ml_tooling/build/tlacroix/ckpt"), quiet=False, fast=False,)
df = pandas.read_csv('/checkpoint/tlacroix/datasets/lean_3.30/v1_full_no_full_names/data.csv')

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
    step: int

@dataclass
class ApplyTactic:
    goal: ParseGoal
    single_goal_pp: str
    node_id: int

req_id_to_request = {}

success, total = 0, 0
waiting_for_session_creation, waiting_for_parse_goal, waiting_for_tactic = 0, 0, 0
next_session = 0
for i, (decl, rows) in enumerate(rows_per_decl.items()):
    filename = rows[0].filename
    req_id = api.new_session(filename, decl)
    req_id_to_request[req_id] = Session(decl_name=decl, filename=filename)
    waiting_for_session_creation += 1

to_graph = {}
node_stats = {
    "size": [],
    "n_subgoals": [],
    "n_metavars": [],
    "n_repeated_hyps": [],
}
eval_times = {
    "tactic": [],
    "parse_goal": [],
    "parse_command": []
}

parse_goal_timeouts = 0
added_with_parse_command = 0

apply_tactic_total, apply_tactic_success = 0, 0
parse_goal_total, parse_goal_success, parse_goal_command_success = 0, 0, 0

with open("parse_goal_errors.json", "w") as parse_goal_error:
    with open("parse_command_errors.json", "w") as parse_command_error:
        with open("apply_tactic_errors.json", "w") as output_tac_error:

            while waiting_for_session_creation > 0 or waiting_for_parse_goal > 0 or waiting_for_tactic > 0:
                res = api.recv()
                req = req_id_to_request[res["req_id"]]
                if isinstance(req, Session):
                    if "error" not in res:
                        to_graph[(req.decl_name, req.filename)] = []
                        for row in rows_per_decl[req.decl_name]:
                            if isinstance(row.goal_pp, str):
                                req_id = api.parse_goal(goal_pp=row.goal_pp, session_name=res["name"], timeout=100_000)
                                waiting_for_parse_goal += 1
                                total += 1
                                req_id_to_request[req_id] = ParseGoal(
                                    session=req, goal_pp=row.goal_pp, tactic=row.human_tactic_code, session_name=res["name"], step=0
                                )
                    waiting_for_session_creation -= 1
                elif isinstance(req, ParseGoal):
                    if req.step == 0:
                        parse_goal_total += 1                    
                    if "error" in res:
                        if 'timeout' in res['error']:
                            parse_goal_timeouts += 1
                        if req.step == 0:
                            # waiting_for_parse_goal += 1
                            # req.step = 1
                            # req_id = api.parse_command(goal_pp=req.goal_pp, session_name=req.session_name, timeout=100_000)
                            # req_id_to_request[req_id] = req
                            parse_goal_error.write(json.dumps(dict(
                                error=res['error'], decl_name=req.session.decl_name, filename=req.session.filename, goal_pp=req.goal_pp
                            )) + '\n')
                        else:
                            parse_command_error.write(json.dumps(dict(
                                error=res['error'], decl_name=req.session.decl_name, filename=req.session.filename, goal_pp=req.goal_pp
                            )) + '\n')
                    else:
                        if req.step == 0:
                            eval_times["parse_goal"].append(res["eval_time"])
                            parse_goal_success += 1
                        else:
                            eval_times["parse_command"].append(res["eval_time"])
                            parse_goal_command_success += 1

                        nodes = res['nodes']
                        for node in nodes:
                            req_id = api.send_tactic(req.session_name, node["node_id"], req.tactic, timeout=100_000, max_size=1000 ** 3, max_subgoals=10_000, max_metavars=10_000, type_check=False)
                            waiting_for_tactic += 1
                            req_id_to_request[req_id] = ApplyTactic(goal=req, single_goal_pp=node["full_pp"], node_id=node["node_id"])
                        success += 1
                    waiting_for_parse_goal -= 1
                elif isinstance(req, ApplyTactic):
                    apply_tactic_total += 1
                    if "error" in res:
                        output_tac_error.write(json.dumps(dict(
                            error=res['error'],
                            decl_name=req.goal.session.decl_name,
                            filename=req.goal.session.filename,
                            goal_pp=req.goal.goal_pp,
                            tactic=req.goal.tactic,
                            node_id=req.node_id
                        )) + '\n')
                    else:
                        apply_tactic_success += 1
                        node_stats["n_repeated_hyps"].append(res["repeated_hyps"])
                        eval_times["tactic"].append(res["eval_time"])
                        to_graph[(req.goal.session.decl_name, req.goal.session.filename)].append({"goal_pp": req.single_goal_pp, "tactic": req.goal.tactic, "res": res})
                        for x in res["nodes"]:
                            node_stats["size"].append(x["size"])
                            node_stats["n_subgoals"].append(x["n_subgoals"])
                            node_stats["n_metavars"].append(x["n_metavars"])
                        if req.goal.step == 1:
                            added_with_parse_command += 1
                    waiting_for_tactic -= 1

                print("Waiting")
                print(waiting_for_session_creation, waiting_for_parse_goal, waiting_for_tactic)
                print(f"Success {success} / Total {total} / added with parse command {added_with_parse_command}")
                for k, v in node_stats.items():
                    if len(v) == 0:
                        continue
                    print(f"{k}: avg={np.mean(v):.3f}, std={np.std(v):.3f}, min={np.min(v)}, max={np.max(v)}")
                for k, v in eval_times.items():
                    if len(v) == 0:
                        continue
                    print(f"eval_time : {k}: avg={np.mean(v):.3f}, std={np.std(v):.3f}, min={np.min(v)}, max={np.max(v)}")
                print(f"Parse attempts {parse_goal_total}, success {parse_goal_success} with command {parse_goal_command_success}")
                print(f"Apply tactic attempts {apply_tactic_total}, success {apply_tactic_success}")
            print(total, success, flush=True)
            print("OVER AND OUT", flush=True)

pickle.dump(to_graph, open('to_graph_single_node.pkl', 'wb'))
pickle.dump(node_stats, open('node_stats.pkl', 'wb'))
pickle.dump(eval_times, open('eval_times.pkl', 'wb'))
