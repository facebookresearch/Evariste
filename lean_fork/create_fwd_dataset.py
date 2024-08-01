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

from typing import Optional, Any
from pathlib import Path
import os
import pandas
from leanml import get_api
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
import pickle

cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
api = get_api(cur_dir / "build" / "release" / "ml_server", quiet=False, fast=False,)

df = pandas.read_csv('/checkpoint/tlacroix/datasets/lean_3.30/data_full_no_full_names/data.csv')

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
    nodes: Any
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

with open("parse_goal_errors.json", "w") as output_error:
    with open("apply_tactic_errors.json", "w") as output_tac_error:

        while waiting_for_session_creation > 0 or waiting_for_parse_goal > 0 or waiting_for_tactic > 0:
            res = api.recv()
            req = req_id_to_request[res["req_id"]]
            if isinstance(req, Session):
                if "error" not in res:
                    to_graph[(req.decl_name, req.filename)] = []
                    for row in rows_per_decl[req.decl_name]:
                        if isinstance(row.goal_pp, str):
                            req_id = api.parse_goal(goal_pp=row.goal_pp, session_name=res["name"])
                            waiting_for_parse_goal += 1
                            total += 1
                            req_id_to_request[req_id] = ParseGoal(
                                session=req, goal_pp=row.goal_pp, tactic=row.human_tactic_code, session_name=res["name"]
                            )
                waiting_for_session_creation -= 1
            elif isinstance(req, ParseGoal):
                if "error" in res:
                    output_error.write(json.dumps(dict(
                        error=res['error'], decl_name=req.session.decl_name, filename=req.session.filename, goal_pp=req.goal_pp
                    )) + '\n')
                else:
                    nodes = res['nodes']
                    req_id = api.send_tactic(req.session_name, nodes[0]["node_id"], req.tactic, timeout=2000)
                    waiting_for_tactic += 1
                    req_id_to_request[req_id] = ApplyTactic(goal=req, nodes=nodes, node_id=nodes[0]["node_id"])
                    success += 1
                waiting_for_parse_goal -= 1
            elif isinstance(req, ApplyTactic):
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
                    to_graph[(req.goal.session.decl_name, req.goal.session.filename)].append((asdict(req), res))
                waiting_for_tactic -= 1

            print("Waiting")
            print(waiting_for_session_creation, waiting_for_parse_goal, waiting_for_tactic)
            print("Success")
            print(success, total)

        print(total, success, flush=True)
        print("OVER AND OUT", flush=True)
pickle.dump(to_graph, open('to_graph.pkl', 'wb'))



    
