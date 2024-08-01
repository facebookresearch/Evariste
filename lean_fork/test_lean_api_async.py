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

from typing import Dict
from pathlib import Path
import os
import time
import json
import sys
import numpy as np
import pandas
import Levenshtein

from leanml import get_api, WrongModule

assert len(sys.argv) > 1, "Please give path to csv as input"
assert Path(sys.argv[1]).exists(), ""
df = pandas.read_csv(sys.argv[1])
has_mathlib = df.filename.str.contains('mathlib/src').any()

cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
start = time.time()
lean_api = get_api(cur_dir / "build" / "release" / "ml_server", preload=True, fast=not has_mathlib)
print(f"Preloaded in {time.time() - start}")

all_decls = df.decl_name.unique().tolist()
start = time.time()

active_sessions = set()
sessions = []
req_id_to_session = {}


def find_best_match(ref: str, tactics: Dict[str, str]):
    """There are small discrepancies between the pretty printer used in the ml server and in the lean step dataset"""
    res, min_dist = None, float("inf")
    for k, v in tactics.items():
        try:
            dist = Levenshtein.distance(ref, k)
        except TypeError:
            dist = float("inf")
        if dist < min_dist:
            res = v
            min_dist = dist
    return res


finished, success = [0], [0]

expected_answers = {}


class Session:
    def __init__(self, decl_id, my_id):

        self.decl = all_decls[decl_id]
        all_steps = df[df.decl_name == self.decl]
        print(all_steps)
        self.filename = all_steps.iloc[0].filename

        self.goal_pp_to_tactic = {}
        for row in all_steps.iloc:
            # if row.tactic_class in {"named", "semicolon"}:
            self.goal_pp_to_tactic[row.goal_pp] = row.human_tactic_code
        self.session_name = None
        self.goals = []
        self.my_id = my_id
        self.dying = False
        self.outstanding_message = 1
        self.sent_messages = 0
        self.has_error = False
        active_sessions.add(self.my_id)

    def process(self, message):
        del expected_answers[message["req_id"]]
        self.outstanding_message -= 1
        if "initial_goal" in message:
            self.goals.append((1, message["initial_goal"]["full_pp"]))
            self.session_name = message["name"]
        elif "nodes" in message:
            for goal in res["nodes"]:
                self.goals.append((goal["node_id"], goal["full_pp"]))
        elif "erased" in message:
            assert self.dying
            active_sessions.remove(self.my_id)
            if not self.has_error:
                success[0] += 1
            finished[0] += 1
            return 0
        else:
            self.has_error = True
            print("has error", message)
            # Handle error at creation
            if self.session_name is None:
                finished[0] += 1
                active_sessions.remove(self.my_id)
                return 0
        return self.send()

    def send(self):
        assert self.my_id in active_sessions, "sending but session is inactive"
        if len(self.goals) == 0 and self.outstanding_message == 0:
            req_id = lean_api.del_session(self.session_name)
            req_id_to_session[req_id] = self.my_id
            self.dying = True
            self.outstanding_message += 1
            expected_answers[req_id] = ("del_session", self.session_name, time.time())
            return 1
        to_ret = 0
        while self.goals and self.sent_messages < 2 * len(self.goal_pp_to_tactic):
            state_id, state_pp = self.goals.pop()
            tac = find_best_match(state_pp, self.goal_pp_to_tactic)
            req_id = lean_api.send_tactic(self.session_name, state_id, tac)
            req_id_to_session[req_id] = self.my_id
            expected_answers[req_id] = (
                "tactic",
                self.session_name,
                state_id,
                tac,
                time.time(),
            )
            to_ret += 1
            self.outstanding_message += 1
            self.sent_messages += 1
        if self.sent_messages >= 2 * len(self.goal_pp_to_tactic):
            self.has_error = True
            self.goals = []
            if self.outstanding_message == 0:
                return self.send()  # this will send a del_session
        return to_ret


simultaneous = 30  # number of maximum parallel outstanding requests
expected_messages = 0


np.random.seed(0)
np.random.shuffle(all_decls)


cur_decl = 0
max_decl = len(all_decls)

start = time.time()
while cur_decl < max_decl or expected_messages > 0:
    # initialize sessions if needed
    while cur_decl < max_decl and len(active_sessions) < simultaneous:
        new_session = Session(cur_decl % len(all_decls), len(sessions))
        try:
            req_id = lean_api.new_session(
                module_path=new_session.filename, decl_name=new_session.decl
            )
        except WrongModule:
            # mapped file doesn't exist
            cur_decl += 1
            continue
        expected_answers[req_id] = ("new_session", cur_decl, time.time())
        req_id_to_session[req_id] = len(sessions)
        sessions.append(new_session)
        cur_decl += 1
        expected_messages += 1

    # receive and process any incoming messages
    next_messages = 0
    while expected_messages > 0:
        try:
            res = lean_api.recv(timeout=0.1)
        except TimeoutError:
            if len(expected_answers) < 10:
                print("Waiting for ", expected_answers)
            break

        if "req_id" in res:
            req_id = res["req_id"]
            next_messages += sessions[req_id_to_session[req_id]].process(res)
            expected_messages -= 1
    expected_messages += next_messages
print(f"Total {finished[0]} -- Success {success[0]}")
print("finished", time.time() - start)
