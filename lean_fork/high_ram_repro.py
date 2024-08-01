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


from leanml import get_api
from pathlib import Path
import os
import time
import json
import psutil

print("Starting API")
cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
start = time.time()
api = get_api(cur_dir / "build" / "release" / "ml_server", preload=False, fast=False, dump_comms=False, quiet=True)
print("API started")

api.new_session(
    "mathlib/src/algebra/homology/exact.lean", 
    "category_theory.exact_epi_comp"
)
session = api.recv()

proc = psutil.Process(api._proc.pid)
print(f"Mem usage before : {int(proc.memory_info().rss / 1024**3)}GB")
ts_to_node_id = {}
ts_to_node_id[session["initial_goal"]["full_pp"]] = 1


with open("high_ram_repro.jsonl", "r") as f:
    for line in f:
        loaded = json.loads(line)
        th = loaded["th"]["conclusion"]
        tac = loaded["tac"]["str"]
        api.send_tactic(session_name=session["name"], state_id=ts_to_node_id[th], tactic_str=tac, timeout=2000)
        res = api.recv()
        print(f"Added {len(res['nodes'])} children")
        for n in res["nodes"]:
            ts_to_node_id[n["full_pp"]] = n["node_id"]
    final_tacs = [
        "simp [cancel_epi]",
        "simp [zero_comp]",
        "simp [category.assoc]",
        "simp only [category.assoc]",
        "simp [cokernel.condition]",
        "simp [equalizer.condition g h]",
        "simp",
        "simp [h0]",
        "simp only [zero_comp]",
        "simp [←cancel_epi]",
        "simp[cancel_epi]"
    ]
    for final_tac in final_tacs:
        print(f"Applying '{final_tac}'")
        api.send_tactic(
            session_name=session["name"],
            state_id=28,
            tactic_str=final_tac,
            timeout=2000
        )
        start = time.time()
        while True:
            try:
                res = api.recv(timeout=1)
                if "error" in res:
                    print("\t", res["error"])
                else:
                    print("\t", f"Has {len(res['nodes'])} children")
                break
            except TimeoutError:
                print(f"Mem usage [{int(time.time() - start)}s]: {int(proc.memory_info().rss / 1024**3)}GB")
                pass
    print("All good")
