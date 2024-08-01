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

from typing import List, Dict, Union
from dataclasses import dataclass
from pathlib import Path
import os

from leanml.parse_goal import parse_goal, parse_goal_structured
from leanml.comms import get_api


cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
api = get_api(cur_dir / "build" / "release" / "ml_server", quiet=False, fast=False, dump_comms=True)


@dataclass
class ParseGoalAndApplyTacticTest:
    decl_name: str
    filename: str
    to_parse: str
    tactic: str
    expected_parsed_goal: str

to_try = [
    ParseGoalAndApplyTacticTest(
        decl_name="int.sub_nat_nat_add_left",
        filename="lean/library/init/data/int/basic.lean",
        to_parse="m n : ℕ\n⊢ m + n = n + m",
        expected_parsed_goal="m n : ℕ\n⊢ m + n = n + m",
        tactic="rw nat.add_comm"
    ),
    ParseGoalAndApplyTacticTest(
        decl_name="int.sub_nat_nat_add_left",
        filename="lean/library/init/data/int/basic.lean",
        to_parse="α : Type*,\n_inst_1 : linear_order α,\na : α\n⊢ max a a = a",
        expected_parsed_goal="α : Type u_1,\n_inst_1 : linear_order α,\na : α\n⊢ max a a = a",
        tactic="unfold max"
    ),
    ParseGoalAndApplyTacticTest(
        decl_name="int.sub_nat_nat_add_left",
        filename="lean/library/init/data/int/basic.lean",
        to_parse="n: ℕ\nh₀: n > 5\n⊢ n > 12",
        expected_parsed_goal="⊢ ∀ (n : ℕ), n > 5 → n > 12",
        tactic="intros"
    ),
]
   
for example in to_try:
    if None not in {example.filename, example.decl_name}:
        req_id = api.new_session(example.filename, example.decl_name)
        session = api.recv()
        api.parse_goal_and_apply_tactic(example.to_parse, session["name"], example.tactic)
        res = api.recv()

        received = res["parsed_goal"]["full_pp"]
        expected = example.expected_parsed_goal
        print("res", res)

        assert received == expected, f"{received!r} != {expected!r}"
        

        assert "error" not in res, res
