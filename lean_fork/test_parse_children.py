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
api = get_api(
    cur_dir / "build" / "release" / "ml_server",
    quiet=False,
    fast=False,
    dump_comms=True,
)


@dataclass
class ParseChildrenTest:
    decl_name: str
    filename: str
    to_parse: List[str]
    expected_parsed: List[str]


to_try = [
    ParseChildrenTest(
        decl_name="int.sub_nat_nat_add_left",
        filename="lean/library/init/data/int/basic.lean",
        to_parse=["m n : ℕ\n⊢ m + n = n + m"],
        expected_parsed=["m n : ℕ\n⊢ m + n = n + m"],
    ),
    ParseChildrenTest(
        decl_name="int.sub_nat_nat_add_left",
        filename="lean/library/init/data/int/basic.lean",
        to_parse=["α : Type*,\n_inst_1 : linear_order α,\na : α\n⊢ max a a = a"],
        expected_parsed=[
            "α : Type u_1,\n_inst_1 : linear_order α,\na : α\n⊢ max a a = a"
        ],
    ),
    ParseChildrenTest(
        decl_name="int.sub_nat_nat_add_left",
        filename="lean/library/init/data/int/basic.lean",
        to_parse=[
            "m n : ℕ\n⊢ m + n = n + m",
            "α : Type*,\n_inst_1 : linear_order α,\na : α\n⊢ max a a = a",
        ],
        expected_parsed=[
            "m n : ℕ\n⊢ m + n = n + m",
            "α : Type u_1,\n_inst_1 : linear_order α,\na : α\n⊢ max a a = a",
        ],
    ),
    ParseChildrenTest(
        decl_name="int.sub_nat_nat_add_left",
        filename="lean/library/init/data/int/basic.lean",
        to_parse=["n: ℕ\nh₀: n > 5\n⊢ n > 12"],
        expected_parsed=["⊢ ∀ (n : ℕ), n > 5 → n > 12"],
    ),
    ParseChildrenTest(
        decl_name="int.sub_nat_nat_add_left",
        filename="lean/library/init/data/int/basic.lean",
        to_parse=["⊢ ∀ (n : ℕ), n > 5 → n > 12"],
        expected_parsed=["⊢ ∀ (n : ℕ), n > 5 → n > 12"],
    ),
]


failing = [
    ParseChildrenTest(
        decl_name="int.sub_nat_nat_add_left",
        filename="lean/library/init/data/int/basic.lean",
        to_parse=["⊢ ∀ (n : ℕ), n > 5 → max(n, m) > 12"],
        expected_parsed=[],
    ),
    ParseChildrenTest(
        decl_name="int.sub_nat_nat_add_left",
        filename="lean/library/init/data/int/basic.lean",
        to_parse=["⊢ ∀ (n : ℕ), n > 5 → n > 12\n\nm n : ℕ\n⊢ m + n = n + m"],
        expected_parsed=[],
    ),
]

for example in to_try:
    req_id = api.new_session(example.filename, example.decl_name)
    session = api.recv()
    api.parse_children(example.to_parse, session["name"])
    res = api.recv()

    received = [c["full_pp"] for c in res["parsed_children"]]
    expected = example.expected_parsed
    print("res", res)

    assert received == expected, f"{received} != {expected}"

    assert "error" not in res, res


for example in failing:
    _ = api.new_session(example.filename, example.decl_name)
    session = api.recv()
    api.parse_children(example.to_parse, session["name"])
    res = api.recv()

    assert "error" in res, res