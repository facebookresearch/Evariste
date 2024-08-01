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

from leanml.parse_goal import parse_goal_structured
from leanml.comms import get_api

@dataclass
class ParseGoalTest:
    decl_name: str
    filename: str
    to_parse: str
    expected: List[Dict[str, Union[str, int]]]

to_try = [
    ParseGoalTest(
        decl_name="tt_band",
        filename="lean/library/init/data/bool/lemmas.lean",
        to_parse="b : bool\n⊢ bool.tt && b = b",
        expected=[{
            'goal': "∀ b : bool,\nbool.tt && b = b",
            'n_hyps': 1,
        }]
    ),
    ParseGoalTest(
        decl_name="field.to_division_ring",
        filename="mathlib/src/algebra/field.lean",
        to_parse="K : Type u,\n_inst_1 : field K\n⊢ field K",
        expected=[{
            'goal': "∀ K : Type u,\n∀ _inst_1 : field K,\nby {tactic.unfreeze_local_instances, tactic.freeze_local_instances, exact\nfield K\n}",
            'n_hyps': 2,
        }]
    ),
    ParseGoalTest(
        decl_name=None,
        filename=None,
        to_parse="n : ℕ\n⊢ 0 = 1 → n = 0\n\nn : ℕ\n⊢ n = 0 → 0 = 1",
        expected=[{
            'n_hyps':1,
            'goal': "∀ n : ℕ,\n0 = 1 → n = 0",
        },{
            'n_hyps':1,
            'goal': "∀ n : ℕ,\nn = 0 → 0 = 1",
        }]
    )

]

cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
api = get_api(cur_dir / "build" / "release" / "ml_server", quiet=False, fast=False, dump_comms=True)

for example in to_try:
    res = [x.to_expression() for x in parse_goal_structured(example.to_parse)]
    assert len(res) == len(example.expected), (res, example.expected)

    for r,  e in zip(res, example.expected):
        for k in ('goal', 'n_hyps'):
            if r[k] != e[k]:
                print("EXPECTED\n")
                print(e[k])
                print("\nGOT\n")
                print(r[k])
                raise RuntimeError(f"{k}")

for example in to_try:
    if None not in {example.filename, example.decl_name}:
        req_id = api.new_session(example.filename, example.decl_name, pp_opts={'pp.implicit': True, 'pp.notation': False})
        session = api.recv()
        api.parse_goal(example.to_parse, session["name"])
        res = api.recv()
        api.parse_command(example.to_parse, session["name"])
        res = api.recv()

        assert "error" not in res, res



# long
# decl_name = "ideal.quotient_equiv_alg"
# filename = "mathlib/src/ring_theory/ideal/operations.lean"
# to_parse = "R : Type u_1,\nS : Type u_2,\n_inst_1 : comm_ring R,\n_inst_2 : comm_ring S,\nA : Type u_3,\n_inst_3 : comm_ring A,\n_inst_4 : algebra R A,\n_inst_7 : algebra R S,\nI : ideal A,\nJ : ideal S,\nf : A ≃ₐ[R] S,\nhIJ : J = ideal.map ↑f I,\nr : R\n⊢ (I.quotient_equiv J ↑f hIJ).to_fun (⇑(algebra_map R I.quotient) r) = ⇑(algebra_map R J.quotient) r"

# req_id = api.new_session(filename, decl_name)
# session = api.recv()
# print("Session created")

# api.send({"req_type": "parse_goal", "name": session["name"], "pp_goal": to_parse})
# print("Parsing goal")
# print(api.recv())