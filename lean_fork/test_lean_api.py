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

from pathlib import Path
import os

from leanml import get_api


cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
api = get_api(cur_dir / "build" / "release" / "ml_server", quiet=False, fast=True, num_threads=7, dump_comms=True, preload=True, profile_tactic_process=True)


api.run_cmd('test', "cleaning_utils/v1/some_decls.lean" , timeout=10_000)
res = api.recv()
assert res['output'] == 'test1\n', res

api.run_cmd('test', "cleaning_utils/v2/some_decls.lean" , timeout=10_000)
res = api.recv()
assert res['output'] == 'test2\n', res


req_id = api.new_session(
    'others/test_split.lean',
    'ab',
    merge_alpha_equiv = True,
)
session = api.recv()
assert session["req_id"] == req_id, "wrong req_id"
# this test broken in lean 3.48
# node_ids = [0]
# # after trunk, the two branches wrongly lead to a merged node
# trunk = [
#   "have this",
#   "apply @ exists.elim",
#   "simp *",
#   # two next tactic lead to same state in lean 3.48 ?
#   # "classical",
#   # "rwa exists_eq_left'",
# ]
# branch_a = [
#   "swap 4",
#   "assumption",  # this tactic doesn't work
#   "swap",
#   "intros x y, assumption",
# ]
# branch_b = [
#   "swap",
#   "intros x y, assumption",
#   "swap",
#   "exact true",
# ]
# trunk_id = 0
# for tactic in trunk:
#     api.send_tactic(session["name"], trunk_id, tactic)
#     latest = api.recv()
#     assert 'error' not in latest
#     trunk_id += 1

# latest_branch_a = None
# for i, tactic in enumerate(branch_a):
#     api.send_tactic(session["name"], trunk_id + i, tactic)
#     latest_branch_a = api.recv()
#     assert 'error' not in latest_branch_a

# latest_branch_b, n = None, trunk_id
# for tactic in branch_b:
#     api.send_tactic(session["name"], n, tactic)
#     latest_branch_b = api.recv()
#     assert 'error' not in latest_branch_b
#     n = latest_branch_b['nodes'][0]['node_id']
# assert latest_branch_b['nodes'][0]['node_id'] != latest_branch_a['nodes'][0]['node_id'], (latest_branch_a, latest_branch_b)




api.parse_goal('⊢ ∃ m n : ℕ, m < n', session["name"])
node_ids = [n["node_id"] for n in api.recv()["nodes"]]
assert len(node_ids) == 1
api.send_tactic(session['name'], node_ids[0], 'refine ⟨_, _, _⟩')
node_ids = [n["node_id"] for n in api.recv()["nodes"]]
assert len(node_ids) == 1
api.send_tactic(session['name'], node_ids[0], 'swap')
nodes_b = [n["node_id"] for n in api.recv()["nodes"]]
assert len(nodes_b) == 1
assert node_ids[0] != nodes_b[0], (node_ids[0], nodes_b[0]) # swap should not be merged


api.parse_goal('⊢ ∀ h : true, true', session["name"])
node_ids = [n["node_id"] for n in api.recv()["nodes"]]
assert len(node_ids) == 1
api.send_tactic(session['name'], node_ids[0], 'intro h')
nodes_b = [n["node_id"] for n in api.recv()["nodes"]]
assert len(nodes_b) == 1
assert node_ids[0] != nodes_b[0], (node_ids[0], nodes_b[0]) # intro should not be merged




req_id = api.new_session(
    "lean/library/init/data/int/basic.lean",
    "int.sub_nat_nat_add_left",
    merge_alpha_equiv=True
)
session = api.recv()
assert session["req_id"] == req_id, "wrong req_id"
api.send_tactic(session["name"], 0, "have A: false")
nodes_a = api.recv()['nodes']

api.send_tactic(session["name"], 0, "have B: false")
nodes_b = api.recv()['nodes']

api.send_tactic(session["name"], 0, "have C: true")
nodes_c = api.recv()['nodes']
idA = set([x['node_id'] for x in nodes_a])
idB = set([x['node_id'] for x in nodes_b])
idC = set([x['node_id'] for x in nodes_c])
assert idA == idB != idC, (idA, idB, idC)





req_id = api.new_session(
    "lean/library/init/data/int/basic.lean",
    "int.sub_nat_nat_add_left",
    merge_alpha_equiv=False
)
session = api.recv()

assert session["req_id"] == req_id, "wrong req_id"
api.send_tactic(session["name"], 0, "have A: false")
nodes_a = api.recv()['nodes']

api.send_tactic(session["name"], 0, "have B: false")
nodes_b = api.recv()['nodes']

idA = set([x['node_id'] for x in nodes_a])
idB = set([x['node_id'] for x in nodes_b])

assert len(idA.intersection(idB)) == 0, (idA, idB)

req_id = api.new_session(
    "lean/library/init/data/int/basic.lean",
    "int.sub_nat_nat_add_left",
    merge_alpha_equiv=True
)
session = api.recv()
assert session["req_id"] == req_id, "wrong req_id"

tactics = [
  "have : ∃ a b c : ℕ, a = b ∧ b = c",
  "refine ⟨_,_,_,_,_⟩",
  "tactic.rotate 3",
  "refl",
  "refl",
  "exact 42",

  "dunfold sub_nat_nat",
  "rw [nat.sub_eq_zero_of_le]",
  "dunfold sub_nat_nat._match_1",
  "rw [nat.add_sub_cancel_left]",
  "apply nat.le_add_right"
]

node_ids = [0]
for tactic in tactics:
    api.send_tactic(session["name"], node_ids[0], tactic)
    latest = api.recv()
    if 'error' in latest: print(latest['error'])
    node_ids = [n["node_id"] for n in latest["nodes"]] + node_ids[1:]

assert node_ids == []
print("No goals left")



pp_goal = 'h : false,\nn : ℕ\n⊢ n > 5'
tactics = [
  'transitivity 2',
  'transitivity 3',
  'contradiction',
  'contradiction',
  'contradiction',
]
api.parse_goal(pp_goal, session["name"])
node_ids = [n["node_id"] for n in api.recv()["nodes"]]
for tactic in tactics:
    api.send_tactic(session["name"], node_ids[0], tactic)
    latest = api.recv()
    for n in latest['nodes']:
        if n['n_subgoals'] != 1:
            raise Exception('more than one goal after splitting:\n' + repr(n))
    node_ids = [n["node_id"] for n in latest["nodes"]] + node_ids[1:]

assert node_ids == []
print("No goals left")




pp_goal = '⊢ true'
tactics = [
  'have := list',
  'trivial',
]
api.parse_goal(pp_goal, session["name"])
node_ids = [n["node_id"] for n in api.recv()["nodes"]]
for tactic in tactics:
    api.send_tactic(session["name"], node_ids[0], tactic)
    latest = api.recv()
    node_ids = [n["node_id"] for n in latest["nodes"]] + node_ids[1:]
assert node_ids == []
print("No goals left")




# The two goals for the conjunctions should not be split,
# since they both reference the metavariable for n.
pp_goal = '⊢ ∃ n, ∃ a b : fin n, ∃ p : fin n → Prop, p a ∧ p b'
tactics = [
  'refine ⟨_, _, _, _, _, _⟩',
  '(tactic.get_goals >>= tactic.set_goals ∘ list.drop 4)',
]
api.parse_goal(pp_goal, session["name"])
node_ids = [n["node_id"] for n in api.recv()["nodes"]]
for tactic in tactics:
    api.send_tactic(session["name"], node_ids[0], tactic)
    latest = api.recv()
    if len(latest['nodes']) > 1:
        raise Exception('more than one splitted goal')
    node_ids = [n["node_id"] for n in latest["nodes"]] + node_ids[1:]




# The two goals for the suffices should not be split,
# since they both reference the same universe metavariable.
pp_goal = '⊢ false'
tactics = [
    'suffices : ∀ p, subsingleton p',
    '{ rw @subsingleton.elim _ (this _) false, trivial }',
    # The following would succeed and finish the proof if the two goals were split:
    # 'apply subsingleton_prop',
]
api.parse_goal(pp_goal, session["name"])
node_ids = [n["node_id"] for n in api.recv()["nodes"]]
for tactic in tactics:
    api.send_tactic(session["name"], node_ids[0], tactic)
    latest = api.recv()
    if len(latest['nodes']) > 1:
        raise Exception('more than one splitted goal')
    node_ids = [n["node_id"] for n in latest["nodes"]] + node_ids[1:]




pp_goal = 'm n : ℕ\n⊢ m + n = n + m'
tactics = [
  "rw nat.add_comm",
]
api.parse_goal(pp_goal, session["name"])
node_ids = [n["node_id"] for n in api.recv()["nodes"]]
for tactic in tactics:
    api.send_tactic(session["name"], node_ids[0], tactic)
    latest = api.recv()
    node_ids = [n["node_id"] for n in latest["nodes"]] + node_ids[1:]

assert node_ids == []
print("No goals left")




pp_goal = 'm n : ℕ\n⊢ m + n = n + m'
tactics = [
  "have A : false",
  "sorry",
  "rw nat.add_comm",
]
api.parse_goal(pp_goal, session["name"])
node_ids = [n["node_id"] for n in api.recv()["nodes"]]
api.send_tactic(session["name"], node_ids[0], 'have A : false')
latest = api.recv()
node_ids = [n["node_id"] for n in latest["nodes"]] + node_ids[1:]
api.send_tactic(session["name"], node_ids[0], 'sorry')
latest = api.recv()
assert 'proof contains sorry' in latest['error']



pp_goal = '''α : Type*,
_inst_1 : linear_order α,
a : α
⊢ max a a = a
'''
tactics = [
  "unfold max",
  "by_cases a ≤ a",
  "rw if_pos h",
  "rw if_neg h",
]
api.parse_goal(pp_goal, session["name"])
node_ids = [n["node_id"] for n in api.recv()["nodes"]]
for tactic in tactics:
    api.send_tactic(session["name"], node_ids[0], tactic)
    latest = api.recv()
    node_ids = [n["node_id"] for n in latest["nodes"]] + node_ids[1:]

assert node_ids == []
print("No goals left")





pp_goal = '⊢ true\n\n⊢ true'
tactics = [
  "trivial",
  "trivial",
]
api.parse_goal(pp_goal, session["name"])
node_ids = [n["node_id"] for n in api.recv()["nodes"]]
for tactic in tactics:
    api.send_tactic(session["name"], node_ids[0], tactic)
    latest = api.recv()
    node_ids = [n["node_id"] for n in latest["nodes"]] + node_ids[1:]

assert node_ids == []
print("No goals left")






req_id = api.new_session(
    'others/test_split.lean',
    'ab',
    merge_alpha_equiv = True,
)
session = api.recv()
assert session["req_id"] == req_id, "wrong req_id"
node_ids = [0]
tactics = [
  'apply exists.elim',
  'swap',
  'intros x y, assumption',
  'apply eq.rec',
  "classical,rw exists_eq_right'",
  'rotate 29',
  'exact λ a, true ∧ true',
  # the following should fail:
  # 'swap, simp',
  # 'trivial',
]
for tactic in tactics:
    api.send_tactic(session["name"], node_ids[0], tactic)
    latest = api.recv()
    node_ids = [n["node_id"] for n in latest["nodes"]] + node_ids[1:]
api.send_tactic(session["name"], node_ids[0], 'swap, simp')
latest = api.recv()
assert 'contains metavariable' in latest['error']




req_id = api.new_session(
    "lean/library/init/data/bool/lemmas.lean",
    "tt_band",
    merge_alpha_equiv=True
)
session = api.recv()
assert session["req_id"] == req_id, "wrong req_id"

pp_goal = 'b : bool\n⊢ bool.tt && b = b'
tactics = [
  "refl",
]
api.parse_goal(pp_goal, session["name"])
node_ids = [n["node_id"] for n in api.recv()["nodes"]]
for tactic in tactics:
    api.send_tactic(session["name"], node_ids[0], tactic)
    latest = api.recv()
    node_ids = [n["node_id"] for n in latest["nodes"]] + node_ids[1:]

assert node_ids == []
print("No goals left")

req_id = api.new_session(
    "lean/library/init/data/bool/lemmas.lean",
    "tt_band",
    merge_alpha_equiv=True
)
session = api.recv()
assert session["req_id"] == req_id, "wrong req_id"
tactics = [
  "sorry",
  "admit",
  "undefined",
  "exact undefined",
  "exact sorry",  
  "abstract {sorry}",
  "async {sorry}",
  "have : true ∧ true → false := λ ⟨_, _⟩, sorry",
  "let f : ℕ × ℕ → ℕ := λ ⟨_, _⟩, sorry"
]
expected = {}
for tactic in tactics:
    expected[api.send_tactic(session["name"], 0, tactic)] = tactic
for i in range(len(tactics)):
    latest = api.recv()
    assert latest["error"].startswith("kernel exception") or \
        latest['error'].startswith('proof contains ') or \
        latest['error'].startswith('Forbidden token ') or \
        latest['error'].startswith('Async constant')
print("Forbidden tokens ok")


req_id = api.new_session(
    "others/test_slow.lean",
    "p_or_q_or_r",
    merge_alpha_equiv=False  # false here because we test various strip tags, so we need to reprint
)
session = api.recv()
assert session["req_id"] == req_id, "wrong req_id"

api.parse_goal_and_apply_tactic("p q r : Prop,\nh : p ∨ q\n⊢ p ∨ q ∨ r",
  session["name"],
  "cases h with hp hq",
  strip_tags=True
)
subgoals = api.recv()
print(subgoals)
subgoals = subgoals['subgoals']
for g in subgoals:
  assert 'case or.inl' not in g['full_pp'] and 'case or.inr' not in g['full_pp']

api.parse_goal_and_apply_tactic("p q r : Prop,\nh : p ∨ q\n⊢ p ∨ q ∨ r",
  session["name"],
  "cases h with hp hq",
  strip_tags=False
)
subgoals = api.recv()['subgoals']
for g in subgoals:
  assert 'case or.inl' in g['full_pp'] or 'case or.inr' in g['full_pp']

## Check that intros is forbidden unless given names
req_id = api.new_session(
    "lean/library/init/data/bool/lemmas.lean",
    "tt_band",
    merge_alpha_equiv=True
)
session = api.recv()
assert session["req_id"] == req_id, "wrong req_id"
bad_tactics = [
  "intro", "introI", "intros",
  "sorry;intro"
]
good_tactics = [
  "intro h1", "introI some other name", "intros h2 h3",
  "sorry;intro a"
]
expected = {}
for good, tactics in zip([True, False], [good_tactics, bad_tactics]):
  for tactic in tactics:
      expected[api.send_tactic(session["name"], 0, tactic)] = tactic
  for i in range(len(tactics)):
      latest = api.recv()
      if good and "error" in latest:
        assert not latest["error"].startswith("Forbidden intro tactic ")
      if not good:
        assert latest["error"].startswith("Forbidden intro tactic ")
print("Forbidden intros ok")



req_id = api.new_session(
    "others/test_slow.lean",
    "one_eq_zero",
    merge_alpha_equiv=True
)
session = api.recv()
assert session["req_id"] == req_id, "wrong req_id"

tactics = [
  "convert of_mul_one",
  "apply_instance",
]

node_ids = [0]
found_error = False
for tactic in tactics:
    api.send_tactic(session["name"], node_ids[0], tactic)
    latest = api.recv()
    if 'error' in latest:
        assert 'failed to type-check' in latest['error']
        found_error = True
        break
    node_ids = [n["node_id"] for n in latest["nodes"]] + node_ids[1:]

assert found_error
print("Found error!")


# check repeated hyps
# in send_tactic
req_id = api.new_session(
    "others/test_slow.lean",
    "repeat_hyps",
    merge_alpha_equiv=True
)
session = api.recv()
assert session["req_id"] == req_id, "wrong req_id"

api.send_tactic(session["name"], node_ids[0], "induction h₁ : n with n", max_repeated_hyps=0)
latest = api.recv()
assert 'error' in latest and latest['error'] == "Too many repeated hypothesis names.", latest

# in parse_goal_and_apply_tactic
req_id = api.new_session(
    "others/test_slow.lean",
    "p_or_q_or_r",
    merge_alpha_equiv=True
)
session = api.recv()
assert session["req_id"] == req_id, "wrong req_id"

# in the apply
for ok, mrh in zip([True, False], [1, 0]):
  api.parse_goal_and_apply_tactic("n : ℕ\n⊢ n = 3", session["name"], "induction h₁ : n with n", max_repeated_hyps=mrh, strip_tags=True)
  latest = api.recv()
  if not ok:
    assert 'error' in latest and latest['error'] == "Too many repeated hypothesis names.", latest
  else:
    assert 'error' not in latest, latest

# in parse_goal
parse_multi_hyp = "n n : ℕ,\nh: n > 2\n⊢ n = 3"
for ok, mrh in zip([True, False], [1, 0]):
  api.parse_goal(parse_multi_hyp, session["name"], max_repeated_hyps=mrh)
  latest = api.recv()
  if not ok:
    assert 'error' in latest and latest['error'] == "Too many repeated hypothesis names.", latest
  else:
    assert 'error' not in latest, latest

# in the parse
for ok, mrh in zip([True, False], [10, 0]):
  api.parse_goal_and_apply_tactic(parse_multi_hyp, session["name"], "induction h₁ : n with n", max_repeated_hyps=mrh, strip_tags=True)
  latest = api.recv()
  if not ok:
    assert 'error' in latest and latest['error'] == "Too many repeated hypothesis names.", latest
  else:
    assert 'error' not in latest, latest


# in parse_children
for ok, mrh in zip([True, False], [1, 0]):
  api.parse_children([parse_multi_hyp], session["name"], max_repeated_hyps=mrh)
  latest = api.recv()
  if not ok:
    assert 'error' in latest and latest['error'] == "Too many repeated hypothesis names.", latest
  else:
    assert 'error' not in latest, latest


# check no split
req_id = api.new_session(
    "others/test_slow.lean",
    "p_or_q_or_r",
    merge_alpha_equiv=True
)
session = api.recv()
assert session["req_id"] == req_id, "wrong req_id"

api.send_tactic(session["name"], 0, "cases h with hp hq", nosplit=True)
assert len(api.recv()["nodes"]) == 1
api.send_tactic(session["name"], 0, "cases h with hp hq", nosplit=False)
assert len(api.recv()["nodes"]) == 2

# check no split no goals
req_id = api.new_session(
    "others/test_slow.lean",
    "assumption",
    merge_alpha_equiv=True
)
session = api.recv()
assert session["req_id"] == req_id, "wrong req_id"

api.send_tactic(session["name"], 0, "assumption", nosplit=True)
assert len(api.recv()["nodes"]) == 0
