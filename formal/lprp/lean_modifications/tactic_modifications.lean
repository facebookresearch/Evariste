/- This is a staging area for code which will be inserted
into a Lean file. 

The code to be inserted is between the line comments
`PR BEGIN MODIFICATION` and `PR END MODIFICATION`

It will be inserted by `insert_proof_recording_code.py`.

Insert info:
  - file: `_target/lean/library/init/meta/tactic.lean`
  - location: end of file

Most of this code is carefully written, but
any code labeled "BEGIN/END CUSTOMIZABLE CODE"
encourages customization to change what
is being recorded
-/

prelude
import init.meta.tactic

universes u v


--PR BEGIN MODIFICATION

-- proof recording code
namespace pr  

/-- Trace data as JSON preceeded by a `<PR>` flag.
This makes it easy to filter out all the traced data. -/
meta def trace_data (table : string) (key : string) (field : string) (data : string) : tactic unit :=
  tactic.trace $ "<PR> {" 
    ++ "\"table\": " ++ (repr table) ++ ", "
    ++ "\"key\": " ++ (repr key) ++ ", "
    ++ "\"" ++ field ++ "\": " ++ data
    ++ "}"

meta def trace_data_string (table : string) (key : string) (field : string) (str_data : string) : tactic unit :=
  trace_data table key field (repr str_data)

meta def trace_data_num (table : string) (key : string) (field : string) (num_data : nat) : tactic unit :=
  trace_data table key field (repr num_data)

meta def trace_data_bool (table : string) (key : string) (field : string) (bool_data : bool) : tactic unit :=
  trace_data table key field (if bool_data then "true" else "false")

-- tactic recording

/-- Each tactic application within a given file can be keyed by four
numbers.  Combinators allow a tactic to be called more than once, and
some nested tactics use the same line and column position, so
we also include depth to capture nesting and index to capture executed
order.  (A proof can be uniquely keyed by its first tactic 
at depth 1, index 1.) -/
structure tactic_address :=
  -- the line and column of the tactic instance
  -- using 1-indexing like editors do even though lean uses a mix of 0 and 1-indexing
  (line : nat) (column : nat)
  -- depth of tactic block (tactic inside a tactic)
  (depth : nat) 
  -- index indicating the order of execution
  (index : nat)

meta def addr_key (addr : tactic_address) : string :=
  (repr addr.line) 
    ++ ":" ++ (repr addr.column)
    ++ ":" ++ (repr addr.depth)
    ++ ":" ++ (repr addr.index)

meta def trace_tactic_data_string (addr : tactic_address) (field : string) (str_data : string) : tactic unit :=
  trace_data_string "tactic_instances" (addr_key addr) field str_data

meta def trace_tactic_data_num (addr : tactic_address) (field : string) (num_data : nat) : tactic unit :=
  trace_data_num "tactic_instances" (addr_key addr) field num_data

meta def trace_tactic_data_bool (addr : tactic_address) (field : string) (bool_data : bool) : tactic unit :=
  trace_data_bool "tactic_instances" (addr_key addr) field bool_data

meta def trace_tactic_data_addr (addr : tactic_address) (field : string) (addr_data : tactic_address) : tactic unit :=
  trace_data_string "tactic_instances" (addr_key addr) field (addr_key addr_data)

meta def get_tactic_address (o : options) (nm : name) : tactic_address := 
  { tactic_address .
    line   := o.get_nat (name.mk_string "line" nm) 0,
    column := o.get_nat (name.mk_string "column" nm) 0,
    depth  := o.get_nat (name.mk_string "depth" nm) 0,
    index  := o.get_nat (name.mk_string "index" nm) 0,
  }

meta def set_tactic_address (o : options) (nm : name) (addr : tactic_address) : options :=
  let o := o.set_nat (name.mk_string "line" nm) addr.line in
  let o := o.set_nat (name.mk_string "column" nm) addr.column in
  let o := o.set_nat (name.mk_string "depth" nm) addr.depth in
  let o := o.set_nat (name.mk_string "index" nm) addr.index in
  o

def is_same (a b : tactic_address) : bool := (
  a.line = b.line ∧
  a.column = b.column ∧
  a.depth = b.depth ∧
  a.index = b.index
)

-- set the tactic state
meta def set_state (new_state: tactic_state): tactic unit :=
  -- this is in mathlib but easier to recreate
  λ _, interaction_monad.result.success () new_state

meta def trace_tactic_state_data (addr: tactic_address) (finished : bool) := do
-- BEGIN CUSTOMIZABLE CODE
-- customize as needed to record different
-- parts of the tactic state in different formats

-- position data
let t_key := (addr_key addr),
let temporal := if finished then "after" else "before",
let ts_key := t_key ++ ":" ++ temporal,
trace_data_string "tactic_state" ts_key "tactic_instance" t_key,
trace_data_string "tactic_state" ts_key "before_after" temporal,

-- environment (just store fingerprints)
env <- tactic.get_env,
-- store fingerprint as a string to prevent large values errors
trace_data_string "tactic_state" ts_key "env_fingerprint" (repr env.fingerprint),

-- declaration
decl <- tactic.decl_name,
trace_data_string "tactic_state" ts_key "decl_name" (to_string decl),

-- open namespaces
open_nmspaces <- tactic.open_namespaces,
trace_data_string "tactic_state" ts_key "open_namespaces" (string.intercalate " " (open_nmspaces.map repr)),

-- goal stack information
goals <- tactic.get_goals,
trace_data_num "tactic_state" ts_key "goal_count" goals.length,

  let dump_all_goals : tactic unit := do {
  -- dump entire goal stack as a single goal
  (tactic_state_strings : list string) ← goals.mmap $ λ g, do {
    saved_ts ← tactic.read,
    tactic.set_goals [g],
    ts ← tactic.read,
    result ← PRINT_TACTIC_STATE ts,
    set_state saved_ts,
    pure result
  },

  let g_key := ts_key ++ ":0",
  trace_data_string "tactic_state_goal" g_key "tactic_state" ts_key,
  trace_data_num "tactic_state_goal" g_key "ix" 0,
  trace_data_num "tactic_state_goal" g_key "goal_hash" ((goals.map expr.hash).foldr (+) 0),
  trace_data_string "tactic_state_goal" g_key "goal_pp" ("\n\n".intercalate tactic_state_strings)
},

dump_all_goals,

-- -- individual goal information
-- goals.enum.mmap' $ λ ⟨n, g⟩, (do
--   let g_key := ts_key ++ ":" ++ (repr n),
--   trace_data_string "tactic_state_goal" g_key "tactic_state" ts_key,
--   trace_data_num "tactic_state_goal" g_key "ix" n,
--   -- store hash of goal metavariable to know if goal changed
--   trace_data_num "tactic_state_goal" g_key "goal_hash" g.hash,
--   -- pretty print the goal by temporarily making it the only goal
--   saved_ts <- tactic.read, -- copy tactic state
--   tactic.set_goals [g],
--   ts <- tactic.read, -- temporary tactic state
--   (printed_tactic_state : string) ← PRINT_TACTIC_STATE ts,
--   trace_data_string "tactic_state_goal" g_key "goal_pp" printed_tactic_state,
--   set_state saved_ts -- put tactic state back to way it was
-- ),

return ()
-- END CUSTOMIZABLE CODE

meta def store_info_in_tactic_state (finished : bool) (line col : ℕ) : tactic unit := do
let column := col + 1, -- use 1-indexed columns for convience

-- get stored proof trace information
o <- tactic.get_options,
-- pop from the top of the stack
let depth := o.get_nat `proof_rec.depth 0,
let prev_open_addr := get_tactic_address o (mk_num_name `proof_rec.open depth),
let block_addr := get_tactic_address o (mk_num_name `proof_rec.block (depth+1)),
-- find the previous tactic
let prev_addr := get_tactic_address o `proof_rec.prev,
-- find the start of the proof
let proof_addr := get_tactic_address o `proof_rec.proof,

-- there are three cases.  Handle each seperately
match (finished, is_same prev_addr prev_open_addr) with
| ⟨ ff, tt ⟩ := do
  -- starting a new tactic block

  -- calculate address
  let new_depth := depth + 1,
  let new_addr := { tactic_address .
    line   := line,
    column := column,
    depth  := new_depth,
    index  := 1,
  },
  let new_block_addr := new_addr,
  let new_proof_addr := if new_depth = 1 then new_addr else proof_addr,

  -- trace data to stdout
  trace_tactic_data_bool new_addr "executed" tt,
  trace_tactic_data_num new_addr "line" new_addr.line,
  trace_tactic_data_num new_addr "column" new_addr.column,
  trace_tactic_data_num new_addr "depth" new_addr.depth,
  trace_tactic_data_num new_addr "index" new_addr.index,
  trace_tactic_data_addr new_addr "proof" new_proof_addr,
  trace_tactic_data_addr new_addr "block" new_block_addr,
  trace_tactic_data_addr new_addr "parent" prev_open_addr, -- will be ⟨0,0,0,0⟩ if no parent
  trace_tactic_data_addr new_addr "prev" prev_addr,  -- previous completed tactic (not same depth)
  
  -- trace data about the state beforehand
  trace_tactic_state_data new_addr ff,

  -- update proof trace information
  o <- tactic.get_options,
  let o := o.set_nat `proof_rec.depth new_depth,
  let o := set_tactic_address o (mk_num_name `proof_rec.open new_depth) new_addr,
  let o := set_tactic_address o (mk_num_name `proof_rec.block new_depth) new_block_addr,
  let o := set_tactic_address o `proof_rec.proof new_proof_addr,
  let o := set_tactic_address o `proof_rec.prev new_addr,
  tactic.set_options o,

  return ()
| ⟨ ff, ff ⟩ := do
  -- starting new tactic at same depth as previous tactic
  
  -- calculate address
  assert $ (prev_addr.depth = depth + 1),
  assert $ (block_addr.depth = depth + 1),
  let new_depth := prev_addr.depth,
  let new_addr := { tactic_address .
    line   := line,
    column := column,
    depth  := new_depth,
    index  := prev_addr.index + 1,
  },

  -- trace data to stdout
  trace_tactic_data_bool new_addr "executed" tt,
  trace_tactic_data_num new_addr "line" new_addr.line,
  trace_tactic_data_num new_addr "column" new_addr.column,
  trace_tactic_data_num new_addr "depth" new_addr.depth,
  trace_tactic_data_num new_addr "index" new_addr.index,
  trace_tactic_data_addr new_addr "proof" proof_addr,
  trace_tactic_data_addr new_addr "block" block_addr,
  trace_tactic_data_addr new_addr "parent" prev_open_addr, -- will be ⟨0,0,0,0⟩ if no parent
  trace_tactic_data_addr new_addr "prev" prev_addr,

  -- trace data about the state beforehand
  trace_tactic_state_data new_addr ff,

  -- update proof trace information (only information which changes)
  o <- tactic.get_options,
  let o := o.set_nat `proof_rec.depth new_depth,
  let o := set_tactic_address o (mk_num_name `proof_rec.open new_depth) new_addr,
  let o := set_tactic_address o `proof_rec.prev new_addr,
  tactic.set_options o,

  return ()
| ⟨ tt, _ ⟩ := do
  -- tactic completed successfully

  -- calculate address
  assert $ (line = prev_open_addr.line) ∧ (column = prev_open_addr.column) ∧ (depth = prev_open_addr.depth),
  let new_addr := prev_open_addr,
  
  -- trace data to stdout
  trace_tactic_data_bool prev_open_addr "succeeded" tt,
  
  -- trace data about the state afterward
  trace_tactic_state_data new_addr tt,

  -- update proof trace information (only information which changes)
  o <- tactic.get_options,
  let o := o.set_nat `proof_rec.depth (depth - 1),
  let o := set_tactic_address o `proof_rec.prev new_addr,
  tactic.set_options o,

  return ()
end

meta def step_and_record {α : Type u} (line col : ℕ) (t : tactic α) : tactic unit := do
-- only record if the pp.colors flag is set to false
-- we can't make our own system option, so re-using
-- one built in.  (Even thought we are setting it to
-- the default, we can still check that it is set.)

o <- tactic.get_options,
if bnot (o.get_bool `pp.colors tt) then do
  store_info_in_tactic_state ff line col, -- before
  tactic.step t,
  store_info_in_tactic_state tt line col  -- after
else tactic.step t

end pr

-- redefined istep to do proof recording
meta def tactic.istep {α : Type u} (line0 col0 : ℕ) (line col : ℕ) (t : tactic α) : tactic unit :=
λ s, (@scope_trace _ line col (λ _, pr.step_and_record line col t s)).clamp_pos line0 line col
--PR END MODIFICATION
