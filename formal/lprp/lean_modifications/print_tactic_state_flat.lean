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

--PR BEGIN MODIFICATION
private meta def set_bool_option (n : name) (v : bool) : tactic unit :=
do s ← tactic.read,
   tactic.write $ tactic_state.set_options s (options.set_bool (tactic_state.get_options s) n v)

private meta def enable_full_names : tactic unit := do {
  set_bool_option `pp.full_names true
}

private meta def with_full_names {α} (tac : tactic α) : tactic α :=
tactic.save_options $ enable_full_names *> tac

private meta def enable_full_pp : tactic unit := do {
  -- set_bool_option `pp.all  true
  set_bool_option `pp.implicit true,
  set_bool_option `pp.universes false,
  set_bool_option `pp.notation true,
  set_bool_option `pp.numerals true
}

private meta def with_full_pp {α} (tac : tactic α) : tactic α :=
tactic.save_options $ enable_full_pp *> tac


meta def tactic_state.to_flattened_string (ts : tactic_state) : tactic string := do {
  ts₀ ← tactic.read,
  tactic.write ts,
  -- result ← (tactic.read >>= λ ts, pure ts.to_format.to_string),
  result ← with_full_names $ (tactic.read >>= λ ts, pure ts.to_format.to_string),
  tactic.write ts₀,
  pure result
}

local notation `PRINT_TACTIC_STATE` := tactic_state.to_flattened_string
--PR END MODIFICATION
