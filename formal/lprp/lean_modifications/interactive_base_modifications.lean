/- This is a staging area for code which will be inserted
into a Lean file. 

The code to be inserted is between the line comments
`PR BEGIN MODIFICATION` and `PR END MODIFICATION`

It will be inserted by `insert_proof_recording_code.py`.

Insert info:
  - file: `_target/lean/library/init/meta/interactive_base.lean`
  - location: end of file

Most of this code is carefully written, but
any code labeled "BEGIN/END CUSTOMIZABLE CODE"
encourages customization to change what
is being recorded.
-/

prelude
import init.data.option.basic
import init.meta.lean.parser init.meta.tactic init.meta.has_reflect
import .tactic_modifications


--PR BEGIN MODIFICATION

namespace interactive.pr

meta def trace_parser_state (arg_num : nat) (parser_text : string) (is_after: bool) : lean.parser unit := do
  -- TODO: Flesh out
  pos <- lean.parser.cur_pos,
  tactic.trace pos

/-- A wrapper around the parser which records information about 
the parsed value. -/
meta def record {α : Type} /-(tactic_name : string) (arg_num : nat) (parser_text : string) -/
(p : lean.parser (reflected_value α)) : lean.parser (reflected_value α) := do
o <- tactic.get_options,
if bnot (o.get_bool `pp.colors tt) then do
  -- record position information before parsing
  pos <- lean.parser.cur_pos,
  let line := pos.line,
  let column := pos.column + 1,
  let key := (to_string line) ++ ":" ++ (to_string column),
  pr.trace_data_num "tactic_param_pos" key "line" line,
  pr.trace_data_num "tactic_param_pos" key "column" column,

  -- record metadata about the parser
  --pr.trace_data_string "tactic_param" key "tactic_name" tactic_name,
  --pr.trace_data_num "tactic_param" key "ix" arg_num,
  --pr.trace_data_string "tactic_param" key "parser_code" parser_text,

  -- parse
  a <- p,
  
  -- record position information after parsing
  pos <- lean.parser.cur_pos,
  let line := pos.line,
  let column := pos.column + 1,
  pr.trace_data_num "tactic_param_pos" key "end_line" line,
  pr.trace_data_num "tactic_param_pos" key "end_column" column,

  -- record information about the paramater value
  -- this section can be customized as needed.
  fmt <- tactic.pp $ reflected_value.expr a,
  pr.trace_data_string "tactic_param_value" key "reflected_expr_pp" (to_string $ fmt),

  return a
else p

/-- A wrapper around parsers in interactive tactic parameters. -/
meta def recorded {α : Type} /-(tactic_name: string) (arg_num : nat) (parser_text : string)-/ (p : lean.parser α) [r : lean.parser.reflectable p] : lean.parser α :=
lean.parser.val $ record /-tactic_name arg_num parser_text-/ r.full

end interactive.pr

--PR END MODIFICATION


set_option pp.colors false

meta def tactic.interactive.my_try (t : interactive.parse $ interactive.pr.recorded_itactic "my_try" 0) : tactic unit := tactic.try t
meta def tactic.interactive.my_skip (q : interactive.parse $ interactive.pr.recorded "my_skip" 0 "foo" $ optional (lean.parser.tk "%")) : tactic unit := tactic.skip
meta def tactic.interactive.my_apply (q : interactive.parse $ interactive.pr.recorded "my_apply" 0 "texpr" $ interactive.types.texpr) : tactic unit := 
tactic.interactive.concat_tags (do h ← tactic.i_to_expr_for_apply q, tactic.apply h)
meta def rw_core (rs : interactive.parse (interactive.pr.recorded "asdfas" 0 "rw_rules" (tactic.interactive.rw_rules)) ) (l : interactive.parse (interactive.pr.recorded "asdf" 1 "location" (interactive.types.location)) ) (cfg : tactic.rewrite_cfg := {}) : tactic unit := 
return ()

meta def tactic.interactive.my_rewrite (q : interactive.parse (interactive.pr.recorded "rw_core" 0 "rw_rules" (tactic.interactive.rw_rules)) ) (l : interactive.parse (interactive.pr.recorded "rewrite" 1 "location" (interactive.types.location)) ) (cfg : tactic.rewrite_cfg := {}): tactic unit :=
tactic.interactive.propagate_tags (rw_core q l cfg)

#check tactic.interactive.propagate_tags (rw_core (_ : interactive.parse (interactive.pr.recorded "asdfas" 1 "rw_rules" tactic.interactive.rw_rules)) (_ : interactive.parse (interactive.pr.recorded "rewrite" 1 "location" (interactive.types.location)) ) {})

. 
example : true := begin
    my_skip; my_try { my_try {skip}, my_apply true.intro, skip,}
end