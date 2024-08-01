import v1.utils
import v1.proof_running

section cleantactic

declare_trace cleantactic

section pexpr
namespace tactic
open interactive.types lean.parser

meta def texpr' := lean.parser.pexpr tac_rbp tt

meta def simp_arg' : lean.parser simp_arg_type :=
(tk "*" *> return simp_arg_type.all_hyps) <|>
(tk "-" *> simp_arg_type.except <$> ident) <|>
(tk "<-" *> simp_arg_type.symm_expr <$> texpr') <|>
(simp_arg_type.expr <$> texpr')

meta def simp_arg_list' : lean.parser (list simp_arg_type) :=
(tk "*" *> return [simp_arg_type.all_hyps]) <|> list_of simp_arg' <|> return []

end tactic
end pexpr

meta def print_loc : interactive.loc → string
| interactive.loc.wildcard := " at *"
| (interactive.loc.ns [none]) := ""
| (interactive.loc.ns l) := " at" ++ (
    l.foldl (λ s o,
      s ++ " " ++ match o with
      | some nm := to_string nm
      | none := "⊣"
      end
      )
      ""
    )

meta def get_variants_loc : interactive.loc → tactic (list interactive.loc)
| interactive.loc.wildcard := pure [interactive.loc.ns [], interactive.loc.wildcard]
| (interactive.loc.ns [none]) := pure [interactive.loc.ns [none]]
| (interactive.loc.ns l) := do ll ←  l.erase_dup.combinations, pure $ ll.filter_map $ λ ls, match ls with
  | [] := none
  | _ := some $ interactive.loc.ns ls
  end


meta def print_simp_arg_list : list tactic.simp_arg_type → string
| [] := ""
| [tactic.simp_arg_type.all_hyps] := " *"
| l := " " ++ l.to_string  -- with brackets and comma


meta def print_simp_arg_list_string : list string → string
| [] := ""
| ["*"] := " *"
| l := " " ++ l.to_string  -- with brackets and comma


meta def get_variants_simp_arg_list (l : list tactic.simp_arg_type) : tactic (list (list tactic.simp_arg_type)) := do
  let l' := l.erase_dup,
  l'' ← l'.combinations,
  pure l''


meta def get_variants_bool_ff_tt : bool → list bool
| ff := [ff]
| tt := [ff,tt]

meta def get_variants_bool_tt_ff : bool → list bool
| tt := [tt]
| ff := [tt,ff]


meta def print_with_ident_list : list name → string
| [] := ""
| l := " with " ++ (l.foldl (λ s nm, s ++ " " ++ to_string nm) "")


meta def get_variants_with_ident_list : list name → tactic (list (list name))
| l := l.erase_dup.combinations


meta def parser_args_norm_num : lean.parser (list (tactic.simp_arg_type) × interactive.loc) := do
  hs ← tactic.simp_arg_list',
  l ← interactive.types.location,
  pure (hs,l)


meta def tactic.get_variants_norm_num (ts: tactic_state) (action: string) : tactic (list string) :=
  let s := "norm_num" in 
  if ¬(action.to_list.take s.length = s.to_list) then
    tactic.fail format!"not a {s}"
  else do
    tactic.lazy_trace `cleantactic $ pure format!"is a `norm_num`",
    let args := list.as_string $ action.to_list.drop s.length,
    tactic.lazy_trace `cleantactic $ pure format!"args:\t{args}",
    tactic.lazy_trace `cleantactic $ pure format!"ts:\t{ts}",
    (hs, l) ← tactic.run_tac ts (run_on_input parser_args_norm_num args),
    hs_variants ← get_variants_simp_arg_list hs,
    l_variants ← get_variants_loc l,
    -- favor empty loc first
    let variants := l_variants.product hs_variants,
    variants.mmap $ λ ⟨l,hs⟩, pure $ "norm_num" ++ (print_simp_arg_list hs) ++ (print_loc l)


section simp

meta def tactic.interactive.propagate_tags_EVARISTE {α} (tac : tactic α) : tactic α :=
do tag ← tactic.get_main_tag,
   if tag = [] then tac
   else tactic.focus1 $ do
     res ← tac,
     gs ← tactic.get_goals,
     when (bnot gs.empty) $ (do
       new_tag ← tactic.get_main_tag,
       when new_tag.empty $ tactic.with_enable_tags (tactic.set_main_tag tag)),
      pure res


meta def tactic.get_simp_only_hs (use_iota_eqn : interactive.parse $ optional (lean.parser.tk "!")) (no_dflt : interactive.parse interactive.types.only_flag) (hs : interactive.parse tactic.simp_arg_list) (attr_names : interactive.parse interactive.types.with_ident_list)
              (locat : interactive.parse interactive.types.location) (cfg : tactic.simp_config_ext := {}) : tactic (list string) :=
let cfg := match use_iota_eqn with
| none     := cfg
| (some _) := {iota_eqn := tt, ..cfg}
end in
tactic.interactive.propagate_tags_EVARISTE $
do lms ← tactic.interactive.simp_core cfg.to_simp_config cfg.discharger no_dflt hs attr_names locat,
  pure $ lms.to_list.map $ to_string

end simp

meta def parser_args_simp  := do
  use_iota_eqn ← optional (lean.parser.tk "!") >>= λ s, match s with | none := pure ff | _ := pure tt end,
  trace_lemmas ← optional (lean.parser.tk "?") >>= λ s, match s with | none := pure ff | _ := pure tt end,
  no_dflt ← interactive.types.only_flag,
  hs ← tactic.simp_arg_list',
  attr_names ← interactive.types.with_ident_list,
  locat ← interactive.types.location,
  pure (use_iota_eqn, trace_lemmas, no_dflt, hs, attr_names, locat)


meta def tactic.get_variants_simp (ts: tactic_state) (action: string) (simp_only : bool) : tactic (list string) :=
  let s := "simp" in 
  if ¬(action.to_list.take s.length = s.to_list) then
    tactic.fail format!"not a {s}"
  else do
    tactic.lazy_trace `cleantactic $ pure format!"is a `simp`",
    let args := list.as_string $ action.to_list.drop s.length,
    tactic.lazy_trace `cleantactic $ pure format!"args:\t{args}",
    tactic.lazy_trace `cleantactic $ pure format!"ts:\t{ts}",
    (use_iota_eqn, trace_lemmas, no_dflt, hs, attr_names, locat) ← tactic.run_tac ts (run_on_input parser_args_simp args),
    let use_iota_eqn_variants := get_variants_bool_ff_tt use_iota_eqn,
    let no_dflt_variants := get_variants_bool_tt_ff no_dflt,
    attr_names_variants ← get_variants_with_ident_list attr_names,
    simp_only_hs ← if simp_only then tactic.run_tac ts $ tactic.get_simp_only_hs (if use_iota_eqn then some () else none) no_dflt hs attr_names locat else pure [],
    hs_variants_no_simp_only ← get_variants_simp_arg_list hs >>= λ ll, ll.mmap $ λ l, l.mmap $ λ h, pure $ to_string h,
    let hs_variants :=
      if simp_only_hs.empty then
        hs_variants_no_simp_only
      -- include hyps, which do not appear in simp_only_hs
      else
        let simp_only_hs_with_all := hs_variants_no_simp_only.map (λ ls, list.erase_dup $ simp_only_hs ++ ls) in
        simp_only_hs_with_all ++ hs_variants_no_simp_only,
    locat_variants ← get_variants_loc locat,
    -- favor empty loc first
    let variants := use_iota_eqn_variants.product (no_dflt_variants.product (attr_names_variants.product (locat_variants.product hs_variants))),
    variants.mmap $ λ ⟨use_iota_eqn,no_dflt_variants,attr_names,locat,hs⟩, pure $
      "simp" ++
      (if use_iota_eqn then "!" else "") ++
      (if no_dflt_variants then " only" else "") ++
      (print_simp_arg_list_string hs) ++
      (print_with_ident_list attr_names) ++
      (print_loc locat)


meta def parser_args_field_simp  := do
  no_dflt ← interactive.types.only_flag,
  hs ← tactic.simp_arg_list',
  attr_names ← interactive.types.with_ident_list,
  locat ← interactive.types.location,
  pure (no_dflt, hs, attr_names, locat)


meta def tactic.get_variants_field_simp (ts: tactic_state) (action: string) : tactic (list string) :=
  let s := "field_simp" in 
  if ¬(action.to_list.take s.length = s.to_list) then
    tactic.fail format!"not a {s}"
  else do
    tactic.lazy_trace `cleantactic $ pure format!"is a `field_simp`",
    let args := list.as_string $ action.to_list.drop s.length,
    tactic.lazy_trace `cleantactic $ pure format!"args:\t{args}",
    tactic.lazy_trace `cleantactic $ pure format!"ts:\t{ts}",
    (no_dflt, hs, attr_names, locat) ← tactic.run_tac ts (run_on_input parser_args_field_simp args),
    let no_dflt_variants := get_variants_bool_tt_ff no_dflt,
    attr_names_variants ← get_variants_with_ident_list attr_names,
    hs_variants ← get_variants_simp_arg_list hs,
    locat_variants ← get_variants_loc locat,
    -- favor empty loc first
    let variants := no_dflt_variants.product (attr_names_variants.product (locat_variants.product hs_variants)),
    variants.mmap $ λ ⟨no_dflt_variants,attr_names,locat,hs⟩, pure $
      "field_simp" ++
      (if no_dflt_variants then " only" else "") ++
      (print_simp_arg_list hs) ++
      (print_with_ident_list attr_names) ++
      (print_loc locat)


meta def tactic.get_variants_tidy (ts: tactic_state) (action: string) : tactic (list string) :=
  let s := "tidy" in 
  if ¬(action.to_list.take s.length = s.to_list) then
    tactic.fail format!"not a {s}"
  else do
    r ← tactic.run_tac ts tactic.tidy.core,
    pure [r.to_string_aux tt, action]


meta def tactic.get_tactic_variants (ts: tactic_state) (action: string) (simp_only : bool): tactic (list string) :=
  tactic.get_variants_norm_num ts action <|>
  tactic.get_variants_simp ts action simp_only <|>
  tactic.get_variants_field_simp ts action <|>
  tactic.get_variants_tidy ts action <|>
  pure []


/--
Clean tactics individually
Other possible cleaning:
- decompose `simp` in more granular steps when it does not finish the job
- get rid of `revert`
- rename canonically hypotheses `h₀`, `h₁`, etc
-/
meta def tactic.clean_tactic_at_idx
(simp_only : bool)
(ts_base : tactic_state)
(tgt : expr)
(steps : list BwdStep)
(resume_idx : ℕ)
: tactic (list BwdStep) :=
do {
  let actions := steps.map BwdStep.action,
  tactic.lazy_trace `cleantactic $ pure format!"actions:\t{actions}",
  tactic.lazy_trace `cleantactic $ pure format!"resume_idx:\t{resume_idx}",
  focus_action ← actions.nth resume_idx,
  ts_start ← BwdStep.ts_start <$> steps.nth resume_idx,
  tactic.lazy_trace `cleantactic $ pure format!"focus_action:\t{focus_action}",
  variants: list string ← tactic.get_tactic_variants ts_start focus_action simp_only,
  tactic.lazy_trace `cleantactic $ pure format!"variants:\t{variants}",
  actual_steps ← (variants ++ [focus_action]).mfirst (λ ac, do
    let actual_actions := actions.take resume_idx ++ [ac] ++ actions.drop (resume_idx + 1),
    run_actions ts_base tgt actual_actions
  ),
  pure actual_steps
} <|> pure steps


meta def tactic.clean_all_tactics
(simp_only : bool)
(ts_base : tactic_state)
(tgt : expr)
: list BwdStep → ℕ → tactic (list BwdStep) := λ steps resume_idx,
do {
  let resume_idx := if resume_idx > 0 then resume_idx - 1 else (steps.length - 1),
  actual_steps ← tactic.clean_tactic_at_idx simp_only ts_base tgt steps resume_idx,
  if resume_idx = 0 then
    pure actual_steps
  else
    tactic.clean_all_tactics actual_steps resume_idx
} <|> pure steps

end cleantactic


meta def clean_individual_tactic_at_idx (simp_only : bool) (idx : ℕ) (tgt : expr) (steps : list BwdStep) : tactic (list BwdStep) := do
  ts_base ← tactic.read,
  steps ← tactic.clean_tactic_at_idx simp_only ts_base tgt steps idx,
  tactic.write ts_base,
  pure $ steps


meta def clean_all_individual_tactics (simp_only : bool) (tgt : expr) (steps : list BwdStep) : tactic (list BwdStep) := do
  ts_base ← tactic.read,
  steps ← tactic.clean_all_tactics simp_only ts_base tgt steps 0,
  tactic.write ts_base,
  pure steps
