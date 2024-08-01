/-
Lots of utilities copy-pasted from OpenAI's lean-gym + other useful stuff
Unlike lean-gym, it makes use of the `tactic` monad instead of the `io` one
-/

import all

universe u

section string
namespace string

meta def strip (s : string) : string :=
  let sl := s.to_list in
  let sl := sl.drop_while (λ c, c = ' ') in
  let sl := list.reverse $ sl.reverse.drop_while (λ c, c = ' ') in
  sl.as_string


meta def startswith (s : string) (s' : string) : bool := 
  let sl := s.to_list in
  let sl' := s'.to_list in
  sl'.is_prefix_of sl

meta def strip_curly_braces (s : string) : string :=
  let s := s.strip in
  let sl := s.to_list in
  let sl := match sl with | '{'::tl := tl | tl := tl end in
  let slr := sl.reverse in
  let slr := match slr with | '}'::tl := tl | tl := tl end in
  slr.reverse.as_string.strip

end string
end string

section list
namespace list

-- is a `tactic` and not a pure kernel-proof function because of some weird bugs in the Python API
meta def pick_n {α : Type} : list α → ℕ → tactic (list (list α))
| _ 0 := pure [[]]
| l (n+1) := do
  let l₁ := l.take (l.length - n),
  let l₂ := l₁.enum,
  ll ← l₂.mmap  $ λ ⟨i,a⟩, (do
    let l₃ := l.drop (i+1),
    l₄ ← l₃.pick_n n,
    l₄.mmap $ λ l', pure $ a::l'
  ),
  pure ll.join


-- is a `tactic` and not a pure kernel-proof function because of some weird bugs in the Python API
meta def combinations_aux {α : Type} (l : list α) : ℕ → tactic (list (list α))
| 0 := pure []
| (n + 1) := do ll₁ ← combinations_aux n, ll₂ ← l.pick_n n, pure $ ll₁ ++ ll₂


meta def combinations {α : Type}: list α → tactic (list (list α)) := λ l, l.combinations_aux (l.length + 1)

end list
end list

section json

section has_to_json

meta class has_to_json (α : Type u) : Type (u+1) :=
(to_json : α → json)

meta class has_to_tactic_json (α : Type u) : Type (u+1) :=
(to_tactic_json : α → tactic json)

meta class has_from_json (α : Type u) : Type (u+1) :=
(from_json : json → tactic α)

end has_to_json

meta def list.lookup_prod {α β} : (list (α × β)) → (α → bool) → option β
| [] _ := none
| (⟨a,b⟩::xs) p := if p a then pure b else xs.lookup_prod p

meta def json.get_string : json → option string
| (json.of_string str) := pure str
| _ := none

meta def json.get_float : json → option native.float
| (json.of_float str) := pure str
| _ := none

meta def json.lookup : json → string → option json
| (json.object kvs) str := kvs.lookup_prod $ λ k, k = str
| _ _ := none

end json

section formatting

notation `to_tactic_json` := has_to_tactic_json.to_tactic_json

/- TODO(): optionally use `with_verbose` -/
meta instance : has_to_tactic_json expr :=
⟨λ e, (json.of_string ∘ format.to_string ∘ format.flatten) <$> tactic.pp e⟩

meta instance has_to_tactic_json_of_has_coe_json {α} [has_coe α json] : has_to_tactic_json α :=
⟨λ x, pure ↑x⟩

meta instance has_to_tactic_json_list {α} [has_to_tactic_json α] : has_to_tactic_json (list α) :=
let fn : list α → tactic json := λ xs, json.array <$> (xs.mmap to_tactic_json) in
⟨fn⟩

meta instance has_to_tactic_json_prod {α β} [has_to_tactic_json α] [has_to_tactic_json β] : has_to_tactic_json (α × β) :=
let fn : α × β → tactic json := λ ⟨a,b⟩,
json.array <$> ((::) <$> to_tactic_json a <*> pure <$> to_tactic_json b)
in ⟨fn⟩

meta def name.to_json' : name → json := λ nm, json.of_string nm.to_string

meta instance has_to_tactic_json_name : has_to_tactic_json name :=
let fn : name → tactic json := λ nm, pure nm.to_json' in
⟨fn⟩

meta instance has_to_tactic_json_coord : has_to_tactic_json expr.coord :=
let fn : expr.coord → tactic json := λ c, pure $ json.of_string c.repr in
⟨fn⟩

meta instance has_to_tactic_json_addr : has_to_tactic_json expr.address :=
let fn : expr.address → tactic json := λ xs, json.array <$> (xs.mmap to_tactic_json) in
⟨fn⟩

meta instance has_to_tactic_json_option {α} [has_to_tactic_json α] : has_to_tactic_json (option α) :=
let fn : option α → tactic json := λ x,
  match x with
  | (some val) := to_tactic_json val
  | none := pure $ json.null
  end in ⟨fn⟩


meta def fold {α} : list (α × bool) → list α
| [] := []
| ((a,b) :: tl) := if b then a :: fold tl else fold tl

meta def zip {α} {β}: list α → list β → list (α × β)
| (a :: tla) (b :: tlb) := (a,b) :: zip tla tlb
| _ _:= []

end formatting

namespace expr

meta def app_symbol_is (e : expr) (nm : name) : bool :=
match e.get_app_fn with
| (expr.const n _) := n = nm
| _ := ff
end

meta def contains_undefined (e : expr) : bool :=
e.fold ff $ λ e' _ b, if e'.app_symbol_is `undefined then tt else b

end expr

namespace tactic

open interaction_monad interaction_monad.result

/- capture but backtrack the state -/
meta def capture' {α} (t : tactic α) : tactic (tactic_result α) :=
λ s, match t s with
| (success r s') := success (success r s') s
| (exception f p s') := success (exception f p s') s
end

/- API: (success, exc_msg, result) -/
meta def unwrap' {α} [inhabited α](t : tactic_result α) : tactic (bool × string × α) :=
match t with
| (success r s') := pure (tt,"",r)
| (exception f p s') := 
    let msg := match f with
    | some x := to_string (x ())
    | _ := "<no_msg>"
    end in pure (ff, msg, default α)
  end


meta def set_goal_to (goal : expr) : tactic unit :=
mk_meta_var goal >>= set_goals ∘ pure

meta def guard_sorry (e : expr) : tactic unit := guard $ bnot e.contains_sorry

meta def guard_undefined (e : expr) : tactic unit := guard $ bnot e.contains_undefined

end tactic


section validate

meta def kernel_type_check (pf : expr) : tactic unit := do {
  tp ← tactic.infer_type pf,
  env ← tactic.get_env,
  let decl := (declaration.defn `_ (expr.collect_univ_params pf) tp pf reducibility_hints.opaque ff),
  res ← tactic.capture' (env.add decl $> ()),
  match res with
  | (interaction_monad.result.success _ _) := pure ()
  | (interaction_monad.result.exception msg _ _) := let msg := msg.get_or_else (λ _, ("" : format)) in
    tactic.fail format! "kernel type check failed:\n---\n{msg ()}\n---\n"
  end
}

meta def validate_proof (tgt: expr) (pf: expr) : tactic unit := do {
    env ← tactic.get_env,
    pf ← pure $ env.unfold_untrusted_macros pf,
    pft ← tactic.infer_type pf,
    tactic.type_check pf tactic.transparency.all,
    guard (bnot pf.has_meta_var) <|> do {
      tactic.fail format! "proof contains metavariables"
    },
    tactic.guard_sorry pf <|> do {
      tactic.fail format! "proof contains `sorry`"
    },
    tactic.guard_undefined pf <|> do {
      tactic.fail format! "proof contains `undefined`"
    },
    tactic.is_def_eq tgt pft <|> do {
      tgt_fmt ← tactic.pp tgt,
      pft_fmt ← tactic.pp pft,
      tactic.fail format! "proof type mismatch: {tgt_fmt} != {pft_fmt}"
    },
    kernel_type_check pf
}

meta def validate_decl (nm : name) : tactic unit := do {
  env ← tactic.get_env,
  d ← env.get nm,
  validate_proof d.type d.value
}

end validate

section tactic
open interaction_monad.result
namespace tactic

meta def run_tac' {α} (tac :tactic α) : tactic α := do {
  result ← tactic.capture tac,
  match result with
  | (success val _) := pure val
  | (exception m_fmt _ _) := do {
    let fmt_msg := (m_fmt.get_or_else (λ _, format!"n/a")) (),
    let msg := format!"[fatal] {fmt_msg}",
    tactic.fail msg
  }
  end
}

meta def run_tac {α : Type} (ts : tactic_state) (tac : tactic α) : tactic α :=
  run_tac' (do tactic.write ts, tac)


meta def gather_and_apply {α : Type} (some_l : list expr) (n: ℕ) (tac: tactic α) : tactic α :=
  let (head, tail) := some_l.split_at n in
  if list.length head < n then
    do fail "Not enough goals left"
  else
    do
      set_goals head,
      a ← tac,
      gs ← get_goals,
      set_goals (gs ++ tail),
      pure a


meta def focus_for_top_k_goals {α : Type} (top_k : ℕ) (tac : tactic α) : tactic α :=
  do
    gs ← get_goals,
    gather_and_apply gs top_k tac


meta def focus_for_goals_upper_than_k {α : Type} (stack_stop : ℕ) (tac : tactic α) : tactic α :=
  do
    gs ← get_goals,
    if stack_stop ≥ gs.length then
      fail format!"expected substack larger than actual goal stack ({stack_stop} ≥ {gs.length})"
    else
      gather_and_apply gs (gs.length - stack_stop) tac

end tactic
end tactic

section run_with_state'

namespace interaction_monad
open interaction_monad.result
meta def run_with_state' {σ₁ σ₂ : Type} {α : Type*} (state : σ₁) (tac : interaction_monad σ₁ α) : interaction_monad σ₂ α :=
λ s, match (tac state) with
     | (success val _) := success val s
     | (exception fn pos _) := exception fn pos s
     end
end interaction_monad
end run_with_state'

section parse_tac

setup_tactic_parser

open tactic

/-- Run the given parser on the given string input. -/
meta def run_on_input {α} (p : lean.parser α) (s : string) : tactic α :=
lean.parser.run $ do
  get_state >>= λ ps, of_tactic $ do
    tactic.set_env ps.env,
    -- eval_trace format!"[parse_itactic_reflected] TRYING TO PARSE {itactic_string}",
    prod.fst <$> (@interaction_monad.run_with_state' parser_state _ _ ps $ with_input p s)

/-- Parse a reflected interactive tactic from a string.
    The result can be evaluated to a `tactic unit` by using
    `eval_expr (tactic unit)`. -/
meta def parse_itactic_reflected (tactic_string : string) : tactic expr := do
let itactic_string := "{ " ++ tactic_string ++  " }",
r ← run_on_input parser.itactic_reflected itactic_string,
pure $ reflected_value.expr r

/-- Parse an interactive tactic from a string. -/
meta def parse_itactic (tactic_string : string) : tactic (tactic string) :=
do
  rtac ← parse_itactic_reflected tactic_string,
  u ← eval_expr (tactic unit) rtac,
  pure (u *> pure tactic_string)


meta def get_tac_and_capture_result (next_candidate : string) (timeout : ℕ := 5000) (max_heartbeats : ℕ := 200000) : tactic (tactic_result _) := do {
  tac ← do {
    env ← tactic.get_env,
    tac ← parse_itactic next_candidate,
    tactic.set_env env,
    pure tac
  },
  result ← tactic.capture' (tactic.try_for_time timeout $ tactic.try_for max_heartbeats tac), -- if `tac` fails, exception is captured here
  pure result
}

meta def get_tac_wrap_it_and_capture_result
(next_candidate : string)
(wrap : tactic string → tactic string)
(timeout : ℕ := 100000)
(max_heartbeats : ℕ := 2000000) : tactic (tactic_result _) := do {
  tac ← do {
    env ← tactic.get_env,
    tac ← parse_itactic next_candidate,
    tactic.set_env env,
    pure tac
  },
  result ← tactic.capture' (tactic.try_for max_heartbeats $ wrap tac), -- if `tac` fails, exception is captured here
  pure result
}

end parse_tac


meta def tactic.lazy_trace (nm : name) (s : thunk (tactic format)) : tactic unit := do
  s' ← s(),
  tactic.trace_if_enabled nm $ format!"{nm}: " ++ s'


section prettyprint

meta def set_base_verbosity : tactic unit := do {
  -- OPTIONS THAT REMAIN UNCHANGED
  tactic.set_bool_option `pp.structure_projections true,
  tactic.set_bool_option `pp.locals_full_names false,
  tactic.set_bool_option `pp.annotations false,
  tactic.set_bool_option `pp.hide_comp_irrelevant true,
  -- tactic.set_bool_option `pp.unicode true,
  tactic.set_bool_option `pp.colors false,
  tactic.set_bool_option `pp.structure_instances true,
  tactic.set_bool_option `pp.purify_metavars true,
  tactic.set_bool_option `pp.notation true,
  tactic.set_bool_option `pp.preterm false,
  tactic.set_bool_option `pp.links false,
  tactic.set_bool_option `pp.generalized_field_notation false,
  tactic.set_bool_option `pp.private_names false,
  tactic.set_bool_option `pp.goal.compact false,
  tactic.set_bool_option `pp.universes false,
  tactic.set_bool_option `pp.full_names true,
  tactic.set_bool_option `pp.purify_locals true,
  tactic.set_bool_option `pp.strings true,
  -- tactic.set_bool_option `pp.all false,
  tactic.set_bool_option `pp.proofs true,
  tactic.set_bool_option `pp.numerals true,
  tactic.set_bool_option `pp.delayed_abstraction false,
  tactic.set_bool_option `pp.instantiate_mvars true,
  tactic.set_bool_option `pp.use_holes false,
  tactic.set_nat_option `pp.max_depth 128,
  tactic.set_nat_option `pp.goal.max_hypotheses 200,
  tactic.set_nat_option `pp.max_steps 10000,
  tactic.set_nat_option `pp.width 120,
  tactic.set_nat_option `pp.indent 2,

  -- OPTIONS THAT CHANGE WITH VERBOSE,
  tactic.set_bool_option `pp.coercions false,
  tactic.set_bool_option `pp.binder_types false,
  tactic.set_bool_option `pp.implicit false,
  tactic.set_bool_option `pp.beta true,
  tactic.set_bool_option `pp.structure_instances_qualifier false
}


meta def no_verbose {α} (tac : tactic α) : tactic α :=
tactic.save_options (set_base_verbosity >> tac)


end prettyprint

section tactic_state_splitting

meta def tactic.directly_refd_mvars (mvar : expr) : tactic (list expr) :=
  tactic.with_local_goals' [mvar] $ do -- temporarily set goals to [mvar]
  tgt ← tactic.target,
  hyps ← tactic.local_context,
  -- types of assumptions
  hyp_tys ← hyps.mmap tactic.infer_type,
  -- let-values of assumptions (the v in `h : α := v`)
  -- hyp_vals ← hyps.mmap (λ hyp, tactic.local_def_value hyp <|> pure default),
  hyp_vals ← hyps.mfoldl (λ  (acc : list expr) hyp, 
  (do
    v ← tactic.local_def_value hyp,
    pure $ v::acc
  ) <|> pure acc
  ) [],
  -- put everything in a single expression so that we can call `expr.list_meta_vars` on it
  let e := expr.mk_app tgt (hyp_tys ++ hyp_vals),
  e ← tactic.instantiate_mvars e, -- instantiate assigned mvars
  pure e.list_meta_vars


meta def tactic.refd_mvars_core : list expr → list expr → tactic (list expr)
| (mvar :: todo) done :=
  if mvar ∈ done then
    tactic.refd_mvars_core todo done
  else do
    refd ← tactic.directly_refd_mvars mvar,
    tactic.refd_mvars_core (refd ++ todo) (mvar :: done)
| [] done := pure done


meta def tactic.refd_mvars (mvar : expr) : tactic (list expr) :=
tactic.refd_mvars_core [mvar] []


meta def get_connected_component_aux (adj_l : list (list ℕ)) : ℕ → list ℕ → list ℕ := λ i visited,
  if visited.mem i then
    visited
  else
    match adj_l.nth i with
    | some l := l.foldl (
      λ (acc : list ℕ) i', get_connected_component_aux i' acc
    ) (i::visited)
    | _ := []  -- error
    end


meta def get_connected_component (adj_l : list (list ℕ)) (i : ℕ) : list ℕ := get_connected_component_aux adj_l i []

meta def get_all_connected_components (adj_l : list (list ℕ)) : list (list ℕ) :=
  adj_l.enum.foldl (
    λ (acc : list (list ℕ)) ⟨i,_⟩,
      if acc.any (λ cc, cc.mem i) then
        acc
      else
        let new_cc := get_connected_component adj_l i in
        acc ++ [new_cc]
  ) []


/--
Does its best to emulate the `split_tactic_state` from `ml_server.cpp` in `lean_ml_tooling`
-/
meta def tactic.split_tactic_state (ts: tactic_state) : tactic (list (list expr)) := do
  gs ← tactic.get_goals,
  l_mvars ← gs.mmap tactic.refd_mvars,
  -- build adjacency list
  let adj_l := l_mvars.enum.map (
    λ ⟨i,mvars⟩, list.map prod.fst (
      l_mvars.enum.filter (
        λ x, x.1 ≠ i ∧ ¬ list.empty (x.2.inter mvars)
      )
    )
  ),
  -- traverse graph to get connected components
  let ccs : list (list ℕ) := get_all_connected_components adj_l,
  -- cluster by continuous goals
  let bounds := list.reverse $ ccs.foldl (
    λ (acc : list ℕ) cc,
    let cc_sort := cc.merge_sort (≤)in
    if cc_sort.head > acc.head then
      (cc_sort.reverse.head)::acc
    else
      (cc_sort.reverse.head)::(acc.tail)
  )[0],
  let clusters := list.zip (0::((bounds.take $ bounds.length - 1).map $ λ i, i+1)) bounds,
  pure $ clusters.map $ λ ⟨i,j⟩, (gs.take $ j + 1).drop i

  end tactic_state_splitting


meta def parse_decl : string → tactic (name) := lean.parser.run_with_input lean.parser.ident
