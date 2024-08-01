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
-- sexp modification
local notation `SEXP_DEPTH_LIMIT` := some (50 : ℕ)

section sexp
inductive sexp (α : Type u) : Type u
| atom : α → sexp
| list : (list sexp) → sexp

meta def sexp.to_format {α : Type u} [has_to_format α] : sexp α → format
| (sexp.atom val) := has_to_format.to_format val
| (sexp.list ses) := format.of_string $ "(" ++ (" ".intercalate (ses.map $ format.to_string ∘ sexp.to_format)) ++ ")"

meta instance sexp_has_to_format {α} [has_to_format α] : has_to_format (sexp α) :=
⟨sexp.to_format⟩

end sexp

section open_binders
open expr
open tactic
@[simp] 
private def option.elim {α β} : option α → β → (α → β) → β
| (some x) y f := f x
| none     y f := y

private meta def match_lam {elab} : expr elab →
  option (name × binder_info × expr elab × expr elab)
| (lam var_name bi type body) := some (var_name, bi, type, body)
| _ := none

private meta def match_pi {elab} : expr elab →
  option (name × binder_info × expr elab × expr elab)
| (pi var_name bi type body) := some (var_name, bi, type, body)
| _ := none

private meta structure binder :=
  (name : name)
  (info : binder_info)
  (type : expr)

private meta def mk_binder_replacement (local_or_meta : bool) (b : binder) :
  tactic expr :=
if local_or_meta then mk_local' b.name b.info b.type else mk_meta_var b.type

@[inline] meta def get_binder (do_whnf : option (transparency × bool))
  (pi_or_lambda : bool) (e : expr) :
  tactic (option (name × binder_info × expr × expr)) := do
  e ← option.elim do_whnf (pure e) (λ p, whnf e p.1 p.2),
  pure $ if pi_or_lambda then match_pi e else match_lam e

private meta def open_n_binders (do_whnf : option (transparency × bool))
  (pis_or_lambdas : bool) (locals_or_metas : bool) :
  expr → ℕ → tactic (list expr × expr)
| e 0 := pure ([], e)
| e (d + 1) := do
  some (name, bi, type, body) ← get_binder do_whnf pis_or_lambdas e | failed,
  replacement ← mk_binder_replacement locals_or_metas ⟨name, bi, type⟩,
  (rs, rest) ← open_n_binders (body.instantiate_var replacement) d,
  pure (replacement :: rs, rest)

private meta def open_n_pis : expr → ℕ → tactic (list expr × expr) :=
open_n_binders none tt tt

private meta def open_n_lambdas : expr → ℕ → tactic (list expr × expr) :=
open_n_binders none ff tt


section sexp_of_expr
open expr

/-- head-reduce a single let expression -/
private meta def tmp_reduce_let : expr → expr
| (elet _ _ v b) := b.instantiate_var v
| e              := e

meta def sexp.concat {m} [monad m] [monad_fail m] {α} : (sexp α) → (sexp α) → m (sexp α)
| (sexp.list xs) (sexp.list ys) := pure (sexp.list $ xs ++ ys)
| _ _ := monad_fail.fail "sexp.concat failed"

local infix `<+>`:50 := sexp.concat -- TODO(jesse): just write an applicative instance, don't want to think about `seq` now though

meta def sexp.map {α β : Type*} (f : α → β) : sexp α → sexp β
| (sexp.atom x) := (sexp.atom $ f x)
| (sexp.list xs) := (sexp.list $ list.map sexp.map xs)

meta instance : functor sexp :=
{map := λ α β f, sexp.map f}

def mk_type_ascription : sexp string → sexp string → sexp string := λ s₁ s₂, sexp.list [(sexp.atom ":"), s₁, s₂]

-- TODO(jesse): supply version with even more type annotations
meta def sexp_of_expr : (option ℕ) → expr → tactic (sexp string) := λ fuel ex, do {
  match fuel with
  | none := pure ()
  | (some x) := when (x = 0) $ tactic.fail "sexp_of_expr fuel exhausted"
  end,
  match ex with
  | e@(var k) := (sexp.list [sexp.atom "var"]) <+> (sexp.list [sexp.atom (repr k)])
  | e@(sort l) := (sexp.list [sexp.atom "sort"]) <+> (sexp.list [sexp.atom (to_string l)])
  | e@(const nm ls) := pure $ sexp.atom nm.to_string
  | e@(mvar un pt tp) := do tp_sexp ← sexp_of_expr ((flip nat.sub 1) <$> fuel) tp, pure $ mk_type_ascription (sexp.atom pt.to_string) tp_sexp
  | e@(local_const un pt binder_info tp) := do {
    tp_sexp ← sexp_of_expr ((flip nat.sub 1) <$> fuel) tp,
    -- pure $ flip mk_type_ascription tp_sexp $ sexp.list [sexp.atom pt.to_string] -- note: drop binder info for now
   pure (sexp.atom pt.to_string)
  }
  -- version without "APP" head symbol
  | e@(app e₁ e₂) := (λ s₁ s₂, sexp.list [s₁, s₂]) <$> sexp_of_expr ((flip nat.sub 1) <$> fuel) e₁ <*> sexp_of_expr ((flip nat.sub 1) <$> fuel) e₂

  -- version with "APP" head symbol -- switch if needed
  -- | e@(app e₁ e₂) := (λ s₁ s₂, sexp.list [sexp.atom "APP", s₁, s₂]) <$> sexp_of_expr ((flip nat.sub 1) <$> fuel) e₁ <*> sexp_of_expr ((flip nat.sub 1) <$> fuel) e₂

  -- | e@(app e₁ e₂) := sexp.list <$> ((::) <$> (sexp_of_expr ((flip nat.sub 1) <$> fuel) $ get_app_fn e) <*> (get_app_args e).mmap (sexp_of_expr ((flip nat.sub 1) <$> fuel)))
  | e@(lam var_name b_info var_type body) := do {
    ⟨[b], e'⟩ ← open_n_lambdas e 1,
    sexp.list <$> do {
      b_sexp ← sexp_of_expr ((flip nat.sub 1) <$> fuel) b,
      b_tp_sexp ← sexp_of_expr ((flip nat.sub 1) <$> fuel) var_type,
      body_sexp ← sexp_of_expr ((flip nat.sub 1) <$> fuel) e',
      pure $ [sexp.atom "LAMBDA", mk_type_ascription (b_sexp) b_tp_sexp, body_sexp]
    }
  }
  | e@(pi var_name b_info var_type body) := do {
    ⟨[b], e'⟩ ← open_n_pis e 1,
    sexp.list <$> do {
      b_sexp ← sexp_of_expr ((flip nat.sub 1) <$> fuel) b,
      b_tp_sexp ← sexp_of_expr ((flip nat.sub 1) <$> fuel) var_type,
      body_sexp ← sexp_of_expr ((flip nat.sub 1) <$> fuel) e',
      pure $ [sexp.atom "PI", mk_type_ascription (b_sexp) b_tp_sexp, body_sexp]
    }
  }
  -- reduce let expressions before sexpr serialization
  | e@(elet var_name var_type var_assignment body) := sexp_of_expr ((flip nat.sub 1) <$> fuel) (tmp_reduce_let e)
  | e@(macro md deps) := 
    sexp.list <$> do {
      deps_sexp_list ← sexp.list <$> (deps.mmap $ sexp_of_expr (((flip nat.sub 1) <$> fuel))),
      let deps_sexp := sexp.list [sexp.atom "MACRO_DEPS", deps_sexp_list],
      pure $ [sexp.atom "MACRO", sexp.atom (expr.macro_def_name md).to_string, deps_sexp]
    }
  end
}

end sexp_of_expr
end open_binders

section 
private meta def set_bool_option (n : name) (v : bool) : tactic unit :=
do s ← tactic.read,
   tactic.write $ tactic_state.set_options s (options.set_bool (tactic_state.get_options s) n v)

private meta def enable_full_names : tactic unit := do {
  set_bool_option `pp.full_names true
}

private meta def with_full_names {α} (tac : tactic α) : tactic α :=
tactic.save_options $ enable_full_names *> tac

open sexp tactic
meta def tactic_state.to_sexp (ts : tactic_state) : tactic (sexp string) := do
  λ _, interaction_monad.result.success () ts,
  -- do { gs ← tactic.get_goals, guard (gs.length = 1) <|>
  --      tactic.trace ("[tactic_state.to_sexp] WARNING: NUM GOALS" ++ gs.length.repr) },
  with_full_names $ do {
    hyps ← tactic.local_context,
    annotated_hyps ← hyps.mmap (λ h, prod.mk h <$> tactic.infer_type h),
    hyps_sexp ← do {
      hyps_sexps ← annotated_hyps.mmap $
        function.uncurry $ λ hc ht, mk_type_ascription <$>
          sexp_of_expr SEXP_DEPTH_LIMIT hc <*> sexp_of_expr none ht,
      pure $ sexp.list $ [sexp.atom "HYPS"] ++ hyps_sexps
    },
    goal_sexp ← (λ x, sexp.list [sexp.atom "GOAL", x]) <$> (tactic.target >>= sexp_of_expr SEXP_DEPTH_LIMIT),
    pure $ sexp.list [sexp.atom "GOAL_STATE", hyps_sexp, goal_sexp]
  }
end 

local notation `PRINT_TACTIC_STATE` := λ ts, (format.to_string ∘ format.flatten ∘ sexp.to_format) <$> tactic_state.to_sexp ts

-- end sexp modifications
--PR END MODIFICATION
