/-
Inspired from OpenAI's lean-gym, but with the `tactic` monad instead of the `io` one
-/

import v0.utils


meta structure BwdStep :  Type :=
  (ts_start : tactic_state)
  (action : string)
  (ts_end : tactic_state)


meta inductive ProofSteps
| of_actions : list string → ProofSteps
| of_steps : list BwdStep → ProofSteps



declare_trace runproof
declare_trace runproof_rec

open interaction_monad.result

meta def run_proof_rec (ts_init : tactic_state) : tactic_state → list string → tactic (list BwdStep) := λ ts actions, do
  tactic.lazy_trace `runproof_rec $ pure format!"ts_init:\t{ts_init}",
  tactic.lazy_trace `runproof_rec $ pure format!"ts:\t{ts}",
  tactic.lazy_trace `runproof_rec $ pure format!"tactic:\t{actions.head}",
  match (ts, actions) with
  | (ts_start, []) := (tactic.run_tac ts_init (do
    -- confirm that the proof has finished
    [g] ← tactic.get_goals,
    tgt ← tactic.infer_type g,
    tactic.lazy_trace `runproof_rec (do pp ← tactic.pp tgt, pure format!"tgt:\t{pp}"),
    tactic.write ts_start,
    gs ← tactic.get_goals,
    guardb (gs.empty),
    pf ← tactic.get_assignment g >>= tactic.instantiate_mvars,
    tactic.lazy_trace `runproof_rec $ pure format!"pf:\t{pf}",
    validate_proof tgt pf,
    tactic.lazy_trace `runproof $ pure format!"proof validated",
    pure []
  )) <|> (tactic.lazy_trace `runproof (pure format!"failed to validate") >> tactic.fail "failed to validate")
  | (ts_start, (action::actions)) := do 
    result_with_string ← tactic.run_tac ts_start (get_tac_and_capture_result action),
    match result_with_string with
    -- The tactic application was successful.
    | interaction_monad.result.success _ ts_end := do {
      /- TODO: to be added to handle multi-goals - but with it, triggers a segfault
      gs_start ← tactic.run_tac ts_start tactic.get_goals,
      gs_end ← tactic.run_tac ts_end tactic.get_goals,
      cond(gs_start.length > gs_end.length + 1)
      (do
        tactic.lazy_trace `runproof $ pure format!"several goals solved at once, not handled",
        tactic.fail "several goals solved at once, not handled"
      )
      (pure ()),
      -/
      rest ← run_proof_rec ts_end actions,
      pure $ ⟨ts_start, action, ts_end⟩ :: rest
    }
    | interaction_monad.result.exception f p s' := do
      let msg := match f with 
      | some x := format!"failed to run proof: {x ()}"
      | _ := format!"failed to run proof: <no msg>"
      end,
      tactic.lazy_trace `runproof $ pure msg,
      tactic.fail msg
    end
  end


meta def run_proof (ts_base : tactic_state) (tgt : expr) (actions : list string) : tactic (list BwdStep) := do
  tactic.lazy_trace `runproof (do pp ← tactic.pp tgt, pure format!"tgt:\t{pp}"),
  tactic.lazy_trace `runproof $ pure format!"actions:\t{actions}",
  tactic.write ts_base,
  tactic.set_goal_to tgt,
  ts_init ← tactic.read,
  run_proof_rec ts_init ts_init actions


meta def check_proof_from_nm (nm : name) (actions : list string) : tactic unit := do
  ts_base ← tactic.read,
  tgt ← tactic.get_env >>= λ e, e.get nm >>= pure ∘ declaration.type,
  run_proof ts_base tgt actions,
  pure ()

/--
API: (valid_proof: bool, exc: string)
-/
meta def check_proof (nm : name) (actions : list string) : tactic (bool × string) := do
  result ← tactic.capture $ check_proof_from_nm nm actions,
  match result with
  | interaction_monad.result.success _ _ := pure (tt,"")
  | interaction_monad.result.exception f _ _ := 
    let msg := match f with
    | some x := to_string (x ())
    | _ := "<no_msg>"
    end in pure (ff, msg)
  end
