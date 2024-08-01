import v0.utils
import v0.proof_running


section removehaves

declare_trace removehaves

/--
Inspired from OpenAI's lean-gym, but: all `have`s are now tested from removal,
as it appeared that some `have`s did not have their proof term in the final proof terms but
were still useful for resolution (e.g., with `nlinarith` - don't ask me why)
-/
meta def tactic.remove_haves (ts_base : tactic_state) (tgt : expr) : ProofSteps → ℕ → tactic (list BwdStep) := λ ps resume_idx, do
steps ← match ps with
  | ProofSteps.of_actions actions := run_proof ts_base tgt actions
  | ProofSteps.of_steps steps := pure steps
end,
do {
  let ts_final := (steps.last sorry).ts_end,
  ⟨bad_idx, resume_goals⟩ ← steps.enum.reverse.mfirst (λ ⟨idx, ⟨ts_start, action, ts_end⟩⟩, 
    cond(resume_idx > 0 ∧ idx >= resume_idx)
    (tactic.fail "action already handled")
    (do {
      tactic.lazy_trace `removehaves $ pure format!"tactic:\t{action}",
      if ((action.to_list.take 5 = "have ".to_list) || ((action.to_list.reverse.take 4).reverse = "have".to_list)) then do {
        gs_start ← tactic.run_tac ts_start tactic.get_goals,
        gs_end ← tactic.run_tac ts_end tactic.get_goals,
        tactic.lazy_trace `removehaves $ pure format!"gs_start:\t{gs_start}",
        tactic.lazy_trace `removehaves (do pp ← gs_start.mmap tactic.infer_type >>= tactic.pp, pure format!"type of gs_start:\t{pp}"),
        tactic.lazy_trace `removehaves $ pure format!"gs_end:\t{gs_end}",
        tactic.lazy_trace `removehaves (do pp ← gs_end.mmap tactic.infer_type >>= tactic.pp, pure format!"type of gs_end:\t{pp}"),
        (goal_with_assumption, all_in_one) ← do {
          -- `have` step with proof provided
          if gs_start.length = gs_end.length ∧ gs_start.tail = gs_end.tail then tactic.lazy_trace `removehaves (pure "`have` step with proof provided") >> pure (gs_end.head, tt) 
          -- `have` step without proof provided
          else if gs_start.length + 1 = gs_end.length ∧ gs_start.tail = gs_end.tail.tail then tactic.lazy_trace `removehaves (pure"`have` step without proof provided") >> pure (gs_end.tail.head, ff)
          -- something weird, e.g. `have h : <prop>; swap`
          else tactic.lazy_trace `removehaves (pure "something weird, e.g. `have h : <prop>; swap`") >> tactic.fail "weird `have`"
        },
        tactic.lazy_trace `removehaves $ pure format!"goal_with_assumption:\t{goal_with_assumption}",
        tactic.lazy_trace `removehaves (do pp ← tactic.infer_type goal_with_assumption >>= tactic.pp, pure format!"type of goal_with_assumption:\t{pp}"),
        tactic.lazy_trace `removehaves $ pure format!"all_in_one:\t{all_in_one}",
        assumption ← tactic.run_tac ts_end $ do { 
          tactic.set_goals [goal_with_assumption], 
          hs ← tactic.local_context,
          tactic.lazy_trace `removehaves $ pure format!"hs:\t{hs}",
          tactic.lazy_trace `removehaves (do pp ← hs.mmap tactic.infer_type >>= tactic.pp, pure format!"type of hs:\t{pp}"),
          if hs_nnil : hs ≠ [] then pure (hs.last hs_nnil) else tactic.fail "weird `have`"
        },
        tactic.lazy_trace `removehaves $ pure format!"assumption:\t{assumption}",
        tactic.lazy_trace `removehaves (do pp ← tactic.infer_type assumption >>= tactic.pp, pure format!"type of assumption:\t{pp}"),
        proof_of_goal_with_assumption ← tactic.run_tac ts_final (tactic.instantiate_mvars goal_with_assumption),
        tactic.lazy_trace `removehaves $ pure format!"proof_of_goal_with_assumption:\t{proof_of_goal_with_assumption}",
        pure (idx, if all_in_one then gs_end else gs_end.tail)
      } else
        tactic.fail "not a `have` step"
    })
  ),
  tactic.lazy_trace `removehaves $ pure format!"bad_idx:\t{bad_idx}",
  tactic.lazy_trace `removehaves (do pp ← resume_goals.mmap tactic.infer_type >>= tactic.pp, pure format!"type of resume_goals:\t{pp}"),
  goals_per_step : list (list expr) ← steps.mmap (λ ⟨ts_start, action, ts_end⟩, tactic.run_tac ts_start tactic.get_goals),
  tactic.lazy_trace `removehaves (do pp ← goals_per_step.mmap (λ gs, gs.mmap tactic.infer_type) >>= tactic.pp, pure format!"type of goals_per_step:\t{pp}"),
  let resume_idx : ℕ := goals_per_step.enum.find_index (λ idx__goals, idx__goals.fst > bad_idx ∧ idx__goals.snd = resume_goals),
  tactic.lazy_trace `removehaves $ pure format!"resume_idx:\t{resume_idx}",
  if resume_idx = steps.length then
  tactic.lazy_trace `removehaves (pure "weird `have`, possible swap") >> tactic.fail "weird `have`, possible swap" else do {
    let actions := steps.map BwdStep.action,
    let actual_actions := actions.take bad_idx ++ actions.drop resume_idx,
    tactic.lazy_trace `removehaves (do let stripped_actions := (actions.take resume_idx).drop bad_idx, pure format!"stripped_actions:\t{stripped_actions}"),
    tactic.lazy_trace `removehaves $ pure format!"actual_actions:\t{actual_actions}",
    do {
      actual_steps ← run_proof ts_base tgt actual_actions,
      tactic.remove_haves (ProofSteps.of_steps actual_steps) bad_idx
    }
    --  sometimes an hyp is not used directly in the proof term BUT still matters for find the solution
    <|> tactic.remove_haves (ProofSteps.of_steps steps) bad_idx
  }
} <|> pure steps

end removehaves


section removeuselessactionsbuthaves

declare_trace removeuselessactionsbuthaves

/--
After all `have`s have been tested for removal, all other tactics are tested
-/
meta def tactic.remove_useless_actions_but_haves (ts_base : tactic_state) (tgt : expr) : ProofSteps → ℕ → tactic (list BwdStep) := λ ps resume_idx, do
steps ← match ps with
  | ProofSteps.of_actions actions := run_proof ts_base tgt actions
  | ProofSteps.of_steps steps := pure steps
end,
do {
  let ts_final := (steps.last sorry).ts_end,
  ⟨bad_idx, resume_goals⟩ ← steps.enum.reverse.mfirst (λ ⟨idx, ⟨ts_start, action, ts_end⟩⟩, 
    cond(resume_idx > 0 ∧ idx >= (resume_idx-1))
    (tactic.fail "action already handled")
    (do {
      tactic.lazy_trace `removeuselessactionsbuthaves $ pure format!"tactic:\t{action}",
      if action.to_list.take 5 = "have ".to_list then
        tactic.fail "action is a `have`"
      else (do
        gs_end ← tactic.run_tac ts_end tactic.get_goals,
        pure (idx, gs_end)
      )
    })
  ),
  tactic.lazy_trace `removeuselessactionsbuthaves $ pure format!"bad_idx:\t{bad_idx}",
  tactic.lazy_trace `removeuselessactionsbuthaves (do pp ← resume_goals.mmap tactic.infer_type >>= tactic.pp, pure format!"type of resume_goals:\t{pp}"),
  goals_per_step : list (list expr) ← steps.mmap (λ ⟨ts_start, action, ts_end⟩, tactic.run_tac ts_start tactic.get_goals),
  tactic.lazy_trace `removeuselessactionsbuthaves (do pp ← goals_per_step.mmap (λ gs, gs.mmap tactic.infer_type) >>= tactic.pp, pure format!"type of goals_per_step:\t{pp}"),
  let resume_idx : ℕ := goals_per_step.enum.find_index (λ idx__goals, idx__goals.fst > bad_idx ∧ idx__goals.snd = resume_goals),
  tactic.lazy_trace `removeuselessactionsbuthaves $ pure format!"resume_idx:\t{resume_idx}",
  let actions := steps.map BwdStep.action,
  let actual_actions := actions.take bad_idx ++ actions.drop resume_idx,
  tactic.lazy_trace `removeuselessactionsbuthaves (do let stripped_actions := (actions.take resume_idx).drop bad_idx, pure format!"stripped_actions:\t{stripped_actions}"),
  tactic.lazy_trace `removeuselessactionsbuthaves $ pure format!"actual_actions:\t{actual_actions}",
  do {
    actual_steps ← run_proof ts_base tgt actual_actions,
    tactic.remove_useless_actions_but_haves (ProofSteps.of_steps actual_steps) (bad_idx + 1)
  }
  --  sometimes an hyp is not used directly in the proof term BUT still matters for find the solution
  <|> tactic.remove_useless_actions_but_haves (ProofSteps.of_steps steps) (bad_idx + 1)
} <|> pure steps

end removeuselessactionsbuthaves


/--
Other possible cleaning:
- split the tactic `;`
- check arguments of `simp`
- decompose `simp` in more granular steps when it does not finish the job
- remove `try`
- remove `repeat`
- remove `* at *`, `[]`
- in case of segfault, retry
- get rid of `revert`
- rename canonically hypotheses `h₀`, `h₁`, etc
-/
meta def clean_proof_from_expr (tgt : expr) (actions : list string) : tactic (list BwdStep) := do
  ts_base ← tactic.read,
  steps ← tactic.remove_haves ts_base tgt(ProofSteps.of_actions actions) 0,
  steps ← tactic.remove_useless_actions_but_haves ts_base tgt(ProofSteps.of_steps steps) 0,
  tactic.write ts_base,
  pure $ steps


/--
Clean proof and return the `Proof` format expected by Evariste
-/
meta def clean_proof_from_nm (nm : name) (actions : list string) : tactic (list (string × string × list string)) := do
  tgt ← tactic.get_env >>= λ e, e.get nm >>= pure ∘ declaration.type,
  steps ← clean_proof_from_expr tgt actions,
  steps.mmap $ λ ⟨ts_start, action, ts_end⟩, do
    gs_start ← tactic.run_tac ts_start tactic.get_goals,
    str_start ← tactic.run_tac ts_start $ do { 
      tactic.set_goals [gs_start.head],
      tactic.read >>= no_verbose ∘ tactic.pp >>= λ s, pure $ to_string s
    },
    gs_end ← tactic.run_tac ts_end tactic.get_goals,
    str_end ← do {
      -- subgoals created
      if gs_start.length ≤ gs_end.length then (gs_end.take (gs_end.length - gs_start.length + 1)).mmap $ λ g, tactic.run_tac ts_end $ do {
        tactic.set_goals [g],
        tactic.read >>= no_verbose ∘ tactic.pp >>= λ s, pure $ to_string s
      }
      -- goal solved
      else if gs_start.length = gs_end.length + 1 then pure []
      else tactic.fail "several goals solved at once, not handled"
    },
    pure (str_start, action, str_end)


/--
API: (valid_cleaned: bool, exc: string, cleaned_proof: list (string × string × list string))
-/
meta def clean_proof (nm : name) (actions : list string) : tactic (bool × string × list (string × string × list string)) := do
  result ← tactic.capture $ clean_proof_from_nm nm actions,
  match result with
  | interaction_monad.result.success pfl _ := pure (tt,"",pfl)
  | interaction_monad.result.exception f _ _ := 
    let msg := match f with
    | some x := to_string (x ())
    | _ := "<no_msg>"
    end in pure (ff, msg,[])
  end


/--
Clean proof and return a json object
API: {valid_proof: bool, valid_cleaned: bool, exc: string, cleaned_proof: list (string × string × list string)}
-/

meta def check_and_clean_proof (nm : string) (actions : list string) : tactic json := do
  name ← parse_decl nm,
  (valid_proof, exc) ← check_proof name actions,
  if ¬valid_proof then
    pure $ json.object [
      ("valid_proof", json.of_bool ff),
      ("valid_cleaned", json.of_bool ff),
      ("exc", json.of_string exc),
      ("cleaned_proof", json.array [])
    ]
  else
    do
      (valid_cleaned, exc, proof) ← clean_proof name actions,
      proof_json ← to_tactic_json proof,
      pure $ json.object [
        ("valid_proof", json.of_bool tt),
        ("valid_cleaned", json.of_bool valid_cleaned),
        ("exc", json.of_string exc),
        ("cleaned_proof", proof_json)
      ]
