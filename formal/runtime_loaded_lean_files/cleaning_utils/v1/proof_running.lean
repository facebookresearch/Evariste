/-
Inspired from OpenAI's lean-gym, but with the `tactic` monad instead of the `io` one
-/

import v1.utils


meta structure BwdStep :  Type :=
  (ts_start : tactic_state)
  (defocus_substack : ℕ)
  (action : string)
  (ts_end : tactic_state)


meta instance : has_to_format BwdStep :=
let fn : BwdStep → format := λ s, format!"<{s.ts_start}, {s.defocus_substack}, {s.action}, {s.ts_end}>" in
⟨fn⟩


declare_trace runactions
declare_trace runactions_rec

open interaction_monad.result

def enclosing_toks := [
  ('{','}'),
  ('[',']'),
  ('(',')'),
  ('⟨','⟩')
]

def opening_toks := enclosing_toks.map prod.fst
def closing_toks := enclosing_toks.map prod.snd


meta def opening_to_closing_tok_aux (c : char) : list (char × char) → char := λ l,
  match l with
  | [] := ' '
  | (c₁, c₂)::tl := if c₁= c then c₂ else  opening_to_closing_tok_aux tl
  end


meta def opening_to_closing_tok (c : char) : char := opening_to_closing_tok_aux c enclosing_toks


meta def balance_enclosing_toks (action : string) : string :=
  list.as_string $ prod.fst $ action.to_list.foldl (
    -- acc : string in construction × stack × to_be_balanced
    λ (acc : list char × list char × bool) c,
    let ⟨cs, stack, b⟩ := acc in
    if ¬ b then
      (cs, [], b)
    else
      if opening_toks.mem c then
        (cs ++ [c], (opening_to_closing_tok c)::stack, b)
      else if closing_toks.mem c then
        if (¬stack.empty) ∧ stack.head = c then
          (cs ++ [c], stack.drop 1, b)
        else
          (cs, [], ff)
      else
        (cs ++ [c], stack, b)
  ) ([], [], tt)


meta def prevent_alternative_aux : list char → list char := λ l,
  if l.length < 3 then
    l
  else if l.take 3 = "<|>".to_list then
    []
  else
    l.head :: (prevent_alternative_aux $ l.drop 1)


meta def prevent_alternative (action : string) : string :=
  string.strip $ list.as_string $ prevent_alternative_aux action.to_list


meta def split_into_atomic_actions : string → bool → tactic (list string) := λ action split_on_commas,
  let action := balance_enclosing_toks action in
  let action := action.strip in (
  if action.startswith "{" then do
    guardb $ action.to_list.reverse.head = '}',
    let action := list.as_string $ (action.to_list.take (action.length - 1)).drop 1,
    rest ← split_into_atomic_actions action tt,
    pure $ "{"::rest ++ ["}"]
  else if action.startswith "solven " then do
    let l := action.to_list,
    let ⟨l₁,l₂⟩ := action.to_list.span $ λ c, c ≠ 'b',  -- begin / end
    let ac_solven := l₁.as_string.strip,
    if l₂.empty then
      pure [ac_solven]
    else do
      guardb $ (l₂.reverse.take 3).reverse.as_string = "end",
      let action := list.as_string $ (l₂.take $ l₂.length - 3).drop 5,
      rest ← split_into_atomic_actions action tt,
      pure $ ac_solven::rest ++ ["}"] 
  else if split_on_commas then do
    ⟨acs, cur_ac, stack⟩← action.to_list.mfoldl (
      -- acc : actions, cur_action, stack
      λ (acc : list string × list char × list char) c,
      let ⟨acs, cur_ac, stack⟩ := acc in
      if c = ',' ∧ stack.empty then do
        cur_acs ← split_into_atomic_actions cur_ac.as_string ff,
        pure (acs ++ cur_acs, [], [])
      else
        let cur_ac := cur_ac ++ [c] in
        if opening_toks.mem c then
          pure $ (acs, cur_ac, (opening_to_closing_tok c)::stack)
        else if closing_toks.mem c then do
          guardb $ stack.head = c,
          pure $ (acs, cur_ac, stack.drop 1)
        else
          pure $ (acs, cur_ac, stack)
    ) ([], [], []),
    guardb stack.empty,
    if cur_ac.empty then
      pure acs
    else do
      cur_acs ← split_into_atomic_actions cur_ac.as_string ff,
      pure $ acs ++ cur_acs
  else do
    pure [prevent_alternative action]
  ) <|> pure [action]


meta def get_action_spans : list ℕ → list string → list (list string) × list string
| [] acs := ⟨[], acs⟩
| (n::tl) acs := 
  let idx := prod.fst $ acs.foldl (λ (acc : ℕ × ℕ × ℕ) ac,
  let ⟨cur_idx, cur_open, cur_done⟩ := acc in
  if cur_done = n then
    acc
  else
    let ac := ac.strip in
    let cur_idx := cur_idx + 1 in
    let cur_open := 
      if ac = "{" then 
        cur_open + 1
      else if ac = "}" then
        cur_open - 1
      else
        cur_open in
    let cur_done :=
      if (ac = "}") ∧ (cur_open = 0) then
        cur_done + 1
      else
        cur_done in
    (cur_idx, cur_open, cur_done)
  ) (0,0,0) in
  let ⟨acs_spans, acs_after⟩ := get_action_spans tl (acs.drop idx) in
  ⟨(acs.take idx)::acs_spans, acs_after⟩


meta def run_actions_rec (ts_init : tactic_state) : tactic_state → list string → list ℕ → tactic (list BwdStep) := λ ts actions defocus_substacks, do
  tactic.lazy_trace `runactions_rec $ pure format!"ts_init:\t{ts_init}",
  tactic.lazy_trace `runactions_rec $ pure format!"ts:\t{ts}",
  tactic.lazy_trace `runactions_rec $ pure format!"tactic:\t{actions.head}",
  match (ts, actions, defocus_substacks) with
  | (ts_start, [], []) := (tactic.run_tac ts_init (do
      -- confirm that the proof has finished
      [g] ← tactic.get_goals,
      tgt ← tactic.infer_type g,
      tactic.lazy_trace `runactions_rec (do pp ← tactic.pp tgt, pure format!"tgt:\t{pp}"),
      tactic.write ts_start,
      gs ← tactic.get_goals,
      guardb (gs.empty),
      pf ← tactic.get_assignment g >>= tactic.instantiate_mvars,
      tactic.lazy_trace `runactions_rec $ pure format!"pf:\t{pf}",
      validate_proof tgt pf,
      tactic.lazy_trace `runactions $ pure format!"proof validated",
      pure []
    )) <|> (tactic.lazy_trace `runactions (pure format!"failed to validate") >> tactic.fail "failed to validate")

  | (ts_start, [], n::_) := do
    let msg := format!"failed to run proof - {n} goals expected in top focus goal substack after running all actions",
    tactic.lazy_trace `runactions $ pure msg,
    tactic.fail msg

  | (ts_start, (action::actions), _) := do
    cur_stack_sz ← tactic.run_tac ts_start (do
          gs ← tactic.get_goals,
          pure gs.length
    ),
    let current_focus := match defocus_substacks with
    | [] := cur_stack_sz
    | h::tl := cur_stack_sz - h
    end,
    tactic.lazy_trace `runactions_rec $ pure format!"cur_stack_sz = {cur_stack_sz}",
    tactic.lazy_trace `runactions_rec $ pure format!"defocus_substacks = {defocus_substacks}",
    tactic.lazy_trace `runactions_rec $ pure format!"current_focus = {current_focus}",

    if action = "{" then do
      let new_stop := cur_stack_sz - 1,
      tactic.lazy_trace `runactions_rec $ pure format!"defocus_substacks.push {new_stop} -> {new_stop::defocus_substacks}",
      rest ← run_actions_rec ts_start actions (new_stop::defocus_substacks),
      pure $ ⟨ts_start, new_stop, action, ts_start⟩ :: rest

    else if action.startswith "solven " then do
      focus ← lean.parser.run_with_input lean.parser.small_nat $ (action.to_list.drop 7).as_string,
      let new_stop := cur_stack_sz - focus,
      guardb (defocus_substacks.empty ∨ defocus_substacks.head ≤ new_stop),
      tactic.lazy_trace `runactions_rec $ pure format!"defocus_substacks.push {new_stop} -> {new_stop::defocus_substacks}",
      rest ← run_actions_rec ts_start actions (new_stop::defocus_substacks),
      pure $ ⟨ts_start, new_stop, action, ts_start⟩ :: rest

    else if (action = "}") ∨ (action = "end") then
      match defocus_substacks with
      | [] := do
        let msg := format!"failed to run proof - goal substack not empty",
        tactic.lazy_trace `runactions $ pure msg,
        tactic.fail msg
      | h::tl := 
        if h = cur_stack_sz then do
          tactic.lazy_trace `runactions_rec $ pure format!"defocus_substacks.pop -> {tl}",
          rest ← run_actions_rec ts_start actions tl,
          let prev_stop := if tl.empty then 0 else tl.head,
          pure $ ⟨ts_start, prev_stop, action, ts_start⟩ :: rest
        else do
          let msg := format!"failed to run proof - goal stack size: {cur_stack_sz} while it was expected to be {h}",
          tactic.lazy_trace `runactions $ pure msg,
          tactic.fail msg
      end

    else if current_focus = 1
      ∧ ((action.startswith "any_goals") ∨ (action.startswith "all_goals") ) then
      let single_ac := (action.to_list.drop 9).as_string.strip_curly_braces in
      if single_ac = "" then run_actions_rec ts_start actions defocus_substacks else do
      tactic.lazy_trace `runactions_rec $ pure format!"only one goal in focus, cleaned `{action}`",
      run_actions_rec ts_start (single_ac::actions) defocus_substacks

    else
    -- TODO: do the same with `any_goals`
      if (action.startswith "all_goals") then do
        tactic.lazy_trace `runactions_rec $ pure format!"`all_goals`, splitting",
        let single_ac := (action.to_list.drop 9).as_string.strip_curly_braces,
        if single_ac = "" then run_actions_rec ts_start actions defocus_substacks else do
        let defocus := match defocus_substacks with | h::_ := h | _ := 0 end,
        tactic.lazy_trace `runactions_rec $ pure format!"defocus:\t{defocus}",
        -- apply single_ac to each goal to see which ones are solved
        gs_post ← tactic.run_tac ts_start (do
          gs ← tactic.get_goals,
          (gs.take current_focus).mmap ( λ g, do
            tactic.set_goals [g],
            tactic.read >>= λ ts, tactic.lazy_trace `runactions $ pure format!"ts:\t{ts}",
            result_with_string ← get_tac_and_capture_result single_ac,
            match result_with_string with
            -- The tactic application was successful.
            | interaction_monad.result.success _ ts_end := tactic.run_tac ts_end tactic.get_goals
            | interaction_monad.result.exception f p s' := do
              let msg := match f with 
              | some x := format!"failed to run proof on action `{single_ac}`: {x ()}"
              | _ := format!"failed to run proof on action `{single_ac}`: <no msg>"
              end,
              tactic.lazy_trace `runactions $ pure msg,
              tactic.fail msg
            end
          )
        ),
        ts_post_pp ← gs_post.mmap $ λ gl, tactic.set_goals gl >> tactic.read,
        tactic.lazy_trace `runactions_rec $ pure format!"ts_post_pp:\t{ts_post_pp}",
        -- retrieve span of actions for each subgoals
        let ⟨actions_span,actions_after⟩ := get_action_spans (gs_post.map list.length) actions,
        -- copy actions accordingly
        let all_acs := (actions_span.foldl (λ acc acs, acc ++ ["{", single_ac] ++ acs ++ ["}"]) []) ++ actions_after,
        tactic.lazy_trace `runactions_rec $ pure format!"`all_goals` all_acs:\t{all_acs}",
        run_actions_rec ts_start all_acs defocus_substacks
      else do
        atomic_acs ← split_into_atomic_actions action tt,
        tactic.lazy_trace `runactions $ pure format!"action `{action}` is splitted into {atomic_acs}",
        if atomic_acs.length = 0 then do
          let msg := format!"failed to run proof - action `{action}` was splitted into []",
          tactic.lazy_trace `runactions $ pure msg,
          tactic.fail msg
        else if atomic_acs.length = 1 then do
          let stop := match defocus_substacks with
            | [] := 0
            | h::_ := h
            end,
          result_with_string ← tactic.run_tac ts_start (get_tac_wrap_it_and_capture_result atomic_acs.head $ tactic.focus_for_top_k_goals current_focus),
          match result_with_string with
          -- The tactic application was successful.
          | interaction_monad.result.success _ ts_end := do {
            rest ← run_actions_rec ts_end actions defocus_substacks,
            pure $ ⟨ts_start, stop, atomic_acs.head, ts_end⟩ :: rest
          }
          | interaction_monad.result.exception f p s' := do
            let msg := match f with 
            | some x := format!"failed to run proof on action `{atomic_acs.head}`: {x ()}"
            | _ := format!"failed to run proof on action `{atomic_acs.head}`: <no msg>"
            end,
            tactic.lazy_trace `runactions $ pure msg,
            tactic.fail msg
          end
        else
          run_actions_rec ts_start (atomic_acs ++ actions) defocus_substacks
  end


meta def run_actions (ts_base : tactic_state) (tgt : expr) (actions : list string) : tactic (list BwdStep) := do
  tactic.lazy_trace `runactions (do pp ← tactic.pp tgt, pure format!"tgt:\t{pp}"),
  tactic.lazy_trace `runactions $ pure format!"actions:\t{actions}",
  tactic.write ts_base,
  tactic.set_goal_to tgt,
  ts_init ← tactic.read,
  run_actions_rec ts_init ts_init actions []


meta def run_bwd_steps (decl_nm : name) (actions : list string) : tactic (list BwdStep) := do
  ts_base ← tactic.read,
  tgt ← tactic.get_env >>= λ e, e.get decl_nm >>= pure ∘ declaration.type,
  run_actions ts_base tgt (actions)


meta def is_noop (action : string) : bool :=
  if action = "{" ∨ action = "}" ∨ action = "end" ∨ action.startswith "solven" then
    tt
  else
    ff


/--
Return the `Proof` format expected by Evariste
TODO: include it in the `run_actions_rec` as to avoir double proof running
-/
meta def steps_to_proof (steps : list BwdStep) : tactic (list (string × string × list string)) :=
  steps.mfoldl
  (λ
    (acc : list (string × string × list string))
    ⟨ts_start, defocus_substack, action, ts_end⟩,
      if is_noop action then
        pure acc
      else do
        (s_start, ls_end) ← tactic.run_tac ts_start (do
          nb_goals_start ← list.length <$> tactic.get_goals,
          lgs_start ← list.head <$> tactic.split_tactic_state ts_start,
          tactic.set_goals lgs_start,
          s_start ← tactic.read >>= no_verbose ∘ tactic.pp >>= λ s, pure $ to_string s,
          tac ← do {
            env ← tactic.get_env,
            tac ← parse_itactic action,
            tactic.set_env env,
            pure tac
          },
          tac,
          lgs_end ← tactic.read >>= tactic.split_tactic_state,
          ls_end ←  (
            if lgs_end.length = 1 ∧  lgs_end.head.empty then
              pure []
            else
              lgs_end.mmap $ λ gs_end, (do
                tactic.set_goals gs_end,
                tactic.read >>= no_verbose ∘ tactic.pp >>= λ s, pure $ to_string s
              )
          ),
          pure (s_start, ls_end)
        ),
        pure $ acc ++ [(s_start, action, ls_end)]
  ) []


/--
API: (valid_proof: bool, exc: string, proof: list (string × string × list string), steps: list BwdStep)
-/
meta def check_and_get_proof_from_steps
(nm : string)
(steps : list BwdStep)
(bwd_steps_runner : name → list BwdStep → tactic (list BwdStep))
: tactic (bool × string × list (string × string × list string) × list BwdStep) := do
  decl_nm ← parse_decl nm,
  (b,exc,steps) ← (tactic.capture' $ bwd_steps_runner decl_nm steps) >>= tactic.unwrap',
  if ¬ b then
    pure (b, exc, [], [])
  else do
      (b',exc',proof) ← (tactic.capture' $ steps_to_proof steps) >>= tactic.unwrap',
      pure (b', exc', proof, steps)


/--
API: (valid_proof: bool, exc: string, proof: list (string × string × list string), steps: list BwdStep)
-/
meta def check_and_get_proof
(nm : string) 
(actions : list string)
: tactic (bool × string × list (string × string × list string) ×  list BwdStep) := do
  decl_nm ← parse_decl nm,
  (b,exc,steps) ← (tactic.capture' $ run_bwd_steps decl_nm actions) >>= tactic.unwrap',
  if ¬ b then
    pure (b, exc, [], [])
  else do
    (b',exc',proof) ← (tactic.capture' $ steps_to_proof steps) >>= tactic.unwrap',
    pure (b', exc', proof, steps)
