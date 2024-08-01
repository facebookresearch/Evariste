import all
import minif2f_test
import minif2f_test_false
import minif2f_valid
import minif2f_valid_false
import oai_curriculum
import oai_curriculum_false
import annotations_v1
import annotations_v1_false
import v1.proof_running
import v1.individual_tactic_cleaning
import v1.useless_tactics_removal
import v1.test_evariste
import autof_codex
import ladder_synth

open_locale big_operators
open_locale euclidean_geometry
open_locale nat
open_locale real
open_locale topological_space
open_locale asymptotics


meta def dummy_clean_proof_from_expr (tgt : expr) (steps : list BwdStep) : tactic (list BwdStep) := pure steps


/--
Clean proof and return a json object
API: {
    valid_proof: bool,
    proof: list (string × string × list string)
    valid_cleaned: bool,
    cleaned_proof: list (string × string × list string)
    exc: string,
}
-/
meta def get_and_clean_proof_json
  (nm : string)
  (actions : list string)
  (clean_proof_from_expr: expr → list BwdStep → tactic (list BwdStep)) : tactic json := let actions := "repeat{intro}"::actions in do
  (valid_proof, exc, proof_with_repeatintro, steps) ← check_and_get_proof nm actions,
  proof_json ← to_tactic_json proof_with_repeatintro.tail,  -- remove `repeat{intro}`
  if ¬valid_proof then
    pure $ json.object [
      ("valid_proof", json.of_bool ff),
      ("proof", proof_json),
      ("valid_cleaned", json.of_bool ff),
      ("cleaned_proof", json.array []),
      ("exc", json.of_string exc)
    ]
  else
    do
      (valid_cleaned, exc, cleaned_proof, _) ← check_and_get_proof_from_steps nm steps $ λ nm' steps', do {
        tgt ← tactic.get_env >>= λ e, e.get nm' >>= pure ∘ declaration.type,
        clean_proof_from_expr tgt steps'
      },
      -- remove `repeat{intro}`
      let cleaned_proof :=
        if cleaned_proof.head.snd.fst = "repeat{intro}" then
          cleaned_proof.tail
        else
          cleaned_proof,
      cleaned_proof_json ← to_tactic_json cleaned_proof,
      pure $ json.object [
        ("valid_proof", json.of_bool tt),
        ("proof", proof_json),
        ("valid_cleaned", json.of_bool valid_cleaned),
        ("cleaned_proof", cleaned_proof_json),
        ("exc", json.of_string exc)
      ]


meta def do_remove_useless_tactics (nm : string) (actions : list string) : tactic unit := do
    cleaned_proof ← get_and_clean_proof_json nm actions remove_useless_tactics,
    tactic.trace "🦑",
    tactic.trace cleaned_proof


meta def do_clean_individual_tactic_at_idx (idx : ℕ) (nm : string) (actions : list string) (simp_only : bool) : tactic unit := do
    cleaned_proof ← get_and_clean_proof_json nm actions $ clean_individual_tactic_at_idx simp_only (idx + 1),  -- to account for `repeat{intro}`
    tactic.trace "🦑",
    tactic.trace cleaned_proof


meta def do_clean_all_individual_tactics (nm : string) (actions : list string) (simp_only : bool) : tactic unit := do
    cleaned_proof ← get_and_clean_proof_json nm actions $ clean_all_individual_tactics simp_only,
    tactic.trace "🦑",
    tactic.trace cleaned_proof


meta def do_get_proof (nm : string) (actions : list string) : tactic unit := let actions := "repeat{intro}"::actions in do
    (valid_proof, exc, proof_with_repeatintro, _) ← check_and_get_proof nm actions,
    proof_json ← to_tactic_json proof_with_repeatintro.tail,  -- remove `repeat{intro}`
    let result := json.object [
      ("valid_proof", json.of_bool valid_proof),
      ("exc", json.of_string exc),
      ("proof", proof_json)
    ],
    tactic.trace "🦑",
    tactic.trace result
