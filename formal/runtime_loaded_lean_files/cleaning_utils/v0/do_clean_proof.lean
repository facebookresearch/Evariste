import v0.solven
import v0.proof_cleaning
import all
import minif2f_test
import minif2f_test_false
import minif2f_valid
import minif2f_valid_false
import oai_curriculum
import oai_curriculum_false
import autof_codex
import autof_lela
import annotations
import testsplit
import annotations_v1
import annotations_v1_false
import autof_codex
import ladder_synth

open_locale big_operators
open_locale euclidean_geometry
open_locale nat
open_locale real
open_locale topological_space
open_locale asymptotics


meta def do_clean_proof (d : string) (a : list string) : tactic unit := do
    cleaned_proof ← check_and_clean_proof d a,
    tactic.trace "🦑",
    tactic.trace cleaned_proof


meta def test_run_cmd : tactic unit := do
  tactic.trace "test"
