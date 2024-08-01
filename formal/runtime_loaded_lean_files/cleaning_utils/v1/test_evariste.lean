import all

open_locale big_operators
open_locale euclidean_geometry
open_locale nat
open_locale real
open_locale topological_space
open_locale asymptotics


theorem EVARISTE_test_1 {p q : Prop} (h : p ∧ q) : q ∧ p :=
begin
  have hp : p,
  from and.left h,
  have hq,
  from and.right h,
  exact and.intro hq hp,
end
