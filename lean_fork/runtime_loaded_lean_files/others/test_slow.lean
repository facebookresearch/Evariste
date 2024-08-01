import algebra.group.type_tags

theorem p_or_q_or_r (p q r: Prop) (h: p ∨ q) : p ∨ q ∨ r :=
begin
  cases h with hp hq,
  {
    left, assumption
  },
  {
    right,left,assumption
  } 
end

theorem one_eq_zero : 1 = 0 := begin sorry end

-- induction h₁ : n with n,
theorem repeat_hyps (n : ℕ) : n = 3 := begin sorry end

theorem assumption (n : ℕ) (n > 5) : n > 5 := begin sorry end