import common
open_locale real

/-- This file must be run in an API like folder (with access to common.lean)
    run this to get only theorem declaratio in the namespace imo_th : 
    lean imo_th.lean 3>&1 1>&2 2>&3 3>&- | grep imo_th
-/

/-- redefine this here because the one in mathlib is not tagged norm_cast  -/
@[simp, norm_cast] lemma rpow_nat_cast (x : ℝ) (n : ℕ) : x ^ (n : ℝ) = x ^ n :=
by simp only [real.rpow_def, (complex.of_real_pow _ _).symm, complex.cpow_nat_cast,
  complex.of_real_nat_cast, complex.of_real_re]


lemma sqrt_eq_rpow (x : ℝ) : real.sqrt x = x ^ (1/(2:ℝ)) :=
begin
  obtain h | h := le_or_lt 0 x,
  { rw [← mul_self_inj_of_nonneg (real.sqrt_nonneg _) (real.rpow_nonneg_of_nonneg h _), real.mul_self_sqrt h,
      ← sq, ← rpow_nat_cast, ← real.rpow_mul h],
    norm_num },
  { have : 1 / (2:ℝ) * π = π / (2:ℝ), ring,
    rw [real.sqrt_eq_zero_of_nonpos h.le, real.rpow_def_of_neg h, this, real.cos_pi_div_two, mul_zero] }
end

theorem sqrt_div_self' {x:ℝ} : real.sqrt x / x = 1 / real.sqrt x :=
by rw [←real.div_sqrt, one_div_div, real.div_sqrt]


namespace imo_th

theorem CS_2 (a b c d: ℝ) : (a^(2:ℝ)+b^(2:ℝ)) *(c^(2:ℝ)+d^(2:ℝ)) ≥ (a*c+b*d)^(2:ℝ) := begin
  have := @abs_real_inner_le_norm ((euclidean_space ℝ (fin 2))) _ ![a,b] ![c,d],
  simp [euclidean_space.norm_eq, fin.sum_univ_succ] at this,
  rw [←real.sqrt_mul,←real.sqrt_sq_eq_abs, real.sqrt_le] at this,
  simp [real.norm_eq_abs] at this,
  norm_cast,
  assumption,
  repeat {nlinarith},
end


theorem CS_3 (a b c d e f: ℝ) : (a^(2:ℝ)+b^(2:ℝ)+c^(2:ℝ)) *(d^(2:ℝ)+e^(2:ℝ)+f^(2:ℝ)) ≥ (a*d+b*e+c*f)^(2:ℝ) := begin
  have := @abs_real_inner_le_norm ((euclidean_space ℝ (fin 3))) _ ![a,b,c] ![d,e,f],
  simp [euclidean_space.norm_eq, fin.sum_univ_succ] at this,
  rw [←real.sqrt_mul,←real.sqrt_sq_eq_abs, real.sqrt_le] at this,
  simp [real.norm_eq_abs] at this,
  {
    have k : a * d + (b * e + c * f) = a*d+b*e+c*f, by ring,
    rw k at this, clear k,
    have k : (a ^ 2 + (b ^ 2 + c ^ 2)) * (d ^ 2 + (e ^ 2 + f ^ 2)) = (a^2+b^2+c^2) *(d^2+e^2+f^2), by ring,
    rw k at this, clear k,
    norm_cast,
    assumption,
  },
  repeat {nlinarith},
end


theorem CS_4 (a b c d e f g h: ℝ) : (a^(2:ℝ)+b^(2:ℝ)+c^(2:ℝ)+d^(2:ℝ)) *(e^(2:ℝ)+f^(2:ℝ)+g^(2:ℝ)+h^(2:ℝ)) ≥ (a*e+b*f+c*g+d*h)^(2:ℝ) := begin
  have := @abs_real_inner_le_norm ((euclidean_space ℝ (fin 4))) _ ![a,b,c,d] ![e,f,g,h],
  simp [euclidean_space.norm_eq, fin.sum_univ_succ] at this,
  rw [←real.sqrt_mul,←real.sqrt_sq_eq_abs, real.sqrt_le] at this,
  simp [real.norm_eq_abs] at this,
  {
    have k : a * e + (b * f + (c * g + d * h)) = a*e+b*f+c*g+d*h, by ring,
    rw k at this, clear k,
    have k : (a ^ 2 + (b ^ 2 + (c ^ 2 + d ^ 2))) * (e ^ 2 + (f ^ 2 + (g ^ 2 + h ^ 2))) = (a^2+b^2+c^2+d^2) * (e^2+f^2+g^2+h^2), by ring,
    rw k at this, clear k,
    norm_cast,
    assumption,
  },
  repeat {nlinarith},
end


theorem CS_5 (a b c d e f g h i j: ℝ) : (a^(2:ℝ)+b^(2:ℝ)+c^(2:ℝ)+d^(2:ℝ)+e^(2:ℝ)) * (f^(2:ℝ)+g^(2:ℝ)+h^(2:ℝ)+i^(2:ℝ)+j^(2:ℝ)) ≥ (a*f+b*g+c*h+d*i+e*j)^(2:ℝ) := begin
  have := @abs_real_inner_le_norm ((euclidean_space ℝ (fin 5))) _ ![a,b,c,d,e] ![f,g,h,i,j],
  simp [euclidean_space.norm_eq, fin.sum_univ_succ] at this,
  rw [←real.sqrt_mul,←real.sqrt_sq_eq_abs, real.sqrt_le] at this,
  simp [real.norm_eq_abs] at this,
  {

    have k : a * f + (b * g + (c * h + (d * i + e * j))) = a*f+b*g+c*h+d*i+e*j, by ring,
    rw k at this, clear k,
    have k : (a ^ 2 + (b ^ 2 + (c ^ 2 + (d ^ 2 + e ^ 2)))) * (f ^ 2 + (g ^ 2 + (h ^ 2 + (i ^ 2 + j ^ 2)))) = (a^2+b^2+c^2+d^2+e^2) *(f^2+g^2+h^2+i^2+j^2), by ring,
    rw k at this, clear k,
    norm_cast,
    assumption,
  },
  repeat {nlinarith},
end


theorem AM_QM_2
  (a b : ℝ) :
  real.sqrt((a^(2:ℝ) + b^(2:ℝ)) / (2:ℝ)) ≥ (a + b) / (2:ℝ) :=
begin
  have := CS_2 a b 1 1,
  simp at this,
  norm_num at this,
  norm_cast at this,
  rw [←real.sqrt_le, real.sqrt_sq_eq_abs] at this,
  have key := le_of_abs_le this,
  have t_le_t : (2:ℝ) ≤ (2:ℝ), by refl,
  have t_zero : 0 < (2:ℝ), by norm_num,
  have s_nneg := real.sqrt_nonneg ((a ^ 2 + b ^ 2) * (2:ℝ)),
  have multiplied := div_le_div s_nneg key t_zero t_le_t,
  norm_cast at *,
  have : real.sqrt ((a ^ 2 + b ^ 2) * 2) / 2 = real.sqrt ((a ^ 2 + b ^ 2)  / 2),
  {
    have hab : 0 ≤ (a ^ 2 + b ^ 2) := by nlinarith,
    field_simp [real.sqrt_div hab, real.div_sqrt, mul_assoc, real.sqrt_mul],
  },
  rw this at multiplied,
  exact multiplied,
  nlinarith,
end

theorem AM_QM_3
  (a b c : ℝ) :
  real.sqrt((a^(2:ℝ) + b^(2:ℝ) + c^(2:ℝ)) / (3:ℝ)) ≥ (a + b + c) / (3:ℝ) :=
begin
  have := CS_3 a b c 1 1 1,
  simp at this,
  norm_num at this,
  norm_cast at this,
  rw [←real.sqrt_le, real.sqrt_sq_eq_abs] at this,
  have key := le_of_abs_le this,
  have t_le_t : (3:ℝ) ≤ (3:ℝ), by refl,
  have t_zero : 0 < (3:ℝ), by norm_num,
  have s_nneg := real.sqrt_nonneg ((a ^ 2 + b ^ 2 + c^2) * (3:ℝ)),
  have multiplied := div_le_div s_nneg key t_zero t_le_t,
  norm_cast at *,
  have : real.sqrt ((a ^ 2 + b ^ 2 + c^2) * 3) / 3 = real.sqrt ((a ^ 2 + b ^ 2 + c^2)  / 3),
  {
    clear key t_le_t t_zero s_nneg multiplied this,
    norm_num,
    rw [mul_div_assoc, (@sqrt_div_self' (3:ℝ)), one_div],
    field_simp,
    rw real.sqrt_div,
    nlinarith,
  },
  rw this at multiplied,
  exact multiplied,
  nlinarith,
end

theorem AM_QM_4
  (a b c d : ℝ) :
  real.sqrt((a^(2:ℝ) + b^(2:ℝ) + c^(2:ℝ) + d^(2:ℝ)) / (4:ℝ)) ≥ (a + b + c + d) / (4:ℝ) :=
begin
  have := CS_4 a b c d 1 1 1 1,
  simp at this,
  norm_num at this,
  norm_cast at this,
  rw [←real.sqrt_le, real.sqrt_sq_eq_abs] at this,
  have key := le_of_abs_le this,
  have t_le_t : (4:ℝ) ≤ (4:ℝ), by refl,
  have t_zero : 0 < (4:ℝ), by norm_num,
  have s_nneg := real.sqrt_nonneg ((a ^ 2 + b ^ 2 + c^2 + d^2) * (4:ℝ)),
  have multiplied := div_le_div s_nneg key t_zero t_le_t,
  norm_cast at *,
  have : real.sqrt ((a ^ 2 + b ^ 2 + c^2 + d^2) * 4) / 4 = real.sqrt ((a ^ 2 + b ^ 2 + c^2 + d^2)  / 4),
  {
    clear key t_le_t t_zero s_nneg multiplied this,
    norm_num,
    rw [mul_div_assoc, (@sqrt_div_self' (4:ℝ)), one_div],
    field_simp,
    rw ← real.sqrt_mul,
    ring_nf,
    nlinarith,
  },
  rw this at multiplied,
  exact multiplied,
  nlinarith,
end

theorem AM_QM_5
  (a b c d e : ℝ) :
  real.sqrt((a^(2:ℝ) + b^(2:ℝ) + c^(2:ℝ) + d^(2:ℝ) + e^(2:ℝ)) / (5:ℝ)) ≥ (a + b + c + d + e) / (5:ℝ) :=
begin
  have := CS_5 a b c d e 1 1 1 1 1,
  simp at this,
  norm_num at this,
  norm_cast at this,
  rw [←real.sqrt_le, real.sqrt_sq_eq_abs] at this,
  have key := le_of_abs_le this,
  have t_le_t : (5:ℝ) ≤ (5:ℝ), by refl,
  have t_zero : 0 < (5:ℝ), by norm_num,
  have s_nneg := real.sqrt_nonneg ((a ^ 2 + b ^ 2 + c^2 + d^2 + e^2) * (5:ℝ)),
  have multiplied := div_le_div s_nneg key t_zero t_le_t,
  norm_cast at *,
  have : real.sqrt ((a ^ 2 + b ^ 2 + c^2 + d^2 + e^2) * 5) / 5 = real.sqrt ((a ^ 2 + b ^ 2 + c^2 + d^2 + e^2)  / 5),
  {
    clear key t_le_t t_zero s_nneg multiplied this,
    norm_num,
    rw [mul_div_assoc, (@sqrt_div_self' (5:ℝ)), one_div],
    field_simp,
    rw real.sqrt_div,
    nlinarith,
  },
  rw this at multiplied,
  exact multiplied,
  nlinarith,
end


theorem GM_AM_2
  (a b : ℝ)
  (h₀ : a ≥ 0 ∧ b ≥ 0) :
  (a + b) / (2:ℝ) ≥ real.sqrt(a * b) :=
begin
  have weight : 0 ≤ 1/(2:ℝ), by norm_num,
  have sum : 1/(2:ℝ) + 1/(2:ℝ) = 1, by norm_num,
  have := real.geom_mean_le_arith_mean2_weighted weight weight h₀.left h₀.right sum,
  norm_cast at *,
  rw [←sqrt_eq_rpow, ←sqrt_eq_rpow, ←real.sqrt_mul] at this,
  convert this,
  field_simp,
  exact h₀.1,
end

theorem GM_AM_3
  (a b c : ℝ)
  (h₀ : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) :
  (a + b + c) / (3:ℝ) ≥ (a * b * c)^((1 : ℝ) / (3:ℝ)) :=
begin
  have weight : 0 ≤ 1/(3:ℝ), by norm_num,
  have sum : 1/(3:ℝ) + 1/(3:ℝ) + 1/(3:ℝ) = 1, by norm_num,
  have := real.geom_mean_le_arith_mean3_weighted weight weight weight h₀.1 h₀.right.left h₀.right.right sum,
  norm_cast at *,
  convert this,
  rw [← real.mul_rpow, ← real.mul_rpow],
  repeat {nlinarith},
end

theorem GM_AM_4
  (a b c d : ℝ)
  (h₀ : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) :
  (a + b + c + d) / (4:ℝ) ≥ (a * b * c * d)^((1 : ℝ) / (4:ℝ)) :=
begin
  have weight : 0 ≤ 1/(4:ℝ), by norm_num,
  have sum : 1/(4:ℝ) + 1/(4:ℝ) + 1/(4:ℝ) + 1/(4:ℝ) = 1, by norm_num,
  have := real.geom_mean_le_arith_mean4_weighted weight weight weight weight h₀.left h₀.right.left h₀.right.right.left h₀.right.right.right sum,
  norm_cast at *,
  convert this,
  rw [← real.mul_rpow, ← real.mul_rpow, ← real.mul_rpow],
  repeat {nlinarith},
  rw mul_assoc,
  exact mul_nonneg h₀.left (mul_nonneg h₀.right.left h₀.right.right.left),
end

end imo_th

section main
meta def mk_decl_msg (d : declaration) : tactic string := do {
    decl_type ← do {
      (format.to_string ∘ format.flatten) <$> tactic.pp d.type
    },
    let msg : json := json.object $ [
      ("decl_name", d.to_name.to_string),
      ("decl_type", (decl_type : string)),
      ("success", "SUCCESS")
    ],
    pure $ json.unparse msg
}

meta def get_decl_names : tactic (list declaration) := do 
  
  this_env ← tactic.get_env,
  pure $ list.filter (λ d, (this_env.in_current_file d.to_name)) (this_env.fold (@list.nil declaration) (λ d l, d::l))

#eval do l ← get_decl_names, l.mmap $ (λ d, do {
  s ← mk_decl_msg d,
  tactic.trace s
})

#eval get_decl_names

end main

-- end imo_th