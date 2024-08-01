import data.real.basic
import data.real.pi
import analysis.normed_space.pi_Lp
open_locale real

namespace imo_th
universe u
variable {α : Type u}

theorem CS_2 [ordered_semiring α] (a b c d: α) : (a^2+b^2) *(c^2+d^2) ≥ (a*c+b*d)^2 := sorry
theorem CS_3 [ordered_semiring α] (a b c d e f: α) : (a^2+b^2+c^2) *(d^2+e^2+f^2) ≥ (a*d+b*e+c*f)^2 := sorry
theorem CS_4 [ordered_semiring α] (a b c d e f g h: α) : (a^2+b^2+c^2+d^2) *(e^2+f^2+g^2+h^2) ≥ (a*e+b*f+c*g+d*h)^2 := sorry
theorem CS_5 [ordered_semiring α] (a b c d e f g h i j: α) : (a^2+b^2+c^2+d^2+e^2) * (f^2+g^2+h^2+i^2+j^2) ≥ (a*f+b*g+c*h+d*i+e*j)^2 := sorry


theorem AM_QM_2 (a b : ℝ) : real.sqrt((a^2 + b^2) / (2:ℝ)) ≥ (a + b) / (2:ℝ) := sorry
theorem AM_QM_3 (a b c : ℝ) : real.sqrt((a^2 + b^2 + c^2) / (3:ℝ)) ≥ (a + b + c) / (3:ℝ) := sorry
theorem AM_QM_4 (a b c d : ℝ) : real.sqrt((a^2 + b^2 + c^2 + d^2) / (4:ℝ)) ≥ (a + b + c + d) / (4:ℝ) := sorry
theorem AM_QM_5 (a b c d e : ℝ) : real.sqrt((a^2 + b^2 + c^2 + d^2 + e^2) / (5:ℝ)) ≥ (a + b + c + d + e) / (5:ℝ) := sorry


theorem GM_AM_2 (a b : ℝ) (h₀ : a ≥ 0 ∧ b ≥ 0) : (a + b) / (2:ℝ) ≥ real.sqrt(a * b) := sorry
theorem GM_AM_3 (a b c : ℝ) (h₀ : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) : (a + b + c) / (3:ℝ) ≥ (a * b * c)^((1 : ℝ) / (3:ℝ)) := sorry
theorem GM_AM_4 (a b c d : ℝ) (h₀ : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) : (a + b + c + d) / (4:ℝ) ≥ (a * b * c * d)^((1 : ℝ) / (4:ℝ)) := sorry

theorem GM_AM_prod1_sumgeqn1_2 (a : ℝ) (h₀ : a > 0) : a + (1:ℝ) / a ≥ (2:ℝ) := sorry
theorem GM_AM_prod1_sumgeqn1_3
  (a b c : ℝ)
  (h₀ : a ≥ 0)
  (h₁ : b ≥ 0)
  (h₂ : c ≥ 0)
  (h₃ : a * b * c = (1:ℝ)) :
  a + b + c ≥ (3:ℝ) := sorry
theorem GM_AM_prod1_sumgeqn2_2
  (a b : ℝ)
  (h₀ : a > 0)
  (h₁ : b > 0) :
  a / b + b / a ≥ (2:ℝ) := sorry
theorem GM_AM_prod1_sumgeqn2_3
  (a b c : ℝ)
  (h₀ : a > 0)
  (h₁ : b > 0)
  (h₂ : c > 0) :
  a / b + b / c + c / a ≥ (3:ℝ) := sorry

theorem GM_AM_ineq1_2
  [ordered_semiring α]
  (a b : α)
  (h₀ : a ≥ 0)
  (h₁ : b ≥ 0) :
  a^3 + (2:α) * b^3 ≥ (3:α) * b^2 * a := sorry

theorem HM_GM_2
  (a b : ℝ)
  (h₀ : a > 0)
  (h₁ : b > 0) :
  real.sqrt(a * b) ≥ (2:ℝ) / ((1:ℝ) / a + (1:ℝ) / b) := sorry

theorem HM_GM_3
  (a b c : ℝ)
  (h₀ : a > 0)
  (h₁ : b > 0)
  (h₂ : c > 0) :
  (a * b * c)^((1 : ℝ) / (3:ℝ)) ≥ (3:ℝ) / ((1:ℝ) / a + (1:ℝ) / b + (1:ℝ) / c) := sorry

theorem HM_GM_4
  (a b c d : ℝ)
  (h₀ : a > 0)
  (h₁ : b > 0)
  (h₂ : c > 0)
  (h₃ : d > 0) :
  (a * b * c * d)^((1 : ℝ) / (4:ℝ)) ≥ (4:ℝ) / ((1:ℝ) / a + (1:ℝ) / b + (1:ℝ) / c + (1:ℝ) / d) := sorry


theorem Peter_Paul
  (a b c : ℝ)
  (h₀ : c > 0) :
  a^2 / ((2:ℝ) * c) + c * b^2 / (2:ℝ) ≥ a * b := sorry

theorem AM_minus_GM_increase_2
  (a b c : ℝ)
  (h₀ : a ≥ 0)
  (h₁ : b ≥ 0)
  (h₂ : c ≥ 0) :
  (2:ℝ) * ((a + b) / (2:ℝ) - (a * b)^((1 : ℝ) / 2)) ≤ (3:ℝ) * ((a + b + c) / (3:ℝ) - (a * b * c)^((1 : ℝ) / (3:ℝ))) := sorry

theorem increase1_2
  [ordered_ring α]
  (a b : α)
  (h₀ : a ≥ 0)
  (h₁ : b ≥ 0)
  (h₂ : a ≤ b) :
  b^2 ≥ a * ((2:α) * b - a) := sorry

theorem increase1_3
  [ordered_ring α]
  (a b c: α)
  (h₀ : a ≥ 0)
  (h₁ : b ≥ 0)
  (h₂ : c ≥ 0)
  (h₃ : a ≤ b)
  (h₄ : b ≤ c) :
  c^3 ≥ a * ((2:α) * b - a) * ((3:α) * c - (2:α) * b) := sorry

theorem rearrangement_2
  [ordered_semiring α]
  (a1 a2 b1 b2 : α)
  (h₀ : a1 ≤ a2)
  (h₁ : b1 ≤ b2) :
  a1 * b2 + a2 * b1 ≤ a1 * b1 + a2 * b2 := sorry

theorem Chebyshev_min_2
  [ordered_semiring α]
  (a1 a2 b1 b2 : α)
  (h₀ : a1 ≤ a2)
  (h₁ : b1 ≤ b2) :
  (a1 * b2 + a2 * b1) * (2:α) ≤ (a1 + a2) * (b1 + b2) := sorry

theorem Chebyshev_max_2
  [ordered_semiring α]
  (a1 a2 b1 b2 : α)
  (h₀ : a1 ≤ a2)
  (h₁ : b1 ≤ b2) :
  (a1 + a2) * (b1 + b2) ≤ (a1 * b1 + a2 * b2) * (2:α) := sorry

theorem Chebyshev_min_3
  [ordered_semiring α]
  (a1 a2 a3 b1 b2 b3 : α)
  (h₀ : a1 ≤ a2)
  (h₁ : a2 ≤ a3)
  (h₂ : b1 ≤ b2)
  (h₃ : b2 ≤ b3) :
  (a1 * b3 + a2 * b2 + a3 * b1) * (3:α) ≤ (a1 + a2 + a3) * (b1 + b2 + b3) := sorry

theorem Chebyshev_max_3
  [ordered_semiring α]
  (a1 a2 a3 b1 b2 b3 : α)
  (h₀ : a1 ≤ a2)
  (h₁ : a2 ≤ a3)
  (h₂ : b1 ≤ b2)
  (h₃ : b2 ≤ b3) :
  (a1 + a2 + a3) * (b1 + b2 + b3) ≤ (a1 * b1 + a2 * b2 + a3 * b3) * (3:α) := sorry

theorem Young
  (a b p q : ℝ)
  (h₀ : a ≥ 0)
  (h₁ : b ≥ 0)
  (h₂ : p > 0)
  (h₃ : q > 0)
  (h₁ : (1:ℝ) / p + (1:ℝ) / q = 1) :
  a * b ≤ a^p / p + b^q / q := sorry

theorem Young2
  (a b p q : ℝ)
  (h₀ : a > 0)
  (h₁ : b > 0)
  (h₂ : p > 0)
  (h₃ : q > 0)
  (h₁ : (1:ℝ) / p + (1:ℝ) / q = 1) :
  real.log(a) / p + real.log(b) / q ≤ real.log(a / p + b / q) := sorry

theorem Cauchy2_2
  (a1 a2 b1 b2 : ℝ)
  (h₀ : b1 > 0)
  (h₁ : b2 > 0) :
  (a1 + a2)^2 ≤ (a1^2 / b1 + a2^2 / b2) * (b1 + b2) := sorry

theorem Cauchy2_3
  (a1 a2 a3 b1 b2 b3 : ℝ)
  (h₀ : b1 > 0)
  (h₁ : b2 > 0)
  (h₂ : b3 > 0) :
  (a1 + a2 + a3)^2 ≤ (a1^2 / b1 + a2^2 / b2 + a3^2 / b3) * (b1 + b2 + b3) := sorry

theorem Cauchy3_2
  (a1 a2 b1 b2 : ℝ)
  (h₀ : a1 * b1 > 0)
  (h₁ : a2 * b2 > 0) :
  (a1 + a2)^2 ≤ (a1 / b1 + a2 / b2) * (a1 * b1 + a2 * b2) := sorry

theorem Cauchy3_3
  (a1 a2 a3 b1 b2 b3 : ℝ)
  (h₀ : a1 * b1 > 0)
  (h₁ : a2 * b2 > 0)
  (h₂ : a3 * b3 > 0) :
  (a1 + a2 + a3)^2 ≤ (a1 / b1 + a2 / b2 + a3 / b3) * (a1 * b1 + a2 * b2 + a3 * b3) := sorry

end imo_th