import data.real.basic
import tactic.linarith
import system.io
import data.list

open tactic nat io list

meta def gather_and_apply {α} (some_l : list expr) (n: ℕ) (tac: tactic α) : tactic α :=
  let (head, tail) := split_at n some_l in
  if list.length head < n then
    do fail "Not enough goals left"
  else
    do
      set_goals head,
      a ← tac,
      gs ← get_goals,
      match gs with
        | []      := 
          do
            set_goals (append gs tail),
            pure a
        | l := fail "solven tactic failed. Remaining subgoals"
      end

namespace tactic.interactive
open lean.parser interactive

meta def solven (n : parse small_nat) (tac : itactic) : tactic unit :=
do gs ← get_goals,
   gather_and_apply gs n tac

end tactic.interactive
