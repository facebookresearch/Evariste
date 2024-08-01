import init.meta.lean.parser

--set_option pp.colors false  

namespace parser
open lean

meta def get_options : parser options :=
λ s, interaction_monad.result.success s.options s

meta def myparser : parser unit := do
p <- parser.cur_pos,
o <- get_options,
tactic.trace "Inside parser",
tactic.trace o,
tactic.trace p,
return ()

@[reducible] protected meta def my_itactic : parser (tactic unit) := parser.val (do
pos <- parser.cur_pos,
tactic.trace pos,
a <- parser.itactic_reflected,
pos <- parser.cur_pos,
tactic.trace pos,
return a)

meta def pr_parser {α : Type} (p : parser α) : parser α := do
pos <- parser.cur_pos,
tactic.trace pos,
a <- p,
pos <- parser.cur_pos,
tactic.trace pos,
return a

meta instance pr_parser.reflectable {α} (p : parser α) [p.reflectable] :
    (pr_parser p).reflectable :=
⟨pr_parser (parser.reflectable.full p)⟩

end parser

meta def tactic.interactive.my_try (t : interactive.parse parser.my_itactic) : tactic unit := tactic.try t
meta def tactic.interactive.my_apply (q : interactive.parse $ parser.pr_parser $ interactive.types.texpr) : tactic unit := 
tactic.interactive.concat_tags (do h ← tactic.i_to_expr_for_apply q, tactic.apply h)

set_option pp.all true

#check interactive.parse $ interactive.types.texpr
#check interactive.parse $ parser.pr_parser $ interactive.types.texpr
#check interactive.parse $ parser.my_itactic
#check interactive.parse $ lean.parser.itactic
#check interactive.parse $ parser.pr_parser $ lean.parser.itactic
#check @lean.parser.reflectable.cast (tactic.{0} unit) lean.parser.itactic_reflected
example : true := begin
    my_try { my_apply true.intro },
end