/- This is a staging area for code which will be inserted
into a Lean file. 

The code to be inserted is between the line comments
`PR BEGIN MODIFICATION` and `PR END MODIFICATION`

It will be inserted by `insert_proof_recording_code.py`.

Insert info:
  - file: `_target/deps/lean/library/init/meta/interactive.lean`
  - location: after the itactic definition

Most of this code is carefully written, but
any code labeled "BEGIN/END CUSTOMIZABLE CODE"
encourages customization to change what
is being recorded
-/

prelude
import init.meta.tactic init.meta.type_context init.meta.rewrite_tactic init.meta.simp_tactic
import init.meta.smt.congruence_closure init.control.combinators
import init.meta.interactive_base init.meta.derive init.meta.match_tactic
import init.meta.congr_tactic init.meta.case_tag
import .interactive_base_modifications

namespace tactic
namespace interactive

--PR BEGIN MODIFICATION

@[reducible] meta def pr.recorded_itactic /-(tactic_name: string) (arg_num : nat)-/ : lean.parser (tactic unit) := 
lean.parser.val $ interactive.pr.record /-tactic_name arg_num "itactic"-/ lean.parser.itactic_reflected

--PR END MODIFICATION

end interactive
end tactic