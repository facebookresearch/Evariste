/- This is a staging area for code which will be inserted
into a Lean file. 

The code to be inserted is between the line comments
`PR BEGIN MODIFICATION` and `PR END MODIFICATION`

It will be inserted by `insert_proof_recording_code.py`.

Insert info:
  - file: `_target/lean/library/init/meta/converter/interactive.lean`
  - location: after the itactic definition

Most of this code is carefully written, but
any code labeled "BEGIN/END CUSTOMIZABLE CODE"
encourages customization to change what
is being recorded
-/

prelude
import init.meta.interactive init.meta.converter.conv
import .interactive_base_modifications

namespace conv
namespace interactive

--PR BEGIN MODIFICATION

@[reducible] meta def pr.recorded_itactic /-(tactic_name: string) (arg_num : nat)-/ : lean.parser (conv unit) := 
lean.parser.val $ interactive.pr.record /-tactic_name arg_num "itactic"-/ lean.parser.itactic_reflected

--PR END MODIFICATION

end interactive
end conv