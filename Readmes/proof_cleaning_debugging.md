### Debugging proof cleaning

Error : `substring not found: no "🦑" in `

- print what's sent to the api and repro in a script with "debug" on.
```py
from pathlib import Path

from params import ConfStore
from evariste.backward.env.lean.env import LeanExpanderEnv, DeclNotFound
from evariste.backward.goal_factory import get_labels


dataset = ConfStore["lean_v1.1"].get_materialized()
env = LeanExpanderEnv(
    dataset=dataset,
    dump_path=Path("."),
    debug=True,
)
to_run = 'do_clean_proof "aopsbook_v2_c6_p98" ["intros"]'
env.api.eval_cmd(to_run, "cleaning_utils/v0/do_clean_proof.lean", timeout=300_000)
print(env.api.recv())
```

- Running this script shows no output. Create a lean folder with cleaning_utils and its imports as roots. My `leanpkg.path` looks like 
```bash
path /datasets/lean_3.30/prebuilt/library
path /datasets/lean_3.30/prebuilt/mathlib
path ./src
path ./others
path ./cleaning_utils
```

- run the command in lean
```bash
import v0.do_clean_proof
#eval do_clean_proof "aopsbook_v2_c6_p98" ["intros"]
```

- Got import errors. Fixed them. API doesn't raise on import errors.
