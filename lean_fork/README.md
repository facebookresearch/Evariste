# FB only: Updating master using upstream

```
git remote add public git@github.com:facebookresearch/lean_ml_tooling.git
git pull --no-recurse-submodules --rebase=i public master # DANGER: rewrite master history
```

Then push force on master (!)
Warning: if submodules have been updated, make sure the --no-recuse-submodules didnt mess this up.
If this is a problem, just keep lean_ml_tooling submodules up to date.
TODO: find a better workflow?

# FB only :

- Added checkpoint-process as a submodule, link statically 
Most fb specific changes are there to accomodate the checkpoint creation which doesn't work well with ongoing threads. Thus :
- Lean changes :
    - lean/src/util/log_tree.* : added clear_listeners
    - lean/src/util/task.cpp : allow task_queue reset (safe if we've cleared it and nothing is running)
- ML Server changes :
    - setup / clear handlers

## Set-up
Initializing the submodules, patching and building lean as well as the ml_server can all be done by running 
```
/bin/bash scripts/build.sh
```

Once this is done, we can build the `leanml` package and test it.
```
pip install -e .
python test_lean_api.py
```

For a more involved example showing how to handle async requests, run all tactic steps in the [lean steps dataset](https://github.com/jesse-michael-han/lean-step-public) with :
```
python test_lean_api_async /path/to/lean_steps.csv
```
On a server with 80 cpus, this takes ~1h30min to build the entire mathlib, then 10min to run all tactics in the dataset.
The server will fail to prove declarations using mixed tactic / term proofs, private declarations or declarations that are not theorems.

## Acknowledgements
None of this would have been possible without patient help from Gabriel Ebner.

## License
lean_ml_tooling is APACHE 2 licensed, as found in the LICENSE file.
