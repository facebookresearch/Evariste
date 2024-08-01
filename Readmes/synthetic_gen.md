### Generating statements

**
NOTE: LOADING CHANGED
split files are not used anymore, now, all we need is a `{split}.lean` file, where all declarations are unique and wrapped in a `namespace {split} // end {split}`. 
This file needs to be located in the folder pointed to by `dataset.statement_splits_path`.
If we need this property to be a comma separated list of folder, don't hesitate to change this !

For example, we could have
```
dataset.statement_splits_path = '/path/to/basic,/path/to/generated'
```

Proof cleaner might complain that your new split doesn't appear.
Add it to the imports in cleaning_utils/v1/do_clean_proof.lean (this doesn't mean that everyone will have to use your statement_splits_path. an import is a noop if the file isn't found in lean path roots)
**

Here, we're only looking to extract lean statements, not the proofs, so no "extraction" is needed.
This command : 
```bash
python maxi_gen.py --random_gen lean_gen_rwalk_real --random_gen.filter_hard 0 --random_gen.simplified 0 --random_gen.no_proof 1 --run_extraction 0 --exp_name test --n_jobs 2 --get_all_exercises 1 --random_gen.n_generations 2
```
- `exp_name` will be the split_name. Make it unique.
- `--get_all_exercises 1` will collate all declarations and create the proper files for loading.

Files `DATA_DIR/split.{split_name}` and `/path/to/others/{split_name}.lean` will be created.
Finally, add your split_name to `evariste/envs/lean/splits.py`, and test that everything can be loaded using `scripts.tim.lean.check_load_minif2f`

Or use a yaml sweep like this
```yaml
random_gen: [lean_gen_rwalk_real]
random_gen.filter_hard: [1]
random_gen.simplified: [1]

# generate exercises only
get_all_exercises: [1]
randomgen.no_proof: [1]
run_extraction: [0]

# 200 exercises
n_jobs: [20]
random_gen.n_generations: [10]
```
and launch on 1 CPU.


### Generating train data
- random_gen.no_proof must be set to 0
- run_extraction must be set to 1
- get_all_train must be set to 1
- n_shard is the number of shards the csv will be split into.
```bash
python maxi_gen.py --random_gen lean_gen_rwalk_real --random_gen.filter_hard 0 --random_gen.simplified 0 --run_extraction 1 --exp_name test --n_jobs 2 --get_all_train 1 --random_gen.n_generations 2
```

Finally add your new split name in `new_generated_data` in `model/data/envs/lean.py`:
```python
    new_generated_data: List[str] = [
        # Add here split names of recently (after 6/7/22) generated with maxi_gen.py
        ...,
        'test'  # == --exp_name
    ]
```

Test your training data with:
```bash
python train.py --cfg bwd_lean_debug --tasks "lean_{split_name}_statement--tactic-EOU_seq2seq"
```