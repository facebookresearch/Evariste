# HOL-Light

## Initial setup

Note: running `opam update` may be necessary to find some packages. Also never forget to run `eval $(opam env)` after the switch create.

```
opam switch create hol_light 4.07.1
opam install camlp5=7.10
opam install ocamlfind
opam install num
opam install base64  # to encode HOL-Light states
opam install dune.2.8.2
opam install dune-configurator.2.8.2
opam install dune-private-libs.2.8.2

# camlp5 7.10 is working (camlp5=7.10), 7.11 is failing
# ocaml -I `camlp5 -where` camlp5o.cma
# ocaml -I /.opam/hol_light/lib/camlp5
```

## Packages summary

```
# Packages matching: installed
# Name              # Installed # Synopsis
base-bigarray       base
base-threads        base
base-unix           base
camlp5              7.10        Preprocessor-pretty-printer of OCaml
conf-m4             1           Virtual package relying on m4
ocaml               4.05.0      The OCaml compiler (virtual package)
ocaml-base-compiler 4.05.0      Official 4.05.0 release
ocaml-config        1           OCaml Switch Configuration
ocamlfind           1.8.1       A library manager for OCaml
```

In ~/.ocamlinit, add:

```
#use "topfind";;
#require "camlp5";;
```

## Install and run HOL-Light

```
cd resources
mkdir -p HOL_Light/jrh13
mv HOL_Light/jrh13

git clone https://github.com/jrh13/hol-light.git
cd hol-light
make
ocaml
#use "hol.ml";;
```

## Create checkpoints

```

#### save checkpoint
## #load "resources/HOL_Light/checkpoint/checkpoint.cma";;
#load "resources/HOL_Light/checkpoint-process/ocaml/checkpoint.cma";;
open Checkpoint;;
checkpoint_save("checkpoint-hl");;

#### load Multivariate
needs "Multivariate/make_complex.ml";;
checkpoint_save("checkpoint-hl-multivariate-complex");;

#### load flyspeck
Sys.chdir("resources/HOL_Light/flyspeck");;
needs "load_flyspeck.ml";;

#### save checkpoint
checkpoint_save("checkpoint-hl-flyspeck");;
```

## Export proofs

```
cd resources/HOL_Light/hol-light/

HOL_TRACING=yes rlwrap ocaml
#use "hol.ml";;
Trace.export "Proofsplit/hol.bin";;

HOL_TRACING=yes rlwrap ocaml
#use "hol.ml";;
needs "Multivariate/make_complex.ml";;
Trace.export "Proofsplit/Multivariate_make_complex.bin";;

######needs "resources/HOL_Light/flyspeck/load_flyspeck.ml";;
```

## Create datasets and train a model

```
# run splitted proofs
python hol_light_parse_goals.py

# create train / valid / test split and export datasets
python hol_light_create_dataset.py --run_proofs True

# train
python train.py  \
--hl_data_path "resources/HOL_Light/mpu/hol-light/DATASET" \
--max_len 2048 \
--batch_size 8 \
--epoch_size 500 \
--max_epoch 100000 \
--hl_tasks "hl_pred_tact_goal" \
--debug
```

## Data augmentation

```
N_SHARDS=200
for SHARD_ID in $(seq 0 $((N_SHARDS-1))); do
    python hol_light_augm_generate_cmds.py --run_commands 1 \
          --api_log_level debug --console_log_level info --silent_timeouts 1 \
          --n_shards $N_SHARDS --shard_id $SHARD_ID \
          --corrupt_mode list_subset --max_list_subsets 200
done

N_SHARDS=200
for SHARD_ID in $(seq 0 $((N_SHARDS-1))); do
    python hol_light_augm_generate_cmds.py --run_commands 1 \
          --api_log_level debug --console_log_level info --silent_timeouts 1 \
          --n_shards $N_SHARDS --shard_id $SHARD_ID \
          --corrupt_mode list_split
done

N_SHARDS=400
for SHARD_ID in $(seq 0 $((N_SHARDS-1))); do
    python hol_light_augm_generate_cmds.py --run_commands 1 \
          --api_log_level debug --console_log_level info --silent_timeouts 1 \
          --n_shards $N_SHARDS --shard_id $SHARD_ID \
          --corrupt_mode remove_cmds --max_remove_cmds 5 --max_diff_remove_cmds 200
done


python hol_light_augm_generate_cmds.py --run_commands 1 \
          --api_log_level debug --console_log_level info --silent_timeouts 1 \
          --n_shards 200 --shard_id 12 \
          --corrupt_mode list_subset --max_list_subsets 200

python hol_light_augm_generate_cmds.py --run_commands 1 \
          --api_log_level debug --console_log_level info --silent_timeouts 1 \
          --n_shards 400 --shard_id 172 \
          --corrupt_mode list_split --max_list_subsets 200

python hol_light_augm_generate_cmds.py --run_commands 1 \
          --api_log_level debug --console_log_level info --silent_timeouts 1 \
          --n_shards 200 --shard_id 54 \
          --corrupt_mode remove_cmds --max_list_subsets 200


python hol_light_augm_merge_dataset.py --n_shards 200 --corrupt_mode list_subset
python hol_light_augm_merge_dataset.py --n_shards 200 --corrupt_mode list_split
python hol_light_augm_merge_dataset.py --n_shards 400 --corrupt_mode remove_cmds


resources/HOL_Light/mpu/hol-light/SYNTHETIC/list_subset/list_subset.all.0200.tok
resources/HOL_Light/mpu/hol-light/SYNTHETIC/list_split/list_split.all.0200.tok
resources/HOL_Light/mpu/hol-light/SYNTHETIC/remove_cmds/remove_cmds.all.0400.tok
```

## Summary

```
### https://github.com/jrh13/hol-light
- Official repo




############### Old stuff


### https://github.com/mpu/hol-light
Initially with proof split. Can be ignored.

```


## Debug

checkpoint_save("checkpoint-hl-multivariate-complex");;

Hashtbl.iter (fun a b -> (Format.print_string(a^" "^b); Format.print_newline())) loaded_files;;
visualizer/mm_session.py
