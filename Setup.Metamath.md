# Metamath

## Retrieve splits from Holophrasm

https://github.com/dwhalen/holophrasm/blob/24642f175b2361ba7047227df1c91f23fa9175a0/data_utils5.py#L139-L153

```python
list_of_propositions = database.propositions_list[:]
np.random.seed(seed=121451345)
list_of_propositions = np.random.permutation(list_of_propositions)

num_validation = len(list_of_propositions) // 10
num_test = num_validation
num_training = len(list_of_propositions) - num_test - num_validation

training_propositions = list_of_propositions[:num_training]
training_propositions = [_ for _ in training_propositions if _.type == 'p']
validation_propositions = list_of_propositions[num_training:num_training + num_validation]
validation_propositions = [_ for _ in validation_propositions if _.type == 'p']
test_propositions = list_of_propositions[num_training + num_validation:]
test_propositions = [_ for _ in test_propositions if _.type == 'p']

print(len(training_propositions))    # 21788
print(len(validation_propositions))  # 2712
print(len(test_propositions))        # 2720

with open('resources/metamath/DATASET_HOLOPHRASM/split.train', 'w') as f:
    f.write('\n'.join(p.label for p in training_propositions) + '\n')

with open('resources/metamath/DATASET_HOLOPHRASM/split.valid', 'w') as f:
    f.write('\n'.join(p.label for p in validation_propositions) + '\n')

with open('resources/metamath/DATASET_HOLOPHRASM/split.test', 'w') as f:
    f.write('\n'.join(p.label for p in test_propositions) + '\n')
```

## Create dataset

```bash
cd resources/metamath
git clone https://github.com/metamath/set.mm.git

mkdir resources/metamath/DATASET

## New split (5b748471ba493a7f38207da0033006f5716dc4b5 - Fri May 22 21:42:32 2020 -0400)
## Easy split, valid & test are easy theorems not used anywhere.

python -m scripts.datasets.metamath.create_dataset \
--console_log_level info --api_log_level info \
--dataset_dir resources/metamath/DATASET \
--database_path resources/metamath/set.mm/set.mm \
--build_splits True --data proof_trees

python -m scripts.datasets.metamath.create_dataset \
--console_log_level info --api_log_level info \
--dataset_dir resources/metamath/DATASET \
--database_path resources/metamath/set.mm/set.mm \
--build_splits True --data compressed_steps

## Holophrasm split
## Random split

python -m scripts.datasets.metamath.create_dataset \
--console_log_level info --api_log_level info \
--dataset_dir resources/metamath/DATASET_HOLOPHRASM \
--database_path resources/metamath/holophrasm/set.mm \
--build_splits False --data proof_trees

python -m scripts.datasets.metamath.create_dataset \
--console_log_level info --api_log_level info \
--dataset_dir resources/metamath/DATASET_HOLOPHRASM \
--database_path resources/metamath/holophrasm/set.mm \
--build_splits False --data compressed_steps

## New split 2 (861bd3552636dcdb9cbc8df59d01b14520c72f82 - Tue Oct 13 09:27:50 2020 -0400)
## Random split

python -m scripts.datasets.metamath.create_dataset \
--console_log_level info --api_log_level info \
--dataset_dir resources/metamath/DATASET_2 \
--database_path resources/metamath/set.mm_2/set.mm \
--valid_never_used False \
--build_splits True --data proof_trees --valid_test_size 1000

python -m scripts.datasets.metamath.create_dataset \
--console_log_level info --api_log_level info \
--dataset_dir resources/metamath/DATASET_2 \
--database_path resources/metamath/set.mm_2/set.mm \
--build_splits True --data compressed_steps

## New split 3 (861bd3552636dcdb9cbc8df59d01b14520c72f82 - Tue Oct 13 09:27:50 2020 -0400)
## Ensure that valid & test only contains unused theorems

python -m scripts.datasets.metamath.create_dataset \
--console_log_level info --api_log_level info \
--dataset_dir resources/metamath/DATASET_3 \
--database_path resources/metamath/set.mm_2/set.mm \
--valid_never_used True --valid_test_size 1000 \
--build_splits True --valid_no_duplicates True --data proof_trees --valid_test_size 1000 \
--remove_syntactic True

python -m scripts.datasets.metamath.create_dataset \
--console_log_level info --api_log_level info \
--dataset_dir resources/metamath/DATASET_3 \
--database_path resources/metamath/set.mm_2/set.mm \
--valid_never_used True --valid_test_size 1000 \
--build_splits True --valid_no_duplicates True --data compressed_steps
```



## Informal dataset

The following script will extract informal descriptions associated with Metamath theorems, and convert them into LaTeX. It will generate a compilable LaTeX file, `metamath_informal.tex`, that you can compile into PDF.

```bash
python -m scripts.datasets.metamath.informal  # create metamath_informal.tex
pdflatex metamath_informal.tex                # create metamath_informal.pdf
```

## syntax_gen_100k split

```
python -m scripts.datasets.metamath.create_dataset \
--console_log_level info --api_log_level info \
--dataset_dir resources/metamath/syntax_gen_100k/DATASET \
--database_path resources/metamath/syntax_gen_100k/set.mm \
--build_splits False --data proof_trees
```

## Smaller subset

For fast debugging, you can use the following script:

```bash
python -m scripts.datasets.metamath.create_subset --dataset_dir resources/metamath/DATASET_HOLOPHRASM --subset_size 100
```

This will create a subset of the full dataset in `resources/metamath/DATASET_HOLOPHRASM/100`. Providing this directory to `--mm_data_dir' will start experiments very quickly (without loading the 2GB+ proof_trees.pkl file).


## Compression test

Investigating a good trade-off number of stored subproofs / generation length.

```bash
##### --min_stat_counts 1 --min_proof_len 2 -> stores everything
##### results in more stored proofs, and equally long compressions
python -m src.envs.metamath_stat_history --n_labels 5000 --min_stat_counts 1 --min_proof_len 2

        ===== 4997 Compressing elcnv2 (106 tokens) ...
        Reference: 7 subproofs. Proof length: 50 (57 including -1 indexing tokens).
        S-History: 20 subproofs. Proof length: 50
        ===== 4998 Compressing nfcnv (43 tokens) ...
        Reference: 3 subproofs. Proof length: 34 (37 including -1 indexing tokens).
        S-History: 11 subproofs. Proof length: 34
        ===== 4999 Compressing opelcnvg (78 tokens) ...
        Reference: 4 subproofs. Proof length: 68 (72 including -1 indexing tokens).
        S-History: 21 subproofs. Proof length: 68
        ===== 5000 Compressing brcnvg (43 tokens) ...
        Reference: 1 subproofs. Proof length: 41 (42 including -1 indexing tokens).
        S-History: 14 subproofs. Proof length: 41
        ===== Summary =====
        Average proof length ratio: 1.000 (+/- 0.000)
        Average subproofs ratio: 4.120 (+/- 2.149)
        Ref compressed tokens: 193916
        New compressed tokens: 193916
        Ref subproofs: 17907
        New subproofs: 63385

##### --min_stat_counts 3 --min_proof_len 2 -> only nodes that appeared 3 times
##### results in less stored proofs, but longer compressions
python -m src.envs.metamath_stat_history --n_labels 5000 --min_stat_counts 3 --min_proof_len 2

        ===== 4997 Compressing elcnv2 (106 tokens) ...
        Reference: 7 subproofs. Proof length: 50 (57 including -1 indexing tokens).
        S-History: 7 subproofs. Proof length: 80
        ===== 4998 Compressing nfcnv (43 tokens) ...
        Reference: 3 subproofs. Proof length: 34 (37 including -1 indexing tokens).
        S-History: 2 subproofs. Proof length: 41
        ===== 4999 Compressing opelcnvg (78 tokens) ...
        Reference: 4 subproofs. Proof length: 68 (72 including -1 indexing tokens).
        S-History: 2 subproofs. Proof length: 76
        ===== 5000 Compressing brcnvg (43 tokens) ...
        Reference: 1 subproofs. Proof length: 41 (42 including -1 indexing tokens).
        S-History: 1 subproofs. Proof length: 43
        ===== Summary =====
        Average proof length ratio: 1.184 (+/- 0.211)
        Average subproofs ratio: 0.617 (+/- 0.421)
        Ref compressed tokens: 193916
        New compressed tokens: 259828
        Ref subproofs: 17907
        New subproofs: 13322
```

## Notes

### Cycles in theorems

```
G: A very small number of theorems, (e.g. onfrALTlem2, a9e2nd, e233, trsspwALT2, etc.) rely on the "idi" theorem, which proves a statement by assuming the statement itself (not sure why this theorem exists) and this creates infinite loops, and "RecursionError: maximum recursion depth". These theorems create issues when computing the depth or the number of nodes with `set_nodes_and_depth`, but also for the `random_topological_sort` function.

Examples of problematic theorems (Holophrasm split): a9e2nd, a9e2ndVD, e233, onfrALTlem2, sspwimpALT, sspwimpcf, sspwtrALT, sstrALT2, suctrALT3, suctrALTcf, trsspwALT2, undif3VD, uun0.1

We ignore these theorems when creating proof trees in `create_dataset.py` using the `has_cycle` function.
```
