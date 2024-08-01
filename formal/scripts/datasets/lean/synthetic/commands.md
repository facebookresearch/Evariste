# Generate synthetic data

## Run generation pipeline
using sweep `scripts/datasets/lean/synthetic/first_gen.json`

## Stitch data
```
mkdir /datasets/synthetic/v2
python -m scripts.datasets.lean.synthetic.maxi_stitch generated_data_dir /datasets/synthetic/v2/all.csv
```

## Split data and tokenize it
```
python -m scripts.datasets.lean.synthetic.split_csv_lean --path /datasets/synthetic/v2/all.csv --n_chunks 8 --chunk_id 0 &
python -m scripts.datasets.lean.synthetic.split_csv_lean --path /datasets/synthetic/v2/all.csv --n_chunks 8 --chunk_id 1 &
python -m scripts.datasets.lean.synthetic.split_csv_lean --path /datasets/synthetic/v2/all.csv --n_chunks 8 --chunk_id 2 &
python -m scripts.datasets.lean.synthetic.split_csv_lean --path /datasets/synthetic/v2/all.csv --n_chunks 8 --chunk_id 3 &
python -m scripts.datasets.lean.synthetic.split_csv_lean --path /datasets/synthetic/v2/all.csv --n_chunks 8 --chunk_id 4 &
python -m scripts.datasets.lean.synthetic.split_csv_lean --path /datasets/synthetic/v2/all.csv --n_chunks 8 --chunk_id 5 &
python -m scripts.datasets.lean.synthetic.split_csv_lean --path /datasets/synthetic/v2/all.csv --n_chunks 8 --chunk_id 6 &
python -m scripts.datasets.lean.synthetic.split_csv_lean --path /datasets/synthetic/v2/all.csv --n_chunks 8 --chunk_id 7 &
```
