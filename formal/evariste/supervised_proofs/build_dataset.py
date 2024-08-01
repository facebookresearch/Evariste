# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any
import yaml

from evariste import json
from evariste.supervised_proofs.common import (
    get_repo,
    SupervisedDataRepo,
    dump_as_lean_proofs,
    dump_as_json_proofs,
)


def build_dataset(dataset_name: str):
    repo = get_repo()
    dataset_dir = repo.dataset_dir() / dataset_name
    dataset_cfg_path_str = dataset_dir / "dataset_cfg.yaml"

    print("dataset_dir", dataset_dir)

    dataset_cfg_path = Path(dataset_cfg_path_str)
    assert dataset_cfg_path.exists(), dataset_cfg_path
    with dataset_cfg_path.open("r") as fp:
        dataset_cfg = yaml.safe_load(fp.read())
    print(dataset_cfg)
    check_dataset_cfg(repo, dataset_cfg)
    splits_cfgs: Dict[str, List[Tuple[str, str]]] = dataset_cfg["splits"]
    label_to_proof = {}
    for split, to_load in splits_cfgs.items():
        for model, dump in to_load:
            _, proofs = repo.load_dump(split, model, dump)
            for proof in proofs:
                if proof.label not in label_to_proof:
                    label_to_proof[proof.label] = proof
    proofs = list(label_to_proof.values())

    lean_path = dataset_dir / f"dataset.lean"
    json_path = dataset_dir / f"dataset.json"
    metadata_path = dataset_dir / f"dataset.metadata"
    assert not lean_path.exists(), f"rm {dataset_dir / 'dataset.*'}"
    assert not json_path.exists(), json_path
    assert not metadata_path.exists(), metadata_path

    dump_as_lean_proofs(lean_path, proofs=proofs)
    dump_as_json_proofs(json_path, proofs=proofs, lean_proof_path=lean_path)

    n_proofs_by_split: Dict[str, int] = defaultdict(int)
    for p in proofs:
        n_proofs_by_split[p.split] += 1

    metadata = {
        "n_proofs": len(proofs),
        "splits": list({p.split for p in proofs}),
        "n_proofs_by_split": dict(n_proofs_by_split),
        "dataset_cfg": dataset_cfg,
    }

    with metadata_path.open("w") as fp:
        json.dump(metadata, fp)

    print("To commit")
    print(f"cd {repo.root}")
    print(f'git add -A && git commit -m "dataset {dataset_dir.name}"')


def check_dataset_cfg(repo: SupervisedDataRepo, dataset_cfg: Dict[str, Any]):
    splits_cfgs: Dict[str, List[Tuple[str, str]]] = dataset_cfg["splits"]
    available_splits = repo.available_splits()
    for split, to_load in splits_cfgs.items():
        assert split in available_splits, (split, available_splits)
        for model, dump in to_load:
            assert model in repo.available_models_for_split(split), (
                model,
                repo.available_models_for_split(split),
            )
            assert dump in repo.available_dumps_for_models_and_split(
                model=model, split=split
            ), (
                dump,
                repo.available_dumps_for_models_and_split(model=model, split=split),
            )


if __name__ == "__main__":
    import fire

    fire.Fire(build_dataset)
