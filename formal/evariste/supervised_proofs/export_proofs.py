# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
python -m evariste.supervised_proofs.export_proof_from_eval YOUR_PATH
"""
import re
from pathlib import Path
from typing import Any, Dict

from evariste import json
from evariste.backward.env.lean.utils import to_lean_proof
from evariste.backward.graph import Proof
from evariste.datasets import LeanDatasetConf
from evariste.model.data.envs.lean_utils import load_first_proofs
from evariste.model_zoo import ZOO
from params import ConfStore
from evariste.supervised_proofs.common import (
    get_repo,
    make_label_to_module,
    ProofWithStatement,
    EvalMetadata,
)


class Export:
    def from_eval(self, eval_path_str: str, dataset_name: str = ""):
        export_proof_from_eval(eval_path_str, dataset_name)

    def from_train(
        self,
        train_path_str: str,
        model_name: str,
        split: str,
        dataset_name: str,
        dump_name: str = "proof_dump0",
    ):
        train_path = Path(train_path_str)
        assert train_path.exists()
        proof_dir = train_path / "prover_dumps" / "first_proofs" / f"{split}"
        make_dump(
            dataset_name,
            train_path_str,
            model_name,
            proof_dir,
            split,
            dump_name=dump_name,
        )


def export_proof_from_eval(eval_path_str: str, dataset_name: str):

    eval_path = Path(eval_path_str)
    eval_conf_path = eval_path / "eval_one_params.json"
    assert eval_conf_path.exists()

    with eval_conf_path.open("r") as fp:
        cfg: Dict[str, Any] = json.load(fp)

    split = cfg["split"]
    model_name = cfg["model_name"]

    print(cfg.keys())

    dataset_name = (
        (
            cfg["override_dataset"]
            if cfg["override_dataset"] != ""
            else ZOO.get_model(cfg["model_name"])
        )
        if dataset_name == ""
        else dataset_name
    )
    print(split)
    proof_dir = eval_path / "first_proofs" / f"{split}"
    assert proof_dir.exists(), proof_dir

    make_dump(dataset_name, eval_path_str, model_name, proof_dir, split, "proofs_dump0")


def make_dump(
    dataset_name: str,
    path: str,
    model_name: str,
    proof_dir: Path,
    split: str,
    dump_name: str,
):
    repo = get_repo()
    repo.check_not_dump_exists(split, model_name, dump_name)
    dataset = ConfStore[dataset_name]
    name = dump_name
    metadata = EvalMetadata(
        path=path, dataset_name=dataset_name, model_name=model_name, split=split,
    )
    label_to_module = make_label_to_module(dataset, split)
    proofs_with_statement = []
    for label, proof in load_first_proofs(proof_dir):
        proofs_with_statement.append(
            ProofWithStatement.from_proof(
                dataset, label_to_module, proof, label, split=split
            )
        )
    repo.make_dump(
        split=split,
        model_name=model_name,
        dump_name=name,
        proofs=proofs_with_statement,
        metadata=metadata,
    )
    tgt_dir = repo.model_dir(split, model_name)
    print("tgt_dir", tgt_dir)
    print("To commit")
    print(f"cd {repo.root}")
    print(f'git add -A && git commit -m "proofs from {model_name} {split} {name}"')


def make_label_to_proof_str(
    dataset: LeanDatasetConf,
    label_to_module: Dict[str, str],
    label_to_proof: Dict[str, Proof],
) -> Dict[str, str]:
    label_to_proof_str = {}

    for label, proof in label_to_proof.items():
        others_path = Path(dataset.checkpoint_path).parent.parent.parent
        lean_file = others_path / label_to_module[label]
        assert lean_file.exists(), f"{lean_file} not found"
        with open(lean_file, "r") as f:
            match = re.match(
                rf"""(?P<imports>(.*?))theorem {label}(\n| )(?P<hyp>(.*?)) :=\nbegin\n(?P<proof>(.*?))\nend""",
                f.read(),
                flags=re.DOTALL,
            )
            assert match is not None

        th_with_proof = f"theorem {label}\n{match.group('hyp')} :=\nbegin\n{to_lean_proof(proof).strip()}\nend"
        label_to_proof_str[label] = th_with_proof
    return label_to_proof_str


if __name__ == "__main__":
    import fire

    fire.Fire(Export())
