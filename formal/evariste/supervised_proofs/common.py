# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import getpass
import hashlib
import re
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Set, Optional, Dict, Tuple, List, Any

import pandas

from evariste import json
from evariste.backward.env.lean.graph import LeanTheorem, LeanTactic, LeanContext
from evariste.backward.env.lean.utils import to_lean_proof
from evariste.backward.graph import Proof
from evariste.datasets import LeanDatasetConf
from evariste.model.data.envs.lean_utils import post_order_traversal


regex = re.compile(
    rf"theorem (?P<label>(.*?))(\n| )(?P<hyp>(.*?)) :=\nbegin\n(?P<proof>(.*?))\nend",
    flags=re.DOTALL,
)


class Suffix(str, Enum):
    json = "json"
    lean = "lean"
    metadata = "metadata"


@dataclass
class CleaningMetadata:
    cleaner_version: str
    succeed: int
    failed: int


@dataclass
class EvalMetadata:
    path: str
    split: str
    model_name: str
    dataset_name: str
    cleaning_metadata: Optional[CleaningMetadata] = None

    @staticmethod
    def from_dict(metadata: Dict) -> "EvalMetadata":
        if isinstance(metadata.get("cleaning_metadata"), dict):
            metadata["cleaning_metadata"] = CleaningMetadata(
                **metadata["cleaning_metadata"]
            )
        return EvalMetadata(**metadata)


class SupervisedDataRepo:
    def __init__(self, root: Path):
        self.root = root

    @property
    def split_by_model_dir(self) -> Path:
        return self.root / "split_by_model"

    def split_dir(self, split: str) -> Path:
        return self.split_by_model_dir / split

    def model_dir(self, split: str, model: str) -> Path:
        return self.split_dir(split) / model

    def dump_dir(
        self, split: str, model: str, dump_name: str, suffix: Suffix = Suffix.json
    ) -> Path:
        return self.model_dir(split, model) / f"{dump_name}.{suffix}"

    def available_splits(self) -> Set[str]:
        return {p.name for p in self.split_by_model_dir.iterdir()}

    def available_models_for_split(self, split: str) -> Set[str]:
        return {p.name for p in self.split_dir(split).iterdir()}

    def available_dumps_for_models_and_split(self, model: str, split: str) -> Set[str]:
        return {p.stem for p in self.model_dir(split, model).iterdir()}

    def load_dump(
        self, split: str, model: str, dump_name: str
    ) -> Tuple[EvalMetadata, List["ProofWithStatement"]]:
        data_dir = self.model_dir(split, model)
        meta_dict, proofs = load_dataset(data_dir, dataset_name=dump_name)
        return EvalMetadata.from_dict(meta_dict), proofs

    def check_not_dump_exists(self, split: str, model_name: str, dump_name: str):
        for suffix in Suffix:
            assert isinstance(suffix, Suffix)
            p = self.dump_dir(
                split, model=model_name, dump_name=dump_name, suffix=suffix
            )
            if p.exists():
                raise RuntimeError(
                    f"{p} exists, to delete\nrm {p.parent / f'{dump_name}.*'}"
                )

    def make_dump(
        self,
        split: str,
        model_name: str,
        dump_name: str,
        proofs: List["ProofWithStatement"],
        metadata: EvalMetadata,
    ):
        tgt_dir = self.model_dir(split, model_name)
        tgt_dir.mkdir(exist_ok=True, parents=True)

        metadata_path = self.dump_dir(
            split, model=model_name, dump_name=dump_name, suffix=Suffix.metadata
        )
        lean_path = self.dump_dir(
            split, model=model_name, dump_name=dump_name, suffix=Suffix.lean
        )
        json_path = self.dump_dir(
            split, model=model_name, dump_name=dump_name, suffix=Suffix.json
        )

        assert not metadata_path.exists(), tgt_dir
        assert not lean_path.exists(), tgt_dir
        assert not json_path.exists(), tgt_dir

        with metadata_path.open("w") as fp:
            json.dump(asdict(metadata), fp, indent=2)

        dump_as_lean_proofs(lean_path, proofs=proofs)
        dump_as_json_proofs(json_path, proofs=proofs, lean_proof_path=lean_path)

    def dataset_dir(self):
        return self.root / "datasets"


@dataclass
class ProofWithStatement:
    label: str
    statement: str
    proof: Proof
    split: str

    @staticmethod
    def from_proof(
        dataset: LeanDatasetConf,
        label_to_module: Dict[str, str],
        proof: Proof,
        label: str,
        split: str,
    ):
        others_path = Path(dataset.checkpoint_path).parent.parent.parent
        lean_file = others_path / label_to_module[label]
        assert lean_file.exists(), f"{lean_file} not found"
        with open(lean_file, "r") as f:
            match = re.match(
                rf"(?P<imports>(.*?))theorem {label}(\n| )(?P<hyp>(.*?)):=\nbegin\n(?P<proof>(.*?))\nend",
                f.read(),
                flags=re.DOTALL,
            )
            assert match is not None
            assert "theorem" not in match.group("hyp")
            assert "sorry" not in match.group("hyp")
            assert match is not None
        return ProofWithStatement(
            label=label, statement=match.group("hyp"), proof=proof, split=split
        )

    def proof_str(self) -> str:
        return f"theorem {self.label}\n{self.statement} :=\nbegin\n{to_lean_proof(self.proof).strip()}\nend"

    def to_dict(self) -> Dict[str, Any]:
        steps = proof_to_flatten_proof_steps(self.proof)
        dict_steps = [asdict(s) for s in steps]
        return {
            "label": self.label,
            "split": self.split,
            "statement": self.statement,
            "n_proof_steps": len(dict_steps),
            "proof_steps": dict_steps,
        }

    @staticmethod
    def from_dict(proof_dict: Dict[str, Any]) -> "ProofWithStatement":
        label = proof_dict["label"]
        statement = proof_dict["statement"]
        split = proof_dict["split"]
        proof = proof_from_proof_steps(
            [ProofStep(**entry) for entry in proof_dict["proof_steps"]]
        )
        return ProofWithStatement(
            label=label, statement=statement, proof=proof, split=split
        )


def get_repo(repo_path_str: Optional[str] = None) -> SupervisedDataRepo:
    if repo_path_str is not None:
        repo = Path(repo_path_str)
    else:
        repo = Path("DEFAULT_REPO_PATH")
    assert repo.exists()
    return SupervisedDataRepo(repo)


@dataclass
class ProofStep:
    id: int
    theorem: str
    tactic: str
    children_ids: List[int]


def proof_to_flatten_proof_steps(proof: Proof) -> List[ProofStep]:
    flatten_proof = post_order_traversal(proof)
    n_steps = len(flatten_proof)
    offset = n_steps - 1
    steps: List[ProofStep] = []
    for i, (thm, tac, children) in enumerate(reversed(flatten_proof)):
        assert isinstance(thm, LeanTheorem)
        assert isinstance(tac, LeanTactic)
        assert tac.is_valid
        step = ProofStep(
            id=i,
            theorem=thm.conclusion,
            tactic=tac.to_dict()["str"],
            children_ids=[offset - cid for cid in children],
        )
        steps.append(step)
    return steps


def proof_from_proof_steps(steps: List[ProofStep]) -> Proof:
    id2proof: Dict[int, Proof] = {}
    for step in reversed(steps):
        proof = (
            LeanTheorem(
                conclusion=step.theorem,
                context=LeanContext(namespaces=set()),
                state=None,
            ),
            LeanTactic(step.tactic),
            [id2proof[cid] for cid in step.children_ids],
        )
        id2proof[step.id] = proof
    return id2proof[0]


SPLIT_TO_CSV = {
    "oai_curriculum": "oai_curriculum.csv",
    "annotations": "annotations.csv",
    # "data.csv",
    # "minif2f.csv",
    # "annotations.csv",
    # "annotations_false.csv",
    # "oai_curriculum.csv",
}


def make_label_to_module(dataset: LeanDatasetConf, split: str) -> Dict[str, str]:
    assert isinstance(dataset, LeanDatasetConf)
    csv_name = SPLIT_TO_CSV[split]
    csv_path = Path(dataset.data_dir) / csv_name
    assert csv_path.exists()
    loaded_data = pandas.read_csv(csv_path)
    print(f"Loaded {len(loaded_data)} rows from {csv_path}")
    label_to_module = {}
    for row in loaded_data.iloc:
        label_to_module[row.decl_name] = row.filename
    return label_to_module


def dump_as_lean_proofs(tgt_path: Path, proofs: List[ProofWithStatement]):
    assert not tgt_path.exists()
    proof_lean_str = "\n\n".join([proof.proof_str() for proof in proofs])
    with tgt_path.open("w") as fp:
        fp.write(proof_lean_str)


def dump_as_json_proofs(
    tgt_path: Path, proofs: List[ProofWithStatement], lean_proof_path: Path
):

    assert not tgt_path.exists()
    with lean_proof_path.open("rb") as fp:
        md5 = hashlib.md5(fp.read()).hexdigest()

    with tgt_path.open("w") as fp:
        json.dump(
            {"lean_md5": md5, "proofs": [p.to_dict() for p in proofs]}, fp, indent=2
        )


def load_dataset(
    data_dir: Path, dataset_name: str
) -> Tuple[Dict, List[ProofWithStatement]]:
    json_dump_path = data_dir / f"{dataset_name}.json"
    lean_dump_path = data_dir / f"{dataset_name}.lean"
    meta_dump_path = data_dir / f"{dataset_name}.metadata"

    with meta_dump_path.open("r") as fp:
        metadata = json.load(fp)

    with lean_dump_path.open("rb") as fp:
        md5 = hashlib.md5(fp.read()).hexdigest()

    with json_dump_path.open("r") as fp:
        json_dump = json.load(fp)
        expected_md5 = json_dump["lean_md5"]
        assert md5 == expected_md5, (lean_dump_path, md5, expected_md5)
    proofs = [ProofWithStatement.from_dict(d) for d in json_dump["proofs"]]

    return metadata, proofs
