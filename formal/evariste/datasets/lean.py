# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Tuple, List, Set, Dict
from dataclasses import dataclass, field
from collections import defaultdict
from subprocess import Popen, PIPE
from copy import deepcopy
from pathlib import Path
import os
import re
import time
import pandas
import subprocess

from params import Params, ConfStore
from evariste.utils import formal_workdir
from evariste.clusters.utils import clusterify_path
from evariste.backward.env.lean.filter_tactics import TacticFilter
from evariste.backward.env.lean.tokenizer import LeanTokenizer


LEAN_SPLITS: Dict[str, Optional[str]] = {
    "train": "data.csv",
    "valid": "data.csv",
    "test": "data.csv",
    "minif2f_valid": "minif2f.csv",
    "minif2f_test": "minif2f.csv",
    "annotations": None,
    "annotations_false": "annotations_false.csv",
    "oai_curriculum": "oai_curriculum.csv",
    "synthetic_rwalk_20_10": None,
    "imo": None,
    "autof_codex": None,
    "autof_lela": None,
}

VIRTUAL_SPLITS: Dict[str, List[str]] = {
    "imo": ["minif2f_valid", "minif2f_test"],
}
# to create a new version vXX based on a specific commit Y, update
# this mapping  {"vXX": "Y} then run
# `python -m scripts.tim.lean.update_annotations my/path vXX`
ANNOTATION_VERSIONS = {"v1": "842a4eae10a1dcb23e6deb4c5020e86320915a17"}

for version in ANNOTATION_VERSIONS:
    VIRTUAL_SPLITS[f"annotations_{version}-imo"] = [f"annotations_{version}"]

LEAN_CONDITIONING = {
    "proofterm": clusterify_path(""),
    "tactic": clusterify_path(""),
    "premise": clusterify_path(""),
}

SYNTHETIC_DATA_DIR = "YOUR_PATH/synthetic/"
DEFAULT_STATEMENT_SPLITS = clusterify_path("")
DEFAULT_SPLIT_STR = "minif2f_valid,minif2f_test,oai_curriculum,ladder_synth"


def cpu_fingerprint():
    return (
        subprocess.check_output(
            "cat /proc/cpuinfo | grep flags | tail -n 1 | md5sum", shell=True
        )
        .decode()
        .split()[0]
    )


def find_ckpt_path(path: str) -> str:
    if "release" in path:
        return path
    elif "<cpu_fp>" in path:
        final = path.replace("<cpu_fp>", cpu_fingerprint())
        ckpt_dir = Path(final).parent
        try:
            # Race condition if several jobs need the same checkpoint at once
            # if we create the dir, then we create the checkpoint. Otherwise we wait for the checkpoint to exist.
            print(f"Attempting to load checkpoint in {ckpt_dir} ...")
            os.makedirs(ckpt_dir, exist_ok=False)
        except PermissionError:
            print("PermissionError")
            if not os.path.exists(ckpt_dir):
                raise
        except OSError as e:
            print(f"Folder already exists: {e}")
            pass
        else:
            print("Didn't find checkpoint. Creating...", flush=True)
            root = ckpt_dir.parent.parent  # root/build/<cpu_fp>/ckpt
            lean_path = ":".join(
                [
                    f"{root}/{x}"
                    for x in [
                        "lean/library",
                        "mathlib/src",
                        "others",
                        "cleaning_utils",
                    ]
                    if Path(f"{root}/{x}").exists()
                ]
            )
            os.environ["LEAN_PATH"] = lean_path
            old_dir = os.getcwd()
            os.chdir(ckpt_dir)
            print(
                f"Running LEAN_PATH={os.environ.get('LEAN_PATH')} {root}/build/release/ml_server preload"
            )
            proc = Popen(
                [f"{root}/build/release/ml_server", "preload"],
                encoding="utf-8",
                stdin=PIPE,
                stdout=PIPE,
            )
            assert proc.stdout is not None
            output = proc.stdout.readline().strip()
            assert output == "lean ml server ready", f"{output} {output!r}"
            os.chdir(old_dir)
            print("Checkpoint created.", flush=True)

        while not Path(final).exists():
            print(f"{final} not found. Waiting ...", flush=True)
            time.sleep(60)

        return final
    else:
        return path


@dataclass
class LeanDatasetConf(Params):
    statement_splits_path: str = field(
        default=DEFAULT_STATEMENT_SPLITS, metadata={"help": "Path to lean checkpoint"},
    )
    splits_str: str = DEFAULT_SPLIT_STR

    checkpoint_path: str = field(
        default="", metadata={"help": "Path to lean checkpoint"},
    )
    tokenizer: str = field(
        default="bpe_arxiv_utf8", metadata={"help": "Tokenizer version"},
    )
    data_dir: str = field(
        default="",
        metadata={
            "help": "Path to dataset files (data.csv and split.{train|test|valid})"
        },
    )
    pact_data_dir: str = field(
        default="", metadata={"help": "Path to PACT dataset files"},
    )
    synthetic_data_dir: str = field(
        default=SYNTHETIC_DATA_DIR,
        metadata={"help": "Path to synthetic dataset files"},
    )
    max_size: int = field(
        default=1_000_000, metadata={"help": "Maximum Lean size for a goal state"}
    )  # <= 1_000_000 -> ~99.80% mathlib training goal states
    max_subgoals: int = field(
        default=10, metadata={"help": "Maximum number of allowed subgoals"}
    )  # <= 10 -> ~99.96% mathlib training goal states
    max_metavars: int = field(
        default=10,
        metadata={"help": "Maximum number of allowed metavariables (?m_0, ?m_1, ...)"},
    )  # <= 10 -> ~99.87% mathlib training goal states
    max_repeated_hyps: int = 3  # otherwise we get bugged proofs. 3 handles > 99% of cases.
    max_inst: int = 20  # < 20 ~99.7% mathlib training goal states
    fast: bool = False
    path_to_fifos: Optional[Tuple[Path, Path]] = None
    timeout: int = 2000
    parse_goal_timeout: int = 20000
    num_threads: int = 10
    pp_full_names: bool = False
    nosplit: bool = False
    merge_alpha_equiv: bool = False

    # if True, match on conclusion instead of matching on fingerprint,
    # WARNING: should be True if we want to be sure not to have false positive proofs
    # but we might miss some true positive if False
    fwd_match_on_conclusion: bool = True
    fwd_use_parsed_pp_as_conclusion: bool = True
    # if True, discard goals is parse_goal(goal).fp != goal.fp
    fwd_same_fp_for_parsed: bool = False

    parsable_bwd: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, ensure that all tactic children are parsable with parse_goal"
            )
        },
    )
    parsable_bwd_same_fp_for_parsed: bool = field(
        default=False,
        metadata={"help": "If True, ensure that parse(g).fingerprint = g.fingerprint"},
    )
    pact_alpha_weight: float = field(
        default=1.0,
        metadata={
            "help": (
                "Re-weight tasks for PACT training "
                "0: each task is sampled uniformly, "
                "1: each sample overall is sampled uniformly"
            )
        },
    )

    # dump all active lean proof interactions. Useful to repro segfaults and weird memory usage
    # False by default because disk usage can get quite high with many workers.
    dump_proof_logs: bool = False

    # dump all
    dump_tactic_batches: bool = False
    dump_tactic_times: bool = False

    # what key to use for fingerprints
    fingerprint: str = "node_id"  # 'fingerprint_new' or 'fingerprint_with_locals'

    # no 'repeat, clear, try, ';' or tactics starting with {', '<|>'
    filter_tactics: TacticFilter = field(default_factory=lambda: TacticFilter())

    # Potentially request a Lean cluster
    lean_cluster: bool = False
    num_instances: int = 1
    partition: str = "local"

    conditioning: str = ""

    past_tactics: int = 0  #: skip past tactics
    skip_statements: bool = True  #: skip statements in collocated .skip files

    def __post_init__(self):
        # check that leanml isn't from lean_fork
        import leanml
        import evariste

        assert (
            Path(leanml.__file__).parent.parent == Path(evariste.__file__).parent.parent
        )

        self.data_dir = clusterify_path(self.data_dir)
        self.pact_data_dir = clusterify_path(self.pact_data_dir)
        self.synthetic_data_dir = clusterify_path(self.synthetic_data_dir)
        self.checkpoint_path = clusterify_path(self.checkpoint_path)
        if self.merge_alpha_equiv:
            assert self.fingerprint == "node_id"
        assert self.pact_alpha_weight >= 0
        assert self.fingerprint in {
            "node_id",
            "fingerprint",
            "fingerprint_new",
            "fingerprint_with_locals",
        }, self.fingerprint

    def _check_and_mutate_args(self):
        LeanTokenizer.build(self.tokenizer)
        assert os.path.isdir(self.data_dir), self.data_dir
        assert (
            Path(self.data_dir) / "data.csv"
        ).exists(), f"No data.csv in {self.data_dir}"
        for split in {"train", "valid", "test"}:
            assert (
                Path(self.data_dir) / f"split.{split}"
            ).exists(), f"No split.{split} in {self.data_dir}"
        if self.path_to_fifos is not None:
            assert len(self.path_to_fifos) == 2
            for p in self.path_to_fifos:
                assert p.exists(), f"{p} not found"

        if self.conditioning != "":
            assert self.conditioning_path is not None
            assert self.conditioning in LEAN_CONDITIONING, repr(self.conditioning)
            assert Path(self.conditioning_path).exists()
        else:
            assert self.conditioning_path is None

    def get_materialized(self) -> "LeanDatasetConf":
        to_ret = deepcopy(self)
        to_ret.checkpoint_path = find_ckpt_path(self.checkpoint_path)
        to_ret.data_dir = clusterify_path(self.data_dir)
        # TODO check that all used splits are present ?
        return to_ret

    @property
    def is_old(self) -> bool:
        return (
            "v31" in self.checkpoint_path
            or "v28" in self.checkpoint_path
            or "v33" in self.checkpoint_path
        )

    @property
    def conditioning_path(self) -> Optional[str]:
        return LEAN_CONDITIONING.get(self.conditioning, None)

    @property
    def splits(self) -> List[str]:
        return self.splits_str.split(",")

    def get_labels_and_splits(self) -> Dict[str, List[Tuple[Path, str, str]]]:
        """
        Load all decls from files in statement_splits.
        Load all decls from data.csv (train/test/valid)
        Also check that all these files are imported in all cleaning scripts
        Create "virtual" IMO splits
        """
        if getattr(self, "_labels_and_split", None) is not None:
            # undeclared in __init__ to avoid breaking params since type is complex
            return self._labels_and_split  # type: ignore
        res = defaultdict(list)
        assert (Path(self.data_dir) / "data.csv").exists()
        loaded_data = pandas.read_csv(Path(self.data_dir) / "data.csv")

        added = set()
        for row in loaded_data.iloc:  # type: ignore
            if row.decl_name in added:
                continue
            res[row.split].append((row.filename, row.decl_name, row.split))
            added.add(row.decl_name)

        file_must_be_imported = set()
        p = Path(self.statement_splits_path)
        assert p.exists(), p

        to_load_splits: Set[str] = set(self.splits)
        for split, to_load in VIRTUAL_SPLITS.items():
            if split in to_load_splits:
                to_load_splits.update(to_load)

        for split in to_load_splits:
            if split in VIRTUAL_SPLITS:
                continue
            fn = p / f"{split}.lean"
            assert fn.exists(), f"Missing {fn}"
            file_must_be_imported.add(split)
            skip_statements: Set[str] = set()
            if self.skip_statements:
                try:
                    with open(p / f"{split}.skip") as f:
                        for l in f:
                            ls = l.strip()
                            if ls and not ls.startswith("#"):
                                skip_statements.add(ls)
                except FileNotFoundError:
                    pass
            print(f"Skipping {len(skip_statements)} statements for the split `{split}`")
            with open(fn, "r") as f:
                these_decls = [
                    x[0]
                    for x in re.findall(
                        r"theorem (?P<label>(\d|\w|_)+)(\n| :\n)", f.read()
                    )
                ]
                for x in these_decls:
                    if x not in skip_statements:
                        res[split].append((fn, x, split))

        if "imo" in self.splits:
            for split_ in ["minif2f_valid", "minif2f_test"]:
                for fn, decl, namespace in res[split_]:
                    if "imo" not in decl:
                        continue
                    res["imo"].append((fn, decl, namespace))
            print(
                f"Found {len(res['imo'])}/{len(res['minif2f_valid']) + len(res['minif2f_test'])} IMO problems."
            )

        for version in ANNOTATION_VERSIONS:
            if f"annotations_{version}" in self.splits:
                for fn, decl, namespace in res[f"annotations_{version}"]:
                    if "imo" not in decl:
                        continue
                    res[f"annotations_{version}-imo"].append((fn, decl, namespace))
                l1 = len(res[f"annotations_{version}-imo"])
                l2 = len(res[f"annotations_{version}"])
                print(f"Found {l1}/{l2} IMO problems.")

        to_keep = set(self.splits)
        to_keep.update({"train", "valid", "test"})
        res_kept = {x: y for x, y in res.items() if x in to_keep}
        assert set(res_kept.keys()) == to_keep, f"{set(res_kept.keys())} != {to_keep}"

        # check that all decls are referenced only once
        seen: Dict[str, str] = {}
        for split, to_check in res_kept.items():
            for _, decl, _ in to_check:
                assert (
                    decl not in seen
                ), f"{split} declares {decl} previously declared in {seen[decl]}"
                seen[decl] = split

        print("Lean dataset loaded :")
        for x, y in res_kept.items():
            print(f"\tSplit {x} : {len(y)} statements")

        self._labels_and_split = res_kept

        for fn in (
            formal_workdir() / "runtime_loaded_lean_files" / "cleaning_utils"
        ).glob("*/do_clean_proof.lean"):
            with open(fn, "r") as f:
                content = f.read()
            # TODO only check depending on cleaning version
            for x in file_must_be_imported:
                assert f"import {x}\n" in content, f"{x} can't be cleaned from {fn}"

        return res_kept


PACT_VERSION = 1
PACT_DATA_DIR = {1: "YOUR_PATH/pact", 3: "YOUR_PATH/pact3",}[PACT_VERSION]

# V2 == 1c4b8bffebdf052260d11d47a5ff96dc7d1fb8db -> 1 file / split and a few fixes. still 244/244
LEAN_FULL_NAMES_DIR_V2_TACTIC_FULL_NAMES = ""

# less permissive forward, enforcing max_repeated_hyps=0, fwd_match_on_conclusion=False,
# fwd_same_fp_for_parsed=True should avoid false proofs, but number of real proofs
# are expected to be lower too.
# THIS CHECKPOINT IS OLD, but was the latest used for fwd.
# fwd dataset files should be properly rebuilt (with albert's script for conjecturing) on a more recent checkpoint
LEAN_DATASET_V3_FWD = LeanDatasetConf(
    statement_splits_path=DEFAULT_STATEMENT_SPLITS,
    splits_str=DEFAULT_SPLIT_STR,
    checkpoint_path="",
    data_dir="",
    pact_data_dir=PACT_DATA_DIR,
    conditioning="",
    tokenizer="",
    max_repeated_hyps=0,  # mrh=0 for fwd
    pp_full_names=True,
    fwd_match_on_conclusion=False,
    fwd_same_fp_for_parsed=True,
)

LEAN_DATASET_LATEST_FWD = LEAN_DATASET_V3_FWD


def mk_lean_conf(
    version: str, merge: bool = True,
):
    return LeanDatasetConf(
        statement_splits_path=DEFAULT_STATEMENT_SPLITS,
        splits_str=DEFAULT_SPLIT_STR,
        checkpoint_path=f"",
        data_dir=LEAN_FULL_NAMES_DIR_V2_TACTIC_FULL_NAMES,
        pact_data_dir=PACT_DATA_DIR,
        conditioning="",
        tokenizer="",
        pp_full_names=True,
        fingerprint="node_id",
        max_repeated_hyps=1,
        merge_alpha_equiv=merge,
    )


LEAN_DATASETS = {}
for version in [
    "v28",
    "v31",  # add thread ids
    "v33",  # v31 + proof cleaning
    "v1.0",  # symlinked others/
    "v1.1",  # use cwd others/cleaning_utils
    "v1.2",  # do not split shared universe metavars
]:
    LEAN_DATASETS[f"{version}"] = mk_lean_conf(version)
    LEAN_DATASETS[f"{version}_no_merge"] = mk_lean_conf(version, merge=False)


LEAN_DATASET_LATEST = LEAN_DATASETS["v1.2"]


def register_lean_datasets():
    ConfStore[f"lean_latest"] = LEAN_DATASET_LATEST
    for x, y in LEAN_DATASETS.items():
        ConfStore[f"lean_{x}"] = y
