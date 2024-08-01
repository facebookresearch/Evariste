# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, List, Dict, Tuple
import os
import os.path
import subprocess
from evariste import json as json
from multiprocessing import Pool
from dataclasses import asdict
import re
import multiprocessing as mp

from evariste.envs.hl.api import (
    HOLLightAPI,
    HOLLightToken,
    HOLLightSample,
    HOLLightException,
)
from evariste.envs.hl.tokenizer import Lexer
from evariste.logger import create_logger


ALWAYS_EXCLUDED = {
    "basics.ml",
    "bool.ml",
    "database.ml",
    "drule.ml",
    "equal.ml",
    "fusion.ml",
    "help.ml",
    "hol.ml",
    "json.ml",
    "lib.ml",
    "make.ml",
    "nets.ml",
    "pa_j.ml",
    "pa_j_3.1x_5.xx.ml",
    "pa_j_3.1x_6.02.1.ml",
    "pa_j_3.1x_6.02.2.ml",
    "pa_j_3.1x_6.11.ml",
    "pa_j_3.1x_6.xx.ml",
    "pa_j_3.07.ml",
    "pa_j_3.08.ml",
    "pa_j_3.09.ml",
    "pa_j_4.xx_7.06.ml",
    "pa_j_4.xx_7.xx.ml",
    "parser.ml",
    "preterm.ml",
    "printer.ml",
    "spawn_repls.ml",
    "system.ml",
    "tactics.ml",
    "update_database.ml",
    "update_database_3.ml",
    "update_database_4.ml",
    "Arithmetic/make.ml",
    "Complex/make.ml",
    "Functionspaces/make.ml",
    "Jordan/make.ml",
    "Logic/make.ml",
    "Multivariate/make_complex.ml",
    "Multivariate/make.ml",
    "Permutation/make.ml",
    "Quaternions/make.ml",
    "Rqe/make.ml",
    # "Boyer_Moore",
    # "Evariste",
    # "Formal_ineqs",
    # "Help",
    # "IEEE",
    # "IsabelleLight",
    # "LP_arith",
    # "Minisat",
    # "miz3",
    # "Mizarlight",
    # "Model",
    # "Proofrecording",
    # "ProofTrace",
    # "QBF",
    # "RichterHilbertAxiomGeometry",
    # "Tutorial",
    # "Unity"
}


class HOLLightDatasetError(HOLLightException):
    pass


def get_filepaths(
    hol_light_dirpath: str,
    include_dirs: Optional[List[str]] = None,  # dirs
    include: Optional[List[str]] = None,  # files
    exclude: Optional[List[str]] = None,  # files
) -> List[str]:
    """
    Get the filepaths to be processed relative to the hol_light_dirpath
    Args:
        hol_light_dirpath: str  # path to the HOL-Light directory
        include_dirs: Optional[List[str]] = None  # directories to include
        include: Optional[List[str]] = None  # relative filepaths to include
        exclude: Optional[List[str]] = None  # relative filepaths to exclude
    Returns:
        abs_paths: List[str]  # List[abspath]
    NOTE: this is not recursive.
    """
    included = set()
    include_dirs = set() if include_dirs is None else set(include_dirs)
    include = set() if include is None else set(include)
    exclude = set() if exclude is None else set(exclude)
    exclude |= ALWAYS_EXCLUDED
    for relative_dirpath in include_dirs:
        absdirpath = os.path.join(hol_light_dirpath, relative_dirpath)
        for name in os.listdir(absdirpath):
            relative_path = os.path.join(relative_dirpath, name)
            abspath = os.path.join(hol_light_dirpath, relative_path)
            if os.path.isdir(abspath):
                continue
            if os.path.splitext(name)[-1] not in {".ml", ".hl"}:
                continue
            if relative_path in exclude:
                continue
            included.add(relative_path)
    for relative_path in include:
        if relative_path not in included and relative_path not in exclude:
            included.add(relative_path)
    relative_paths = list(included)
    abs_paths = [
        os.path.join(hol_light_dirpath, relative_path)
        for relative_path in relative_paths
    ]
    assert all(os.path.isfile(path) for path in abs_paths)
    return abs_paths


def _call_psplit(psplit_path: str, traces_path: str, path: str):
    print(f"psplit on {path}")
    assert os.path.isfile(path)
    p = subprocess.Popen(
        [psplit_path, traces_path, path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = p.communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")
    if stderr:
        return {"status": "nok", "message": stderr}
    try:
        d = json.loads(stdout)
    except json.JSONDecodeError as e:
        return {
            "status": "nok",
            "message": f'{type(e).__name__}: {str(e)} with stdout "{stdout}"',
        }
    return d[path]


def split_proofs(
    hol_light_dirpath: str,
    filepaths: List[str],
    psplit_path: Optional[str] = None,
    traces_path: Optional[str] = None,
) -> Dict:
    """
    Run the psplit tool and retrive the json outputs
    Args:
        hol_light_dirpath: str  # path to the HOL-Light directory
        filepaths: List[Tuple[str, str]]  # List[(abspath, relative_path)]
        psplit_path: Optional[str] = None  # path to the psplit binary
        traces_path: Optional[str] = None  #path to the file that contains the traces of the prove calls
    Returns:
        raw_proofs: Dict[
            relative_path: {  # filepath relative to the root of the HOL-Light directory
                "status": str,  # "ok" or "nok"
                "message": str  # only if status is "nok"
                "content" : List [  # only if status is "ok"
                    {
                        "line": int,
                        "status": str,  # "ok", "nok" or "skip"
                        "message": str,  # only if status if "nok" ok "skip"
                        "name": str,  # only if status if "ok" or "nok"
                        "goal": str,  # only if status if "ok" or "nok"
                        "tactics": List[str]  # only if status if "ok"
                    }
                ]
            }
        ]
    """
    raw_proofs = {}
    included = set()

    if psplit_path is None:
        psplit_path = os.path.join(
            hol_light_dirpath, "Evariste", "Proofsplit", "psplit"
        )
    if traces_path is None:
        traces_path = os.path.join(
            hol_light_dirpath, "Evariste", "Proofsplit", "traces.bin"
        )
    assert os.path.isfile(psplit_path)
    assert os.path.isfile(traces_path)

    raw_proofs = {}
    for path in filepaths:
        psplit_result = _call_psplit(
            psplit_path=psplit_path, traces_path=traces_path, path=path
        )
        raw_proofs.update({path: {**psplit_result, **{"filename": path}}})

    return raw_proofs


_LEXER = Lexer()
_LEXER.build()


def _tokenize_tactic(tactic: str) -> List[HOLLightToken]:
    """Ideally should be integrated in psplit"""
    global _LEXER
    lexcodes = _LEXER.run(tactic)
    return [lx[1] for lx in lexcodes]


def _run_proof(content: Dict) -> Dict:
    global ENV
    try:
        if content["status"] in {"skip", "nok"}:
            raise HOLLightDatasetError(
                f"To run a proof the content status cannot be {content['status']}"
            )
        filename = content["filename"]
        line = content["line"]
        goal = content["goal"]
        tactics = content["tactics"]
        name = content["name"]
        goal_tokens = goal.split()  # as terms are already nicely tokenized
        # TODO: psplit to directly tokenize the goal
        goalstack = ENV.set_bwd_proving(goal_tokens)
        goalstacks = [goalstack]
        idx = -1
        for tact in tactics:
            idx += 1
            tactic_tokens = _tokenize_tactic(
                tact
            )  # TODO: ideally psplit should return tactics as list of tokens directly
            goalstack = ENV.bwd_apply_tactic_to_goalstack(tactic_tokens=tactic_tokens)
            goalstacks.append(goalstack)
        expected_proving_seq = HOLLightSample.normalize_proof(
            goalstacks=goalstacks,
            tactics=[_tokenize_tactic(tact) for tact in tactics],
            origin=(filename, line, name),
        )
    except HOLLightException as e:
        print(f"\t{filename}\t#{line}\t{name} ...".ljust(180) + "FAILS".rjust(20))
        return {
            **content,
            **{"status": "nok", "message": f"{type(e).__name__}: {str(e)}"},
        }
    # To be 100% it worked, you have to replay the sequence sample by sample
    proving_seq = []
    try:
        for idx, sample in enumerate(expected_proving_seq):
            actual_sample = ENV.bwd_apply_tactic(
                tactic_tokens=sample.tactic,
                hyps_tokens=sample.goal.hyps_tokens,  # normalized
                concl_tokens=sample.goal.concl_tokens,  # normalized
            )
            if actual_sample != sample:
                raise HOLLightDatasetError(
                    f'different samples: FOUND "{json.dumps(asdict(actual_sample))}", '
                    f'EXPECTED "{json.dumps(asdict(sample))}"'
                )
            actual_sample.origin = sample.origin
            proving_seq.append(sample.light)
    except HOLLightException as e:
        print(f"\t{filename}\t#{line}\t{name} ...".ljust(180) + "FAILS".rjust(20))
        return {
            **content,
            **{
                "proving_sequence": [asdict(s) for s in expected_proving_seq],
                "status": "nok",
                "message": f"sample #{idx}: {type(e).__name__}: {str(e)}",
            },
        }

    print(f"\t{filename}\t#{line}\t{name} ...".ljust(180) + "SUCCESS")
    return {**content, **{"proving_sequence": proving_seq}}


def create_dataset_from_split_proofs(
    raw_proofs: dict, checkpoint: str, n_envs: int = 1, log_path: Optional[str] = None
) -> Tuple[List[HOLLightSample], Dict]:
    """
    Creates the dataset from the json output by the psplit tool
    Args:
        raw_proofs: Dict[
            relative_path: {  # filepath relative to the root of the HOL-Light directory
                "status": str,  # "ok" or "nok"
                "message": str  # only if status is "nok"
                "content" : List [  # only if status is "ok"
                    {
                        "line": int,
                        "status": str,  # "ok", "nok" or "skip"
                        "message": str,  # only if status if "nok" ok "skip"
                        "name": str,  # only if status if "ok" or "nok"
                        "goal": str,  # only if status if "ok" or "nok"
                        "tactics": List[str]  # only if status if "ok"
                    }
                ]
            }
        ]
        checkpoint: str  # path to checkpoint
    Returns:
        dataset: List[
            {
                "filename" str,
                "line": int,
                "name": str,
                "steps": List[
                    {
                        "goal": {
                            "hyps": List[Tuple[Optional[HOLLightToken], str]],
                            "concl": str
                        },
                        "tactic: str,
                        "subgoals": List[
                            {
                                "hyps": List[Tuple[Optional[HOLLightToken], str]],
                                "concl": str
                            }
                        ]
                    }
                ]
            }
        ]
        error_analysis: {
            "success": List[
                {
                    "filename": str,  # relative_path
                    "line": int,
                    "name": str,
                    "goal": str,
                    "tactics": List[str],
                    "proving_sequence": List[HOLLightSample]
                }
            ],
            "file_fails": List[
                {
                    "filename: str,
                    "message": str
                }
            ]
            "fails": List[
                {
                    "filename": str,  # relative_path
                    "line": int,
                    "message": str,  # error message,
                    "name": str,
                    "goal": str,
                    "tactics": List[str],
                    "proving_sequence": List[HOLLightSample]
                }
            ],
            "skip": List[
                {
                    "filename": str,  # relative_path
                    "line": int,
                    "message": str,
                }
            ]
        ]
    """
    dataset = []
    error_analysis = {"success": [], "file_fails": [], "fails": [], "skip": []}
    contents_to_run = []
    for filename in raw_proofs:
        print(f"extract data from {filename}")
        if raw_proofs[filename]["status"] == "nok":
            error_analysis["file_fails"].append(
                {
                    **{
                        k: v
                        for k, v in raw_proofs[filename].items()
                        if k not in {"status"}
                    },
                    **{"filename": filename},
                }
            )
            continue
        for content in raw_proofs[filename]["content"]:
            content["filename"] = filename
            if content["status"] == "skip":
                error_analysis["skip"].append(
                    {k: v for k, v in content.items() if k not in {"status"}}
                )
                continue
            if content["status"] == "nok":
                error_analysis["fails"].append(
                    {k: v for k, v in content.items() if k not in {"status"}}
                )
                continue
            contents_to_run.append(content)

    def _initializer():
        global ENV

        match = re.search(r"(?P<rank>\d+)$", mp.current_process().name)
        rank = None if not match else int(match.group("rank"))
        ENV = HOLLightAPI(
            checkpoint_path=checkpoint,
            timeout=10.0,
            logger=create_logger(
                f"{log_path}/hol{'-' + str(rank) if rank is not None else ''}.log"
            ),
            rank=rank,
        )

    if n_envs == 1:
        _initializer()
        results = [_run_proof(content) for content in contents_to_run]
    else:
        with Pool(processes=n_envs, initializer=_initializer) as pool:
            results = pool.map(_run_proof, contents_to_run, 1)
    for updated_content in results:
        if updated_content["status"] == "nok":
            error_analysis["fails"].append(
                {k: v for k, v in updated_content.items() if k != "status"}
            )
        else:
            error_analysis["success"].append(
                {
                    k: v
                    for k, v in updated_content.items()
                    if k not in {"status", "proving_sequence"}
                }
            )
            dataset.append(
                {
                    "filename": updated_content["filename"],
                    "line": updated_content["line"],
                    "name": updated_content["name"],
                    "steps": updated_content["proving_sequence"],
                }
            )

    return dataset, error_analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate the dataset with psplit")
    parser.add_argument(
        "--checkpoint", type=str, help="Path to checkpoint", default="",
    )
    parser.add_argument(
        "--hol_light_dirpath",
        type=str,
        help="Path to HOL-Light directory",
        required=True,
    )
    parser.add_argument(
        "--psplit_path",
        type=str,
        help="Path to psplit binary - default to hol_light_dir/Evariste/Proofsplit/psplit",
    )
    parser.add_argument(
        "--traces_path",
        type=str,
        help="Path to trace file - default to hol_light_dir/Evariste/Proofsplit/traces.bin",
    )
    parser.add_argument(
        "--include_dirs",
        type=str,
        help="HOL-Light directories to include - separated by commas",
    )
    parser.add_argument(
        "--include",
        type=str,
        help='HOL-Light files to include in addition to the ones in "include_dirs" - separated by commas',
    )
    parser.add_argument(
        "--exclude",
        type=str,
        help='HOL-Light files to exclude from the ones in "include_dirs" - separated by commas',
    )
    parser.add_argument(
        "--dataset_path", type=str, help="Path to the dataset dump file", required=True
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        help="Number of HOL-Light environments running concurrently",
        required=True,
    )

    args = parser.parse_args()

    for field in {"include_dirs", "include", "exclude"}:
        if getattr(args, field) is not None:
            setattr(
                args,
                field,
                [
                    "" if k in {"core", "hol"} else k
                    for k in getattr(args, field).split(",")
                ],
            )
    try:
        os.makedirs(os.path.dirname(args.dataset_path))
    except FileExistsError:
        pass
    if os.path.exists(args.dataset_path):
        raise FileExistsError(f'The file "{args.dataset_path}" already exists')
    if os.path.exists(args.dataset_path + ".errors"):
        raise FileExistsError(
            f"The file \"{args.dataset_path + '.errors'}\" already exists"
        )
    filepaths = get_filepaths(
        hol_light_dirpath=args.hol_light_dirpath,
        include_dirs=args.include_dirs,
        include=args.include,
        exclude=args.exclude,
    )
    raw_proofs = split_proofs(
        hol_light_dirpath=args.hol_light_dirpath,
        psplit_path=args.psplit_path,
        traces_path=args.traces_path,
        filepaths=filepaths,
    )
    dataset, error_analysis = create_dataset_from_split_proofs(
        raw_proofs=raw_proofs,
        checkpoint=args.checkpoint,
        n_envs=args.n_envs,
        log_path=os.path.dirname(args.dataset_path),
    )
    with open(args.dataset_path, "w") as fp:
        for thm_samples in dataset:
            json.dump(thm_samples, fp)
            fp.write("\n")
    with open(args.dataset_path + ".errors", "w") as fp:
        json.dump(error_analysis, fp, indent="\t")

    nb_total_files = len(filepaths)
    nb_successful_files = nb_total_files - len(error_analysis["file_fails"])
    nb_successful_theorems = len(error_analysis["success"])
    nb_total_theorems = len(error_analysis["fails"]) + nb_successful_theorems
    nb_samples = sum(len(th["steps"]) for th in dataset)

    print("DATASET BUILDING - STATS:")
    print(
        f"\t- {nb_successful_files} successful files / {nb_total_files} "
        f"files (success rate: {100 * nb_successful_files / nb_total_files:.1f} %)"
    )
    print(
        f"\t- {nb_successful_theorems} successful theorems / {nb_total_theorems} "
        f"theorems in successful files (success rate: "
        f"{100 * nb_successful_theorems / nb_total_theorems:.1f} %)"
    )
    print(
        f"\t- {nb_samples} samples created ({nb_samples / nb_successful_theorems:.1f} "
        f"samples / theorem)"
    )
