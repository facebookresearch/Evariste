# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Tuple, Optional, TypedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
import re
import shutil
import subprocess

from leanml.parse_goal import parse_goal_structured

import evariste
from evariste.backward.graph import Proof, NonPicklableProof, fold_proof
from evariste.backward.env.lean.graph import LeanTactic
from evariste.utils import clusterify_path
from evariste.json import json


class ExportedProof(TypedDict):
    name: Optional[str]
    thm: str
    actions: List[str]
    decl_signature: str


def mk_check_folder(target_folder: Path):
    assert target_folder.exists()
    root_folder = Path(os.path.dirname(evariste.__file__)).parent
    os.makedirs(target_folder / "src", exist_ok=True)
    os.makedirs(target_folder / "lib", exist_ok=True)
    for x in ["others", "cleaning_utils"]:
        shutil.copytree(
            root_folder / "runtime_loaded_lean_files" / x,
            target_folder / "lib",
            dirs_exist_ok=True,
        )

    with open(target_folder / "leanpkg.path", "w") as f:
        f.write(
            f"path {clusterify_path('YOUR_PATH/library')}\n"
            f"path {clusterify_path('YOUR_PATH/src')}\n"
            f"path ./src\n"
            f"path ./lib\n"
        )
        f.flush()

    with open(target_folder / "leanpkg.toml", "w") as f:
        f.write(
            """
[package]
name = "lean_proof_check"
version = "0.1"
lean_version = "leanprover-community/lean:3.30.0"
path = "src"

[dependencies]
mathlib = {git = "https://github.com/leanprover-community/mathlib", rev = "9a8dcb9be408e7ae8af9f6832c08c021007f40ec"}
"""
        )
        f.flush()

    check_lean_paths(target_folder)


def check_lean_paths(target_folder: Path):
    assert target_folder.is_dir()
    result = subprocess.run(
        ["lean", "--path"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=target_folder,
    )
    stdout = result.stdout.decode("utf-8")
    print(stdout)
    try:
        parsed = json.loads(stdout)
    except json.decoder.JSONDecodeError:
        print(f"Could not parse JSON object from stdout: {stdout}")
        raise
    assert type(parsed["path"]) is list
    print(parsed["path"])
    if not any("mathlib" in path for path in parsed["path"]):
        raise RuntimeError("mathlib is not found!")


def check_lean_proof(target_folder: Path, fname: str, timeout=60):
    assert str(fname).endswith(".lean")
    assert os.path.exists(os.path.join(target_folder, "src", fname))
    proc = subprocess.Popen(
        ["lean", os.path.join(target_folder, "src", fname)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=target_folder,
    )
    try:
        outs, errs = proc.communicate(timeout=timeout)
        timeout = False
    except subprocess.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()
        timeout = True
    return {
        "path": fname,
        "stdout": outs,
        "stderr": errs,
        "timeout": timeout,
    }


def check_lean_proofs(target_folder: Path, max_workers: int = 40, timeout: int = 120):
    """
    Check Lean proofs by running Lean on .lean files.
    """
    n_valid = 0
    n_total = 0
    failures = []
    timeouts = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        fnames = os.listdir(target_folder / "src")
        fnames = [
            x
            for x in fnames
            if x
            not in {
                "proof_cleaning.lean",
                "proof_running.lean",
                "solven.lean",
                "utils.lean",
            }
        ]
        remaining = set(fnames)
        for res in executor.map(
            check_lean_proof,
            [target_folder] * len(fnames),
            fnames,
            [timeout] * len(fnames),
            chunksize=1,
        ):
            print(f"========== {n_total + 1}/{len(fnames)} {res['path']}")
            stdout = res["stdout"]
            stderr = res["stderr"]
            print(f"stdout: {stdout}")
            print(f"stderr: {stderr}")
            if stdout == b"" and not res["timeout"]:
                n_valid += 1
            else:
                print("FAIL")
                failures.append(res["path"])
                if res["timeout"]:
                    print("TIMEOUT")
                    timeouts.append(res["path"])
            n_total += 1
            print(f"{n_valid}/{n_total} ({100 * n_valid / n_total:.2f}%)")
            remaining.remove(res["path"])
            print(remaining)

    print(f"Found {n_valid}/{n_total} ({100 * n_valid / n_total:.2f}%) valid proofs.")
    if failures:
        print(f"{len(failures)} failures:")
        print("\t" + "\n\t".join(failures))
    if timeouts:
        print(f"{len(timeouts)} timeouts:")
        print("\t" + "\n\t".join(timeouts))


def extract_tactics_from_proof(proof: Proof) -> List[LeanTactic]:
    """
    Extracts the list of LeanTactic from the proof

    :param proof: proof object
    :type proof: Proof
    :return: the list of tactics
    :rtype: List[LeanTactic]
    """
    return fold_proof(proof, [], lambda p, l: l + [p[1]])  # type: ignore


def extract_ident_and_actions_from_proof(
    proof: Proof, all_curly_braces: bool = False
) -> List[Tuple[int, str]]:
    """
    Extracts the list of pairs (identation, action) from the proof, including
    curly braces and `solven k begin ... end`

    :param proof: proof object
    :type proof: Proof
    :param all_curly_braces: indicates if all curly braces should be added (vs. only the necessary ones)
    :type all_curly_braces: bool, defaults to False
    :return: the list of pairs (identation, action)
    :rtype: List[Tuple[int, str]]
    """

    def _walk_proof(
        cur_elem: Proof, one_child_or_last_child_no_curl=True, indent=0
    ) -> List[Tuple[int, str]]:
        assert not isinstance(cur_elem, NonPicklableProof)
        this_res: List[Tuple[int, str]] = []
        cur_th, cur_tac, children = cur_elem
        # This hack should be improved
        # TODO: use the number of goals from the Lean expander env and
        # add a `n_goals` attribute to `LeanTheorem`
        solven = cur_th.conclusion.count("\n\n") + 1
        if not one_child_or_last_child_no_curl:
            if solven == 1:
                this_res += [(indent, "{"), (indent + 1, str(cur_tac))]
            else:
                this_res += [
                    (indent, f"solven {solven} begin"),
                    (indent + 1, str(cur_tac)),
                ]
        else:
            this_res.append((indent, str(cur_tac)))
        assert cur_tac.is_valid, cur_tac
        has_one_child = len(children) == 1
        is_last_child = lambda idx: idx == len(children) - 1
        for idx, c in enumerate(children):
            this_res.extend(
                _walk_proof(
                    c,
                    one_child_or_last_child_no_curl=(
                        has_one_child or (is_last_child(idx) and not all_curly_braces)
                    ),
                    indent=indent + (not one_child_or_last_child_no_curl),
                )
            )
        if not one_child_or_last_child_no_curl:
            this_res.append((indent, "}" if solven == 1 else "end"))
        return this_res

    return _walk_proof(proof, not all_curly_braces)


def extract_actions_from_proof(
    proof: Proof, all_curly_braces: bool = False
) -> List[str]:
    """
    Extracts the list of actions from the proof, including
    curly braces and `solven k begin ... end`

    :param proof: proof object
    :type proof: Proof
    :param all_curly_braces: indicates if all curly braces should be added (vs. only the necessary ones)
    :type all_curly_braces: bool, defaults to False
    :return: the list of actions
    :rtype: List[str]
    """
    return [
        ac for _, ac in extract_ident_and_actions_from_proof(proof, all_curly_braces)
    ]


def to_lean_proof(proof: Proof, starts_with_intros=False) -> str:
    """
    Serializes a proof

    :param proof: proof to be serialized
    :type proof: Proof
    :param starts_with_intros: indicates if the proof starts with `repeat{intro}`
    :type starts_with_intros: bool, defaults to False
    :return: proof serialized
    :rtype: str
    """
    s = "repeat{intro},\n" if starts_with_intros else ""
    for (indent, ac) in extract_ident_and_actions_from_proof(proof):
        s += "  " * indent + ac.strip()
        if not (ac.strip().endswith("{") or ac.strip().endswith("begin")):
            s += ","
        s += "\n"
    return s


def export_lean_proof(
    proof: Proof,
    *,
    decl_signature: Optional[str] = None,
    thm_name: Optional[str] = None,
) -> ExportedProof:
    assert not isinstance(proof, NonPicklableProof)
    concl = proof[0].conclusion
    structured_goal = parse_goal_structured(concl)[0]
    cmd = structured_goal.to_command()

    if decl_signature is not None:
        if thm_name is not None:
            decl_signature = re.sub(
                r"(?P<thm>theorem|lemma)\s+(?P<name>\w+)(?P<type>.*)",
                rf"\g<thm> {thm_name}\g<type>",
                decl_signature,
                flags=re.DOTALL,
            )
    else:
        assert isinstance(cmd["decl_name"], str)
        assert isinstance(cmd["command"], str)
        thm_name = thm_name if thm_name else cmd["decl_name"]
        decl_signature = re.sub(
            r"def\s_root_\.\w{6}",
            f"theorem {thm_name}",
            cmd["command"],
            flags=re.DOTALL,
        )
        match = re.match(r"(?P<signature>([^:]|:(?!=))*)", decl_signature)
        assert match
        decl_signature = match.group("signature").strip()
    match = re.match(r"(?:theorem|lemma)\s+(?P<name>\w+)", decl_signature)
    assert match, f"'{decl_signature}' is not a valid lean theorem signature"
    thm_name = match.group("name")

    actions = [ac for _, ac in extract_ident_and_actions_from_proof(proof)]

    return {
        "name": thm_name,
        "thm": concl,
        "actions": actions,
        "decl_signature": decl_signature,
    }


def export_lean_proof_to_file(
    proof: Proof,
    *,
    decl_signature: Optional[str] = None,
    thm_name: Optional[str] = None,
) -> str:
    """
    Returns the printed proof as if on a lean file
    If `decl_signature` or `thm_name` are not filled, a default
    name is given to the theorem
    If both are filled, `thm_name` comes to replace the theorem name
    in the `decl_signature`
    If only `thm_name` is filled, the signature is inferred

    :param proof: proof to be exported
    :type proof: Proof
    :param decl_signature: the signature of the theorem
    :type decl_signature: str, optional
    :param thm_name: the theorem name
    :param thm_name: str, optional
    :return: printed proof as if on a lean file
    :rtype: str
    """
    exp_pf = export_lean_proof(proof, decl_signature=decl_signature, thm_name=thm_name,)
    serialized_proof = to_lean_proof(proof)
    return (
        exp_pf["decl_signature"]
        + " :=\nbegin\n"
        + "\n".join([f"  {x}" for x in serialized_proof.split("\n")]).rstrip()
        + "\nend\n"
    )


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description="Print lean proof from .pkl")
    parser.add_argument("--src", type=str, help="Source pkl file", required=True)

    args = parser.parse_args()
    src = os.path.abspath(args.src)
    assert src.endswith(".pkl"), src

    thm_name = os.path.basename(src)
    if "__" in thm_name:
        thm_name = thm_name.split("__")[0]
    else:
        thm_name = thm_name[:-4]
    with open(src, "rb") as f:
        proof = pickle.load(f)

    exp_pf = export_lean_proof(proof, thm_name=thm_name)
    print(exp_pf["actions"])
