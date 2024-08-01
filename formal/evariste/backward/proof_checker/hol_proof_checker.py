# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from multiprocessing import Pool
from typing import List, Tuple
import re

from evariste.datasets.hol import ConfStore
from evariste.backward.graph import Proof
from evariste.envs.hl.api import (
    HOLLightAPI,
    _build_tactic_from_tokens,
    _hol_tokens_to_ocaml,
)
from evariste.envs.ocaml.api import OCamlError


class _InvalidTactic(Exception):
    pass


def convert_proof_to_script(proof: Proof) -> str:
    concl = _hol_tokens_to_ocaml(proof[0].conclusion.split())
    proof_script = f"prove (\n\t`{concl}`,"

    def dfs(proof, ident):
        tactic, subproofs = proof[1], proof[2]
        if not tactic.is_valid:
            raise _InvalidTactic
        tactic_tokens = tactic._tactic.split()
        tactic_str = _build_tactic_from_tokens(tactic_tokens)
        subproof_script = "\n" + "\t" * ident + tactic_str
        if len(subproofs) == 0:
            return subproof_script
        elif len(subproofs) == 1:
            return subproof_script + " THEN" + dfs(subproofs[0], ident)
        return (
            subproof_script
            + " THENL ["
            + ";".join([dfs(subproof, ident + 1) for subproof in subproofs[::-1]])
            + "\n"
            + "\t" * ident
            + "]"
        )

    proof_script += dfs(proof, 1)
    proof_script += "\n);;\n"
    return proof_script


def _check_proof_script(proof_script: str) -> bool:
    global ENV
    try:
        with open(proof_script) as fp:
            reply = ENV._send(fp.read())
        match = re.match(r"val it : thm =", reply)
        return match is not None
    except OCamlError:
        return False


def check_proof_scripts(
    proof_scripts: List[str], checkpoint: str, n_envs: int = 1
) -> List[Tuple[str, bool]]:
    def _initializer():
        global ENV
        ENV = HOLLightAPI(checkpoint_path=checkpoint, timeout=20.0)

    if n_envs == 1:
        _initializer()
        results = [_check_proof_script(proof_script) for proof_script in proof_scripts]
    else:
        with Pool(processes=n_envs, initializer=_initializer) as pool:
            results = pool.map(_check_proof_script, proof_scripts, 1)
    proof_names = [
        os.path.splitext(os.path.basename(proof_script))[0]
        for proof_script in proof_scripts
    ]
    return list(zip(proof_names, results))


def check_proofs_in_dir(
    proof_dirpath: str, checkpoint: str, n_envs: int = 1
) -> List[Tuple[str, bool]]:
    proof_scripts = []
    failed_conversions = []
    for prooffile in os.listdir(args.proof_dirpath):
        if not (prooffile.endswith(".proof") or prooffile.endswith(".pkl")):
            continue
        proof_name = os.path.splitext(prooffile)[0]
        proof_script = os.path.join(proof_dirpath, proof_name + ".ml")
        proof_scripts.append(proof_script)
        proof = pickle.load(open(os.path.join(proof_dirpath, prooffile), "rb"))
        try:
            with open(proof_script, "w") as fd:
                fd.write(convert_proof_to_script(proof))
        except _InvalidTactic:
            failed_conversions.append((proof_name, False))

    return (
        check_proof_scripts(
            proof_scripts=proof_scripts, checkpoint=checkpoint, n_envs=n_envs
        )
        + failed_conversions
    )


if __name__ == "__main__":
    import argparse
    import os
    import os.path
    import pickle
    from getpass import getuser

    def get_default_proof_dirpath():
        root = f"hol-light-logs"
        return os.path.join(root, sorted(os.listdir(root))[-1], "proofs")

    parser = argparse.ArgumentParser(description="Check pickled proofs")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint",
        default=ConfStore["hl_plus_default_dataset"].checkpoint_path,
    )
    parser.add_argument(
        "--proof_dirpath",
        type=str,
        help="Path to HOL-Light directory",
        default=get_default_proof_dirpath(),
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        help="Number of HOL-Light environments running concurrently",
        default=10,
    )

    args = parser.parse_args()

    results = check_proofs_in_dir(
        proof_dirpath=args.proof_dirpath, checkpoint=args.checkpoint, n_envs=args.n_envs
    )
    nb_valid_proofs = sum(r[1] for r in results)
    nb_proofs = len(results)
    actual_proof_names = [r[0] for r in results if r[1]]
    invalid_proof_names = [r[0] for r in results if not r[1]]
    print(f"There are {nb_valid_proofs} / {nb_proofs} valid proofs")
    if invalid_proof_names:
        print(
            f"\nThe invalid proof{'s are' if len(invalid_proof_names) > 1 else ' is'}:"
        )
        for proof_name in invalid_proof_names:
            print(f"\t{proof_name}")

    print(f"\nThe actual proof{'s are' if len(actual_proof_names) > 1 else ' is'}:")
    for proof_name in actual_proof_names:
        print(f"\t{proof_name}")
