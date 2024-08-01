# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from pathlib import Path
import re


FORMER_EXPECTED_HEADERS = [
    """import common
open_locale big_operators
open_locale euclidean_geometry
open_locale nat
open_locale real
open_locale topological_space\n\n"""
]

EXPECTED_HEADER = """import common
open_locale big_operators
open_locale euclidean_geometry
open_locale nat
open_locale real
open_locale topological_space
open_locale asymptotics\n\n"""


def get_final_minif2f_proof(minif2f_dir: str, label: str, proof: str):
    lean_file = Path(minif2f_dir) / f"{label}.lean"
    assert lean_file.exists(), f"{lean_file} not found"
    with open(lean_file, "r") as f:
        original = f.read()
        new = re.sub(
            rf"""\/-(.*)-\/(?P<imports>(.*))theorem {label}(?P<hyp>(.*)) :=\nbegin\n(?P<proof>(.*))\nend""",
            f"""\g<imports>theorem {label}\g<hyp> :=\nbegin\n{proof}\nend""",
            original,
            flags=re.DOTALL,
        )
        return new


def get_sigsegv_proof_logs(log_file: str):
    log = open(log_file, "r").read()

    all_paths = re.findall(
        r"raise DeadLean\(f\"Proc\.poll returned \{proc_poll\}\"\)\n"
        r"\s+leanml\.comms\.DeadLean: Proc\.poll returned -11\n\s+-+\n"
        r"\s+You can check full logs with 'job\.stderr\(0\)' and 'job\.stdout\(0\)'or at paths:\n"
        r"\s+-\s(?P<err_path>(\/\w+)+.err)",
        log,
    )
    all_sigsegv_path = [x[0] for x in all_paths]
    all_proof_log_dirs = []
    for p in all_sigsegv_path:
        llog = open(p[: -len(".err")] + ".out", "r").read()
        match = re.search(r"(?P<dir>(\/\w+)+\/proof_logs_(\w+))", llog)
        assert match is not None
        all_proof_log_dirs.append(match.group("dir"))
    return all_proof_log_dirs
