# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import defaultdict
from subprocess import check_output
from typing import Set
import re
import sys

from evariste import json as json


def cancel_group(job_id: str, children):
    """Cancels an mcts job and all its descendent"""
    n_total_cancelled = 1
    for n, c in children.items():
        n_total_cancelled += len(c)
    print(f"Cancelling {n_total_cancelled} jobs. Continue ? [yn]")
    r = input()
    if r == "y":
        for n in children.keys():
            check_output(f"scancel -n {n}", shell=True)
        try:
            int(job_id)  # will work if job_id is from slurm, won't work on local
            check_output(f"scancel {job_id}", shell=True)
        except ValueError:
            print("Don't forget to stop the local controller ...")
    else:
        print(".. aborted")


args = sys.argv
if len(args) == 1:
    args.append("show")
args = sys.argv[1:]

available_actions = {"show", "cancel"}
assert args[0] in available_actions, f"action {args[0]} not in {available_actions}"
action = args[0]

if action == "cancel":
    assert len(args) >= 2, "expected at least 1 job id after action cancel"


sacct = check_output(
    "sacct --noheader --format=JobID,JobName%150,State --state=PENDING,RUNNING",
    shell=True,
).decode()


job_ids: Set[str] = set()
jobs = defaultdict(list)

for s in sacct.split("\n"):
    if len(s) > 0:
        job_id, name, state = re.sub(" +", " ", s).strip().split(" ")
        job_ids.add(job_id)
        if "." in job_id:
            continue
        # \w+ matches word so won't match things launched via slurm
        match = re.match(r"prover_gang_mcts_training_(\w+)", name)
        if match is not None:
            # job was launched manually
            job_ids.add(match.group(1))
        jobs[name].append((job_id, name, state))

job_group = defaultdict(lambda: defaultdict(list))

for name in jobs.keys():
    for job_id in job_ids:
        if job_id in name:
            job_group[job_id][name] += jobs[name]


if action == "show":
    print(
        json.dumps(
            {
                launcher_id: {
                    name: sum((val[2] == "RUNNING" for val in vals), 0)
                    for name, vals in children.items()
                }
                for launcher_id, children in job_group.items()
            },
            indent=2,
        )
    )
elif action == "cancel":
    for job_id in args[1:]:
        if job_id in job_group:
            print(f"Cancelling {job_id}")
            cancel_group(job_id, job_group[job_id])
        else:
            print(f"{job_id} not found")
else:
    raise NotImplementedError(action)
