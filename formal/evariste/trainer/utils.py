# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# courtesy of glample's notebook of doom.
from typing import Any, Tuple, List, Dict
from collections import Counter
from logging import getLogger
from pathlib import Path
import os
import math
import time
import shutil
import traceback
import numpy as np

from evariste import json as json
from params.params import flatten_dict

logger = getLogger()
Metrics = List[Dict[str, Any]]


class IncompleteEval(Exception):
    pass


def module_reload_paths(s: str, module_names: List[str]) -> Dict[str, Path]:
    split = [x.split(":") for x in s.split(",")]
    assert all(
        name in module_names and os.path.isfile(path) for name, path in split
    ), f"{module_names} {s}"
    return {name: Path(path) for name, path in split}


def get_secs(s):
    def parse_h_m_s(s):
        assert len(s) in [7, 8]
        if len(s) == 7:
            assert s[1] == ":" and s[4] == ":"
            h = s[0]
            m = s[2:4]
            s = s[-2:]
        else:
            assert s[2] == ":" and s[5] == ":"
            h = s[:2]
            m = s[3:5]
            s = s[-2:]
        assert h.isdigit() and m.isdigit() and s.isdigit()
        return int(s) + 60 * int(m) + 3600 * int(h)

    if len(s) in [7, 8]:
        return parse_h_m_s(s)
    else:
        i = s.find(" day")
        d = s[:i]
        assert d.isdigit()
        return 86400 * int(d) + parse_h_m_s(s[s.find(", ") + 2 :])


def get_prover_bwd_score(dump_path: str, do_raise=True):
    if not os.path.isfile(os.path.join(dump_path, "done")) and do_raise:
        raise IncompleteEval
    path = os.path.join(dump_path, "results.json")
    assert os.path.isfile(path)
    proved, failed = set(), set()
    with open(path, "r") as f:
        for line in f:
            loaded = json.loads(line)
            if loaded["success"]:
                proved.add(loaded["label"])
            else:
                failed.add(loaded["label"])
    assert len(proved.intersection(failed)) == 0, dump_path
    return (len(proved) / (len(proved) + len(failed))), len(proved)


def pass_at_k(n, c, k):
    """
    Unbiased estimate of the % solved for k trials given n >= k trials
    Taken from: https://arxiv.org/pdf/2107.03374.pdf

    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def compute_pass_at_k(infos, k=32):
    names = set(info["name"] for info in infos)
    trials = Counter([info["name"] for info in infos])
    succeed = Counter([info["name"] for info in infos if info["solved"]])
    scores = []
    for name in names:
        score = pass_at_k(n=trials[name], c=succeed.get(name, 0), k=k)
        scores.append(score)
    return np.mean(scores)


def get_prover_fwd_score(dump_path: Path) -> Dict[str, Any]:
    from evariste.comms.zip_store import ZipStore

    results_path = dump_path / "results_v3_1.json"

    # cached results
    if results_path.exists():
        with results_path.open("r") as fp:
            results = json.load(fp)
        return results

    job_dirs = [p for p in dump_path.iterdir() if p.name.startswith("job_")]
    if len(job_dirs) == 0:
        raise IncompleteEval
    for job_dir in job_dirs:
        if not (job_dir / "done").exists():
            raise IncompleteEval

    infos = []
    for job_path in job_dirs:

        store = ZipStore(job_path)
        try:
            infos.extend(store.read_jsonl("infos"))
        except FileNotFoundError:
            raise IncompleteEval

    names = set(info["name"] for info in infos)
    solved_names = {info["name"] for info in infos if info["solved"]}
    n_solved = len(solved_names)
    total = len(names)
    solved_sizes = [
        info["solved_size"] for info in infos if info.get("solved_size") is not None
    ]

    min_proof_size_by_label: Dict[str, int] = {}
    for info in infos:
        label = info["name"]
        size = info.get("solved_size")
        if size is None:
            continue
        if label not in min_proof_size_by_label:
            min_proof_size_by_label[label] = size
        else:
            min_proof_size_by_label[label] = min(min_proof_size_by_label[label], size)
    min_proof_size_by_label_values = list(min_proof_size_by_label.values())

    pass_at_32 = compute_pass_at_k(infos, k=32)

    results = {
        "proven": n_solved / total,
        "proven-n": n_solved,
        "pass@32": pass_at_32,
        "proof_size": np.mean(solved_sizes) if solved_sizes else -0.1,
        "min_proof_size_by_label": np.mean(min_proof_size_by_label_values)
        if min_proof_size_by_label
        else -0.1,
    }

    # caching results
    tmp = dump_path / "results.json.tmp"
    with tmp.open("w") as fp:
        json.dump(
            results, fp,
        )
    tmp.rename(results_path)

    return results


CACHE: Dict[str, Tuple[float, Any]] = {}


def parse_logs(path: str, force_reload: bool = False) -> List[Dict]:
    """Load logs. Only reload file if last modification time has changed."""
    t = os.path.getmtime(path)
    if force_reload or path not in CACHE or CACHE[path][0] < t:
        CACHE[path] = (t, _parse_logs(path))
    return CACHE[path][1]


def _parse_logs(path: str) -> List[Dict]:
    lines = [line.rstrip() for line in open(path, "r")]
    logs = []
    prev_epoch = -1
    log_pattern = "__log__:"
    for line in lines:
        try:
            pos = line.find(log_pattern)
            if pos < 0:
                continue
            split = line.split(" - ")
            data = json.loads(line[pos + len(log_pattern) :])
            for k in list(data.keys()):
                if (
                    k.endswith("-proven")
                    and isinstance(data[k], str)
                    and not k.startswith("fwd-")
                ):
                    try:
                        acc_proven, n_proven = get_prover_bwd_score(dump_path=data[k])
                        data[k] = acc_proven
                        data[k + "-n"] = n_proven
                    except IncompleteEval:
                        data[k] = -1
                elif (
                    k.startswith("fwd-")
                    and k.endswith("-proven")
                    and isinstance(data[k], str)
                ):
                    try:
                        results = get_prover_fwd_score(dump_path=Path(data[k]))
                        data[k] = results["proven"]
                        data[k + "-pass@32"] = results["pass@32"]
                        data[k + "-n"] = results["proven-n"]
                        data[k + "-proof-size"] = results["proof_size"]
                        data[k + "-min-proof-size-by-label"] = results[
                            "min_proof_size_by_label"
                        ]
                    except IncompleteEval:
                        data[k] = -0.1
                        data.pop(k)
            assert len(split) == 4, split
            assert "seconds" not in data and "hours" not in data
            data["seconds"] = get_secs(split[2])
            data["hours"] = data["seconds"] * 1.0 / 3600
            if data["epoch"] < prev_epoch:
                print(f"WARNING, EPOCH WENT BACK ({prev_epoch} -> {data['epoch']})")
                continue
            prev_epoch = data["epoch"]
            logs.append(data)
        except Exception as e:
            print(str(e))
            continue
    return logs


def get_exp_ids(exp_names, dump_paths):
    exp_ids = []
    for exp_name in exp_names:
        for dump_path in dump_paths:
            current = []
            if not os.path.isdir(os.path.join(dump_path, exp_name)):
                continue
            # print("Looking into: %s" % os.path.join(dump_path, exp_name))
            for exp_id in sorted(os.listdir(os.path.join(dump_path, exp_name))):
                if not os.path.isdir(os.path.join(dump_path, exp_name, exp_id)):
                    print('"%s/%s" is not a directory' % (exp_name, exp_id))
                    continue
                current.append((dump_path, exp_name, exp_id))
            # if current:
            #     print('%s - %s (%i)' % (dump_path, exp_name, len(current)))
            exp_ids.extend(current)
    return exp_ids


def parse_experiments(experiments, exp_ids, overwrite: bool):
    for dump_path, exp_name, exp_id in exp_ids:
        try:
            exp_path = os.path.join(dump_path, exp_name, exp_id)
            log_path = os.path.join(exp_path, "train.log")
            params_file = "params.json"
            if os.path.exists(os.path.join(exp_path, "launcher_params.json")):
                params_file = "launcher_params.json"
            params_path = os.path.join(exp_path, params_file)
            if not os.path.isfile(log_path):
                # print("No log file in %s" % exp_path)
                continue
            if not os.path.isfile(params_path):
                print("No parameter file in %s" % exp_path)
                continue
            with open(params_path, "r") as f:
                params = flatten_dict(json.load(f))
            if exp_id in experiments:
                if overwrite:
                    del experiments[exp_id]
                else:
                    assert False, exp_id
            assert exp_id not in experiments
            params = dict(params)
            logs = parse_logs(log_path)
            if "name" not in params:
                params["name"] = exp_name
            assert params["name"] == exp_name
            experiments[exp_id] = {
                "id": exp_id,
                "name": exp_name,
                "params": params,
                "logs": logs,
                "exp_path": exp_path,
            }
        except Exception:
            print("Failed %s - %s" % (exp_name, exp_id))
            print(traceback.format_exc())
    return experiments


def get_experiments(
    names: List[str], dump_paths: List[str], use_metrics_jsonl: bool = False
):
    """
    For adversarial training, it is recommended to use use_metrics_jsonl=True
    for the moment. It will parse metrics from actors and learners.
    """
    exp_ids = get_exp_ids(names, dump_paths)
    if not use_metrics_jsonl:
        return parse_experiments({}, exp_ids, overwrite=True)
    else:
        return parse_experiments_with_metrics(
            exp_ids,
            # needed for the moment for adversarial training
            allow_missing_params=True,
        )


def parse_experiments_with_metrics(exp_ids, allow_missing_params: bool = False):
    experiments = {}
    for dump_path, exp_name, exp_id in exp_ids:
        exp_path = Path(dump_path) / exp_name / exp_id
        if not exp_path.exists():
            print(f"{exp_path} doesn't exist")
            continue
        params_path = exp_path / "params.json"
        if params_path.exists():
            with params_path.open("r") as f:
                params = flatten_dict(json.load(f))
        elif allow_missing_params:
            # useful for the moment for adversarial training
            print(f"Missing params json {params_path}, replacing by empty params")
            params = {}
        else:
            print(f"Missing params json: {params_path}")
            continue
        metrics = parse_metrics_jsonl(str(exp_path))
        experiments[exp_id] = {
            "id": exp_id,
            "name": exp_name,
            "exp_path": str(exp_path),
            "params": params,
            "metrics": metrics,
        }
    return experiments


def parse_metrics_jsonl(
    dump_path: str,
    root_tag: str = "",
    ignore_prefixes: Tuple[str, ...] = ("comms", "generations_", "tb"),
) -> Metrics:
    """
    Parse recursively all folders in dump_path, looking for metrics.{tag}.jsonl files
    to load them. (The recursion is needed for adversarial training folders)

    All keys in metrics.json are prefixed by 1) the subfolder (if not in root path)
    2) the tag in "metrics.{tag}.jsonl" (if any)

    So for adversarial training, metrics look like:
    "prover_trainer/mm_valid_xxx".


    For the moments metrics are just concatenated, the user would be responsible
    for filtering, sorting, merging (ineed it is not trivial to merge since
    global_steps are not the same across metrics.jsonl files)
    """
    root = Path(dump_path)
    root_tag = _add_trailing_if_not_empty(root_tag, "/")
    metrics = []
    for path in root.iterdir():
        if path.name.startswith("metrics.") and path.name.endswith(".jsonl"):
            parsed_tag = path.name[len("metrics.") : -len(".jsonl")]
            tag = f"{root_tag}{_add_trailing_if_not_empty(parsed_tag, '/')}"
            with path.open("r") as fp:
                for line in fp.readlines():
                    entry = json.loads(line.strip())
                    # we add the tag into the key of the dict
                    if tag:
                        entry = {f"{tag}{k}": v for k, v in entry.items()}
                    metrics.append(entry)
        elif path.is_dir():
            if path.name.startswith(ignore_prefixes):  # no logs in comms:
                continue
            metrics.extend(
                parse_metrics_jsonl(
                    str(path),
                    root_tag=f"{root_tag}{path.name}",
                    ignore_prefixes=ignore_prefixes,
                )
            )
    return metrics


def _add_trailing_if_not_empty(a_str: str, suffix: str) -> str:
    """
    "tag" + "/" -> "tag/"
    "" + "/" -> ""
    """
    if a_str:
        return f"{a_str}{suffix}"
    return a_str


def clean_fwd_eval_folders(dump_path: Path):
    for folder in (
        f
        for f in dump_path.iterdir()
        if (f.name.startswith("fwd-") and f.name.endswith("-proven"))
        or f.name == "fwd_eval"
    ):

        metric = folder.name if folder.name != "fwd_eval" else "fwd-valid-lean-proven"

        def lstrip(prefix: str, name: str) -> str:
            assert name.startswith(prefix)
            return name[len(prefix) :]

        epochs = sorted([int(lstrip("epoch_", p.name)) for p in folder.iterdir()])

        best_score = -math.inf
        for epoch in epochs:
            subfolder = folder / f"epoch_{epoch}"
            try:
                results = get_prover_fwd_score(subfolder)
            except IncompleteEval:
                continue
            score = results["proven"]
            checkpoint_path = subfolder / f"checkpoint.{epoch}.pth"
            tgt_path = dump_path / f"best-{metric}.pth"

            if score > best_score:
                best_score = score

                if checkpoint_path.exists():
                    checkpoint_path.rename(tgt_path)
            else:
                if checkpoint_path.exists():
                    checkpoint_path.unlink()


def nfs_barrier(
    dump_path: str,
    rank: int,
    world_size: int,
    timeout_s: float = 3600,
    name: str = "nfs_barrier",
):
    """NFS barrier - instead of using NCCL barrier that timeout after 30min"""

    lock_dir = Path(dump_path) / name
    trash_dir = Path(dump_path) / f"{name}.trash"
    if rank == 0:
        try:
            assert not lock_dir.exists(), lock_dir
            lock_dir.mkdir()
            _wait_for_all(lock_dir, rank, world_size, timeout_s)
            _wait_for_all_workers_to_see_that_everybody_is_ready(
                lock_dir, world_size, max_wait=3.0
            )
        finally:
            lock_dir.rename(trash_dir)  # atomic ?
            shutil.rmtree(trash_dir)
    else:
        _wait_for_all(lock_dir, rank, world_size, timeout_s)


OK_SUFFIX = ".ok"


def _wait_for_all(lock_dir: Path, rank: int, world_size: int, timeout_s: float):
    log_duration = 60
    last_log = time.time()
    start = time.time()

    if rank == 0:
        assert lock_dir.exists()
    else:
        logger.info(f"Barrier: Waiting for lock_dir: {lock_dir}")
        while not lock_dir.exists():
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Barrier: Timeout reached, no lock_dir {lock_dir}")
            time.sleep(1.0)
        logger.info(f"Barrier: lock_dir exists")
    rank_file = lock_dir / f"{rank}"
    rank_file.touch()
    logger.info(f"Barrier: wrote {rank_file}")

    all_ready = False
    while not all_ready:
        present = {
            int(p.name) for p in lock_dir.iterdir() if not p.name.endswith(OK_SUFFIX)
        }
        if present == set(range(world_size)):
            all_ready = True
            # showing that you saw everybody was ready
            done_file = lock_dir / f"{rank}{OK_SUFFIX}"
            done_file.touch()
        elif time.time() - last_log > log_duration:
            logger.info(
                f"Barrier: {len(present)}/{world_size} ready "
                f"{time.time() - start:.02f}s/{timeout_s:.02f}s"
            )
            log_duration *= 2
            last_log = time.time()

        if not all_ready and time.time() - start > timeout_s:
            raise TimeoutError(
                f"Barrier: timeout {timeout_s} reached, "
                f"waiting for {set(range(world_size)) - present}"
            )

    logger.info("Barrier: all ready!")


def _wait_for_all_workers_to_see_that_everybody_is_ready(
    lock_dir: Path, world_size: int, max_wait: float
):
    """
    For master, waiting for everybody to show that
    they saw that everybody was ready...
    Like this master can delete the lock folder
    """
    all_ok = False
    start = time.time()
    while not all_ok:
        present = {
            int(p.name[: -len(OK_SUFFIX)])
            for p in lock_dir.iterdir()
            if p.name.endswith(OK_SUFFIX)
        }
        if present == set(range(world_size)):
            all_ok = True

        if not all_ok and time.time() - start > max_wait:
            raise TimeoutError(f"Not all workers wrote their .ok. present: {present}")
