# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import Counter, defaultdict
from evariste import json as json
from dataclasses import dataclass
from typing import List, cast
from pathlib import Path
from pprint import pprint
import numpy as np


from params import Params
from params.params import cfg_from_cli
from evariste.comms.zip_store import ZipStore
from evariste.trainer.utils import compute_pass_at_k
from evariste.forward.common import GenerationHistory


@dataclass
class Config(Params):
    path: str
    ignore_missing_jobs: bool = False


def analyse_(cfg: Config):
    generation_path = Path(cfg.path)
    assert generation_path.exists(), generation_path

    job_dirs = [p for p in generation_path.iterdir() if p.name.startswith("job_")]
    if len(job_dirs) == 0:
        job_dirs = [generation_path]

    with (job_dirs[0] / "config.json").open("r") as fp:
        config = json.load(fp)

    pprint(config)

    infos = []
    for job_path in job_dirs:
        store = ZipStore(job_path)

        try:
            infos.extend(store.read_jsonl("infos"))
        except FileNotFoundError:
            if cfg.ignore_missing_jobs:
                continue
            raise

    print(f"For info: {len([item for item in infos if item['solved']])} / {len(infos)}")

    names = set(info["name"] for info in infos)
    solved_names = {info["name"] for info in infos if info["solved"]}
    not_solved_names = {name for name in names if name not in solved_names}
    n_solved = len(solved_names)
    total = len(names)
    print(f"Solved: {n_solved} / {total} - {100 * n_solved / total:.02f}%")

    failure_reasons = [item["stopped"][0] for item in infos if not item["solved"]]
    print(f"Failure reasons: {Counter(failure_reasons).most_common()}")

    try:
        with (generation_path / "solved.json").open("w") as fp:
            json.dump(list(solved_names), fp)
        print(generation_path / "solved.json", len(solved_names))
        with (generation_path / "not_solved.json").open("w") as fp:
            json.dump(list(not_solved_names), fp)
        print(generation_path / "not_solved.json", len(not_solved_names))
    except PermissionError:
        pass

    if config.get("n_trials_by_trm", 1) > 1:
        solved_dict = defaultdict(int)
        for info in infos:
            if info["solved"]:
                solved_dict[info["name"]] += 1
        assert n_solved == len(solved_dict)
        print(
            f"In total: {len([item for item in infos if item['solved']])} / {len(infos)}"
        )
        for i in range(1, 11):
            print(i, len([s for s in solved_dict if solved_dict[s] == i]))

    proof_sizes = [
        info["solved_size"] for info in infos if info.get("solved_size") is not None
    ]
    if proof_sizes:
        print(
            f"Avg proof size (solved): {np.mean(proof_sizes)} (max: {np.max(proof_sizes)}, median: {np.median(proof_sizes)})"
        )
        print("sizes", sorted(Counter(proof_sizes).most_common()))

    min_proof_size_by_label = {}
    for info in infos:
        label = info["name"]
        size = info.get("solved_size")
        if size is None:
            continue
        if label not in min_proof_size_by_label:
            min_proof_size_by_label[label] = size
        else:
            min_proof_size_by_label[label] = min(min_proof_size_by_label[label], size)
    min_proof_size_by_label = list(min_proof_size_by_label.values())
    if min_proof_size_by_label:
        print(
            f"Avg min proof size by label: {np.mean(min_proof_size_by_label)} (max: {np.max(min_proof_size_by_label)}, median: {np.median(min_proof_size_by_label)})"
        )
        print("sizes", sorted(Counter(min_proof_size_by_label).most_common()))

    errors = []
    n_generated = 0

    generations = []
    solved_gens = {}
    for job_path in job_dirs:
        store = ZipStore(job_path)
        try:
            these_generations = store.read_pickle_zip("generations")
        except FileNotFoundError:
            if cfg.ignore_missing_jobs:
                continue
            raise
        these_generations = cast(List[GenerationHistory], these_generations)
        generations.extend(these_generations)
        for generation in these_generations:
            errors.extend(generation.errors())
            n_generated += len(generation.forward_steps())
            if generation.info.solved and generation.goal.label not in solved_gens:
                solved_gens[generation.goal.label] = generation
    n_errors = len(errors)

    n_thms_by_generation = compute_n_new_thms_by_generation(generations)
    print(f"n_thms_by_generation {n_thms_by_generation}")

    print(f"n_valid_steps: {n_generated}")
    print(f"n_errors: {n_errors}")
    tot = n_errors + n_generated
    print(f"ratio: {100*n_generated/tot:.2f}")
    print(f"n_generated / stack: {n_generated / len(infos):0.2f}")

    gen_errors = [err.type for err in errors]
    most_common_errs = Counter(gen_errors).most_common()
    print(f"Generation errors:")
    print(f"Count: {most_common_errs}")
    print(
        f"Proportion of generated nodes (%): "
        f"{[(k, round(100* v / tot, 2)) for k, v in most_common_errs]}"
    )

    pass_at_32 = compute_pass_at_k(infos, k=32)
    pass_at_1 = compute_pass_at_k(infos, k=1)
    pass_at_4 = compute_pass_at_k(infos, k=4)
    print(f"pass@1: {pass_at_1}")
    print(f"pass@4: {pass_at_4}")
    print(f"pass@32: {pass_at_32}")


def compute_n_new_thms_by_generation(generations: List[GenerationHistory]):
    label_to_thms = defaultdict(set)
    thms = set()
    for generation in generations:
        label = generation.goal.label
        label_to_thms[label].update([s.generated for s in generation.forward_steps()])
        thms.update((s.generated for s in generation.forward_steps()))

    return len(thms) / len(generations)


if __name__ == "__main__":
    cfg = cfg_from_cli(schema=Config)
    cfg = cast(Config, cfg)
    analyse_(cfg)
