# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, List, Iterable, Dict, Union, Tuple
from dataclasses import dataclass, field
import fire
import getpass
import re
from pathlib import Path
from collections import defaultdict
import subprocess
import numpy as np
import pickle
from datetime import datetime
import os
import shutil

import submitit
from submitit.slurm.slurm import SlurmInfoWatcher
from evariste.forward import forward_model_factory
from evariste.forward.forward_model_configs import SAMPLING_PROVER_CFG
from evariste.forward.fwd_eq.eq_env_helper import EQFwdEnvHelper
from evariste.forward.fwd_eq.eq_helpers import evaluate_eq_generation
from evariste.model.checkpoints import get_latest_checkpoint
from evariste.utils import load_stream
import evariste.json as json
from evariste.adversarial_offline.tooling.prove_generations import one_eval


GOOD_STATUS = {"COMPLETED", "RUNNING"}


@dataclass
class Dirs:
    user_workdir: str
    user_gen_workdir: str
    user_dump_path: str
    user_generations_path: str

    @staticmethod
    def from_user(user: Optional[str] = None) -> "Dirs":
        return Dirs(
            user_workdir=user_workdirs.get(user or getpass.getuser(), f"/workdir"),
            user_gen_workdir=user_gen_workdirs.get(
                user or getpass.getuser(), f"/gen_workdir",
            ),
            user_dump_path=user_dump_paths.get(
                user or getpass.getuser(), f"YOUR_PATH/dumped/",
            ),
            user_generations_path=user_generations_paths.get(
                user or getpass.getuser(), f"YOUR_PATH/generated/",
            ),
        )


@dataclass
class TrainerInfo:
    workdir: Path
    dump_path: Path
    main_job_id: str
    epoch: int
    duration_hours: Optional[float] = None
    duration_slurm: Optional[str] = None
    state: Optional[str] = None

    def show(self, full: bool = False):
        if not full and self.state == "COMPLETED":
            return
        print(f"\tEpoch {self.epoch} ({self.state}): {self.main_job_id}")
        assert self.state is not None
        if self.state != "COMPLETED" and not self.state.startswith("CANCELLED"):
            print(f"\t\t{self.workdir / self.main_job_id}_0_log.err")  # what to less

    def losses(self) -> Dict[str, List[float]]:
        # reparse first because in a notebook we want this to always get updated
        # hopefully this isn't slow as hell
        losses = defaultdict(list)
        with open(self.dump_path / "train.log", "r") as f:
            for l in f:
                idx = l.find("__log__:")
                if idx < 0:
                    continue
                for x, y in json.loads(l[idx + len("__log__:") :]).items():
                    losses[x].append(y)
        return losses

    def has_issue(self):
        return self.state not in GOOD_STATUS


@dataclass
class GenStats:
    hard: float
    invalid_steps: float
    errors: Dict[str, float]
    stats: Dict[str, float]


@dataclass
class GenInfo:
    workdir: Path
    epoch: int
    main_job_id: str
    array_id: str
    duration_hours: Optional[float] = None
    duration_slurm: List[str] = field(default_factory=lambda: [])
    statuses: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    gen_stats: Optional[GenStats] = None

    def show(self, full: bool = False):
        if not full and self.is_complete():
            return
        print(f"\tEpoch {self.epoch}: {self.main_job_id}")
        for x, y in self.statuses.items():
            print(f"\t\t{x}: {len(y)}")
            if x != "COMPLETED" and not x.startswith("CANCELLED"):
                print(f"\t\t\t{self.workdir / y[0]}_0.err")  # what to less

    def has_issue(self):
        return any([x not in GOOD_STATUS for x in self.statuses.keys()])

    def is_complete(self):
        return (
            len(self.statuses) == 1 and next(iter(self.statuses.keys())) == "COMPLETED"
        )

    def get_sample_gens(self, dirs: Dirs, exp_name: str, exp_id: str):
        gen_paths = (
            Path(dirs.user_generations_path) / f"{exp_name}_gen_{self.epoch}"
        ).glob(f"{exp_id}_*/generations/*.pkl")
        to_ret = []
        for f in gen_paths:
            for _name, goal, _pf, proved in pickle.load(open(f, "rb")):
                if not proved:
                    to_ret.append(goal.theorem)
            if len(to_ret) > 50:
                break
        return to_ret


@dataclass
class OfflineIterInfo:
    dirs: Dirs
    exp_name: str
    exp_id: str
    workdir: Path
    stdout: Path
    stderr: Path
    main_job_id: str
    prover_trainers: List[TrainerInfo] = field(default_factory=lambda: [])
    generator_trainers: List[TrainerInfo] = field(default_factory=lambda: [])
    generations: List[GenInfo] = field(default_factory=lambda: [])
    state: Optional[str] = None
    swept_params: Optional[Dict] = None  # set using swept_params

    @property
    def params(self) -> Dict:
        with open(self.stdout, "r") as f:
            for l in f.readlines():
                if l.startswith("__grid_params__: "):
                    return json.loads(l[len("__grid_params__: ") :])
        raise RuntimeError("didn't find grid params")

    def prop_hard(self) -> List[float]:
        with open(self.stdout, "r") as f:
            return [
                float(x)
                for x in re.findall(r"Received \d+ \/ \d+ = (\d+.\d+) \%", f.read())
            ]

    def cross_eval(self, only_hard: bool = True) -> np.ndarray:
        filename = "results.pkl" if only_hard else "results_all.pkl"
        results = pickle.load(
            open(
                Path(self.dirs.user_gen_workdir)
                / f"{self.exp_name}_{self.exp_id}"
                / filename,
                "rb",
            )
        )
        filtered = {x: y for x, y in results.items() if y > 0}
        ge, pe = zip(*list(filtered.keys()))
        res = np.full((max(ge) + 1, max(pe) + 1), float("nan"))

        has_data = set()
        for (x, y), z in filtered.items():
            res[x][y] = z
            has_data.add(x)

        sorted_has_data = sorted(has_data) + [1_000_000]
        # reset off diagonal to nans. It's expected to be ~0 but messes up the overall contrast
        next_data_id = 0
        for p in range(max(pe)):
            while (
                next_data_id < len(sorted_has_data)
                and sorted_has_data[next_data_id] <= p
            ):
                next_data_id += 1
            try:
                for i in range(p + 1, sorted_has_data[next_data_id] + 1):
                    if res[p][i] > 0.01:
                        print(
                            f"HIGH VALUE HIDDEN AS NAN! res[{p},{i}]={res[p][i]} // {self.exp_id}"
                        )
                res[p][p + 1 : sorted_has_data[next_data_id] + 1] = float("nan")
            except:
                pass
        return res[sorted(has_data), :]

    def losses(self, src: str) -> Dict[str, List[float]]:
        res: Dict[str, List[float]] = defaultdict(list)
        the_src = {
            "prover_trainer": self.prover_trainers,
            "generator_trainer": self.generator_trainers,
        }[src]
        for t in the_src:
            for x, y in t.losses().items():
                res[x] += y
        return res

    def has_issue(self):
        return any(
            (
                x.has_issue()
                for x in self.prover_trainers
                + self.generator_trainers
                + self.generations
            )
        )

    def avg_duration(self, kind: str):
        kind_to_list: Dict[str, Union[List[TrainerInfo], List[GenInfo]]] = {
            "generator_trainers": self.generator_trainers,
            "prover_trainers": self.prover_trainers,
            "generations": self.generations,
        }
        return np.mean(
            [
                x.duration_hours
                for x in kind_to_list[kind]
                if x.duration_hours is not None
            ]
        )


def get_gen_stats(
    paths: Iterable[Path],
    subsampling: float = 0.1,
    eq_env_helper: Optional[EQFwdEnvHelper] = None,
) -> GenStats:
    """Globs (*.pkl) in path then parse subsampled generations to get % hard, as well as generation stats

    :param paths: iterable paths in which .glob(*.pkl) gives all generations to load
    :param subsampling: how much subsampling should be done when loading, defaults to 0.1
    :return: the generation stats
    """
    total, hard, invalid, steps = 0, 0, 0, 0
    all_stats = defaultdict(list)
    errors: Dict[str, float] = defaultdict(float)
    for _name, bwd_goal, proof_search, proved in load_stream(
        paths, subsampling=subsampling
    ):
        total += 1
        hard += not proved
        for step in proof_search.generation.stack:
            if not step.step:
                invalid += 1
                errors[step.err.type] += 1
            steps += 1
        if eq_env_helper is not None:
            stats = evaluate_eq_generation(
                bwd_goal, proof_search.generation, proved, eq_env_helper
            )
            stats = {re.sub("^avg_", "", x): y for x, y in stats.items()}
            for x, y in stats.items():
                all_stats[x].append(y)

    return GenStats(
        hard=hard / total,
        invalid_steps=invalid / steps,
        errors={x: y / invalid for x, y in errors.items()},
        stats={x: float(np.mean(y)) for x, y in all_stats.items()},
    )


def grab_job_statuses(offline_exps: Dict[str, OfflineIterInfo]):
    """Grabs slurm statuses for all jobs in offline_exps. Modify offline_exps in place
    :param offline_exps: contains all experiments to grab statuses for. is modified inplace
    """
    all_ids = []
    for exp in offline_exps.values():
        all_ids.append(exp.main_job_id)
        for t in exp.generator_trainers:
            all_ids.append(t.main_job_id)
        for t in exp.prover_trainers:
            all_ids.append(t.main_job_id)
        for g in exp.generations:
            all_ids.append(g.array_id)
    to_add = " ".join([f"-j {j}" for j in all_ids]).split(" ")

    # cmd from submitit/slurm/slurm.py I don't want to figure out how to give multiple jobs to SlurmInfoWatcher
    sacct_output = subprocess.check_output(
        ["sacct", "-o", "JobID,State,NodeList,Elapsed", "--parsable2"] + to_add,
        shell=False,
    )
    job_states = SlurmInfoWatcher().read_info(sacct_output)

    for exp in offline_exps.values():
        exp.state = job_states[exp.main_job_id]["State"]
        for t in exp.generator_trainers:
            t.state = job_states[t.main_job_id]["State"]
            t.duration_slurm = job_states[t.main_job_id]["Elapsed"]
        for t in exp.prover_trainers:
            t.state = job_states[t.main_job_id]["State"]
            t.duration_slurm = job_states[t.main_job_id]["Elapsed"]
        for g in exp.generations:
            for x, y in job_states.items():
                if x.split("_")[0] == g.array_id:
                    g.statuses[y["State"]].append(x)
                    g.duration_slurm.append(y["Elapsed"])


def get_durations(offline_exps: Dict[str, OfflineIterInfo]):
    """Parse main_exp/*out to get durations of each steps.
    TODO: get distinct trainer timings (logs need to be improved)
    TODO: maybe get this from other logs / sacct instead ?

    :param offline_exps: the offline_exps, modified inplace
    """
    for exp_id, exp in offline_exps.items():
        with open(exp.stdout, "r") as f:
            content = f.read()

            trainer_launches = re.findall(
                r"(\d+\/\d+\/\d+ \d+:\d+:\d+) - (?:\d|\s|\w|,|:)*- Launching (?:prover )?retraining in",
                content,
            )
            dt_tl = [
                datetime.strptime(x, "%m/%d/%y %H:%M:%S") for x in trainer_launches
            ]

            trainer_results = re.findall(
                r"(\d+\/\d+\/\d+ \d+:\d+:\d+) - (?:\d|\s|\w|,|:)*- Job results: \[None(?:, None)*\]",
                content,
            )
            dt_tr = [datetime.strptime(x, "%m/%d/%y %H:%M:%S") for x in trainer_results]

            # check that launch of prover trainer / prover are less than a minute apart
            not_ok = False
            for i, j in zip(dt_tl[:-1:2], dt_tl[1::2]):
                if (j - i).total_seconds() >= 60:
                    print(f"[WARNING]: weird trainer times in {exp_id}")
                    not_ok = True
                    break
            if not_ok:
                continue

            # if that's the case, we can only use the generator trainer launch
            start_train = dt_tl[:-1:2]
            if (
                len(start_train)
                != len(exp.prover_trainers)
                != len(exp.generator_trainers)
            ):
                print(
                    f"[WARNING]: {len(start_train)} != {len(exp.prover_trainers)} != {len(exp.generator_trainers)} ... {exp_id} // check {exp.stdout}"
                )
                continue

            # for results, let's take the max for now. we need better logs to understand what took long.
            got_results = dt_tr[1::2]

            if len(start_train) - len(got_results) > 1:
                print(
                    f"[WARNING]: {len(start_train)} // {len(got_results)} ? skipping {exp_id} // check {exp.stdout}"
                )
                continue

            for trainers in [exp.generator_trainers, exp.prover_trainers]:
                for t, (e, s) in zip(trainers, zip(got_results, start_train)):
                    t.duration_hours = (e - s).total_seconds() / 3600

            # Now compute gen duration
            launch_generation = re.findall(
                r"(\d+\/\d+\/\d+ \d+:\d+:\d+) - (?:\d|\s|\w|,|:)*- Launching generator",
                content,
            )
            dt_lg = [
                datetime.strptime(x, "%m/%d/%y %H:%M:%S") for x in launch_generation
            ]

            received_gen = re.findall(
                r"(\d+\/\d+\/\d+ \d+:\d+:\d+) - (?:\d|\s|\w|,|:)*- Received \d+ \/ \d+ = \d+.\d+ % hard statements",
                content,
            )
            dt_rg = [datetime.strptime(x, "%m/%d/%y %H:%M:%S") for x in received_gen]

            # we can have launched but not received once
            if len(dt_lg) - len(dt_rg) > 1:
                print(
                    f"[WARNING]: generators : {len(dt_lg)} // {len(dt_rg)} ? skipping {exp_id} // check {exp.stdout}"
                )
                continue
            for g, (e, s) in zip(exp.generations, zip(dt_rg, dt_lg)):
                g.duration_hours = (e - s).total_seconds() / 3600


user_workdirs: Dict[str, str] = {}
user_gen_workdirs: Dict[str, str] = {}
user_dump_paths: Dict[str, str] = {}
user_generations_paths: Dict[str, str] = {}


def get_all_offline_exps(
    exp_name, exp_id: Optional[List[str]] = None, user: Optional[str] = None
) -> Dict[str, OfflineIterInfo]:
    offline_exps: Dict[str, OfflineIterInfo] = {}
    dirs = Dirs.from_user(user)

    # Gather all job_id for which status needs to be checked
    for fn in (Path(dirs.user_workdir) / exp_name).glob(f"*/*stdout"):
        with open(fn, "r") as f:
            content = f.read()
            this_exp_id = re.findall(r"Experiment ID: (?P<id>\w+)", content)[0]
            if not (exp_id is None or this_exp_id in exp_id):
                continue
            offline_exps[this_exp_id] = OfflineIterInfo(
                dirs=dirs,
                exp_name=exp_name,
                exp_id=this_exp_id,
                workdir=fn.parent,
                stdout=fn,
                stderr=fn.parent / f"{fn.stem}.stderr",
                main_job_id=fn.stem,
            )

    if not offline_exps:
        return {}

    for suffix in ["prover_train", "train", "gen"]:
        for fn in (Path(dirs.user_gen_workdir) / f"{exp_name}_{suffix}").glob("*"):
            this_exp_id = fn.name
            if this_exp_id not in offline_exps:
                continue
            exp = offline_exps[this_exp_id]
            for epoch_id in sorted([int(x.name) for x in fn.glob("*")]):
                epoch_fn = fn / str(epoch_id)
                try:
                    first_err = next(epoch_fn.glob("*err"))
                except StopIteration as e:
                    # job submitted but not started yet
                    continue

                with open(first_err, "r") as f:
                    content = f.read()
                    main_job_id = re.findall(r"srun: jobid\s+: (\d+)", content)[0]

                dump_path = (
                    Path(dirs.user_dump_path)
                    / f"{exp_name}_{suffix}"
                    / this_exp_id
                    / str(epoch_id)
                )
                if suffix == "prover_train":
                    exp.prover_trainers.append(
                        TrainerInfo(
                            workdir=epoch_fn,
                            main_job_id=main_job_id,
                            epoch=epoch_id,
                            dump_path=dump_path,
                        )
                    )
                elif suffix == "train":
                    exp.generator_trainers.append(
                        TrainerInfo(
                            workdir=epoch_fn,
                            main_job_id=main_job_id,
                            epoch=epoch_id,
                            dump_path=dump_path,
                        )
                    )
                elif suffix == "gen":
                    exp.generations.append(
                        GenInfo(
                            workdir=epoch_fn,
                            main_job_id=main_job_id,
                            array_id=first_err.stem.split("_")[0],
                            epoch=epoch_id,
                        )
                    )
                else:
                    raise RuntimeError()

    return offline_exps


def offline_iter(
    exp_name: str,
    exp_id: Optional[List[str]] = None,
    full: bool = False,
    user: Optional[str] = None,
    with_quant_gen_stats: bool = False,
) -> None:
    """Run to get status of all experiments under name `exp_name`.
    First gather all relevant job ids and get their status in saccount.
    For non failed jobs, gather relevant statistics.
    For failed jobs, point to the relevant stderr.
    Check if cross_eval is up to date or not. If not up to date, eventually run-it.

    :param exp_name: exp_name to check
    :param exp_id: comma separated list of exp_id to check, defaults to None, which means check all exp_ids
    :param user: if set, find exp_name using another user than current one. defaults to None. Set global variables user_((gen_)?workdirs|dump_paths) in code if needed.
    :param full: if set, also to display info for "COMPLETED" jobs
    :param with_quant_gen_stats: if set, load a portion of generated proofs, gathers and displays stats
    """

    dirs = Dirs.from_user(user)

    offline_exps: Dict[str, OfflineIterInfo] = {}
    offline_exps = get_all_offline_exps(exp_name, exp_id, user)
    if len(offline_exps) == 0:
        print("No exps found")
        return

    grab_job_statuses(offline_exps)

    get_durations(offline_exps)

    if with_quant_gen_stats:
        print("Collecting generation statistics")
        for this_exp_id, exp in offline_exps.items():
            ckpt, _ = get_latest_checkpoint(str(exp.generator_trainers[0].dump_path))
            assert ckpt is not None

            (_, _, _, env_helper) = forward_model_factory.from_checkpoint(
                ckpt_path=ckpt,
                device_str=f"cpu",
                cfg=SAMPLING_PROVER_CFG,
                overwrite_tasks="eq_gen_notask",
            )

            assert isinstance(env_helper, EQFwdEnvHelper)
            for g in exp.generations:
                if not g.is_complete():
                    continue
                g.gen_stats = get_gen_stats(
                    (
                        Path(dirs.user_generations_path) / f"{exp_name}_gen_{g.epoch}"
                    ).glob(f"{this_exp_id}_*/generations/"),
                    eq_env_helper=env_helper,
                )

    for i, exp in offline_exps.items():
        print("=====================")
        print(i)
        print(f"Main job ({exp.state}): ", exp.main_job_id)
        print(
            f"Generator trainers (avg time {exp.avg_duration('generator_trainers'):.2f}h):"
        )
        for t in exp.generator_trainers:
            t.show(full)
        print(f"Prover trainers (avg time {exp.avg_duration('prover_trainers'):.2f}h):")
        for t in exp.prover_trainers:
            t.show(full)
        print(f"Generations (avg time {exp.avg_duration('generations'):.2f}h):")
        for g in exp.generations:
            g.show(full)

    return None


def trainer(exp_name: str, exp_id: Optional[List[str]] = None):
    """Grab status from a trainer XP. Particularly useful to monitor relaunched jobs ?

    :param exp_name: path after /dumped/ and before /exp_id
    :param exp_id: comma separated list of ids to monitor or None for all, defaults to None
    """
    pass


def cross_eval(
    exp_name: str,
    exp_id: Optional[List[str]] = None,
    subsampling: float = 0.1,
    size_around_diag: int = 10,
    only_gens: Optional[List[str]] = None,
    only_hard: bool = True,
):
    """Launch cross_eval, for all exp_ids in exp_name and in the exp_id list if given.

    :param exp_name: path after /dumped/ and before /exp_id
    :param exp_id: comma separated list of ids to cross_eval or None for all, defaults to None
    :param subsampling: how much to subsample generations when proving
    :param size_around_diag: if == -1 : run all evals. Otherwise, stay within a square of size n around the diagonal.
    :param only_gens: if not None, will ignore size_around diag and launch evals for full columns listed in the arg.
    :param only_hard: if False, will eval on all generations subsampled
    """

    dirs = Dirs.from_user()
    filename = "results.pkl" if only_hard else "results_all.pkl"

    offline_exps: Dict[str, OfflineIterInfo] = {}
    offline_exps = get_all_offline_exps(exp_name, exp_id)
    if len(offline_exps) == 0:
        print("No exps found")
        return

    jobs: Dict[str, Dict[Tuple[int, int], submitit.Job]] = defaultdict(dict)
    pre_existing_results = {}
    cross_eval_root = f"YOUR_PATH/cross_eval/"
    for this_exp_id, exp in offline_exps.items():
        # try reloading if results are there
        try:
            results = pickle.load(
                open(
                    Path(dirs.user_gen_workdir)
                    / f"{exp_name}_{this_exp_id}"
                    / filename,
                    "rb",
                )
            )
            results = {x: y for x, y in results.items() if y > 0}
            print(
                f"{this_exp_id}: Found {len(results)} pre-existing cross-eval results. Re-use ? [Yn]"
            )
            v = input()
            if v == "n":
                results = {}
        except Exception as e:
            print(f"{this_exp_id}: No pre-existing results found.")
            results = {}
        pre_existing_results[this_exp_id] = results

        # prepare / clean slurm log folders
        os.makedirs(Path(cross_eval_root) / exp_name / this_exp_id, exist_ok=True)
        # clean-up log folders otherwise things will be very confusing:
        slurm_root = Path(dirs.user_gen_workdir) / f"{exp_name}_{this_exp_id}"
        for f in slurm_root.glob("*"):
            if not str(f).endswith(filename):
                f.unlink()

        for f in (Path(cross_eval_root) / exp_name / this_exp_id).glob("*"):
            shutil.rmtree(f)

        executor = submitit.AutoExecutor(folder=slurm_root, slurm_max_num_timeout=-1,)

        executor.update_parameters(
            slurm_job_name=f"{exp_name}_{this_exp_id}_cross_eval",
            slurm_partition="Theorem_Proving",
            slurm_cpus_per_task=10,
            slurm_gpus_per_node=1,
            nodes=1,
            tasks_per_node=1,
            slurm_mem_gb=60,
            slurm_srun_args=["-vv"],
            slurm_timeout_min=60 * 24,
        )

        re_used = 0
        with executor.batch():
            for trainer in exp.prover_trainers:
                prover_ckpt, _ = get_latest_checkpoint(trainer.dump_path)
                gen_root = dirs.user_generations_path
                for g in exp.generations:
                    if (
                        size_around_diag > 0
                        and only_gens is None
                        and abs(trainer.epoch - g.epoch) >= size_around_diag
                    ):
                        continue

                    if only_gens is not None and g.epoch not in only_gens:
                        continue

                    if (trainer.epoch, g.epoch) in results:
                        re_used += 1
                        continue

                    jobs[this_exp_id][(trainer.epoch, g.epoch)] = executor.submit(
                        one_eval,
                        ckpt=prover_ckpt,
                        gen_root=Path(gen_root) / f"{exp_name}_gen_{g.epoch}",
                        exp_id=this_exp_id,
                        subsampling=subsampling,
                        only_hard=only_hard,
                        dump_path=Path(cross_eval_root)
                        / exp_name
                        / this_exp_id
                        / f"{trainer.epoch}_{g.epoch}",
                    )
        print(f"Reused {re_used} // Launched {len(jobs[this_exp_id])}")

    for this_exp_id, jjobs in jobs.items():
        result = pre_existing_results[this_exp_id]
        for x, j in jjobs.items():
            try:
                proved, total = j.result()
                print(this_exp_id, x, proved, total)
                result[x] = proved / max(total, 1)
            except Exception as e:
                print(e)
                print("passing")

        pickle.dump(
            result, open(f"/gen_workdir/{exp_name}_{this_exp_id}/{filename}", "wb",),
        )


def swept_params(exps: List[OfflineIterInfo]) -> None:
    """Given a bunch of experiments, set their swept_params
    :param exps: all experiments for which we compare params to create swept_params
    :return: None, mutates exps.
    """

    changing_params = set()
    param_count: Dict[str, int] = defaultdict(int)
    param_value = {}
    for exp in exps:
        for x, y in exp.params.items():
            param_count[x] += 1
            if x not in param_value:
                param_value[x] = y
            else:
                if y != param_value[x]:
                    changing_params.add(x)

    for x, y in param_count.items():
        if y == 1:
            changing_params.add(x)

    for exp in exps:
        exp.swept_params = {x: exp.params[x] for x in changing_params}
    return


if __name__ == "__main__":
    fire.Fire({"offline_iter": offline_iter, "cross_eval": cross_eval})
