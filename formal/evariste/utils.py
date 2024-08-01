# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import (
    Optional,
    Tuple,
    List,
    Dict,
    TypeVar,
    Generic,
    Iterator,
    Iterable,
    Any,
)
from logging.handlers import RotatingFileHandler
from contextlib import AbstractContextManager
from functools import wraps, partial
from pathlib import Path
import logging
import os
import math
import time
import types
import errno
import pickle
import psutil
import random
import signal
import string
import submitit
import contextlib
import subprocess
import multiprocessing as mp
import numpy as np
from enum import Enum

from evariste.clusters.utils import clusterify_path


PROFILE = False
WRAP_TIMER = 1  # (O for nothing, 1 for relatively slow functions, 2 for everything)
TAIL_LOGGER = True
COND_TOK = "CONDTOK"


class DocStrEnum(Enum):
    """
        see https://stackoverflow.com/questions/50473951/how-can-i-attach-documentation-to-members-of-a-python-enum/50473952#50473952
    """

    def __new__(cls, value, doc=None):
        self = str.__new__(cls, value)
        self._value_ = value
        if doc is not None:
            self.__doc__ = doc
        return self


def formal_workdir() -> Path:
    return Path(__file__).parents[1]


def this_job_id():
    try:
        return submitit.JobEnvironment().job_id
    except RuntimeError:
        return "local"


def my_gb_memory() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def print_memory(logger, where: str):
    mem_av_gb = psutil.virtual_memory().available / (1024 ** 3)
    logger.info(
        f"AVAILABLE MEMORY ({where}): {mem_av_gb:.3f}GB -- Used: {my_gb_memory():.3f}GB"
    )


def tail(path: Path, n: int) -> List[str]:
    assert path.is_file()
    proc = subprocess.Popen(
        ["tail", "-n", str(n), str(path)], encoding="utf-8", stdout=subprocess.PIPE
    )
    assert proc.stdout is not None, (path, n)
    lines = proc.stdout.readlines()
    return lines


def search_files(
    dirpaths: List[str], extensions: List[str], ignore_patterns: Optional[List[str]]
) -> List[str]:
    """
    Find all files with specific extensions
    """
    assert type(dirpaths) is list
    assert type(extensions) is list
    assert ignore_patterns is None or type(ignore_patterns) is list
    file_list = []
    for dirpath in dirpaths:
        assert os.path.isdir(dirpath)
        for root, _, files in os.walk(dirpath):
            for file in files:
                if not any(file.endswith(ext) for ext in extensions):
                    continue
                path = os.path.join(root, file)
                if ignore_patterns is None or not any(
                    pattern in path for pattern in ignore_patterns
                ):
                    file_list.append(path)
    return file_list


def load_stream(
    dirs: Iterable[Path],
    verbose_interval: Optional[int] = None,
    max: Optional[int] = None,
    subsampling: Optional[float] = None,
) -> Iterator[Any]:
    """
    Load items from pickled lists in a set of directories into an iterator.
    """
    total = 0
    for dir in dirs:
        if verbose_interval:
            logging.info(f"Loading from directory: {dir}")
        for filename in dir.glob("*"):
            with open(filename, "rb") as f:
                if subsampling is not None and random.random() > subsampling:
                    continue
                try:
                    for item in pickle.load(f):
                        total += 1
                        if verbose_interval and total % verbose_interval == 0:
                            logging.info(f"Loaded item #{total}/{max}")
                        yield item
                        if max is not None and total >= max:
                            return
                except pickle.UnpicklingError:
                    print(f"Couldn't unpickle {filename}")


T = TypeVar("T")


def stream_saver(
    stream: Iterator[T], path: Path, chunk_length: Optional[int] = 100,
) -> Iterator[T]:
    """
    Saves the elements from `stream` into pickled lists of `interval` elements, without consuming the stream data.
    Filenames are of the format path/XXXX.pkl.
    Note: files will be overwritten without warning.
    """
    file_number = 0
    buffer = []
    for x in stream:
        buffer.append(x)
        if len(buffer) == chunk_length:
            with open(path / Path(f"{file_number:04}.pkl"), "wb") as f:
                pickle.dump(buffer, f)
            buffer = []
            file_number += 1
        yield x


def mod_filter(stream: Iterator[T], total: int, offset: int) -> Iterator[T]:
    for i, x in enumerate(stream):
        if i % total == offset:
            yield x


def get_dump_path(
    root_dump_path: str, exp_name: str, given_exp_id: str, overwrite_dump_path: str
) -> Tuple[str, str]:
    """
    Create a directory to store the experiment.
    """
    print(
        f"GET DUMP PATH -- "
        f"root_dump_path={root_dump_path} "
        f"exp_name={exp_name} "
        f"given_exp_id={given_exp_id} "
        f"overwrite_dump_path={overwrite_dump_path}"
    )
    assert os.path.isdir(root_dump_path)
    assert len(exp_name) > 0

    # create the sweep path if it does not exist
    sweep_path = os.path.join(root_dump_path, exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()

    # create an ID for the job if it is not given in the parameters.
    # if we run on the cluster, the job ID is the one of Chronos.
    # otherwise, it is randomly generated
    if given_exp_id == "":
        chronos_job_id = os.environ.get("CHRONOS_JOB_ID")
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        print(f"SLURM / CHRONOS '{slurm_job_id}', '{chronos_job_id}'")
        exp_id = chronos_job_id if chronos_job_id is not None else slurm_job_id
        if exp_id is None:
            chars = "abcdefghijklmnopqrstuvwxyz0123456789"
            while True:
                exp_id = "".join(random.choice(chars) for _ in range(10))
                if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                    break
        else:
            assert exp_id.isdigit()
    else:
        exp_id = given_exp_id

    if overwrite_dump_path:
        dump_path = overwrite_dump_path
    else:
        # create the dump folder / update parameters
        dump_path = os.path.join(sweep_path, exp_id)
    if not os.path.isdir(dump_path):
        subprocess.Popen("mkdir -p %s" % dump_path, shell=True).wait()
    return dump_path, exp_id


class MyTimeoutError(BaseException):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(repeat_id, signum, frame):
            # logger.warning(f"Catched the signal ({repeat_id}) Setting signal handler {repeat_id + 1}")
            signal.signal(signal.SIGALRM, partial(_handle_timeout, repeat_id + 1))
            signal.alarm(seconds)
            raise MyTimeoutError(error_message)

        def wrapper(*args, **kwargs):
            old_signal = signal.signal(signal.SIGALRM, partial(_handle_timeout, 0))
            old_time_left = signal.alarm(seconds)
            assert type(old_time_left) is int and old_time_left >= 0
            if 0 < old_time_left < seconds:  # do not exceed previous timer
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                if old_time_left == 0:
                    signal.alarm(0)
                else:
                    sub = time.time() - start_time
                    signal.signal(signal.SIGALRM, old_signal)
                    signal.alarm(max(0, math.ceil(old_time_left - sub)))
            return result

        return wraps(func)(wrapper)

    return decorator


def wrap_timer(verbose: int = 2):

    assert 0 <= verbose <= 2

    def decorator(func):
        @wraps(func)
        def wrapped(self, *args, **kwargs):
            logger = getattr(self, "tail_logger", None)
            should_log = (
                (logger is not None)
                and not isinstance(logger, NoneLogger)
                and verbose <= WRAP_TIMER
            )
            if not should_log:
                return func(self, *args, **kwargs)
            else:
                func_name = f"{self.__class__.__name__}.{func.__name__}"
                start = time.time()
                logger.info(f"Entering {func_name} ...")
                res = func(self, *args, **kwargs)
                logger.info(f"{func_name}: {time.time() - start:.5f}")
                return res

        return wrapped

    return decorator


class NoneLogger:
    def info(self, *args, **kwargs):
        pass


def get_tail_logger(path: Path, loc_info: bool = False):
    if not TAIL_LOGGER:
        return NoneLogger()
    job_id = os.environ.get("SLURM_JOB_ID")
    log_path = f"{path}.{job_id}"
    dirpath = os.path.dirname(log_path)
    os.makedirs(dirpath, exist_ok=True)
    loc_infos = "" if not loc_info else "%(filename)s %(funcName)s(%(lineno)d)"
    log_formatter = logging.Formatter(
        f"%(asctime)s %(levelname)s {loc_infos} %(message)s"
    )
    # backupCount needs to be > 0. will store at most (maxBytes * backupCount) on disk
    handler = RotatingFileHandler(
        log_path, mode="a", maxBytes=5 * 1024 * 1024, backupCount=1
    )
    handler.setFormatter(log_formatter)
    logger = logging.getLogger(log_path)
    logger.handlers = []
    logger.addHandler(handler)
    logger.propagate = False
    logger.info("====== NEW LOG START ======")
    return logger


class TermIndexer:
    def __init__(self, id2term: Optional[Dict[int, str]] = None):
        if id2term is None:
            self.id2term: Dict[int, str] = {}
            self.term2id: Dict[str, int] = {}
        else:
            assert all(type(i) is int and type(x) is str for i, x in id2term.items())
            self.id2term = id2term
            self.term2id = {v: k for k, v in id2term.items()}
            assert len(self.id2term) == len(self.term2id)

    def __len__(self) -> int:
        return len(self.id2term)

    def __getitem__(self, i: int) -> str:
        return self.id2term[i]

    def __contains__(self, item: str) -> bool:
        return item in self.term2id

    def add(self, x: str) -> None:
        assert type(x) is str
        if x in self.term2id:
            return
        i = len(self.id2term)
        self.id2term[i] = x
        self.term2id[x] = i
        assert len(self.id2term) == len(self.term2id)

    def get(self, x: str) -> int:
        assert type(x) is str
        return self.term2id[x]


def prepare(exp_name: str) -> Path:
    work_dir = Path(clusterify_path(f"{os.environ['USER']}/workdir"))
    assert work_dir.is_dir()
    the_date = time.strftime("%Y_%m_%d_%H_%M_%S/")
    exp_workdir = work_dir / exp_name / the_date
    os.makedirs(exp_workdir)
    print(f"Copying from {os.getcwd()} to {exp_workdir}")
    subprocess.call(
        [
            "rsync -ar --copy-links --exclude='.git/' "
            "--exclude '*.pth' --exclude '*.log' "
            "--exclude checkpoints --exclude notebooks --exclude __pycache__ "
            "%s/ %s" % (os.getcwd(), exp_workdir)
        ],
        shell=True,
    )
    os.chdir(exp_workdir)
    return exp_workdir


def merge_lists(a: List, b: List, rng: np.random.RandomState):
    """
    Merge two lists of elements without modifying their relative order.
    """
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a

    n = len(a) + len(b)
    a_pos = rng.choice(n, size=len(a), replace=False)
    a_pos_uniq = set(a_pos.tolist())

    a_idx = list(range(len(a)))[::-1]
    b_idx = list(range(len(b)))[::-1]

    # merge lists
    res = []
    for i in range(n):
        if i in a_pos_uniq:
            res.append(a[a_idx.pop()])
        else:
            res.append(b[b_idx.pop()])

    # sanity check
    assert len(a_idx) == 0
    assert len(b_idx) == 0

    return res


class logged_closing(AbstractContextManager):
    """
    Log when closing
    """

    def __init__(self, thing, name: str):
        self.thing = thing
        self.name = name

    def __enter__(self):
        logging.info(f"Entering 'closing' context for {self.name}")
        return self.thing

    def __exit__(self, *exc_info):
        try:
            self.close()
        except Exception:
            logging.error(
                f"Caught error while closing {self.name}, aborting closing..."
            )
            raise

    def close(self):
        logging.info(f"Closing {self.name} ...")
        if isinstance(self.thing, dict):
            for t in self.thing.values():
                t.close()
        else:
            self.thing.close()
        logging.info(f"Closed {self.name}")


X = TypeVar("X")


class PicklingQueue(Generic[X]):
    """ For some reason forking pickler is annoying. On input it's slow. On output it hides errors..."""

    def __init__(self, queue: mp.Queue):
        self.queue = queue

    def cancel_join_thread(self) -> None:
        self.queue.cancel_join_thread()

    def put(self, data: X) -> None:
        self.queue.put_nowait(pickle.dumps(data))

    def get_nowait(self) -> X:
        return pickle.loads(self.queue.get_nowait())

    def get(self, block=True, timeout=None) -> X:
        return pickle.loads(self.queue.get(block=block, timeout=timeout))


def rstr(length: int = 6) -> str:
    return "".join(random.choice(string.ascii_letters) for _ in range(length))


def get_mean(values: List) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def get_max(values: List) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.max(values))


def load_and_project_conditioning_vectors(
    path: Path, target_dim: int
) -> Dict[str, np.ndarray]:
    logger = logging.getLogger()
    z_vectors = pickle.load(open(path, "rb"))
    z_vectors_sorted_names = sorted(z_vectors.keys())
    base_dim = z_vectors[z_vectors_sorted_names[0]].shape[0]
    assert base_dim >= target_dim, (
        base_dim,
        target_dim,
    )
    if target_dim < base_dim:
        # if our vectors are bigger than the encoder dimension run a PCA on them to keep principal directions (1600 -> 512 keeps > 99% of the covariance)
        import torch

        a = torch.from_numpy(
            np.vstack([z_vectors[name] for name in z_vectors_sorted_names])
        )
        _, _, v = torch.pca_lowrank(a, q=target_dim)
        projected = a @ v
        for name, row in zip(z_vectors_sorted_names, projected):
            z_vectors[name] = row.numpy()

        # log how much of the cov is explained after the pca
        if base_dim < 5000:
            _, s, v = torch.pca_lowrank(a, q=base_dim, center=True)
            cov = s ** 2 / (base_dim - 1)
            after = torch.sum(cov[:target_dim])
            before = torch.sum(cov)
            logger.info(
                f"Reduced dimension of z-vectors from {base_dim} to {target_dim} kept {after*100/before}% of covariance"
            )
    return z_vectors


DAG = Dict[str, List[str]]  # label to list of directly dependent labels


def find_descendents(dag: Optional[DAG], label: str) -> List[str]:
    """DFS from label in DAG of theorems to find all descendents"""
    if dag is None or label not in dag:
        return []
    descendents = []
    to_explore = [label]
    seen = set()
    while to_explore:
        cur = to_explore.pop()
        if cur in seen:
            continue
        seen.add(cur)
        descendents.append(cur)
        for children in dag[cur]:
            to_explore.append(children)
    return descendents


@contextlib.contextmanager
def environment_variables(**kwargs: str) -> Iterator[None]:
    backup = {x: os.environ[x] for x in kwargs if x in os.environ}
    os.environ.update(kwargs)
    try:
        yield
    finally:
        for x in kwargs:
            del os.environ[x]
        os.environ.update(backup)


class SlurmSignalReceived(Exception):
    pass


class OurSignalHandler:

    # raise Exception on sigterm/sigcont/usr1 to close "properly" rather
    # than being killed, otherwise submitit bypasses this signal
    # We only raise once, when the first signal is received, to avoid raising a
    # second time while closing resources

    already_raised = False
    started = False

    @classmethod
    def start(cls) -> None:  # pylint:disable=unused-argument
        assert not cls.started, "Already started"
        signums = [signal.SIGTERM, signal.SIGCONT, signal.SIGUSR1]
        for signum in signums:
            signal.signal(signum, cls.handle_signal)
        logging.info(
            f"Raising on signals {[signal.Signals(signum).name for signum in signums]}"
        )
        cls.started = True

    @classmethod
    def handle_signal(
        cls, signum: int, frame: Optional[types.FrameType] = None
    ) -> None:  # pylint:disable=unused-argument
        signal_name = signal.Signals(signum).name
        logging.warning(f"Received signal {signal_name}")
        if not cls.already_raised:
            cls.already_raised = True
            logging.error(f"Raising on signal {signal_name}")
            raise SlurmSignalReceived(f"signal {signal_name}")
        else:
            logging.warning(f"Bypassing {signal_name} since already raised")


def set_TMPDIR():
    # this_job_id() didn't work for grab1 -> python train.py
    job_id = os.environ.get("SLURM_JOB_ID", None)
    if os.path.exists("/scratch/slurm_tmpdir/") and job_id is not None:
        tmp_dir = f"/scratch/slurm_tmpdir/{job_id}"
        if not os.path.exists(tmp_dir):
            logging.warning(
                f"/scratch/slurm_tmpdir/{job_id} not found! not setting TMPDIR"
            )
            return
        logging.info(f"Setting TMPDIR to {tmp_dir}")
        os.environ["TMPDIR"] = tmp_dir
    else:
        logging.info("Not setting TMPDIR")
