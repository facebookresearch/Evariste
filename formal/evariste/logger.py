# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Optional, Callable, Dict, Union
import sys
import time
import logging
from datetime import timedelta
from abc import ABC
import psutil
import os

LOGGING_LEVELS: Dict[Union[str, int], int] = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "notset": logging.NOTSET,
}
LOGGING_LEVELS.update({i: i for i in range(51)})
LOGGING_LEVELS.update({str(i): i for i in range(51)})


class LogFormatter(logging.Formatter):
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ""


def create_logger(filepath, rank=0, console_level="info", name: Optional[str] = None):
    """
    Create a logger.
    Use a different log file for each process.
    """

    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = "%s-%i" % (filepath, rank)
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOGGING_LEVELS[console_level])
    console_handler.setFormatter(log_formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger() if name is None else logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(stderr_handler)

    return logger


def log_memory(logger):
    free_mem = psutil.virtual_memory().available / 1024 ** 3
    mem_used = psutil.Process(os.getpid()).memory_info()[0] / 1024 ** 3
    logger.info(f"[Memory] used: {mem_used} / free: {free_mem}")


def _MixinLoggerFactory(
    default_log_level: str,
    log_formatter: Optional[Callable[[object, str, str], str]] = None,
):
    class _MixinLogger(ABC):
        def set_logger(self, logger):
            self._logger = logger

        @property
        def logger(self):
            return self._logger

        def log(self, msg, log_level=default_log_level):
            msg = msg if log_formatter is None else log_formatter(self, log_level, msg)
            self.logger.log(
                LOGGING_LEVELS[log_level], msg,
            )

    return _MixinLogger
