# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import (
    Tuple,
    Generic,
    List,
    TypeVar,
    Dict,
    Any,
    Type,
)

RequestId = int

Output = TypeVar("Output")
Input = TypeVar("Input")


class AsyncWorkerDeadError(Exception):
    pass


W = TypeVar("W", bound="AsyncWorker")


class AsyncWorker(Generic[Input, Output], ABC):
    """
    Interface for async processing.

    An :class:`AsyncWorker` receives inputs via its :meth:`AsyncWorker.submit` method. It should process them asynchronously and return them when ready in the :meth:`AsyncWorker.ready` method.

    AsyncWorker should be called in a `closing` (or even better `logged_closing`) context:

    .. code-block::

        async_worker = MyAsyncWorker()
        with logged_closing(async_worker):
            async_worker.start()
            while ...:
                if not async_worker.is_alive():
                     raise ...
                for inp in ...:
                    req_id = async_worker.submit(inp)
                    ...
                for req_id, output in async_worker.ready():
                    ...
            async_worker.stop()


    An AsyncWorker can be transformed into an iterator using :class:`evariste.async_workers.async_worker_helpers.make_iterator`.

    A local AsyncWorker can be transformed into a remote AsyncWorker using the wrapper
    `evariste.async_workers.zmq_submitit_worker.ZMQSubmititWorker`

    An AsyncWorker can be transformed into a gang of AsyncWorker using
    `evariste.async_workers.worker_gang.AsyncWorkerGang`
    """

    @abstractmethod
    def start(self):
        """
        Start resources for the worker (however resources can be started in the __init__).

        TODO: As we are using now factories when we create workers, we can probably remove this function and
         assume that workers are started when created

        :return:
        """
        pass

    @abstractmethod
    def submit(self, inp: Input) -> RequestId:
        """
        Submit the input to the worker.

        :param inp: generic input for the worker

        :return: request_id: the request_id associated to the input. The output associated with this input should have this request_id
        """
        pass

    @abstractmethod
    def ready(self) -> List[Tuple[RequestId, Output]]:
        """
        Returns the outputs associated to the submitted inputs once they are ready,

        :return: list of (request_id, output)
        """
        pass

    @abstractmethod
    def is_alive(self) -> bool:
        """
        Useful when async worker is distributed or multi-processed, and when we want to relaunch dead workers.
        However you can choose to raise an error if you don't care about your worker being relaunched.

         TODO. As it is only used by ZMQSubmitWorker, probably can be removed from this interface. A standard
          async worker should raise if not alive.

        :return: true if the worker is still alive.
        """
        pass

    @abstractmethod
    def stop(self):
        """
        Stop the worker.

        Mainly useful when we want to differentiate when a worker was properly stopped
        (no more inputs to process) vs closed because of an error.

        TODO. As it is only used by ZMQSubmitWorker, probably can be removed from this interface. Closing
         should be enough for standard async worker.
        """
        pass

    @abstractmethod
    def close(self):
        """
        closing the worker resources
        """
        pass

    @classmethod
    def batch_stats(cls: Type[W], workers: List[W]) -> Dict[str, Any]:
        """
        Optional: Compute aggregated stats for all `workers` (note that this is a class method, so if you call this
        method on an instance you should put all the workers in `workers` , including your instance).

        :param workers: list of async workers of the same type.
        :return: aggregated stats
        """
        return {}


class AsyncWorkerLauncher(Generic[Input, Output], ABC):
    """
    Interface for a launcher able to launch a batch of async workers.

    Needed when async workers need to be launched in batch, and need to share some resource across workers
    (like a queue)
    """

    @abstractmethod
    def launch_workers(
        self, n_workers: int, first_with_full_logging: bool = False
    ) -> List[AsyncWorker[Input, Output]]:
        """
        Launch `n_workers` async_workers started. these are supposed to be started.

        :param n_workers: num workers to launch
        :param first_with_full_logging: if true you should ensure that the first worker that you launch has full logging enabled (like tensorboard etc).
        :return: the workers
        """
        pass

    @abstractmethod
    def close(self):
        """
        If any close the resources created in the __init__.
        Not responsible to close the async workers launched.
        """
        pass
