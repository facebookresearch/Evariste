# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import abc
from typing import Iterator, Optional, Generic, TypeVar, List, Tuple

from evariste.async_workers.async_worker import (
    AsyncWorker,
    Input,
    Output,
    RequestId,
    AsyncWorkerDeadError,
)


def make_iterator(
    async_worker: AsyncWorker[Input, Output],
    max_in_worker: int,
    input_it: Iterator[Optional[Input]],  # not blocking input
) -> Iterator[Output]:
    """
    Create an iterator given an async worker and input_iterator

    .. warning::

        don't forget to put your async_worker in a `logged_closing` context


    raise `AsyncWorkerDeadError` if async worker dies.

    :param async_worker: worker to transform into an iterator.
    :param max_in_worker:
    :param input_it: Iterator of Optional[Input]. If we receive None instead of input, it allows not to block the main loop and to continue to call `ready` on the async_worker and not to block the whole process waiting for more inputs.

    :return: an iterator of output
    """
    in_worker = 0
    no_more_inputs = False

    async_worker.start()

    while True:
        assert max_in_worker >= in_worker >= 0
        if no_more_inputs and in_worker == 0:
            print("No more inputs and in_worker=0. Exiting make iterator loop.")
            break

        if not async_worker.is_alive():
            raise AsyncWorkerDeadError(
                f"Async worker {async_worker.__class__.__name__} died!"
            )

        # submit all inputs we can
        while not in_worker >= max_in_worker and not no_more_inputs:
            try:
                inp = next(input_it)
            except StopIteration:
                no_more_inputs = True
            else:
                if inp is None:
                    break
                in_worker += 1
                async_worker.submit(inp)

        for _, output in async_worker.ready():
            in_worker -= 1
            yield output

    async_worker.stop()


InnerOutput = TypeVar("InnerOutput")


class PostProcessor(abc.ABC, Generic[InnerOutput, Output]):
    @abc.abstractmethod
    def __call__(self, inp: InnerOutput) -> Output:
        pass

    @abc.abstractmethod
    def close(self):
        pass


class PostProcessedAsyncWorker(
    Generic[Input, InnerOutput, Output], AsyncWorker[Input, Output]
):
    """Probably overkill if we see that used once"""

    def __init__(
        self,
        worker: AsyncWorker[Input, InnerOutput],
        post_processor: PostProcessor[InnerOutput, Output],
    ):
        self.worker = worker
        self.post_processor = post_processor

        self.closed: bool = False

    def is_alive(self) -> bool:
        return self.worker.is_alive()

    def start(self):
        self.worker.start()

    def submit(self, inp: Input) -> RequestId:
        return self.worker.submit(inp)

    def ready(self) -> List[Tuple[RequestId, Output]]:
        return [(rid, self.post_processor(out)) for rid, out in self.worker.ready()]

    def stop(self):
        self.worker.stop()

    def close(self):
        if self.closed:
            return
        self.worker.close()
        self.post_processor.close()
        self.closed = True
