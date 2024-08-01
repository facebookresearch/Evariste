.. currentmodule:: evariste.async_workers

Async Workers
=============

.. note::

    freshness:
        - 2022-10-08


Overview
--------
An :class:`async_worker.AsyncWorker` is an interface for async processing.
Is is basically an object that receive some inputs via :meth:`async_worker.Async_worker.submit`
and returns the corresponding ready outputs (when ready) via :meth:`async_worker.Async_worker.ready`

An async worker can be an environment (like the lean one), a prover...

In addition of standardizing the code, using an `AsyncWorker` as interface allows to use existing and tested helpers:

*  :class:`async_worker_helpers.make_iterator`: allows to turn an `AsyncWorker` into an iterator, given an input iterator. Useful for provers when we manipulate iterators.

* :class:`zmq_submitit_worker.ZMQSubmititWorker`: allows to transforms a local `AsyncWorker` into a a remote `AsyncWorker` with the same `submit`/`ready` but that will be executed on a different slurm jobs. Communications are handled via ZMQ.

* :class:`worker_gang.AsyncWorkerGang`: allow to create a pool of `AsyncWorker`. Failed workers (because of host failures or premption when `AsyncWorker` are distributed via :class:`zmq_submitit_worker.ZMQSubmititWorker`) will be restarted and their inputs rescheduled on other workers.


API doc
-------


`async_worker` module
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: evariste.async_workers.async_worker
    :members:

`async_worker_helpers` module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: evariste.async_workers.async_worker_helpers
    :members:

`zmq_submitit_worker` module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: evariste.async_workers.zmq_submitit_worker
    :members:


`worker_gang` module
~~~~~~~~~~~~~~~~~~~~

.. automodule:: evariste.async_workers.worker_gang
    :members: