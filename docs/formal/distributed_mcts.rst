Distributed MCTS
================

.. note::

    freshness:
        - 2022-10-08

Overview
--------

The two main classes to handle distributed MCTS are the :class:`launch_mcts.MCTSGangHandler`
and the :class:`evariste.backward.prover.zmq_prover.ProverHandler`

* :class:`launch_mcts.MCTSGangHandler` allows to manage a MCTS online training (controller + trainers + provers)

* :class:`evariste.backward.prover.zmq_prover.ProverHandler` allows to manage a MCTS evaluation (controller + provers)


API doc
-------

MCTS online training
~~~~~~~~~~~~~~~~~~~~

.. automodule:: launch_mcts
    :members:

MCTS evaluation
~~~~~~~~~~~~~~~

.. automodule:: evariste.backward.prover.zmq_prover
    :members: