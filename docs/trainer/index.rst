Trainer
=======

Args
----

Data loading
------------

For train  and eval we are creating batch iterators from pytoech DataLoader. We create DataLoader by using a pytorch IterableDataset.
This is an iterator over *batches*. This allow us to control how many tokens are within each pad and to control
the padding thanks to *padding queues*: we create a queue with samples and concatenate samples with same size other samples.

In practice use the :class:`evariste.model.data.envs.batch_iterator.BatchIterator` as implementation of the pytorch `IterableDataset`.

.. autoclass:: evariste.model.data.envs.batch_iterator.BatchIterator
    :members:



MCTS Loader
-----------

Envs
----

Lean
~~~~

Equations
~~~~~~~~~

Metamath
~~~~~~~~
