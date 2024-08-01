Sweep files
===========

Launch experiment on cluster
----------------------------

The syntax for the json files in sweeps is a bit particular. Here is an example.

.. code-block:: json

  {
    "some_name": {
      "b": [1,2,3],
      "c": [4,5,6],
      "SUBSWEEP_some_name": {
        "subsweep_1": {
          "optimizer": ["adam"],
          "lr": [1, 10, 100]
        },
        "subsweep_2": {
          "optimizer": ["madgrad"],
          "lr": [100, 10000, 100000]
        }
      }
    }
  }


This is used to launch 3 * 3 * (3 + 3) = 54 experiments.
These experiments will correspond to the cartesian product of properties _b_ and _c_. Then, we will have two options for _optimizer_: `adam` and `madgrad`.
Whereas the first one will be attempted with _lr_ in `[1, 10, 100]`, the second will be attempted with larger learning rates.

If you want to ignore a subsweep, just add `"drop": []`, it will lead to a cartesian with an empty list and thus will be empty.

For example configs check-out some random configs in the `formal/sweeps` folder.

Useful utilities
----------------
