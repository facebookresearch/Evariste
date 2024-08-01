Adversarial
===========

.. note:: Freshness : 7/10/22

Online
------

Code is located in `formal/evariste/adversarial/`. At the time of writing (7/10/22).

The way online adversarial works is by allocating several GPUs for generation, proving, generator training and prover training.

Each of these is a role and `formal/adversarial_train.py` dispatches each slurm task into the right role depending on world_size and global_id.

Best way to monitor is via tensorboard.

Offline
-------

The point of the `Offline` codebase for adversarial training is to better monitor / understand the training behaviours at each steps.

Currently the main entry point is `formal/offline_iter.py` which relies on `evariste/adversarial_offline/generator.py` for generating prover-filtered theorems.

Then, `train.py` is launched on these generations for both prover and generator.

The folder structure is a bit unconventional : 

* Generations are saved in `{USER}/generated/{EXP_NAME}_gen_{EPOCH}/{EXP_ID}_*/generations/*.pkl`

* Trainer dump dirs are `{USER}/dumped/{EXP_NAME}_{prover_train|train}/{EXP_ID}/{EPOCH}`

* Work dirs `~{EXP_NAME}_{gen|prover_train|train}/{EXP_ID}/{EPOCH}`
