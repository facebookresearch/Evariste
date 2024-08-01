Forward Proving and Generation
================================

Note: all commands are given from `Evariste/formal`
Forward model training
----------------------

Train a forward model
~~~~~~~~~~~~~~~~~~~~~

To launch a local debug forward training, you can type :
.. code-block:: console

    python train.py --cfg fwd_mm_debug # fwd_eq_debug, fwd_hl_debug


All configs for forward training are defined in `evariste/trainer/args.py` for the moment.

To launch a real training use the real config
.. code-block:: console
    
    python train.py --cfg fwd_mm # fwd_eq, fwd_hl

8 gpus are good for standard configs.

Metrics
~~~~~~~
Main metric to watch is `{env}_valid_fwd_proving-prop_proved`. It's the proportion of valid theorem proved by the prover with greedy decoding (the prover stop at its first mistake). 


Proving with forward model
--------------------------

Launching
~~~~~~~~~

During training, the proof search is quite simple (greedy decoding). 
If you want to evaluate your model with a more complicated search you can use 
the `evariste/generation/cli/prove.py` script:

To run a local debug evaluation with greedy sampling you can use:

.. code-block:: console
    
    python -m evariste.generation.cli.prove --model.ckpt {my_ckpt_path} --model.name {my_model_name} --prover greedy  --split valid --debug on


The two main kind of prover configs are:
    * ``greedy`` with simple greedy decoding. The prover stops at the first mistake.
    * ``sampling`` with sampling decoding (temperature of 1). The prover stops when it 
      does 5 consecutive mistakes.

You can see prover config details in `evariste/generation/cli/fwd_configs.py`

Note: ``--model_name`` allow to create a nicer output_path for the run.
I usually store models (ckpt_path) and their names in `evariste/generation/cli/fwd_configs.py`.
They are registered in the ConfStore.
With this I can launch a proving with the model name only ``--model new3_v8_5_ckpt107`` for instance.


A large scale evaluation requires to do multiple trials by theorems. You can launch it on slurm with the same command:

.. code-block:: console
    
    python -m evariste.generation.cli.prove --model new3_v2_1_ckpt189 --prover sampling --async_model on --n_trials_by_trm 128  --slurm on --n_jobs 32  --partition


You can have a look at the script to see how to load a ``ForwardProver`` from a checkpoint and use it on an iterator of ``ForwardGoal``

Code pointers
~~~~~~~~~~~~~

    * The main object for forward proving is the `ForwardProver` (`evariste/generation/forward_prover.py`).
    * It takes as input a iterator of `ForwardGoal`  and return an iterator of `ProofSearch`. Proofs are outputed when ready (so the order is not preserved). Typically the forward prover handle ~1024/2048 proofs concurrently. This allow to saturate the gpu even without beam search.
    * The recommended way of creating the forward prover is by creating it from a forward model checkpoint (see `evariste/generation/forward_prover_factory.py`) and a `ProverConfig`. Recommanded ProverConfigs are in  `evariste/generation/cli/fwd_configs.py`.

Joint training and goal conditioned generation
----------------------------------------------

Description
~~~~~~~~~~~

![forward_training_with_goal_conditionned_generation](_assets/forward_training_with_goal_conditionned_generation.png)

The forward prover can be used to generate new proofs. 

Indeed we use current forward prover to reach a set of goals (typically the train theorems statements). The forward prover will sometimes reach the goal and sometimes not, but in both cases it will generate a graph of proved statements. Each node of this generated graph can be used as a new goal for training the model in a supervised learning fashion (this is very similar to goal relabelling technique).

The setup is similar than a RL setting with an actor that generate trajectories and a learner that learn on these trajectories. 



Launch a trainer/actor forward training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You need to change these parameters to your config:

.. code-block:: console

    cfg.slurm_conf.torch_world_size = {num_gpus_for_training}
    cfg.online_fwd_generation = True
    cfg.mm.graph.generated_prob = 0.5 # proportion of generated data in training batch
    cfg.no_eval_on_train = True # not possible to eval on train


or you can use the example sweep file: `sweeps/demo/fwd_and_gen_actor_8gpus.json`

WARNING: this requires to be in a slurm job to work.

This will launch the trainers and the actors within your slurm job. 
You need to set the ``torch_world_size`` to the number of **trainers** that you want. 
The remaining gpus of your slrum job will be used as actors.

I usually use a ration of 4:1 (4 trainers for 1 generator). This improves significantly the perfs on Metamath.



Pointers to code:
~~~~~~~~~~~~~~~~~

    * the code of the actor is in `evariste/generation/cli/generation_worker.py`

    * within the slurm job, the actor is launched n the `train.py`:

    .. code-block:: python

      def main(cfg: TrainerArgs) -> None:
          if is_generation_worker(cfg):
              return run_generation(cfg) # here!
          # run experiment
          train(cfg)
  

    * the actor dumped the new proofs on the disk (using zips of generated nodes)
    * within the trainer the generated samples are gathered using a `OnlineGenerationDataset`  (`evariste/generation/online_generation/generation_datasets.py`). This dataset is initialised in the `MetamathDataEnvironment`. Generated proofs are treated like human data (same data augmentation applied on it).

