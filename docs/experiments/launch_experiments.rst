Launch experiments
==================

In this tutorial, we show how to launch pretraining experiments, as well as supervised training, and `HyperTree Proof Search (HTPS) <https://arxiv.org/pdf/2205.11491>`_ online training.
Note that if you only want to pretrain a model, or train a model in a supervised way, you can just stop at these steps.

Otherwise, the full training pipepline to train an HTPS model is the following:

#. **Pretrain** a model on a large dataset (e.g. arXiv)
      * Model architecture is an encoder - decoder: it is pretrained with `masked sequence to sequence pretraining (MASS) <https://arxiv.org/pdf/1905.02450.pdf>`_
      * Pretraining is node using the `train.py `_
#. **Train a supervised** model on supervised data
      * The supervised data can be either synthetically generated data for environments like Equations/SymbolicRegression, or human proofs for Metamath/Lean.
      * It is possible to co-train on other tasks to reduce overfitting, e.g. the PACT dataset for Lean.
      * Supervised training is done with `train.py `_
#. Do **online HTPS training**
      * Once a model is trained in a supervised way, we can start the HTPS online training. This is important to start from a trained model, otherwise the MCTS will generate random tactics and won't be able to prove anything (so the HTPS will never send any training data).
      * This is done with `launch_mcts.py `_
#. **Evaluate** a model (large scale eval)
      * While online HTPS training launches async HTPS eval every N epoch. But these are small (no decoding sampling, 1 trial per theorem, 8 GPU), so we want to run a large scale evaluation (e.g. 128 trials per theorem on 128 GPUs) at the end of the online training.
      * This is done with `sweep_eval.py `_


Pretraining and Supervised training
-----------------------------------

Commands
~~~~~~~~

Note that some of these sweeps set dump folder as ``/USERNAME/dumped/`` so mkdir it for you.

Pretraining
'''''''''''

Launch a MASS pretraining on arxiv using the sweep file ``sweeps/references/pretrain_latex.json`` (launch on 32 V100s).

Note that if you want to pretrain with other task than MASS, refer to the trainer part to see what's available.

Supervised training
'''''''''''''''''''

Supervised training of a Metamath model with sweep ``sweeps/references/supervised_mm.json``.

Note that these supersived training reload a pretrained model : ``"reload_model": [""]``

Debug
~~~~~

In both cases, before running your XP on the cluster, you should try to run your training command locally with
``--debug.debug 1`` so that we only load a small fraction of the datasets, and the debug xp loads faster.
Also if you have CUDA OOM locally, you should add ``--batch.bptt 256 --batch.size 8 --batch.tokens 500``

.. code-block:: console

   >>> python /Evariste/formal/train.py  \
   --model.fp16 "true" --batch.bptt 512 --model.n_layers 6 --model.n_heads 8 --model.dropout 0 --model.attention_dropout 0 --model.gelu_activation "false" --model.share_inout_emb "true" --model.sinusoidal_embeddings "false" --mlm.word_pred 0.15 --mlm.sample_alpha 0 --mlm.word_mask_keep_rand_str "0.8,0.1,0.1" --env_base_seed 0 --num_workers 1 --log_network_stats_freq 20 --label_smoothing_eps 0 --accumulate_gradients 1 --clip_grad_norm 1 --epoch_size 20000 --max_epoch 100000 --reload_model "" --exp_name "SUPERVISED_JOB_NAME" --model.enc_emb_dim 1600 --model.dec_emb_dim 1024 --model.enc_n_layers 12 --model.dec_n_layers 6 --batch.size 16 --model.enc_layer_dropout 0.2 --model.dec_layer_dropout 0.2 --model.enc_min_layers 6 --model.dec_min_layers 2 --model.mha_learn_scaling "false" --tasks "latex_mass" --stopping_criterion "_valid-latex_mass-tok-ppl,10" --validation_metrics "_valid-latex_mass-tok-ppl" --latex.data_dir "" --optimizer "adam_inverse_sqrt,warmup_updates=30000,lr=0.0001,weight_decay=0.01" \
   --debug.debug 1 --batch.bptt 256 --batch.size 8 --batch.tokens 500
   INFO - 10/03/22 12:51:07 - 0:00:00 - ============ Initialized logger for PID 3032659 ============
   INFO - 10/03/22 12:51:07 - 0:00:00 - The experiment will be stored in dumped/debug/debug_38868110
   ...

Logs and best model
~~~~~~~~~~~~~~~~~~~~

In the dump folder, here, ``YOUR_PATH``, you'll have the train.log file, containing the training loss, ppl, and evaluation scores.
You will also have your best model according to the ``validation_metrics`` you put in your json config. E.g. for pretraining we have ``"validation_metrics": ["_valid-latex_mass-tok-ppl"]``, here the best model saved will be the one that has the lowest mass-tok-ppl on valid over training.



Online HTPS training
--------------------

.. image:: images/online_training.png
  :width: 714
  :alt: HTPS online training


Command
~~~~~~~

Note that some of these sweeps set dump folder as ``/USERNAME/dumped/`` so mkdir it for you.

When we start a HTPS training, we actually start the controller with a sweep like ``sweeps/references/online_mcts_eq.json``.

The controller does not require any GPU, and few CPU / RAM, so you don't need a big SLURM allocation for it.


Debug
~~~~~

      #. If you have modified the prover, you can test prover behavior locally with `python -m scripts.simpler_run` change the parameter in `simpler_run.py `_ directly (or try using `formal/simpler_run.py`).
      #. To run an HTPS training locally, add the flag ``--local 1``. If you have two local GPUs, you will have one for the trainer and one for the prover.
      #. Note that if you want to test the controller and provers together without trainer (which allow to have 2 provers if needed to be tested), you can run the online training without trainer with ``--no_trainer 1``


Logs and saved models
~~~~~~~~~~~~~~~~~~~~~

You will find the following logs:
      #. **Params** used for the XP: ``launcher_params.json``
      #. **Controller** logs: ``workdir_path/JOBID.stdout`` / ``workdir_path/JOBID.stderr``
      #. **Trainer** logs: ``dump_path/train.log`` / ``dump_path/trainer/*.stdout`` / ``dump_path/trainer/*.stderr``
      #. **Prover** main log (for details about prover implementation, see <REF>):  ``dump_path/provers/*.out`` / ``dump_path/provers/*.err``
      #. Prover other logs:
            * batch backward runner log (one per prover):  ``dump_path/prover_dumps/batch_backward_runner.log*``
            * each mcts process (n_simultaneous_proof per prover) logs:  ``dump_path/prover_dumps/MCTS*``, ``dump_path/prover_dumps/MCTSHandler*``, ``dump_path/prover_dumps/one_mcts*``
      #. **Prover results**: ``dump_path/mcts_results.jsonl``
      #. **Async eval** logs and results: ``dump_path/bwd_prover_eval/lang/split/epoch/*``. If the async eval is successful, you have ``done`` file in the eval folder.

Also model is saved at every async eval : ``dump_path/bwd_prover_eval/lang/split/epoch/checkpoint.-1.pth``


HTPS at-scale proving evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similarly to ``launch_mcts.py``, ``sweep_eval.py`` starts a controller that will take care of starting the provers / collecting the proofs, so it does not require a lot of memory / CPUs, and no GPU. The numbers of trials and machines is specified in the ``sweeps/gui/eval_lean_clustering.json`` config.

Example for eq, eval with 16 provers and 32 attemps per theorem, theorem from identities split with the sweep ``sweeps/references/eval.json``.


Analyze experiments
-------------------

Eval metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~

Plot eval metrics over training. For each type of training, the corresponding metrics are given.

   * pretraining : mass ppl / token accuracy
   * supervised training / online mcts training : seq2seq ppl / token accuracy, proving scores (greedy and async)
   * for at-scale proving eval: async proving scores


Tensorboard for statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensorboard is used to monitor online htps training statistics. You have 
Command : 

.. code-block:: console

      tensorboard --host localhost --port 9512 --logdir /YOUR_PATH/mcts_online_refac_compare_master

You'll find statistics about:
      * trainer == trainer / data_loader* e.g.:
            * data_loader*/ingress -> nb of training sample received by trainer from provers / s
      * model == network
      * controller == launcher_stats e.g.:
            * launcher_stats/n_jobs_ready -> number of prover at given time 
            * launcher_stats/solved_labels_acc* -> accuracy of theorems solved in one split over training
      * prover:
            * mcts_stats e.g.:
                  * mcts_stats/time -> avg time of htps algorithm per theorem
                  * mcts_stats/n_nodes -> avg number of nodes per search tree
            * ProofHandler
            * gpu_expander
            * expander_batcher

References
----------
`HyperTree Proof Search for Neural Theorem Proving <https://arxiv.org/pdf/2205.11491>`_

