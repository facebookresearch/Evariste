Lean
====

Overview
--------

For now, we use a fixed version: lean 3.30, all code for the api is in repos that are found on our internal server, please reach out if needed.

The `lean_ml_tooling` repo contains one main cpp file : ``ml_server.cpp``.
Running the *ml_server* binary gives access to the multi-threaded json REPL. Access is facilitated through a python API.

Code for this API lives in two places : 

* the *lean_ml_tooling* repo, in which it can easily be tested with ``test_lean_api.py``

* the *leanml* submodule in `Evariste/formal/leanml`. 

Finally, there are a few datasets associated with lean :

* the supervised and PACT datasets

* *synthetic* datasets 

* statements files : minif2f, annotations, autof, ...

.. note::
    The API is **asynchronous** and must be used this way to be reasonably fast, all requests and responses must be matched using their `req_id` strings.

    For backward proving, use the `LeanExpanderEnv`, not the API directly.


LeanML API Doc
--------------

There are two API provider. The first is LeanAPI in *leanml*, the second is the lean cluster, which distributes *lean*
over a cpu cluster.

They are meant to be a drop-in replacement for each other. So most concepts are common : 

* **Session** : A session is meant to hold a proof search, but is used more widely. Opening a session means loading a specific *lean declaration*.
  Giving a *module_path* and *decl_name*, we re-create the lean parser completely **before** the declaration (so imported modules, open namespaces, etc...) are the same.
  Then we parse the declaration and load it as node id 0.

* **Nodes** : A node corresponds to a tactic state. If *alpha equivalent merging* is activated, tactic states that are considered equal up to renaming. 

* **Applying a tactic** : In order to apply a tactic, we need to provide a *session name* and *node id*. In order : 
  
  * The node is recovered and the tactic is applied then typechecked

  * The resulting tactic state is split into independent (ie, no shared metavars) contiguous lists of subgoals

  * A bunch of filter on node size, hypothesis names, number of metavars are applied to each of these new tactic states.

  * Check if nodes previously existed. Return correct node ids / pretty-printed tactic states

An example interaction, first we create the session :

.. code-block:: console
   
    > {'req_type': 'new_session', 'module_path': '/...lean_ml_tooling/others/test_slow.lean', 'decl_name': 'p_or_q_or_r', 'merge_alpha_equiv': True, 'opts': {}, 'req_id': '115'}
    < {'initial_goal': {'full_pp': 'p q r : Prop,\nh : p ∨ q\n⊢ p ∨ q ∨ r', 'n_metavars': 0, 'n_subgoals': 1, 'node_id': 0, 'size': 24}, 'name': 'wgnjzn', 'req_id': '115', 'thread_id': '140113367525120'}

Key *initial_goal* describes the loaded root node, it always has *node_id* 0.

*session_name* is unique and must be used to reference this session

We can apply a tactic:

.. code-block:: console
   
    > {'req_type': 'tactic', 'name': 'wgnjzn', 'state_id': 0, 'tactic_str': 'cases h with hp hq', 'timeout': 2000, 'max_size': 1000000000, 'max_subgoals': 10000, 'max_metavars': 10000, 'max_repeated_hyps': 10000, 'nosplit': True, 'req_id': '116'}
    < {'eval_time': 8, 'nodes': [{'full_pp': '2 goals\ncase or.inl\np q r : Prop,\nhp : p\n⊢ p ∨ q ∨ r\n\ncase or.inr\np q r : Prop,\nhq : q\n⊢ p ∨ q ∨ r', 'n_metavars': 2, 'n_subgoals': 2, 'node_id': 1, 'size': 75}], 'repeated_hyps': 0, 'req_id': '116', 'thread_id': '140113484957440'}

*nodes* contain the list of children. Here there is only one, since we prevented splitting by using option `nosplit=True`. 

`lean_ml_tooling.test_lean_api` is a good example covering most API uses.

* **Cleaning a proof** : The `proof cleaner at /evariste/backward/env/lean/cleaner.py>`_ aims at getting rid of unecessary tactics or arguments or simplifying too generic tactics such as `simp`.

LeanAPI
~~~~~~~

.. autoclass:: leanml.comms.LeanAPI
   :members:


Lean cluster
~~~~~~~~~~~~

Access to lean instances is done via :class:`LeanClusterClient`, which uses a :class:`LeanCluster` for instantiating :class:`LeanInstance` via submitit.

.. autoclass:: lean_cluster.client.LeanClusterClient
   :members:

.. autoclass:: lean_cluster.instance.LeanCluster
   :members:

.. autoclass:: lean_cluster.instance.LeanInstance
   :members:

Proof cleaner
~~~~~~~~~~~~~

.. autoenum:: evariste.backward.prover.prover_args.CleaningLevel
   :members:

.. autoclass:: evariste.backward.prover.prover_args.ProofCleaningParams
   :members:

.. autoclass:: evariste.backward.env.lean.cleaner.Cleaner
   :members:

.. autoclass:: evariste.backward.env.lean.cleaner.AsyncProofCleaner
   :members:

Some utils
~~~~~~~~~~

.. autofunction:: evariste.backward.env.lean.utils.extract_tactics_from_proof

.. autofunction:: evariste.backward.env.lean.utils.extract_ident_and_actions_from_proof

.. autofunction:: evariste.backward.env.lean.utils.extract_actions_from_proof

.. autofunction:: evariste.backward.env.lean.utils.to_lean_proof

.. autofunction:: evariste.backward.env.lean.utils.export_lean_proof_to_file


Creating datasets
-----------------

Currently, all datasets are extracted for mathlib commit *9a8dcb9be408e7ae8af9f6832c08c021007f40ec*.

Supervised
~~~~~~~~~~

We use `Gabriel's fork of Jason Rute's code <https://github.com/gebner/lean-proof-recording-public>`_.

See *lean-proof-recording-public* for pre-extracted datasets.

Currently, the dataset we use is ``/datasets/lean_3.30/v1_full_names_newminif2f_tactic_full_names``.

It was first extracted, then post-processed to add fully qualified lemmas in human_tactic_str


PACT
~~~~

The `Proof Artifact Co-Training method <https://arxiv.org/abs/2102.06203>`_ is a work by OpenAI that leverages proof terms to propose new training tasks that deliver additional performance to the main task.
The initial code can be found on `Jesse's repo <https://github.com/jesse-michael-han/lean-step-public>`_.
It was improved in our `Evariste version` that is not yet available online:
- removal of the task `proof_step_classification`
- only “unmasked” hypotheses are added to the tactic state (vs all of them previously, including the ones that had nothing to do)
- to make it work, the way the hyps mask is computed is changed, as to include hyps that indicates the inhabited types (e.g., in `p : Prop, h : p |- p`, both `p : Prop` and `h : p` are included, vs previously only `h : p`)
- the pretty-printing options are now made explicit, esp. the following choices: full names (instead of aliases that depend on the open namespaces), generalized_field_notation to false (e.g., `n.succ.succ` => `nat.succ nat.succ n`)
- the tactic state is nwt printed directly from Lean, instead of Python (so we would have `h p : Prop |- p`` instead of `h : Prop, p : Prop |- p`)
- addition of the task `subexpr_type_prediction`
- addition of a new dataset `prove_self`
- addition of a new dataset `forward`


Synthetic
~~~~~~~~~

For synthetic dataset extraction, we run lean-proof-recording-public, abbreviated *lprp* and dumped in our Evariste codebase.
See *formal.maxi_gen.generate_one* for example usage.


Adding a new statement file or folder
-------------------------------------

Since lean_ml_tooling_v1.1, this is easier : 

* Statements are loaded from lean files in `LeanDatasetConf.statement_splits_path`

* Only files included in `LeanDatasetConf.splits_str` are included

* If you use a new lean file name, it must be added to lean proof cleaning imports. You'll get screamed at if that's not the case. Don't worry if your file isn't available in all statement_splits_path, not finding the import won't break anything.

* It's a good idea to use ``formal.scripts.tim.lean.check_load_minif2f.py`` to make sure all statements are loaded properly.

