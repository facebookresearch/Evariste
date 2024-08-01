## Backward prover

### Main Components

The backward prover uses an MCTS tree search. The code for this is single-threaded and lives in `evariste.mcts.mcts`.
After creating an `MCTS` object, the only thing to call is `simulate` which will run the proof search in the main process.

The `MCTS` requires a `VecBackwardEnv` and an `Expander` to be created. The `Expander` runs the GPU in a background process to expand leaves from the tree.
The `VecBackwardEnv` allows to apply a `Tactic` to a `Theorem` to obtain all subgoals.

Everything is sub-classed based on the environment. For example, for the equations environment, one would use `EQVecBackwardEnv`,  `EQTheorems` and `EQTactic`.
By implementing these `3` thing for a new environment, one can immediately run the prover

### BeamSearch object
The MCTS uses a wrapper around the transformer object defined in `prover_common.model.beam_search`.
This object is required by the `Expander` to actually run queries.

### Putting it all together

Take a look at `2p2e4_metamath.py` and run it with `python -m tutorials.bwd_prover.2p2e4_metamath`.
This will dump a `~/2p2e4.pkl` file that you can visualize with `python -m visualizer mm`. In order to learn how to use the visualizer, read the corresponding tutorial.

Then, you can take a look at `run_multi.py` and run it to see how to run multiple MCTS in parallel to better saturate the GPU with a `BackwardProver`.
Expect a 2:30min start time for all processes to initialize, load the models and such, then you should see some output.


`run_multi_zmq` is a small local example of how things work for multiple machines when results and work units are sent via zmq.


You should now be able to parse the `mcts.zmq_prover` file, and then `launcher.launch` which launches an online MCTS.

