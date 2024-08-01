## Visualizer

For set-up instructions and some facts about code head on to `Evariste/proof_explore`, it would be too easy if all explanatory markdown files were located in the same folder...

Here we want to figure out how to *use* the visualizer.

When running `python -m visualizer {LANGUAGE}`, the visualizer web server will start on a port that depends on the chosen language.
A bunch of things (model loaded, port) are hard-coded in the script, feel free to add command line arguments to this script.

### Interactive mode

Once you're connected to the visualizer, you can either load a goal by label, or by writing the conclusion directly.
Some examples: 
- label `2p2e4` for mm
- conclusion `exp x0 == exp inv inv pow2 sqrt atanh inv inv tanh acosh inv inv cosh x0` for eq
- conclusion `! m n . ( m * n = 0 ) <=> ( m = 0 ) \/ ( n = 0 )` for hl

Once a goal is loaded, clicking on the ⚡ icon will run the backward model with the current settings to get tactics.
Rinse and repeat until you prove the main goal. Feel free to click on tactics or goals for more info.

### MCTS explore mode

You can load a pickle file dumped by the mcts prover to analyze the proof tree.
In this mode, you can replay the search by clicking on the play button. Clicking on a goal will show you all instantaneous MCTS statistics.

