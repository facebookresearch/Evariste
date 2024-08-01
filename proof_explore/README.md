# proof_explore

First, we install nodejs. Download and extract
```
wget https://nodejs.org/dist/v14.15.3/node-v14.15.3-linux-x64.tar.xz
tar -xf node-v14.15.3-linux-x64.tar.xz
mv node-v14.15.3-linux-x64 nodejs
```
Add to path `~/nodejs/bin` to YOUR_PATH and check that `which npm` works.

Tell npm to install everything locally :
```npm config set prefix ~/.npm```

Then in the proof_explore folder, install all nodejs stuff with
`npm install`

Finally, to tell npm to build our files anytime there is a change :
`npm run build`

Now a change to a .vue file should lead to rebuilding `dist/proof-explore.js` which should be symlinked to `Evariste/formal/visualizer/static/proof-explore.js`. You also want to add a symlink to `proof-explore.js.map` to facilitate debugging in Chrome developper tools:
```
cd formal/visualizer/static
ln -s ../../../proof_explore/dist/proof-explore.js proof-explore.js
ln -s ../../../proof_explore/dist/proof-explore.js.map proof-explore.js.map
```

# Work environment
Once `npm run build` is running on your devserver, running `python -m visualizer` in `Evariste/formal` should run the tornado server at `localhost:9097`.
When changing a vue file, you need to wait for the file to be saved on the server, then rebuilt by npm, this takes ~4sec, then changes will appear.

If there is a build error, `proof-explore.js` won't exist, so the page will load without any javascript. Just read the build errors and fix them.

# Code architecture - JS
Most of the code is in `proof_explore/src/components`.
The main entry point is App.vue which loads 3 components:
- The Settings (in the `settings` folder).
- The Tree (Tree.vue) which contains Nodes and Tactics (`node` and `tactic` subfolders) which can also be specialized depending on the language.
- The Footer (the `footer` folder) which displays more detailed informations about tactics and nodes.

# Code architecture - python
All the session logic is in `visualizer/session/core.py`. The only language specialization happens in `enrich_results_{mm|eq}`.
This function is called to add /modify fields in the data structure returned by query_model.

The server serves a very simple `main.html` which pretty much just loads the Vue app.

# Rsync
NPM creates a whole bunch of crazy things in the `proof_explore/node_modules` folder. If you ever build the components locally,
I would recommend not rsyncing them, as it takes a long time: `--exclude proof_explore/node_modules`.