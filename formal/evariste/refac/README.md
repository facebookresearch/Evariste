## What is this ?
Moving everything around shot the reloading of old model pickles.
Thankfully, torch.load allows the use of a user specified pickle_module. 
In evariste.refac.pickle, we make the necessary changes to module names before they are looked up.

After a while, we can probably get rid of this.