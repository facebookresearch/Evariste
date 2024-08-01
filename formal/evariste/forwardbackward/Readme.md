## Forward Backward prover
Results of experiments on with this can be found here: 
https://docs.google.com/document/d/1McLJE9OgWWoK-5xyTUvP626E27w4bvvzgYawAwla1-Q/edit
### Forward Backward v0

```python
for goal in goals:
    for _ in range(n_fwd_trials):
        nhyps += run_fwd(goal)

    for _ in range(n_bwd_trials):
        run_bwd(goal, nhyps)
```

### Forward Backward: v0.1

```python
for goal in goals:
    for _ in range(ntrials):
        nhyps += run_fwd(goal)
        run_bwd(goal, nhyps)
```

### Forward backward alternate v1

```python
for goal in goals:
    leaves = [goal]
    for _ in range(ntrials):
        nhyps += run_fwd(leaves)
        leaves = run_bwd(goal, nhyps, nbwdsteps)
```
