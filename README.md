# reward

## requirements
Python 3.8+

## running
plot termination probability for pareto rewards:
```python
python -m reward.pareto [--options]
```

compare different reward functions on Russell/Norvig grid:
```python
python -m reward.pareto.rn [--plot] [--print]
```

run q learning experiments on grid worlds and plot results:
```python
python -m reward.tier.train --env ENV_NAME -t TIER
python -m reward.tier.plot -l LOAD_DIR
```

### plotting the puddle environment
```python
mdp = make_puddle_world()
plot = mdp.plot()
# To plot a policy, you would do:
plot.pP(policy)
# To plot reward values, you would do:
plot.pR(reward)
# You can also set the title with 
plot.title(<enter title>)
# and add text to a particular state with
plot.annotate(state, <enter annotation>)
 
# Finally, call 
plt.show() # or plt.savefig(...)
# followed by 
plt.close()
```