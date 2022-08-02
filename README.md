# reward

## running
plot termination probability for pareto rewards:
```python
python -m reward.pareto [--options]
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