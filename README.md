# reward

## running
plot termination probability for pareto rewards:
```python
python -m reward.pareto [--options]
```

training atari experiments
```python
python -m reward.atari.train --env ENV -t TIER [options]
python -m reward.atari.plot -l loading_dir
```
currently the supported atari environments are: `['Breakout', 'Pong', 'Freeway', 'Boxing', 'Asterix', 'Battlezone']`
