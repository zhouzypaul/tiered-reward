# Tierede Reward
This is the code for the paper [Tiered Reward Functions: Specifying and Fast Learning of Desired Behavior](https://arxiv.org/abs/2212.03733) presented at the [Reinforcement Learning Conference 2024](https://rl-conference.cc).

## requirements
Python 3.8+

## set up and library dependencies
```bash
python -m venv env
source env/bin/activate
pip3 install -U pip wheel setuptools
pip3 install -r requirements.txt
```

## running

### Tabular
running Q-Learning/RMAX experiments on different grid worlds
```bash
python3 -m reward.tier.train --env ENV_NAME -t 2 3 4 5 6 7 8 9 --num_seeds 300 --initial_value 1e5 --lr 0.9 --gamma 0.9
python3 -m reward.tier.train --env flag_grid -t 6 --steps 100000
```
`ENV_NAME` incluse `chain`, `grid`, `frozen_lake`, and `wall_grid`.

plot the visualization of the comparison between different policies on Russell/Norvig grid
```bash
python3 -m reward.pareto.rn --plot
```

plot termination probability for pareto rewards:
```python
python -m reward.pareto [--options]
```

### Atari

training atari experiments
```python
python -m reward.atari.train --env ENV -t TIER [options]
python -m reward.atari.plot -l loading_dir
```
currently the supported atari environments are: `['Breakout', 'Pong', 'Freeway', 'Boxing', 'Asterix', 'Battlezone']`

`Pong` has exactly 22 tiers 
`Boxing` has exactly 15 tiers
`Asterix` has exactly 5 tiers

### MiniGrid

training minigrid experiment
```bash
python3 -m reward.minigrid.train --algo ppo --env MiniGrid-DoorKey-5x5-v0 -e DoorKey --save-interval 10 --frames 1e7 --reward-function step_penalty [-n]
python3 -m reward.minigrid.visualize --env MiniGrid-DoorKey-5x5-v0 -e DoorKey
python3 -m reward.minigrid.evaluate --env MiniGrid-DoorKey-5x5-v0 -e DoorKey

tensorboard --logdir storage/DoorKey/
```

To comapre different reward learning curves
```bash
python3 -m reward.minigrid.plot -l LOADING_DIR
```
