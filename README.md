# reward
This is the code for the paper [Specifying Behavior Preference with Tiered Reward Functions](https://arxiv.org/abs/2212.03733)

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
running DQN experiments on different grid worlds
```bash
python3 -m reward.tier.train --env ENV_NAME -t 2 3 4 5 6 7 8 9 --agent dqn --steps 100000
```
`ENV_NAME` incluse `chain`, `grid`, `frozen_lake`, and `wall_grid`.


