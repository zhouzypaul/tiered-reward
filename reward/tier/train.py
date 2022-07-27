import pickle
import argparse

import numpy as np

from reward.environments import make_one_dim_chain, make_russell_norvig_grid, make_simple_grid
from reward.agents import run_multiprocessing_q_learning


def train_on_env(env_name):
    """
    train on one environment, with one specific reward function
    """
    if env_name == 'chain':
        env = make_one_dim_chain(
            num_states=10,
            goal_reward=1,
        )


if __name__ == "__main__":
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="chain")
    args = parser.parse_args()


