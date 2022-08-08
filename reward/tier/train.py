import argparse

import numpy as np

from reward.environments import make_one_dim_chain, make_russell_norvig_grid, make_simple_grid
from reward.agents.qlearning import run_q_learning, QLearning 
from reward.tier.reward_functions import make_tier_reward


def train_on_env(env, num_steps, seed, verbose=False):
    """
    train on one environment, with one specific reward function
    """
    agent = QLearning(
        num_steps=num_steps,
        rand_choose=0.05,
        seed=seed,
    )
    res = run_q_learning(
        env, 
        agent,
    )

    if verbose:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 1, figsize=(16, 4))
        env.plot(ax=axes).plot_policy(res.policy).title("Policy")
        plt.tight_layout()
        plt.savefig('results/debug.png')
        print('saved to results/debug.png')

    print(res.event_listener_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="chain")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--tier", "-t", type=int, default=5, help="number of tiers to use for reward")
    parser.add_argument("--seed", "-s", type=int, default=0, help="random seed")

    # hyperparams
    parser.add_argument("--gamma", "-g", type=float, default=0.90, help="discount rate")
    parser.add_argument("--delta", "-d", type=float, default=0.1, help="tier offset for reward")

    # debug
    parser.add_argument("--verbose", "-v", action="store_true", default=False)

    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.env == 'chain':
        num_chain_states = 25
        env = make_one_dim_chain(
            num_states=num_chain_states,
            discount_rate=args.gamma,
            goal_reward=1,
            step_reward=-1,
            success_rate=0.8,
            custom_rewards=None,
        )
        tier_r = make_tier_reward(num_states=num_chain_states, num_tiers=args.tier, gamma=args.gamma, delta=args.delta)
        print(tier_r)
        tier_env = make_one_dim_chain(
            num_states=num_chain_states,
            discount_rate=args.gamma,
            goal_reward=None,
            step_reward=None,
            success_rate=0.8,
            custom_rewards=tier_r,
        )
    else:
        raise NotImplementedError
    
    train_on_env(env, num_steps=args.steps, seed=args.seed, verbose=args.verbose)
    train_on_env(tier_env, num_steps=args.steps, seed=args.seed, verbose=args.verbose)

