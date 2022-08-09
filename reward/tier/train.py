import os
import argparse

import numpy as np

from reward.environments import make_one_dim_chain, make_russell_norvig_grid, make_simple_grid
from reward.agents.qlearning import run_multiprocessing_q_learning, run_q_learning, QLearning
from reward.tier.reward_functions import make_tier_reward, potential_based_shaping_reward
from reward.tier.plot import compare_goal_hitting_stat_with_different_tiers
from reward.utils import create_log_dir
from reward import kvlogger


def train_on_env(env, num_steps, seed, multiprocessing=True, verbose=False):
    """
    train on one environment, with one specific reward function
    """
    if multiprocessing:
        results = run_multiprocessing_q_learning(
            env, 
            num_seeds=10,
            num_learning_steps=num_steps,
        )
    else:
        agent = QLearning(
            num_steps=num_steps,
            rand_choose=0.05,
            seed=seed,
        )
        result = run_q_learning(env, agent)
        results = [result]

    if verbose:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 1, figsize=(16, 4))
        env.plot(ax=axes).plot_policy(results[-1].policy).title("Policy")
        plt.tight_layout()
        plt.savefig('results/debug.png')
        print('saved to results/debug.png')

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="chain")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--tiers", "-t", type=int, nargs="+", default=[5], help="number of tiers to use for reward")
    parser.add_argument("--seed", "-s", type=int, default=0, help="random seed")

    # hyperparams
    parser.add_argument("--gamma", "-g", type=float, default=0.90, help="discount rate")
    parser.add_argument("--delta", "-d", type=float, default=0.1, help="tier offset for reward")

    # debug
    parser.add_argument("--verbose", "-v", action="store_true", default=False)

    args = parser.parse_args()

    np.random.seed(args.seed)

    for tier in args.tiers:

        # saving dir
        saving_dir = os.path.join('results', f"{args.env}-qlearning", f"{tier}-tier")
        create_log_dir(saving_dir, remove_existing=True)

        # make env
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
            tier_r = make_tier_reward(num_states=num_chain_states, num_tiers=tier, gamma=args.gamma, delta=args.delta)
            print(tier_r)
            tier_env = make_one_dim_chain(
                num_states=num_chain_states,
                discount_rate=args.gamma,
                goal_reward=None,
                step_reward=None,
                success_rate=0.8,
                custom_rewards=tier_r,
            )
            pbs_env = potential_based_shaping_reward(env)
        else:
            raise NotImplementedError
        
        results = train_on_env(env, num_steps=args.steps, seed=args.seed, verbose=args.verbose)
        print('original reward')
        print([res.NumGoalsHit for res in results])
        tiered_results = train_on_env(tier_env, num_steps=args.steps, seed=args.seed, verbose=args.verbose)
        print('tiered reward')
        print([res.NumGoalsHit for res in tiered_results])
        pbs_results = train_on_env(pbs_env, num_steps=args.steps, seed=args.seed, verbose=args.verbose)
        print('potential-based shaping reward')
        print([res.NumGoalsHit for res in pbs_results])

        # log results
        def log_results(result, reward_type):
            for res in result:
                episodic_reward = res.EpisodicReward
                for step, ep_reward in episodic_reward.items():
                    kvlogger.logkv('step', step)
                    kvlogger.logkv('episodic_reward', ep_reward)
                    kvlogger.logkv('reward_type', reward_type)
                    kvlogger.logkv('time_till_goal', res.TimeAtGoal)
                    kvlogger.logkv('num_goals_hit', res.NumGoalsHit)
                    kvlogger.logkv('seed', res.Seed)
                    kvlogger.dumpkvs()

        kvlogger.configure(saving_dir, format_strs=['csv'], log_suffix='')
        log_results(results, 'original')
        log_results(tiered_results, 'tiered')
        log_results(pbs_results, 'potential based shaping')

    # plot results
    compare_goal_hitting_stat_with_different_tiers(os.path.join('results', f"{args.env}-qlearning"), args.tiers)
