import os
import argparse

import numpy as np

from reward.environments import make_one_dim_chain, make_single_goal_square_grid, make_frozen_lake, make_russell_norvig_grid
from reward.agents import QLearning, RMaxAgent, run_learning, run_multiprocessing_learning
from reward.tier.reward_functions import potential_based_shaping_reward, make_distance_based_tier_reward, _get_tier_reward
from reward.tier.plot import compare_goal_hitting_stat_with_different_tiers
from reward.utils import create_log_dir
from reward import kvlogger


def train_on_env(env, agent_name, num_steps, num_seeds, multiprocessing=True, verbose=False, seed=0):
    """
    train on one specific environment, with one agent
    """
    if multiprocessing:
        agents = [
            make_agent(agent_name, seed=s, num_steps=num_steps)
            for s in range(num_seeds)
        ]
        results = run_multiprocessing_learning(
            env,
            agents=agents,
        )
    else:
        agent = make_agent(agent_name, seed, num_steps)
        result = run_learning(env, agent)
        results = [result]

    if verbose:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 1, figsize=(16, 4))
        env.plot(ax=axes).plot_policy(results[-1].policy).title("Policy")
        plt.tight_layout()
        plt.savefig('results/debug.png')
        print('saved to results/debug.png')

    return results

def make_agent(agent_name, seed, num_steps):
    """
    create the learning agent
    """
    optimistic_value = 1e10
    if agent_name == 'qlearning':
        agent = QLearning(
            num_steps=num_steps,
            rand_choose=0,
            initial_q=optimistic_value,
            seed=seed,
        )
    elif agent_name == 'rmax':
        agent = RMaxAgent(
            num_steps=num_steps,
            rmax=optimistic_value,
            seed=seed,
        )
    else:
        raise NotImplementedError
    return agent


def make_env(env_name, num_tiers, discount, delta):
    """
    create three environments
    returns:
        env, tier_env, tier_based_shaping_env
    """
    if env_name == "chain":
        num_chain_states = 9
        env = make_one_dim_chain(
            num_states=num_chain_states,
            discount_rate=discount,
            goal_reward=1,
            step_reward=-1,
            success_rate=0.8,
            custom_rewards=None,
        )
        tier_r = make_distance_based_tier_reward(
            env,
            num_tiers=num_tiers,
            gamma=discount,
            delta=delta,
        )
        print(tier_r)
        tier_env = make_one_dim_chain(
            num_states=num_chain_states,
            discount_rate=discount,
            goal_reward=None,
            step_reward=None,
            success_rate=0.8,
            custom_rewards=tier_r,
        )
    
    elif env_name == 'grid':
        num_side_states = 9
        env = make_single_goal_square_grid(
            num_side_states=num_side_states,
            discount_rate=discount,
            success_prob=0.8,
            step_cost=-1,
            goal_reward=1,
            custom_reward=None,
        )
        tier_r = make_distance_based_tier_reward(
            env,
            num_tiers=num_tiers,
            gamma=discount,
            delta=delta,
        )
        print(tier_r)
        tier_env = make_single_goal_square_grid(
            num_side_states=num_side_states,
            discount_rate=discount,
            success_prob=0.8,
            step_cost=None,
            goal_reward=None,
            custom_reward=tier_r,
        )
    
    elif env_name == 'frozen_lake':
        env = make_frozen_lake(
            discount_rate=discount,
            goal_reward=1,
            step_cost=-1,
            hole_penalty=-1,
        )
        tier_env = make_frozen_lake(
            discount_rate=discount,
            goal_reward=_get_tier_reward(tier=2, gamma=discount),
            step_cost=_get_tier_reward(tier=1, gamma=discount),
            hole_penalty=_get_tier_reward(tier=0, gamma=discount),
        )
        tier_r = tier_env.reward_vector
    
    elif env_name == "rn_grid":
        env = make_russell_norvig_grid(
            discount_rate=discount,
            goal_reward=1,
            lava_penalty=-1,
            step_cost=-1,
        )
        tier_env = make_russell_norvig_grid(
            discount_rate=discount,
            goal_reward=1,
            lava_penalty=-1,
            step_cost=-0.04,
        )
        tier_r = tier_env.reward_vector

    else:
        raise NotImplementedError

    tier_pbs_env = potential_based_shaping_reward(
        env,
        shaping_func=lambda s: tier_r[env.state_index[s]]
    )

    return env, tier_env, tier_pbs_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="chain",
                        choices=["chain", "grid", "frozen_lake", "rn_grid"],)
    parser.add_argument('--agent', type=str, default='qlearning',
                        choices=['qlearning', 'rmax'])
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--tiers", "-t", type=int, nargs="+", default=[5], help="number of tiers to use for reward")
    parser.add_argument("--seed", "-s", type=int, default=0, help="random seed")
    parser.add_argument("--num_seeds", type=int, default=10, help="number of seeds to use for multiprocessing")

    # hyperparams
    parser.add_argument("--gamma", "-g", type=float, default=0.90, help="discount rate")
    parser.add_argument("--delta", "-d", type=float, default=0.1, help="tier offset for reward")

    # debug
    parser.add_argument("--verbose", "-v", action="store_true", default=False)

    args = parser.parse_args()

    np.random.seed(args.seed)

    for tier in args.tiers:

        # saving dir
        saving_dir = os.path.join('results', f"{args.env}-{args.agent}", f"{tier}-tier")
        create_log_dir(saving_dir, remove_existing=True)

        # make env
        env, tier_env, tier_pbs_env = make_env(args.env, tier, args.gamma, args.delta)
        
        # training
        results = train_on_env(env, agent_name=args.agent, num_steps=args.steps, seed=args.seed, num_seeds=args.num_seeds, verbose=args.verbose)
        print('original reward')
        print([res.TimeAtGoal for res in results])
        tiered_results = train_on_env(tier_env, agent_name=args.agent, num_steps=args.steps, seed=args.seed, num_seeds=args.num_seeds, verbose=args.verbose)
        print('tiered reward')
        print([res.TimeAtGoal for res in tiered_results])
        tiered_pbs_results = train_on_env(tier_pbs_env, agent_name=args.agent, num_steps=args.steps, seed=args.seed, num_seeds=args.num_seeds, verbose=args.verbose)
        print('Tier-based shaping reward')
        print([res.TimeAtGoal for res in tiered_pbs_results])

        # log results
        def log_results(result, reward_type):
            for res in result:
                episodic_reward = res.EpisodicReward
                for step, ep_reward in episodic_reward.items():
                    kvlogger.logkv('step', step)
                    kvlogger.logkv('episodic_reward', ep_reward)
                    kvlogger.logkv('Reward Type', reward_type)
                    kvlogger.logkv('time_till_goal', res.TimeAtGoal)
                    kvlogger.logkv('num_goals_hit', res.NumGoalsHit)
                    kvlogger.logkv('seed', res.Seed)
                    kvlogger.dumpkvs()

        kvlogger.configure(saving_dir, format_strs=['csv'], log_suffix='')
        log_results(results, 'Original')
        log_results(tiered_results, 'Tiered')
        log_results(tiered_pbs_results, 'Tier Based Shaping')

    # plot results
    compare_goal_hitting_stat_with_different_tiers(os.path.join('results', f"{args.env}-{args.agent}"), args.tiers)
