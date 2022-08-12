import math
from copy import deepcopy

import numpy as np
from msdm.algorithms import ValueIteration
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess


def _get_tier_reward(tier, num_total_tiers, gamma, delta=0.1):
    """
    what's the reward for the i-the tier
    tier          reward
    0             0
    1             H^0
    2             H^1 + delta
    k             H^(k-1) + (H^k-2 +...+ H^0)) * delta
    """
    h = 1 / (1 - gamma)
    if tier >= num_total_tiers - 1:
        return 0
    else:
        return _get_tier_reward(tier+1, num_total_tiers, gamma, delta) * h - delta


def _find_feature_index(env, f):
    states = []
    indexes = []
    for s, si in env.state_index.items():
        if env.location_features.get(s, 's') == f:
            states.append(s)
            indexes.append(si)
    return states, indexes


def make_wall_grid_tier_reward(env, num_tiers, gamma, delta):
    """
    tier reward for wall grid
    
    the distance metric of each state to the goal is baked into the env
    and the tiers are based on the distance to the goal
    the single goal will always be the highest tier
    and the lava will be the worst tier
    """
    assert num_tiers >= 3
    # find all lava
    lavas, idx_lavas = _find_feature_index(env, 'x')
    assert len(lavas) == 5
    # find all walls
    walls, idx_walls = _find_feature_index(env, '#')
    assert len(walls) == 7
    # find all goals
    goal, idx_goal = _find_feature_index(env, 'g')
    assert len(goal) == 1
    # bin according to distance
    forbidden_states = lavas + goal + walls 
    state_to_distance = {s: env.location_distances[s] for s in env.state_list if s not in forbidden_states}
    state_to_distance = {k: v for k, v in sorted(state_to_distance.items(), key=lambda item: item[1])}  # sort according to distance
    num_per_bin = math.ceil(len(state_to_distance) / (num_tiers-2))
    state_to_bin_idx = {}
    for i, (s, dist) in enumerate(state_to_distance.items()):
        state_to_bin_idx[s] = i // num_per_bin

    # build the tier rewards
    tier_r = np.zeros(len(env.state_list))
    for s, si in env.state_index.items():
        # check if the state is a wall
        if s in walls:
            tier_r[si] = np.nan
        else:
            if si in idx_goal:
                # top tier
                tier_r[si] = _get_tier_reward(num_tiers-1, num_tiers, gamma, delta)
            elif si in idx_lavas:
                # bottom tier
                tier_r[si] = _get_tier_reward(0, num_tiers, gamma, delta)
            else:
                # middle tiers
                assert state_to_bin_idx[s] < num_tiers - 1
                tier_r[si] = _get_tier_reward(state_to_bin_idx[s]+1, num_tiers, gamma, delta)
    return tier_r


def make_frozen_lake_tier_reward(env, num_tiers, gamma, delta):
    """
    make a tier reward function for frozen lake
    
    the tiers are based on the shortest distance to the goal state
    the holes will always be the worst tiers
    the single goal will always the highest tier
    """
    assert num_tiers >= 3  # at least 3 tiers
    # find all the hole states
    holes, idx_holes = _find_feature_index(env, 'h')
    assert len(idx_holes) == 10
    # find the goal state
    goal, idx_goal = _find_feature_index(env, 'g')
    assert len(idx_goal) == 1
    # bin according to distance
    terminal_states = holes + goal
    state_to_distance = {s: env.location_distances[s] for s in env.state_list if s not in terminal_states}
    state_to_distance = {k: v for k, v in sorted(state_to_distance.items(), key=lambda item: item[1])}  # sort based on distance
    num_per_bin = math.ceil(len(state_to_distance) / (num_tiers-2))
    state_to_bin_idx = {}
    for i, (s, dist) in enumerate(state_to_distance.items()):
        state_to_bin_idx[s] = i // num_per_bin

    # build the tiered reward
    tier_r = np.zeros(len(env.state_list))
    for s, si in env.state_index.items():
        if si in idx_goal:
            # top tier
            tier_r[si] = _get_tier_reward(num_tiers-1, num_tiers, gamma, delta)
        elif si in idx_holes:
            # bottom tier
            tier_r[si] = _get_tier_reward(0, num_tiers, gamma, delta)
        else:
            # middle tiers, according to bin
            assert state_to_bin_idx[s] < num_tiers - 1
            tier_r[si] = _get_tier_reward(state_to_bin_idx[s]+1, num_tiers, gamma, delta)  # +1 to allow the bottom tier
    return tier_r


def make_distance_based_tier_reward(env, num_tiers, gamma, delta):
    """
    given a tabular MDP, return a reward function that is tiered
    the tiers depend on the L1 distance to the goal state

    the goal is always its own tier -- the highest tier
    the rest of the tiers are spread out evenly according to distance metric

    NOTE: this assumes there is only a single goal state
    """
    def get_l1_distance(a, b):
        """
        L1 distance of two msdm states
        a, b: both are frozendict({'x': x, 'y': y})
        """
        return abs(a['x'] - b['x']) + abs(a['y'] - b['y'])
    # find the distances of every state to the goal 
    assert len(env.absorbing_states) == 1, "Only one goal state is supported"
    goal_state = env.absorbing_states[0]  # assume only one goal
    idx_goal = env.state_index[goal_state]
    state_to_distance = {s: get_l1_distance(s, goal_state) for s in env.state_list if s != goal_state}
    distances = np.array(list(state_to_distance.values()))
    bined_distance = np.digitize(distances, np.linspace(max(distances), 0, num_tiers-1))  # indices of the bins to which each value in input array belongs
    if num_tiers == 2:
        # a small hack to prevent there bing two bins
        bined_distance = np.where(bined_distance >= num_tiers-1, num_tiers-2, bined_distance)  # if the value is greater than the highest bin, put it in the highest bin
    state_to_bin_idx = {s: bined_distance[i] for i, s in enumerate(state_to_distance.keys())}

    # build the tiered reward
    tier_r = np.zeros(len(env.state_list))
    for s, si in env.state_index.items():
        if si == idx_goal:
            tier_r[si] = _get_tier_reward(num_tiers-1, num_tiers, gamma, delta)
        else:
            assert state_to_bin_idx[s] < num_tiers - 1
            tier_r[si] = _get_tier_reward(state_to_bin_idx[s], num_tiers, gamma, delta)
    return tier_r


def potential_based_shaping_reward(env: TabularMarkovDecisionProcess, shaping_func=None):
    """
    take in a TabularMDP, and output the potential based shaped reward
    NOTE: this wrapper should only be used for Q Learning, 
          don't use the wrapper for anything else, because it only overwrites the reward
          function, the reward matrix is off, and others may be too.
    """
    if shaping_func is None:
        vi = ValueIteration()
        res = vi.plan_on(env)
        values = res.V
        shaping_func = lambda s: values[s]
    original_reward_func = deepcopy(env.reward)

    def shaped_reward(s, a, ns):
        shaping = shaping_func(ns) * env.discount_rate - shaping_func(s)
        return shaping + original_reward_func(s, a, ns)
    
    pbs_env = deepcopy(env)
    pbs_env.reward = shaped_reward
    return pbs_env


if __name__ == "__main__":
    # testing this script
    total_tiers = 5
    for i in range(total_tiers):
        print(_get_tier_reward(i, total_tiers, 0.9))

    import argparse
    from pathlib import Path
    from reward.environments import make_single_goal_square_grid, make_frozen_lake, make_wall_grid
    from reward.environments.plot import plot_grid_reward

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", '-e', type=str, default='wall_grid',
                        choices=['grid', 'frozen_lake', 'wall_grid'])
    parser.add_argument("--tier", '-t', type=int, default=5)
    args = parser.parse_args()

    discount = 0.95

    if args.env == 'grid':
        num_side_states = 9
        num_tiers = args.tier
        delta = 0.1
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

        plot_grid_reward(
            tier_env,
            plot_name_prefix='debug_grid',
            results_dir=Path('./results/'),
        )

    elif args.env == 'frozen_lake':
        env = make_frozen_lake(
            discount_rate=discount,
            goal_reward=1,
            step_cost=-1,
            hole_penalty=-1,
        )
        tier_r = make_frozen_lake_tier_reward(
            env, args.tier, discount, 0.1,
        )
        tier_env = make_frozen_lake(
            discount_rate=discount,
            goal_reward=None,
            step_cost=None,
            hole_penalty=None,
            custom_rewards=tier_r,
        )
        plot_grid_reward(
            tier_env,
            plot_name_prefix='debug_frozen_lake',
            results_dir=Path('./results/'),
            reward_vec=tier_r,
        )
    
    elif args.env == 'wall_grid':
        env = make_wall_grid(
            discount_rate=discount,
            success_prob=0.8,
            goal_reward=1,
            step_cost=-1,
            lava_penalty=-1,
        )
        tier_r = make_wall_grid_tier_reward(env, args.tier, discount, 0.1)
        tier_env = make_wall_grid(
            discount_rate=discount,
            success_prob=0.8,
            goal_reward=None,
            step_cost=None,
            lava_penalty=None,
            custom_rewards=tier_r,
        )
        plot_grid_reward(
            tier_env,
            plot_name_prefix='debug_wall_grid',
            results_dir=Path('./results/'),
            reward_vec=tier_r,
        )
