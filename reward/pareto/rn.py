import os

import pandas as pd
import numpy as np
import seaborn as sns
from msdm.algorithms import ValueIteration
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess
from msdm.core.problemclasses.mdp import TabularPolicy
from matplotlib import pyplot as plt 

from reward.environments import make_russell_norvig_grid
from reward.environments import make_puddle_world
from reward.environments.russell_norvig import visualize_rn_grid_policy
from reward.msdm_utils import get_ordered_state_action_list, state_distribution_array
from reward import kvlogger


def get_state_distribution(mdp: TabularMarkovDecisionProcess, policy: TabularPolicy, num_steps: int):
    """
    given an MDP and a corresponding policy, calculate the state ditribution at each timestep
    following that policy
    To calculate, basically just apply the forward equation repeatedly 

    This state distribution will give us the probability of reaching the goal/obstacle
    at each time step

    returns:
        state_dist: (steps, nstates)
    """
    # env
    states, actions = get_ordered_state_action_list(mdp)
    transition_mat = mdp.transition_matrix  # (S, A, S)
    start_state_dist = state_distribution_array(state_list=states, dist=mdp.initial_state_dist())

    # policy
    policy_mat = policy.as_matrix(states, actions)  # (S, A)
    policy_transition = np.einsum('sa,san->sn', policy_mat, transition_mat)  # (S, S)

    # state distribution
    current_dist = start_state_dist
    state_dist = np.zeros((num_steps, len(states)))
    for i in range(num_steps):
        state_dist[i] = current_dist
        current_dist = current_dist @ policy_transition
        assert np.isclose(current_dist.sum(), 1)  # make sure it's a valid distribution
    
    return state_dist


def make_env(env_name, **params):
    """
    make the environement
    """
    if env_name == 'rn_grid':
        return make_russell_norvig_grid(**params)
    elif env_name == 'puddle':
        return make_puddle_world(**params)
    else:
        raise ValueError(f'unknown environment {env_name}')


def make_policy(env_name, gamma, lava_penalty, step_cost, goal_reward, verbose=False):
    """
    make one policy
    """
    assert env_name == 'rn_grid'

    env_params = {
        'discount_rate': gamma,
        'slip_prob': 0.8,
        'goal_reward': goal_reward,
        'lava_penalty': lava_penalty,
        'step_cost': step_cost,
    }
    mdp = make_env(env_name, **env_params)

    vi = ValueIteration()
    result = vi.plan_on(mdp)
    policy = result.policy
    if verbose:
        _, ax = plt.subplots(1, 1)
        if env_name == 'rn_grid':
            visualize_rn_grid_policy(policy, ax=ax)
        else: 
            plot = mdp.plot(True)
            plot.pP(policy) # plots policy
        save_path = f'results/{env_name}/policy_({lava_penalty, step_cost, goal_reward}).png'
        plt.savefig(save_path)
        print(f'policy saved to {save_path}')
        plt.close()
    return mdp, policy


def make_one_direction_policy(env, direction):
    """
    make a policy one the environment such that all policy take the specified direction
    """
    dir_to_action = {
        'left': 0,
        'down': 1,
        'up': 2,
        'right': 3,
    }
    states, actions = get_ordered_state_action_list(env)
    n_states = len(states)
    n_actions = len(actions)
    policy_mat = np.zeros((n_states, n_actions))
    policy_mat[:, dir_to_action[direction]] = 1
    return TabularPolicy.from_matrix(states, actions, policy_mat)


def _accumulate_state_distribution(state_dist):
    """
    if there are multiple states of the same feature (e.g. multiple puddles)
    we want to sum the distribution over all the states of the same feature
    """
    if state_dist.ndim == 1:
        return state_dist
    else:
        return state_dist.sum(axis=1)


def plot_policy_termination_prob(env_name, gamma, num_steps, verbose=False):
    """
    make three policies and compare their termination probabilities
    """
    colors = {
        'R': 'r',
        'G': 'g',
        'B': 'b',
        'up': '#7d34eb',
        'left': 'c',
        'right': 'y',
        'down': '#eb344f',
    }
    rewards = {
        'R': (-1, -0.1, +1),
        'G': (-1, 0, 0.5),
        'B': (-1, -0.9, 0),
    }

    # RBG plot 
    r_mdp, r_policy = make_policy(env_name, gamma, *rewards['R'])
    g_mdp, g_policy = make_policy(env_name, gamma, *rewards['G'])
    b_mdp, b_policy = make_policy(env_name, gamma, *rewards['B'])
    r_state_dist = get_state_distribution(r_mdp, r_policy, num_steps)
    g_state_dist = get_state_distribution(g_mdp, g_policy, num_steps)
    b_state_dist = get_state_distribution(b_mdp, b_policy, num_steps)
    r_goal_probs = _accumulate_state_distribution(r_state_dist[:, GOAL])
    g_goal_probs = _accumulate_state_distribution(g_state_dist[:, GOAL])
    b_goal_probs = _accumulate_state_distribution(b_state_dist[:, GOAL])
    r_lava_probs = _accumulate_state_distribution(r_state_dist[:, LAVA])
    g_lava_probs = _accumulate_state_distribution(g_state_dist[:, LAVA])
    b_lava_probs = _accumulate_state_distribution(b_state_dist[:, LAVA])

    x = range(num_steps)
    alpha = 0.6
    plt.fill_between(x, r_goal_probs, g_goal_probs, facecolor="r", alpha=alpha, label=r'R')
    plt.fill_between(x, g_goal_probs, b_goal_probs, facecolor="y", alpha=alpha, label=r'R $\cap$ G')
    plt.fill_between(x, b_goal_probs, b_lava_probs-1, facecolor="#112596", alpha=alpha, label=r'R $\cap$ G $\cap$ B')
    plt.fill_between(x, b_lava_probs-1, r_lava_probs-1, facecolor="y", alpha=alpha)
    plt.fill_between(x, r_lava_probs-1, g_lava_probs-1, facecolor="g", alpha=alpha, label=r'       G')
    
    plt.legend(loc='upper left')
    plt.title(f'Comparing Policies')
    plt.xlabel('Step')
    plt.ylabel(r'$- 1 + \sum_t o_t$                                      $\sum_t g_t$')
    plt.ylim((-1.05, 1.05))
    plt.xticks(x)
    plt.subplots_adjust(left=0.17, right=0.95)
    save_path = f'results/{env_name}/RGB_prob.png'
    plt.savefig(save_path)
    print(f'plot saved to {save_path}')
    plt.close()

    # one direction policy plot
    left_policy = make_one_direction_policy(r_mdp, direction='left')
    right_policy = make_one_direction_policy(r_mdp, direction='right')
    left_state_dist = get_state_distribution(r_mdp, left_policy, num_steps)
    right_state_dist = get_state_distribution(r_mdp, right_policy, num_steps)
    left_goal_probs = _accumulate_state_distribution(left_state_dist[:, GOAL])
    right_goal_probs = _accumulate_state_distribution(right_state_dist[:, GOAL])
    left_lava_probs = _accumulate_state_distribution(left_state_dist[:, LAVA])
    right_lava_probs = _accumulate_state_distribution(right_state_dist[:, LAVA])

    plt.fill_between(x, r_goal_probs, right_goal_probs, facecolor="r", alpha=alpha, label=r'R')
    plt.fill_between(x, right_goal_probs, left_goal_probs, facecolor="#804000", alpha=alpha, label=r'R $\cap$ right')
    plt.fill_between(x, left_goal_probs, right_lava_probs-1, facecolor="#ff9200", alpha=alpha, label=r'R $\cap$ right $\cap$ left')
    plt.fill_between(x, right_lava_probs-1, r_lava_probs-1, facecolor="#800080", alpha=alpha, label=r'R $\cap$             left')
    plt.fill_between(x, r_lava_probs-1, left_lava_probs-1, facecolor="c", alpha=alpha, label=r'                   left')

    plt.legend(loc='upper left')
    plt.title(f'Comparing Policies')
    plt.xlabel('Step')
    plt.ylabel(r'$- 1 + \sum_t o_t$                                      $\sum_t g_t$')
    plt.ylim((-1.05, 1.05))
    plt.xticks(x)
    plt.subplots_adjust(left=0.17, right=0.95)
    save_path = f'results/{env_name}/one_direction_policy_prob.png'
    plt.savefig(save_path)
    print(f'plot saved to {save_path}')
    plt.close()


def policy_through_reward(env_name: str, gamma: float, mdp: TabularMarkovDecisionProcess, precision: float=-0.01) -> TabularPolicy:
    """
    gather all possible reward functions, and return the policy
    args:
        mdp: not used, keeping it consistent for API 
    """
    env_params = {
        'discount_rate': gamma,
        'slip_prob': 0.8,
    }

    assert env_name == 'rn_grid'

    for goal_reward in np.arange(1, -1, precision):
        for step_cost in np.arange(goal_reward, -1, precision):
            for lava_penalty in np.arange(step_cost, -1, precision):
                r = (lava_penalty, step_cost, goal_reward)
                env_params = {
                    **env_params,
                    'goal_reward': goal_reward,
                    'lava_penalty': lava_penalty,
                    'step_cost': step_cost,
                }
                mdp = make_env(env_name, **env_params)

                vi = ValueIteration()
                result = vi.plan_on(mdp)
                policy = result.policy

                yield policy, r
    


def get_success_and_failure_prob(mdp: TabularMarkovDecisionProcess, policy: TabularPolicy, num_steps: int):
    """
    given an MDP and a policy, turn the policy into a two-point statistic: (goal_reaching_prob, obstacle_reaching_prob)

    note: this currently only supports the Russel/Norvig Gridworld
    """
    state_dist = get_state_distribution(mdp, policy, num_steps=num_steps)
    goal_probs = _accumulate_state_distribution(state_dist[:, GOAL])
    lava_probs = _accumulate_state_distribution(state_dist[:, LAVA])
    
    return goal_probs, lava_probs


def determine_whether_pareto(goal_timestep_probs, lava_timestep_probs):
    """
    given a bunch of goal and lava timestep-wise probabilities, determine which of the policies are on the pareto frontier
    args:
        goal_timestep_probs: (n, num_timesteps)
        lava_timestep_probs: (n, num_timesteps)
    """
    assert len(goal_timestep_probs) == len(lava_timestep_probs)
    n = len(goal_timestep_probs)

    def _domination(goal_prob_1, lava_prob_1, goal_prob_2, lava_prob_2):
        """
        given two policies 1 and 2, determine whether 1 dominates 2
        returns:
            True if 1 dominates 2, False otherwise
        """
        dominating = np.all(goal_prob_1 >= goal_prob_2) and np.all(lava_prob_1 <= lava_prob_2)
        # prevent the edge case of two policies being identical
        identical = np.allclose(goal_prob_1, goal_prob_2) and np.allclose(lava_prob_1, lava_prob_2)
        return dominating and not identical

    # create a nxn matrix to keep track of the domination relationships
    # mat[i, j] = 1 if policy i dominates policy j, 0 otherwise
    # diagonal is left to be 0, because of how we are labeling stuff below
    domination_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # [i] is being compared to [j]
            i_goal_prob = goal_timestep_probs[i]
            i_lava_prob = lava_timestep_probs[i]
            j_goal_prob = goal_timestep_probs[j]
            j_lava_prob = lava_timestep_probs[j]
            i_dominate_j = _domination(i_goal_prob, i_lava_prob, j_goal_prob, j_lava_prob)
            domination_matrix[i, j] = int(i_dominate_j)
    
    # labels
    labels = np.zeros((n,), dtype=object)
    # if it gets dominated by something, it is non-pareto
    for i in range(n):
        if np.any(domination_matrix[:, i]):
            labels[i] = 'pareto dominated'
        else:
            labels[i] = 'pareto optimal'
    
    return labels


def check_all_rewards_for_paretoness(env_name, gamma):
    """
    generate a bunch of randon policies, and plot each policy in a 2D grid as (prob_success, prob_fail)
    """
    # only need this for state, action list, transition matrix, initial state distribution
    env_params = {
        'discount_rate': gamma,
        'slip_prob': 0.8,  # matters for the transition matrix
    }
    pseudo_mdp = make_env(env_name, **env_params)

    num_examples = 0
    precision = -0.1
    for i in np.arange(1, -1, precision):
        for j in np.arange(i, -1, precision):
            for _ in np.arange(j, -1, precision):
                num_examples += 1
    
    print(f'{num_examples} examples to check')
    p_goals_all = np.zeros((num_examples, NUM_STEPS))
    p_lava_all = np.zeros((num_examples, NUM_STEPS))
    stats = np.zeros((num_examples, 2))
    reward_funcs = np.zeros((num_examples, 3))
    for i, (policy, r) in enumerate(policy_through_reward(env_name, gamma, pseudo_mdp, precision=precision)):
        p_goal, p_lava = get_success_and_failure_prob(pseudo_mdp, policy, num_steps=NUM_STEPS)
        reward_funcs[i, :] = r
        p_goals_all[i, :] = p_goal
        p_lava_all[i, :] = p_lava
        stats[i] = [p_goal[-1], p_lava[-1]]  # last timestep prob
    
    labels = determine_whether_pareto(p_goals_all, p_lava_all)

    # logging
    for i in range(len(labels)):
        r_lava, r_step, r_goal = reward_funcs[i]
        p_goals, p_lavas = p_goals_all[i], p_lava_all[i]
        label = labels[i]
        p_goal_final = p_goals[-1]
        p_lava_final = p_lavas[-1]
        kvlogger.logkv('p_goal_final', p_goal_final)
        kvlogger.logkv('p_lava_final', p_lava_final)
        kvlogger.logkv('r', reward_funcs[i])
        kvlogger.logkv('r_lava', r_lava)
        kvlogger.logkv('r_step', r_step)
        kvlogger.logkv('r_goal', r_goal)
        kvlogger.logkv('label', label)
        kvlogger.logkv('p_goal', p_goals)
        kvlogger.logkv('p_lava', p_lavas)
        kvlogger.dumpkvs()
    
    # plotting
    df = pd.DataFrame(stats, columns=['goal', 'lava'])
    df['label'] = labels
    df.sort_values(by='goal', inplace=True)

    # plot
    grid = sns.jointplot(data=df, x='goal', y='lava', hue='label', alpha=0.5)
    grid.ax_joint.set_xlim(0, 1)
    grid.ax_joint.set_ylim(0, 1)
    grid.ax_joint.set_xlabel('Probability of Success')
    grid.ax_joint.set_ylabel('Probability of Failure')
    grid.fig.subplots_adjust(right=0.95)
    # save
    save_path = f'results/{env_name}/random_policy_paretoness.png'
    plt.savefig(save_path)
    print(f'Saved to {save_path}')
    plt.close()


def print_interesting_reward(results_path, find_good_rewards=False):
    """
    print the reward function that are interesting
    """
    df = kvlogger.read_json(results_path)

    # round the rewards to the nearest 0.1
    df.r_goal = df.r_goal.apply(lambda x: round(x, 2))
    df.r_step = df.r_step.apply(lambda x: round(x, 2))
    df.r_lava = df.r_lava.apply(lambda x: round(x, 2))
    df.r = df.apply(lambda row: np.array([row.r_lava, row.r_step, row.r_goal]), axis=1)

    # intuitive constraints
    df = df[df.r_lava < df.r_step]
    df = df[df.r_step < df.r_goal]

    if find_good_rewards:
        df = df[df['label'] == 'pareto optimal']
        df = df[df.r_lava < 0]
        df = df[df.r_step <= 0]
        df = df[df.r_goal >= 0]

        df = df[df.r_lava == -1]
        # df = df[df.r_goal == 1]
        df = df.sort_values(by='p_lava_final', ascending=True)

    else:
        df = df[df['label'] == 'pareto dominated']
        df = df[df.r_lava < 0]
        df = df[df.r_lava == -1]
        # df = df[df.r_step <= 0]
        df = df[df.r_goal >= 0]
        
        df = df[df.p_goal_final < 0.2]
        # df = df[df.p_goal_final > 0]
        # df = df[df.p_lava_final < 0.2]


    print(df[['r', 'r_lava', 'r_step', 'r_goal', 'label', 'p_goal_final', 'p_lava_final']])



if __name__ == '__main__':
    np.random.seed(0)

    import argparse
    parser = argparse.ArgumentParser()
    # experiments
    parser.add_argument('--env', type=str, default='rn_grid')
    parser.add_argument('--print', '-p', action='store_true', help='print interesting reward functions')
    parser.add_argument('--plot', action='store_true', help='plot interesting reward functions')
    parser.add_argument('--find_good_rewards', '-g', action='store_true', help='find good reward functions')

    parser.add_argument('--gamma', type=float, default=0.90)
    
    # debug
    parser.add_argument("--verbose", "-v", action="store_true", default=False)

    args = parser.parse_args()
    
    results_dir = f'results/{args.env}'
    os.makedirs(results_dir, exist_ok=True)

    # environmental constant 
    if args.env == 'rn_grid':
        GOAL = 10
        LAVA = 9
        NUM_STEPS = 20
    
    if args.print:
        print_interesting_reward(results_dir + '/progress.json', find_good_rewards=args.find_good_rewards)
    elif args.plot:
        plot_policy_termination_prob(args.env, args.gamma, num_steps=10, verbose=args.verbose)
    else:
        kvlogger.configure(results_dir, format_strs=['json'])
        print(f"logging to {results_dir}/progress.json")
        check_all_rewards_for_paretoness(args.env, args.gamma)
