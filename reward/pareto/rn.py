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


def make_policy(env_name, is_pareto, verbose=False):
    """
    make a policy for the RN gridworld
    Pareto policy results from reward goal +1, lava -1, step cost -0.04
    Non-Pareto policy results from reward goal +1, lava -1, step cost -0.5
    args:
        is_pareto: whether the policy is pareto
    """
    gamma = 0.95
    env_params = {
        'discount_rate': gamma,
        'slip_prob': 0.8,
    }

    if env_name == 'rn_grid':
        goal_reward = 1 if is_pareto else 1
        lava_penalty = -1 if is_pareto else 1
        step_cost = -0.04 if is_pareto else 0
        if is_pareto:
            assert lava_penalty < 1/(1-gamma) * step_cost < goal_reward
        env_params = {
            **env_params,
            'goal_reward': goal_reward,
            'lava_penalty': lava_penalty,
            'step_cost': step_cost,
        }

    elif env_name == 'puddle':
        goal_reward = 1 if is_pareto else -1
        step_cost = 0 if is_pareto else 0
        puddle_cost = -0.3 if is_pareto else 0.5
        env_params = {
            **env_params,
            'goal_reward': goal_reward,
            'step_cost': step_cost,
            'puddle_cost': puddle_cost,
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
        save_path = f'results/{env_name}/policy_pareto_{is_pareto}.png'
        plt.savefig(save_path)
        print(f'policy saved to {save_path}')
        plt.close()
    return mdp, policy


def _accumulate_state_distribution(state_dist):
    """
    if there are multiple states of the same feature (e.g. multiple puddles)
    we want to sum the distribution over all the states of the same feature
    """
    if state_dist.ndim == 1:
        return state_dist
    else:
        return state_dist.sum(axis=1)


def plot_pareto_policy_termination_prob(env_name, num_steps, verbose=False):
    """
    make the plot comparing a pareto policy with a non-pareto one wrt its termination probability
    """
    # pareto
    pareto_mdp, pareto_policy = make_policy(env_name, is_pareto=True, verbose=verbose)
    pareto_state_dist = get_state_distribution(pareto_mdp, pareto_policy, num_steps=num_steps)
    pareto_goal_probs = _accumulate_state_distribution(pareto_state_dist[:, GOAL])
    pareto_lava_probs = _accumulate_state_distribution(pareto_state_dist[:, LAVA])  # (nsteps, )

    # non-pareto
    non_pareto_mdp, non_pareto_policy = make_policy(env_name, is_pareto=False, verbose=verbose) 
    non_pareto_state_dist = get_state_distribution(non_pareto_mdp, non_pareto_policy, num_steps=num_steps)
    non_pareto_goal_probs = _accumulate_state_distribution(non_pareto_state_dist[:, GOAL])
    non_pareto_lava_probs = _accumulate_state_distribution(non_pareto_state_dist[:, LAVA])

    # plot
    pareto_color = 'c'
    non_pareto_color = 'm'
    plt.plot(pareto_goal_probs, color=pareto_color, label='goal: pareto')
    plt.plot(non_pareto_goal_probs, color=non_pareto_color, label='goal: non-pareto')
    plt.plot(pareto_lava_probs, '-.', color=pareto_color, label='lava: pareto')
    plt.plot(non_pareto_lava_probs, '-.', color=non_pareto_color, label='lava: non-pareto')
    plt.legend()
    plt.title('Probability of Reaching Goal/Lava')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Probability')
    save_path = f'results/{env_name}/pareto_prob.png'
    plt.savefig(save_path)
    print(f'Saved to {save_path}')
    plt.close()



def policy_through_reward(env_name: str, mdp: TabularMarkovDecisionProcess, precision: float=-0.01) -> TabularPolicy:
    """
    gather all possible reward functions, and return the policy
    args:
        mdp: not used, keeping it consistent for API 
    """
    gamma = 0.95
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


def check_all_rewards_for_paretoness(env_name):
    """
    generate a bunch of randon policies, and plot each policy in a 2D grid as (prob_success, prob_fail)
    """
    # only need this for state, action list, transition matrix, initial state distribution
    env_params = {
        'discount_rate': 0.95,
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
    for i, (policy, r) in enumerate(policy_through_reward(env_name, pseudo_mdp, precision=precision)):
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


def debug():
    """ testing this script """
    from reward.environments import make_one_dim_chain
    from msdm.algorithms import ValueIteration
    chain = make_one_dim_chain(num_states=10, goal_reward=1, step_reward=-1, discount_rate=0.9)
    vi = ValueIteration()
    result = vi.plan_on(chain)
    policy = result.policy

    state_distribution = get_state_distribution(chain, policy, num_steps=15)
    print(state_distribution)

    from reward.environments.plot import visualize_grid_world_and_policy
    visualize_grid_world_and_policy(chain, plot_name_prefix='debug', results_dir='results', policy=policy)


if __name__ == '__main__':
    np.random.seed(0)

    import argparse
    parser = argparse.ArgumentParser()
    # experiments
    parser.add_argument('--env', type=str, default='rn_grid')
    
    # debug
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument('--debug', action='store_true', default=False)

    args = parser.parse_args()
    
    results_dir = f'results/{args.env}'
    os.makedirs(results_dir, exist_ok=True)
    kvlogger.configure(results_dir, format_strs=['json'])
    print(f"logging to {results_dir}/progress.json")

    # environmental constant 
    if args.env == 'rn_grid':
        GOAL = 10
        LAVA = 9
        NUM_STEPS = 20
    
    check_all_rewards_for_paretoness(args.env)
    # plot_pareto_policy_termination_prob(args.env, num_steps=NUM_STEPS, verbose=args.verbose)
