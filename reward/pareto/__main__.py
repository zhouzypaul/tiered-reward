import pandas as pd
import numpy as np
import seaborn as sns
from msdm.algorithms import ValueIteration
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess
from msdm.core.problemclasses.mdp import TabularPolicy
from matplotlib import pyplot as plt 

from reward.environments import make_russell_norvig_grid
from reward.environments.russel_norvig import visualize_rn_grid_policy
from reward.msdm_utils import get_ordered_state_action_list, state_distribution_array


# constants for the russell/norvig gridworld
GOAL = 10
LAVA = 9
NUM_STEPS = 20


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


def make_policy(is_pareto, verbose=False):
    """
    make a policy for the RN gridworld
    Pareto policy results from reward goal +1, lava -1, step cost -0.04
    Non-Pareto policy results from reward goal +1, lava -1, step cost -0.5
    args:
        is_pareto: whether the policy is pareto
    """
    gamma = 0.95
    goal_reward = 1 if is_pareto else 1
    lava_penalty = -1 if is_pareto else 1
    step_cost = -0.04 if is_pareto else 0
    if is_pareto:
        assert lava_penalty < 1/(1-gamma) * step_cost < goal_reward
    
    mdp = make_russell_norvig_grid(
        discount_rate=gamma,
        slip_prob=0.8,
        goal_reward=goal_reward,
        lava_penalty=lava_penalty,
        step_cost=step_cost,
    )
    vi = ValueIteration()
    result = vi.plan_on(mdp)
    policy = result.policy
    if verbose:
        _, ax = plt.subplots(1, 1)
        visualize_rn_grid_policy(policy, ax=ax)
        plt.savefig(f'results/rn_grid_pareto_{is_pareto}.png')
        plt.close()
    return mdp, policy


def plot_pareto_policy_termination_prob(num_steps=NUM_STEPS):
    """
    make the plot comparing a pareto policy with a non-pareto one wrt its termination probability
    """
    # pareto
    pareto_mdp, pareto_policy = make_policy(is_pareto=True, verbose=False)
    pareto_state_dist = get_state_distribution(pareto_mdp, pareto_policy, num_steps=num_steps)
    pareto_goal_probs = pareto_state_dist[:, GOAL]
    pareto_lava_probs = pareto_state_dist[:, LAVA]

    # non-pareto
    non_pareto_mdp, non_pareto_policy = make_policy(is_pareto=False, verbose=False) 
    non_pareto_state_dist = get_state_distribution(non_pareto_mdp, non_pareto_policy, num_steps=num_steps)
    non_pareto_goal_probs = non_pareto_state_dist[:, GOAL]
    non_pareto_lava_probs = non_pareto_state_dist[:, LAVA]

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
    save_path = 'results/pareto_prob.png'
    plt.savefig(save_path)
    print(f'Saved to {save_path}')
    plt.close()


def sample_random_policy(mdp: TabularMarkovDecisionProcess) -> TabularPolicy:
    """
    sample a random policy from the MDP
    """
    states, actions = get_ordered_state_action_list(mdp)
    n_states = len(states)
    n_actions = len(actions)
    random_actions = np.random.choice(n_actions, size=n_states)
    policy_mat = np.zeros((n_states, n_actions))
    policy_mat[np.arange(n_states), random_actions] = 1
    return TabularPolicy.from_matrix(states, actions, policy_mat)


def sample_random_policy_through_random_reward(mdp: TabularMarkovDecisionProcess) -> TabularPolicy:
    """
    sample a random policy by sampling a random reward, and them optimizing it to make the policy
    args:
        mdp: not used, keeping it consistent for API 
    """
    gamma = 0.95
    goal_reward = np.random.uniform(-1, 1)
    lava_penalty = np.random.uniform(-1, goal_reward)
    # sample step cost
    upper = (1-gamma) * goal_reward
    lower = (1-gamma) * lava_penalty
    step_cost = np.random.uniform(lower, upper)
    
    mdp = make_russell_norvig_grid(
        discount_rate=gamma,
        slip_prob=0.8,
        goal_reward=goal_reward,
        lava_penalty=lava_penalty,
        step_cost=step_cost,
    )
    vi = ValueIteration()
    result = vi.plan_on(mdp)
    policy = result.policy

    return policy


def get_success_and_failure_prob(mdp: TabularMarkovDecisionProcess, policy: TabularPolicy, num_steps: int=NUM_STEPS):
    """
    given an MDP and a policy, turn the policy into a two-point statistic: (goal_reaching_prob, obstacle_reaching_prob)

    note: this currently only supports the Russel/Norvig Gridworld
    """
    state_dist = get_state_distribution(mdp, policy, num_steps=num_steps)
    goal_probs = state_dist[:, GOAL]
    lava_probs = state_dist[:, LAVA]
    
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
        return np.all(goal_prob_1 >= goal_prob_2) and np.all(lava_prob_1 <= lava_prob_2)

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


def random_policy_paretoness_plot(num_polices=5000, policy_sample_method=sample_random_policy):
    """
    generate a bunch of randon policies, and plot each policy in a 2D grid as (prob_success, prob_fail)
    """
    # only need this for state, action list, transition matrix, initial state distribution
    pseudo_mdp = make_russell_norvig_grid(
        discount_rate=0.95,
        slip_prob=0.8,
        goal_reward=1,
        lava_penalty=-1,
        step_cost=-0.04,
    )
    
    # collect the data
    stats = np.zeros((num_polices, 2))
    prob_of_reaching_goals = np.zeros((num_polices, NUM_STEPS))
    prob_of_reaching_lava = np.zeros((num_polices, NUM_STEPS))
    for i in range(num_polices):
        policy = policy_sample_method(pseudo_mdp)
        p_goals, p_lavas = get_success_and_failure_prob(pseudo_mdp, policy)
        prob_of_reaching_goals[i, :] = p_goals
        prob_of_reaching_lava[i, :] = p_lavas
        stats[i] = [p_goals[-1], p_lavas[-1]]  # last timestep prob
    
    # label the policies
    labels = determine_whether_pareto(prob_of_reaching_goals, prob_of_reaching_lava)

    df = pd.DataFrame(stats, columns=['goal', 'lava'])
    df['label'] = labels

    # plot
    grid = sns.jointplot(data=df, x='goal', y='lava', hue='label', alpha=0.5)
    grid.ax_joint.set_xlim(0, 1)
    grid.ax_joint.set_ylim(0, 1)
    grid.ax_joint.set_xlabel('Probability of Success')
    grid.ax_joint.set_ylabel('Probability of Failure')
    grid.fig.subplots_adjust(right=0.95)
    # save
    save_path = 'results/random_policy_paretoness.png'
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
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--random_policy_paretoness', '-r', action='store_true', default=False)
    parser.add_argument('--num_policies', '-n', type=int, default=5000)
    parser.add_argument('--sample_with_reward', '-s', action='store_true', default=False)
    args = parser.parse_args()

    if args.debug:
        debug()
    elif args.random_policy_paretoness:
        if args.sample_with_reward:
            sample_method = sample_random_policy_through_random_reward
        else:
            sample_method = sample_random_policy
        random_policy_paretoness_plot(num_polices=args.num_policies, policy_sample_method=sample_method)
    else:
        plot_pareto_policy_termination_prob()
