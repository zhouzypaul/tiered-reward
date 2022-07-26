import numpy as np
import msdm
from msdm.algorithms import ValueIteration
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess
from msdm.core.problemclasses.mdp import TabularPolicy
from matplotlib import pyplot as plt 

from reward.environments import make_russell_norvig_grid
from reward.environments.russel_norvig import visualize_rn_grid_policy
from reward.msdm_utils import get_ordered_state_action_list, state_distribution_array


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
    goal_reward = 1
    lava_penalty = -1
    step_cost = -0.04 if is_pareto else -0.9
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
    return mdp, policy


def plot_pareto_policy_termination_prob(num_steps=20):
    # index of state in the state list
    goal = 10
    lava = 9

    # pareto
    pareto_mdp, pareto_policy = make_policy(is_pareto=True, verbose=False)
    pareto_state_dist = get_state_distribution(pareto_mdp, pareto_policy, num_steps=num_steps)
    pareto_goal_probs = pareto_state_dist[:, goal]
    pareto_lava_probs = pareto_state_dist[:, lava]

    # non-pareto
    non_pareto_mdp, non_pareto_policy = make_policy(is_pareto=False, verbose=False) 
    non_pareto_state_dist = get_state_distribution(non_pareto_mdp, non_pareto_policy, num_steps=num_steps)
    non_pareto_goal_probs = non_pareto_state_dist[:, goal]
    non_pareto_lava_probs = non_pareto_state_dist[:, lava]

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    if args.debug:
        debug()
    else:
        plot_pareto_policy_termination_prob()
