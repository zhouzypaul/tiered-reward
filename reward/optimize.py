import numpy as np
import cvxpy as cp

from optre.visitation import get_discountcounted_visitation


def get_linear_program_reward(objective_gamma, 
                                subjective_gamma, 
                                feature_matrix, 
                                transition_matrix, 
                                target_policy_matrix,
                                ignore_state_idx=[],
                                mono_increasing_rewards=False):
    """
    run a linear program to obtain the optimal reward function of a grid world 

    ignore_state_idx is designed so that we can ignore constraints on the terminal state
    """
    # visitation values
    objective_state_visitation, objective_state_action_visitation = get_discountcounted_visitation(
        discount=objective_gamma,
        policy_matrix=target_policy_matrix,
        transition_fn=transition_matrix,
        feature_fn=feature_matrix
    )
    subjective_state_visitation, subjective_state_action_visitation = get_discountcounted_visitation(
        discount=subjective_gamma,
        policy_matrix=target_policy_matrix,
        transition_fn=transition_matrix,
        feature_fn=feature_matrix
    )

    # build the action gap constraints for the LP
    objective_constraints = build_action_gap_constraint(
        state_visitation=objective_state_visitation,
        state_action_visitation=objective_state_action_visitation,
        policy_matrix=target_policy_matrix,
        ignore_state_idx=ignore_state_idx,
    )
    subjective_constraints = build_action_gap_constraint(
        state_visitation=subjective_state_visitation,
        state_action_visitation=subjective_state_action_visitation,
        policy_matrix=target_policy_matrix,
        ignore_state_idx=ignore_state_idx,
    )
    stacked_constraints = np.vstack([objective_constraints, subjective_constraints])

    # do the Linear Program
    r, delta = solve_linear_program(
        action_gap_constraints=stacked_constraints, 
        n_rewards=len(feature_matrix), 
        mono_increasing_rewards=mono_increasing_rewards,
    )

    # Ri is a variable representing the immediate reward received when
    # acting from a state with feature i. Or, put another way, the reward in
    # state s is sum_i Fi(s) Ri.
    rewards = np.dot(np.transpose(feature_matrix), r)

    return r, rewards, delta


def build_action_gap_constraint(state_visitation, state_action_visitation, policy_matrix, ignore_state_idx=[]):
    """
    given a particular D and DA, build the matrix constraints for the LP
    """
    # arrange the matrices so that feature_dim goes first
    n_states, n_actions, n_feats = state_action_visitation.shape

    # build the advantage matrix: (Di(s) - Dai(s)) for all S and A\{pi(s)}
    gap = []  # create a place holder, to be reshaped later
    for s in range(n_states):
        if s in ignore_state_idx:
            continue
        action_probs = policy_matrix[s]
        assert np.count_nonzero(action_probs == 1) == 1  # there's only one optimal action for each s
        optimal_action = np.argmax(action_probs)
        for a in range(n_actions):
            # don't constrain equivalent actions
            if not np.array_equal(state_action_visitation[s, a, :], state_action_visitation[s, optimal_action, :]):
                gap.append(state_visitation[s, :] - state_action_visitation[s, a, :])
    gap = np.vstack(gap)  # stack each pair (s, a) vertically

    return gap


def solve_linear_program(action_gap_constraints,
                        n_rewards=3, 
                        mono_increasing_rewards=False):
    """
    solve the linear program to get the optimal reward

    scheme:
        The Ris will be the variables. We’ll also have another variable called delta.

        forall s in S, we want the value of action a to exceed that of the other actions by at least delta.
        We want, for all s in S, a in A\{pi(s)}:
            sum_{i} { Ri * (Di(s) - Dai(s)) } - delta >= 0
        We’d add constraints on R: 0 <= Ri <= 1.

        sometimes we want to ignore certain states that are indexed by `ignore_state_idx`.
        For example, in terminal states g, we always have Di(g) - Dai(g) = 0. This ruins the LP
        therefore, we ignore those state.

        The objective is simple: maximize delta.
    """
    # set up variables of LP
    rewards = cp.Variable((n_rewards,))
    delta = cp.Variable()

    all_constraints = [
        (action_gap_constraints @ rewards) - delta >= 0,
        rewards <= 1,
        -1 <= rewards,
    ]
    if mono_increasing_rewards:
        all_constraints += [
            rewards[1:] - rewards[:-1] >= 0,
        ]
    obj = cp.Maximize(delta)
    problem = cp.Problem(obj, all_constraints)
    problem.solve()
    assert (problem.status == 'optimal'), problem.status
    return rewards.value, delta.value
