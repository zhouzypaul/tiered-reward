import numpy as np

from optre.msdm_utils import ModifiedValueIteration


def get_discountcounted_visitation(discount,
                                    policy_matrix, 
                                    transition_fn, 
                                    feature_fn, 
                                    max_iterations=10000, 
                                    converge_delta=1e-40):
    """
    compute the discountcouted expected visitation for states and state-action pairs
    expected visitation is the expected amount of times the policy is going to visit 
    a certain state. It is used to calculate the value function:
    value = expected_visitation * rewards

    expected state visitation:
        Di(s) represents the discounted expected visitation of feature i starting from state s following the target policy.
        D_i(s) = F_i(S) + gamma * sum_{s'}{ T(s, pi(s), s') * D_i(s') }
    
    expected state-action visitation:
        a closely related quantity that represents the discounted expected visitation 
        of feature i starting from s, taking action a, then following pi thereafter:
        DA_i(s) = F_i(s) + gamma * sum_{s'}{ T(s, a, s') * D_i(s') }
    Args:
        transition_fn: of shape (state_dim, action_dim, state_dim)
        feature_fn: of shape (feature_dim, state_dim)
    """
    state_dim, action_dim, _ = transition_fn.shape
    feature_dim = len(feature_fn)
    state_visitation = np.zeros((state_dim, feature_dim))

    # computate state visitation
    # D_i(s) = F_i(S) + gamma * sum_{s'}{ T(s, pi(s), s') * D_i(s') }  TODO: solve this using matrix form
    # use value iteration
    for t in range(max_iterations):
        prev_state_visitation = np.copy(state_visitation)

        for idx_s in range(state_dim):
            for idx_feat in range(feature_dim):
                idx_a = policy_matrix[idx_s].argmax()
                next_state_visitation = transition_fn[idx_s, idx_a, :] * prev_state_visitation[:, idx_feat]
                assert next_state_visitation.shape == (state_dim,)
                state_visitation[idx_s, idx_feat] = feature_fn[idx_feat, idx_s] + discount * np.sum(next_state_visitation)

        # check for convergence
        delta = np.abs(state_visitation - prev_state_visitation)
        if np.all(delta < converge_delta):
            break
    
    if t == max_iterations - 1:
        raise RuntimeError(f"Warning, D not converged after {t} iterations with delta {delta}")

    # compute state_action_visitation
    # DA_i(s) = F_i(s) + gamma * sum_{s'}{ T(s, a, s') * D_i(s') }
    state_action_visitation = np.zeros((state_dim, action_dim, feature_dim))
    for idx_s in range(state_dim):
        for idx_a in range(action_dim):
            for idx_feat in range(feature_dim):
                next_state_visitation = transition_fn[idx_s, idx_a, :] * state_visitation[:, idx_feat]
                state_action_visitation[idx_s, idx_a, idx_feat] = feature_fn[idx_feat, idx_s] + discount * np.sum(next_state_visitation)

    return state_visitation, state_action_visitation


def validate_visitation_values(mdp, policy_matrix, transition_fn, feature_fn, reward_values):
    """
    validate that the state_visitation (D) and state_action_visitation (DA) is correct
    given a reward function R:
        D * R = state value function
        DA * R = q values
        (the above are matrix multiplication that sums out the feature dimension)
    NOTE: the function assumes that input reward_values is of length num_states,
            instead of num_features. So it always uses an identity feature matrix.
    """
    # get visitation values that doesn't depend on the input feature groups
    # aka always use the identity feature matrix for this function
    state_visitation, state_action_visitation = get_discountcounted_visitation(
        discount=mdp.discount_rate, 
        policy_matrix=policy_matrix, 
        transition_fn=transition_fn, 
        feature_fn=feature_fn,
    )

    value_iter = ModifiedValueIteration()
    planning_results = value_iter.plan_on(mdp)
    true_value_func = planning_results._valuevec  # (n_states, )
    true_q_values = planning_results._qvaluemat  # (n_states, n_actions)

    value_func =  state_visitation @ reward_values
    q_values = state_action_visitation @ reward_values

    assert np.allclose(value_func, true_value_func, atol=1e-4)
    assert np.allclose(q_values, true_q_values, atol=1e-4)
