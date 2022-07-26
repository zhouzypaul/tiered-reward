import enum
import warnings
import numpy as np
from collections import defaultdict

from msdm.core.problemclasses.mdp import TabularPolicy
from msdm.algorithms import ValueIteration
from msdm.core.algorithmclasses import PlanningResult
from msdm.core.distributions.distributions import FiniteDistribution


def state_distribution_array(state_list, dist: FiniteDistribution):
    """
    given a FiniteDistrubution, make it into a numpy array
    """
    state_dist = np.zeros((len(state_list),))
    for i, s in enumerate(state_list):
        state_dist[i] = dist.prob(s)
    return state_dist


class ModifiedValueIteration(ValueIteration):
    """
    modify the value iteration so that we use our definition of the transition matrix
    """
    def __init__(self, 
                iterations=None, 
                convergence_diff=0.00001, 
                check_unreachable_convergence=True):
        super().__init__(iterations=iterations, 
                        convergence_diff=convergence_diff, 
                        check_unreachable_convergence=check_unreachable_convergence)
    
    def plan_on(self, mdp):
        ss = mdp.state_list
        tf = transition_matrix_terminal_state_hack(mdp)  # difference!
        rf = mdp.reward_matrix
        nt = mdp.nonterminal_state_vec
        rs = mdp.reachable_state_vec
        am = mdp.action_matrix

        iterations = self.iterations
        if iterations is None:
            iterations = max(len(ss), int(1e5))

        terminal_sidx = np.where(1 - nt)[0]

        def get_qvalues(R, T, V, gamma):  # difference!
            """
            Q(s, a) = R(s, a) + sum_{s'} * {gamma * T(s, a, s') * V(s')}
            """
            s_dim, a_dim, _ = R.shape
            q = np.zeros((s_dim, a_dim))
            for s in range(s_dim):
                for a in range(a_dim):
                    try:
                        # this state receives a nonzero reward
                        existing_reward_idx = np.nonzero(R[s, a, :])[0]
                        assert len(existing_reward_idx) >= 1
                        state_reward = R[s, a, existing_reward_idx[0]]
                        assert np.all(R[s, a, existing_reward_idx] == state_reward)
                        q[s, a] += state_reward + gamma * np.sum(T[s, a, :] * V[:])
                    except AssertionError:
                        # the reward for this state is 0
                        assert len(existing_reward_idx) == 0
                        state_reward = 0
                        q[s, a] += state_reward + gamma * np.sum(T[s, a, :] * V[:])
            return q

        v = np.zeros(len(ss))
        for i in range(iterations):
            # q = np.einsum("san,san->sa", tf, rf + mdp.discount_rate * v[None, None, :])  # difference!
            q = get_qvalues(rf, tf, v, mdp.discount_rate)  # difference!
            nv = np.max(q + np.log(am), axis=-1)
            # nv[terminal_sidx] = 0 #terminal states are always 0 reward  # difference!
            if self.check_unreachable_convergence:
                diff = (v - nv)
            else:
                diff = (v - nv)*rs
            if np.abs(diff).max() < self.convergence_diff:
                break
            v = nv

        validq = q + np.log(am)
        pi = TabularPolicy.from_q_matrix(mdp.state_list, mdp.action_list, validq)

        # create result object
        res = PlanningResult()
        if i == (iterations - 1):
            warnings.warn(f"VI not converged after {iterations} iterations")
            res.converged = False
        else:
            res.converged = True
        res.mdp = mdp
        res.policy = res.pi = pi
        res._valuevec = v
        vf = dict()
        for s, vi in zip(mdp.state_list, v):
            vf[s] = vi
        res.valuefunc = res.V = vf
        res._qvaluemat = q
        res.iterations = i
        res.max_bellman_error = diff
        qf = defaultdict(lambda : dict())
        for si, s in enumerate(mdp.state_list):
            for ai, a in enumerate(mdp.action_list):
                qf[s][a] = q[si, ai]
        res.actionvaluefunc = res.Q = qf
        res.initial_value = sum([res.V[s0]*p for s0, p in mdp.initial_state_dist().items()])
        return res


def transition_matrix_terminal_state_hack(gw):
    """
    given a gw, msdm defines that the terminal state always have probability 1
    to transition to itself
    here, we define terminal state to be actually terminal: regardless of what
    actions the agent takes, it has probability 0 to transition to any other state

    return the transition matrix with out definition
    """
    msdm_transition_mat = gw.transition_matrix

    # find out which states are terminal
    terminal_states = []
    all_states = gw.state_list
    for s in all_states:
        if gw.is_terminal(s):
            terminal_states.append(s)
    terminal_states_idx = list(map(lambda s: gw.state_index[s], terminal_states))

    # modify the transition matrix
    transition_mat = np.copy(msdm_transition_mat)
    n_states, n_actions, _ = transition_mat.shape
    for idx in terminal_states_idx:
        transition_mat[idx] = np.zeros((n_actions, n_states))
    
    return transition_mat


def get_policy_value(policy, mdp, num_iters=10):
    """
    given a policy, figure out the value of following that policy in the mdp
    this is used to determine if two policies are equivalent in the mdp

    execute the policy for num_iter times, because some policies are stochastic
    """
    # get the value of following the policy
    policy_values = np.zeros((num_iters,))
    for i in range(num_iters):
        step = 0
        s = mdp.initial_state_dist().sample()
        while not mdp.is_terminal(s):
            # control
            a = policy.action_dist(s).sample()
            ns = mdp.next_state_dist(s, a).sample()
            r = mdp.reward(s, a, ns)
            policy_values[i] += r * (mdp.discount_rate**step)
            step += 1
            s = ns
    return np.mean(policy_values)


def check_policy_equality(a, b, mdp):
    """
    given 2 policies, check whether they are the same policy
    a, b might be deterministic, or stochastic

    the policy equality doesn't care about the terminal state, because policy check in
    terminal state doesn't matter
    """
    return get_policy_value(a, mdp) == get_policy_value(b, mdp)


def make_policy_deterministic(policy, mdp):
    """
    for a msdm.core.problemclasses.mdp.policy.TabularPolicy
    if a certain state has an equal probability for several optimal actions
    enforce the determinism by choosing the first of those actions to be deterministic
    """
    states = mdp.state_list  # NOTE: order is not guaranteed on this list, but not a problem here

    deterministic_states_to_actions = {}
    for s in states:
        action_dist = policy[s]
        sorted_actions = sorted(action_dist.support, key=lambda a: (a['dx'], a['dy']))
        a = sorted_actions[-1]
        deterministic_states_to_actions[s] = a
    
    return TabularPolicy.from_deterministic_map(deterministic_states_to_actions)


def get_ordered_state_action_list(mdp):
    """
    return state_list, action_list
    both of which are guranteed to be in the order of mdp.state_index, mdp.action_index
    this method is needed because mdp.state_list, mdp.action_list doesn't guaranteed the
    order of the lists
    """
    state_list = mdp.state_list
    action_list = mdp.action_list
    state_list = sorted(state_list, key=lambda s: mdp.state_index[s])
    action_list = sorted(action_list, key=lambda a: mdp.action_index[a])
    return state_list, action_list
