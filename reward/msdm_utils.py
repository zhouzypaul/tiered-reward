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
