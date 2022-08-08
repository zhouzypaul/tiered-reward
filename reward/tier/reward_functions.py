import math

import numpy as np


def _get_tier_reward(tier, gamma, delta):
    """
    what's the reward for the i-the tier
    tier          reward
    0             0
    1             H^0
    2             H^1 + delta
    k             H^(k-1) + (H^k-2 +...+ H^0)) * delta
    """
    h = 1 / (1 - gamma)
    if tier == 0:
        return 0
    elif tier == 1:
        return 1
    else:
        return h ** (tier-1) + delta * (h **(tier-1) - 1) / (h - 1)


def make_tier_reward(num_states, num_tiers, gamma=0.95, delta=0.1):
    """
    return a numpy array of length num_states
    NOTE: the goal has to be the single state that's the highest tier
    """
    r = np.zeros(num_states)
    states_per_tier = math.ceil((num_states-1) / (num_tiers-1))
    # final tier
    r[-1] = _get_tier_reward(num_tiers-1, gamma, delta)
    # other tiers
    for i in range(num_tiers-1):
        for j in range(states_per_tier):
            idx = i * states_per_tier + j
            if idx >= num_states-1:
                break
            r[idx] = _get_tier_reward(i, gamma, delta)
    return r


if __name__ == "__main__":
    # testing this script
    r = make_tier_reward(60, 7)
    print(r)
