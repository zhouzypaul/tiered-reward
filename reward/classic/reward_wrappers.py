from functools import cached_property

import gym
import numpy as np
import math


def get_k_tiered_reward(i_tier, total_num_tiers, H, delta):
    """
    k-tiered reward as defined in the paper
    what's the reward for the i-the tier
    tier          reward
    k-1             0
    k-2           H * r_k-1 - delta
    k-3           H * r_k-2 - delta
    ...
    Note that we start indexing the tiers from 0
    """
    if i_tier >= total_num_tiers:
        raise ValueError
    elif i_tier == total_num_tiers - 1:
        return 0
    else:
        return H * get_k_tiered_reward(i_tier + 1, total_num_tiers, H, delta) - delta


class NormalizedRewardWrapper(gym.Wrapper):
    """
    Normalize the reward so that its absolute value is between [0, 1]
    For TieredReward that is designed to be negative, this wrapper should normalize the
    reward values to be between [-1, 0]
    """

    def __init__(self, env, max_r_value):
        super().__init__(env)
        self.max_r_value = max_r_value

    def reward(self, r):
        """
        divide by max reward value
        """
        new_r = r / self.max_r_value
        assert -1 <= new_r <= 1
        return new_r

    def reset(self, **kwargs):
        """
        need to override this method so that args can be passed to TierRewardWrapper
        """
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = self.reward(reward)
        return obs, reward, done, info


class TierRewardWrapper(gym.Wrapper):
    """
    modify the reward() function to make the reward tiered
    """

    def __init__(self, env, num_tiers, gamma, delta, keep_original_reward=False):
        super().__init__(env)
        self.num_tiers = num_tiers
        self.gamma = gamma
        self.keep_original_reward = keep_original_reward

        self.h = 1 / (1-gamma)
        self.delta = delta  # good tier r = H * prev tier + delta
        self.tiers_hitting_count = np.zeros(num_tiers, dtype=np.int32)

    def reset_hitting_count(self):
        """"""
        assert self.keep_original_reward
        self.tiers_hitting_count = np.zeros(self.num_tiers, dtype=np.int32)

    def reset(self, reset_count=False, **kwargs):
        if reset_count:
            self.reset_hitting_count()
        return self.env.reset(**kwargs)

    def _get_tier_reward(self, tier):
        """
        what's the reward for the i-the tier
        tier          reward
        k             0
        k-1           H * r_k - delta
        k-2           H * r_k-1 - delta
        ...
        """
        return get_k_tiered_reward(tier, self.num_tiers, self.h, self.delta)

    def reward(self, reward, obs, info):
        info['original_reward'] = float(reward)

        if self.keep_original_reward:
            return reward

        tier = self._get_tier(obs)
        return self._get_tier_reward(tier)

    def log_tier_hitting_count(self, obs, info):
        tier = self._get_tier(obs)
        self.tiers_hitting_count[tier] += 1
        info['tiers_hitting_count'] = self.tiers_hitting_count

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = self.reward(reward, obs, info)
        self.log_tier_hitting_count(obs, info)
        return obs, reward, done, info

class CartPoleTierReward(TierRewardWrapper):

    def _get_tier(self, obs):
        angle = abs(obs[2])
        tier = math.floor((angle / 0.418) * self.num_tiers)
        return tier

class AcrobotTierReward(TierRewardWrapper):

    def _get_tier(self, obs):
        cost1, sint1, cost2, sint2, velt1, velt2 = obs
        t1 = math.acos(cost1)
        t2 = math.acos(cost2)

        height = -cost1 - math.cos(t1+t2)
        n_height = (height + 2.0) / 4.0

        tier = math.floor(n_height * self.num_tiers)
        return tier

def wrap_tier_rewards(env, num_tiers, gamma, delta, keep_original_reward=False, normalize_reward=False):
    env_id = (env.spec.id).lower()
    if 'cartpole' in env_id:
        env = CartPoleTierReward(env, num_tiers=num_tiers, gamma=gamma,
                                 delta=delta, keep_original_reward=keep_original_reward)
    if 'acrobot' in env_id:
        env = AcrobotTierReward(env, num_tiers=num_tiers, gamma=gamma,
                                 delta=delta, keep_original_reward=keep_original_reward)
    else:
        raise NotImplementedError

    # normalize reward
    if normalize_reward:
        max_reward = get_k_tiered_reward(
            0, total_num_tiers=num_tiers, H=1/(1-gamma), delta=delta)
        max_reward = abs(max_reward)
        env = NormalizedRewardWrapper(env, max_r_value=max_reward)

    return env


if __name__ == "__main__":
    # for testing the k-tier reward
    H = 1/(1-0.9)
    delta = 0.1

    assert get_k_tiered_reward(0, 1, H, delta) == 0
    try:
        get_k_tiered_reward(1, 1, H, delta)
    except ValueError:
        print('ValueError raised as expected')

    k1 = get_k_tiered_reward(0, 5, H, delta)  # -111.1
    k2 = get_k_tiered_reward(1, 5, H, delta)  # -11.1
    k3 = get_k_tiered_reward(2, 5, H, delta)  # -1.1
    k4 = get_k_tiered_reward(3, 5, H, delta)  # -0.1
    k5 = get_k_tiered_reward(4, 5, H, delta)  # 0
    print(k1, k2, k3, k4, k5)
