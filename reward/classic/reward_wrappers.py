from functools import cached_property

import gym
import numpy as np
import math

epsilon = 0.001

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



class TierRewardWrapper(gym.Wrapper):
    """
    modify the reward() function to make the reward tiered
    """

    def __init__(self, env, num_tiers, gamma, delta, keep_original_reward=False, normalize_reward=False):
        super().__init__(env)
        self.num_tiers = num_tiers
        self.gamma = gamma
        self.keep_original_reward = keep_original_reward
        self.normalize_reward = normalize_reward

        self.h = 1 / (1-gamma)
        self.delta = delta  # good tier r = H * prev tier + delta
        self.tiers_hitting_count = np.zeros(num_tiers, dtype=np.int32)

        if normalize_reward:
            max_reward = get_k_tiered_reward(
            0, total_num_tiers=num_tiers, H=1/(1-gamma), delta=delta)
            
            self.max_reward = abs(max_reward)

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

    def reward(self, og_reward, obs, info):
        info['original_reward'] = float(og_reward)

        if self.keep_original_reward:
            return og_reward

        tier = self._get_tier(obs)
        reward = self._get_tier_reward(tier)

        if self.normalize_reward:
            norm_reward = reward / self.max_reward
            assert -1 <= norm_reward <= 1
            reward = norm_reward

        # print(f'og: {og_reward}', f'tier: {reward}')
        
        return reward

    def log_tier_hitting_count(self, obs, info):
        tier = self._get_tier(obs)
        self.tiers_hitting_count[tier] += 1
        info['tiers_hitting_count'] = self.tiers_hitting_count

    def step(self, action):
        obs, og_reward, done, info = self.env.step(action)
        reward = self.reward(og_reward, obs, info)
        self.log_tier_hitting_count(obs, info)
        return obs, reward, done, info

class CartPoleTierReward(TierRewardWrapper):

    def _get_tier(self, obs):
        angle = abs(obs[2])
        tier = math.floor(((0.418-angle) / (0.418+ epsilon)) * self.num_tiers)
        return tier

class AcrobotTierReward(TierRewardWrapper):

    def _get_tier(self, obs):
        cost1, sint1, cost2, sint2, velt1, velt2 = obs
        t1 = math.acos(cost1)
        t2 = math.acos(cost2)

        height = -cost1 - math.cos(t1+t2)
        n_height = (height + 2.0) / (4.0 + epsilon)

        tier = math.floor(n_height * self.num_tiers)

        # print(f'height: {height}')
        # print(f'tier: {tier}')
        return tier

def wrap_tier_rewards(env, num_tiers, gamma, delta, keep_original_reward=False, normalize_reward=False):
    env_id = (env.spec.id).lower()
    if 'cartpole' in env_id:
        env = CartPoleTierReward(env, num_tiers=num_tiers, gamma=gamma,
                                 delta=delta, keep_original_reward=keep_original_reward, 
                                 normalize_reward=normalize_reward)
    elif 'acrobot' in env_id:
        env = AcrobotTierReward(env, num_tiers=num_tiers, gamma=gamma,
                                 delta=delta, keep_original_reward=keep_original_reward, 
                                 normalize_reward=normalize_reward)
    else:
        raise NotImplementedError

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
