from os import environ
import gym
import numpy as np


class TierRewardWrapper(gym.Wrapper):
    """
    modify the reward() function to make the reward tiered
    """
    def __init__(self, env, num_tiers, gamma, keep_original_reward=False):
        super().__init__(env)
        self.num_tiers = num_tiers
        self.gamma = gamma
        self.keep_original_reward = keep_original_reward

        self.h = 1 / (1-gamma)
        self.delta = 0.1  # good tier r = H * prev tier + delta 
        self.tiers_hitting_count = np.zeros(num_tiers, dtype=np.int32)
    
    def reset_hitting_count(self):
        """"""
        assert self.keep_original_reward
        self.tiers_hitting_count = np.zeros(self.num_tiers, dtype=np.int32)

    def reset(self, reset_count=False, **kwargs):
        if reset_count:
            self.reset_hitting_count()
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = self.reward(reward, info)
        self.log_tier_hitting_count(info)
        return obs, reward, done, info

    def reward(self, reward, info):
        raise NotImplementedError
    
    def log_tier_hitting_count(self, info):
        raise NotImplementedError


class BreakoutTierReward(TierRewardWrapper):
    """
    There are 18 brick for each row, the game is playing through the two sequential walls.

    The original reward is:
        There are six rows of bricks.  The color of a brick determines the
        points you score when you hit it with your ball.
        Red - 7 points         Orange - 7 points        Yellow - 4 points
        Green - 4 points       Aqua - 1 point           Blue - 1 point
    We modify the reward to be:
        tiers are defined in terms of the number of bricks hit
        the lowest tier has 0 point, each tier would increase the reward by *H* times + delta
    """
    num_total_bricks = 18 * 6 * 2

    def _get_tier(self, block_hit_count):
        bricks_per_tier = self.num_total_bricks / self.num_tiers
        tier = int(block_hit_count / bricks_per_tier)
        return tier

    def reward(self, reward, info):
        info['original_reward'] = float(reward)

        if self.keep_original_reward:
            return reward

        tier = self._get_tier(info['labels']['blocks_hit_count'])
        if tier == 0:
            reward = 0
        else:
            reward = self.h ** (tier-1) + self.delta
        return reward
    
    def log_tier_hitting_count(self, info):
        tier = self._get_tier(info['labels']['blocks_hit_count'])
        self.tiers_hitting_count[tier] += 1
        info['tiers_hitting_count'] = self.tiers_hitting_count


def wrap_tier_rewards(env, num_tiers, gamma, keep_original_reward=False):
    env_id = (env.spec.id).lower()
    if 'breakout' in env_id:
        env = BreakoutTierReward(env, num_tiers=num_tiers, gamma=gamma, keep_original_reward=keep_original_reward)
    else:
        raise NotImplementedError
    return env
