from functools import cached_property

import gym
import numpy as np


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

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = self.reward(reward, info)
        self.log_tier_hitting_count(info)
        return obs, reward, done, info

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
    """
    num_total_bricks = 18 * 6 * 2

    @cached_property
    def bricks_per_tier(self):
        return self.num_total_bricks / (self.num_tiers-1)

    def _get_tier(self, block_hit_count):
        if block_hit_count == self.num_total_bricks:
            tier = self.num_tiers - 1
        else:
            tier = int(block_hit_count / self.bricks_per_tier)
            try:
                assert 0 <= tier < self.num_tiers - 1
            except AssertionError:
                tier = self.num_tiers - 2  # numerical error
        return tier

    def reward(self, reward, info):
        info['original_reward'] = float(reward)

        if self.keep_original_reward:
            return reward

        tier = self._get_tier(info['labels']['blocks_hit_count'])
        return self._get_tier_reward(tier)

    def log_tier_hitting_count(self, info):
        tier = self._get_tier(info['labels']['blocks_hit_count'])
        self.tiers_hitting_count[tier] += 1
        info['tiers_hitting_count'] = self.tiers_hitting_count


class FreewayTierReward(TierRewardWrapper):
    """
    In FreewayNoFrameskip-v4, there is only three actions: NOOP, UP, DOWN
    the chicken on the right cannot be moved. 
    each action moves the chickin by 1 up/down along the y-axis, as indicated by the ram position

    The original reward is:
        +1 when the chicken is across the road, 0 else
        (this is gotten from playing, the atari manual didn't document the reward function)
    We modify the reward to be:
        tiers are defined in terms of the agent's y position (which car lane it's in)
        the final tier is at the maximum y_position, because we want to encourage the agent to get
        there instead of infinitely sicking around the the goal position and not getting there
    """
    y_max = 177  # after this, the agent is transitioned back to y_min
    y_min = 6

    @cached_property
    def y_block_per_tier(self):
        return (self.y_max - self.y_min) / (self.num_tiers-1)

    def _get_tier(self, y_pos):
        if y_pos >= self.y_max:
            tier = self.num_tiers-1
        else:
            tier = int((y_pos - self.y_min) / self.y_block_per_tier)
            try:
                assert 0 <= tier < self.num_tiers-1
            except AssertionError:
                tier = self.num_tiers-2  # in case of numerical error
        return tier

    def reward(self, reward, info):
        info['original_reward'] = float(reward)

        if self.keep_original_reward:
            return reward

        tier = self._get_tier(int(info['labels']['player_y']))
        return self._get_tier_reward(tier)

    def log_tier_hitting_count(self, info):
        tier = self._get_tier(int(info['labels']['player_y']))
        self.tiers_hitting_count[tier] += 1
        info['tiers_hitting_count'] = self.tiers_hitting_count


class PongTierReward(TierRewardWrapper):
    """
    in Pong, you win when you get to 21 points first

    The original reward is:
        -1 for every ball you lose, +1 for every ball you win
    We modify the reward to be:
        tiers are defined in terms of the agent's score
    """

    def _get_tier(self, score):
        return score

    def reward(self, reward, info):
        info['original_reward'] = float(reward)

        if self.keep_original_reward:
            return reward

        tier = self._get_tier(int(info['labels']['player_score']))
        return self._get_tier_reward(tier)

    def log_tier_hitting_count(self, info):
        tier = self._get_tier(int(info['labels']['player_score']))
        self.tiers_hitting_count[tier] += 1
        info['tiers_hitting_count'] = self.tiers_hitting_count


class BoxingTierReward(PongTierReward):
    """
    in Boxing, you want to get as many points as possible in 2 minutes

    the original reward is:
        -1 for every hit you take, and +1 for every hit you make
    we modify the reward to be:
        tiers are defined purely in terms of the agent's score
    """
    points_per_tier = 4

    def _get_tier(self, score):
        if score == 0:
            return 0
        tier = score // self.points_per_tier + 1
        if tier > self.num_tiers - 1:
            tier = self.num_tiers - 1
        return tier


class AsterixTierReward(TierRewardWrapper):
    """
    The original reward is:
        OBJECT (for Asterix)	POINTS
        nothing     0
        Cauldron:	50
        Helmet:	100
        Shield:	200
        Lamp:	300
        (through experimentatio there seems to be reward other than these)
    we modify the reward to be:
        tiers are defined exactly as the 5 tiers as in the original reward
        but the reward value changes
    """

    def _get_tier(self, reward):
        if reward <= 0:
            return 0
        elif reward <= 50:
            return 1
        elif reward <= 100:
            return 2
        elif reward <= 200:
            return 3
        else:
            return 4

    def reward(self, reward, info):
        info['original_reward'] = float(reward)

        if self.keep_original_reward:
            return reward

        tier = self._get_tier(reward)
        return self._get_tier_reward(tier)

    def log_tier_hitting_count(self, info):
        tier = self._get_tier(info['original_reward'])
        self.tiers_hitting_count[tier] += 1
        info['tiers_hitting_count'] = self.tiers_hitting_count


class BattleZoneTierReward(TierRewardWrapper):
    """
    the original reward seems to be:
        +1 when you hit an enemy, and 0 otherwise
        The scoring system, on the other hand, is:
            TARGET          POINTS
            Tank            1,000
            Fighter         2,000
            Supertank       3,000
            Saucer          5,000
            Bonus Tank      At 50,000 and 100,000 points
    we modify the reward to be:
        tiers are dependent on the cumulative enemies you hit (aka the current score)
    """
    max_num_hit = 200

    @cached_property
    def _num_hit_per_tier(self):
        return self.max_num_hit / self.num_tiers

    def _get_tier(self, score):
        if score == 0:
            return 0
        # each basic tank you hit gets score 1000, other enemies give more score
        num_hit = int(score / 1000)
        tier = int(num_hit / self._num_hit_per_tier) + 1
        if tier > self.num_tiers-1:
            tier = self.num_tiers-1
        return tier

    def reward(self, reward, info):
        info['original_reward'] = float(reward)

        if self.keep_original_reward:
            return reward

        tier = self._get_tier(int(info['labels']['score']))
        return self._get_tier_reward(tier)

    def log_tier_hitting_count(self, info):
        tier = self._get_tier(int(info['labels']['score']))
        self.tiers_hitting_count[tier] += 1
        info['tiers_hitting_count'] = self.tiers_hitting_count


class CartPoleTierReward(TierRewardWrapper):

    num_tiers = 4

    def _get_tier(self, angle):
        # TODO
        return angle

    def reward(self, reward, info):
        info['original_reward'] = float(reward)

        if self.keep_original_reward:
            return reward

        tier = self._get_tier(int(info['labels']['score']))
        return self._get_tier_reward(tier)

    def log_tier_hitting_count(self, info):
        tier = self._get_tier(int(info['labels']['score']))
        self.tiers_hitting_count[tier] += 1
        info['tiers_hitting_count'] = self.tiers_hitting_count


def wrap_tier_rewards(env, num_tiers, gamma, delta, keep_original_reward=False, normalize_reward=False):
    env_id = (env.spec.id).lower()
    if 'breakout' in env_id:
        env = BreakoutTierReward(env, num_tiers=num_tiers, gamma=gamma,
                                 delta=delta, keep_original_reward=keep_original_reward)

    elif 'freeway' in env_id:
        env = FreewayTierReward(env, num_tiers=num_tiers, gamma=gamma,
                                delta=delta, keep_original_reward=keep_original_reward)

    elif 'pong' in env_id:
        try:
            assert num_tiers == 22
        except AssertionError:
            num_tiers = 22
            print(
                f'Warning: Pong has 22 tiers, but you specified {num_tiers} tiers. MODIFYING IT TO BE 22 TIERS.')
        env = PongTierReward(env, num_tiers=num_tiers, gamma=gamma,
                             delta=delta, keep_original_reward=keep_original_reward)

    elif 'boxing' in env_id:
        try:
            assert num_tiers == 15
        except AssertionError:
            num_tiers = 15
            print(
                f'Warning: Boxing has 15 tiers, but you specified {num_tiers} tiers. MODIFYING IT TO BE 15 TIERS.')
        env = BoxingTierReward(env, num_tiers=num_tiers, gamma=gamma,
                               delta=delta, keep_original_reward=keep_original_reward)

    elif 'asterix' in env_id:
        try:
            assert num_tiers == 5
        except AssertionError:
            num_tiers = 5
            print(
                f'Warning: Asterix has 5 tiers, but you specified {num_tiers} tiers. MODIFYING IT TO BE 5 TIERS.')
        env = AsterixTierReward(env, num_tiers=num_tiers, gamma=gamma,
                                delta=delta, keep_original_reward=keep_original_reward)

    elif 'battlezone' in env_id:
        env = BattleZoneTierReward(env, num_tiers=num_tiers, gamma=gamma,
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
