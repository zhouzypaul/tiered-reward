import d4rl
import gym
from gym import Wrapper

from reward.minigrid.minigrid_wrappers import SparseRewardWrapper, StepPenaltyRewardWrapper, \
    TierBasedShapingReward, NormalizeRewardWrapper, get_k_tiered_reward


class TruncationWrapper(Wrapper):
    """d4rl only supports the old gym API, where env.step returns a 4-tuple without
    the truncated signal. Here we explicity expose the truncated signal."""

    def __init__(self, env):
        super().__init__(env)
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        s = self.env.reset()
        return s, {}

    def step(self, a):
        s, r, done, info = self.env.step(a)
        truncated = info.get("TimeLimit.truncated", False)
        return s, r, done, truncated, info


def environment_builder(
    level_name='antmaze-umaze-v0',
    seed=42,
    gamma=0.99,
    delta=0.1,
    num_tiers=5,
    reward_fn='original',
    normalize_reward=False,
    max_steps=None,
    render_mode=None,
    **kwargs,
):
    
    assert level_name == 'antmaze-umaze-v0', "Only antmaze-umaze-v0 is supported currently"
    
    if max_steps is not None and max_steps > 0:
        env = gym.make(level_name, max_steps=max_steps, render_mode=render_mode, seed=seed)
    else:
        env = gym.make(level_name, render_mode=render_mode, seed=seed)
        
    env = TruncationWrapper(env)

    # different reward functions
    if reward_fn == 'sparse':
        env = SparseRewardWrapper(env)
    elif reward_fn == 'step_penalty':
        env = StepPenaltyRewardWrapper(env)
    
    elif reward_fn in ('tier', 'tier_based_shaping'):
        
        # TODO:
        # wrap the environment here 
        
        if reward_fn == 'tier_based_shaping':
            env = TierBasedShapingReward(env, gamma=gamma)
        
    else:
        assert reward_fn == 'original'
        
    # normalize reward
    if normalize_reward:
        assert reward_fn in ('tier', 'tier_based_shaping')
        max_reward = get_k_tiered_reward(0, total_num_tiers=num_tiers, H=1/(1-gamma), delta=delta)
        max_abs_reward = abs(max_reward)
        env = NormalizeRewardWrapper(env, max_r_value=max_abs_reward)

    return env
