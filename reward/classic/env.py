import functools

import gym
import pfrl
# from atariari.benchmark.wrapper import AtariARIWrapper
from pfrl.wrappers import atari_wrappers

from reward.classic.reward_wrappers import wrap_tier_rewards
from reward.classic.vec_env import VectorFrameStack, MultiprocessVectorEnv


def make_env(env_id, gamma, delta, seed, num_tiers=15, original_reward=False, normalize_reward=False, test=False):
    # Use different random seeds for train and test envs
    env_seed = int(2**32 - 1 - seed if test else seed)

    env = gym.make(env_id)
    # is_atari = hasattr(gym.envs, 'atari') and isinstance(
    #     env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
    # assert is_atari
    # try:
    #     env = AtariARIWrapper(env)
    # except AssertionError:
    #     # env is not supported by AtariARIWrapper
    #     pass

    env = wrap_tier_rewards(env, num_tiers=num_tiers, gamma=gamma, delta=delta, keep_original_reward=original_reward, normalize_reward=not test and not original_reward and normalize_reward)

    # if test:
    #     # Randomize actions like epsilon-greedy in evaluation as well
    #     env = pfrl.wrappers.RandomizeAction(env, args.eval_epsilon)
    env.seed(env_seed)

    return env


def make_batch_env(env_id, gamma, delta, num_envs, frame_stack_size, seeds, num_tiers, original_reward, normalize_reward, test):
    if original_reward:
        print('making environment with original reward function')
    assert len(seeds) == num_envs
    vec_env = MultiprocessVectorEnv(
        [
            functools.partial(make_env, env_id, gamma, delta, seeds[idx], num_tiers, original_reward, normalize_reward, test)
            for idx, env in enumerate(range(num_envs))
        ]
    )
    vec_env = VectorFrameStack(vec_env, frame_stack_size)
    return vec_env
