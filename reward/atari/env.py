import functools

import gym
import pfrl
from atariari.benchmark.wrapper import AtariARIWrapper
from pfrl.wrappers import atari_wrappers

from reward.atari.reward_wrappers import wrap_tier_rewards
from reward.atari.vec_env import VectorFrameStack, MultiprocessVectorEnv


def make_env(env_id, seed, max_frames, num_tiers=15, original_reward=False, test=False):
    # Use different random seeds for train and test envs
    env_seed = int(2**32 - 1 - seed if test else seed)

    env = atari_wrappers.make_atari(env_id, max_frames=max_frames)
    is_atari = hasattr(gym.envs, 'atari') and isinstance(
        env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
    assert is_atari
    env = AtariARIWrapper(env)

    env = wrap_tier_rewards(env, num_tiers=num_tiers, gamma=0.99, keep_original_reward=original_reward)

    env = atari_wrappers.wrap_deepmind(
        env,
        episode_life=not test,
        clip_rewards=not test,
        frame_stack=False,  # because we are doing vector frame stack
        scale=False,
        fire_reset=True,
    )

    # if test:
    #     # Randomize actions like epsilon-greedy in evaluation as well
    #     env = pfrl.wrappers.RandomizeAction(env, args.eval_epsilon)
    env.seed(env_seed)

    return env


def make_batch_env(env_id, num_envs, seeds, max_frames, num_tiers, original_reward, test):
    if original_reward:
        print('making environment with original reward function')
    assert len(seeds) == num_envs
    vec_env = MultiprocessVectorEnv(
        [
            functools.partial(make_env, env_id, seeds[idx], max_frames, num_tiers, original_reward, test)
            for idx, env in enumerate(range(num_envs))
        ]
    )
    vec_env = VectorFrameStack(vec_env, 4)
    return vec_env
