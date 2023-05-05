import math
import pickle
import numpy as np
from PIL import Image
import gymnasium as gym
from gymnasium.core import Wrapper, ObservationWrapper
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, ReseedWrapper, FullyObsWrapper


class MinigridInfoWrapper(Wrapper):
    """Include extra information in the info dict for debugging/visualizations."""

    def __init__(self, env):
        super().__init__(env)
        self._timestep = 0

        # Store the test-time start state when the environment is constructed
        self.official_start_obs, self.official_start_info = self.reset()

    def reset(self):
        obs, info = self.env.reset()
        info = self._modify_info_dict(info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._timestep += 1
        info = self._modify_info_dict(info, terminated, truncated)
        return obs, reward, terminated, truncated, info

    def _modify_info_dict(self, info, terminated=False, truncated=False):
        info['player_pos'] = tuple(self.env.agent_pos)
        info['player_x'] = self.env.agent_pos[0]
        info['player_y'] = self.env.agent_pos[1]
        info['truncated'] = truncated
        info['terminated'] = terminated
        info['needs_reset'] = truncated  # pfrl needs this flag
        info['timestep'] = self._timestep  # total number of timesteps in env
        info['has_key'] = self.env.unwrapped.carrying is not None
        if info['has_key']:
            assert self.unwrapped.carrying.type == 'key', self.env.unwrapped.carrying
        info['door_open'] = determine_is_door_open(self)
        return info


class ResizeObsWrapper(ObservationWrapper):
    """Resize the observation image to be (84, 84) and compatible with Atari."""

    def observation(self, observation):
        img = Image.fromarray(observation)
        return np.asarray(img.resize((84, 84), Image.BILINEAR))


class TransposeObsWrapper(ObservationWrapper):
    def observation(self, observation):
        assert len(observation.shape) == 3, observation.shape
        assert observation.shape[-1] == 3, observation.shape
        return observation.transpose((2, 0, 1))


class SparseRewardWrapper(Wrapper):
    """Return a reward of 1 when you reach the goal and 0 otherwise."""

    def step(self, action):
        # minigrid discounts the reward with a step count - undo that here
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, float(reward > 0), terminated, truncated, info


class StepPenaltyRewardWrapper(Wrapper):
    """Return a reward of -1 for each step, and 0 for you reach the goal."""
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        penalty_reward = 0 if reward > 0 else -1
        return obs, penalty_reward, terminated, truncated, info


class GrayscaleWrapper(ObservationWrapper):
    def observation(self, observation):
        observation = observation.mean(axis=0)[np.newaxis, :, :]
        return observation.astype(np.uint8)


class RandomStartWrapper(Wrapper):
    def __init__(self, env, start_loc_file='start_locations_4rooms.pkl'):
        """Randomly samples the starting location for the agent. We have to use the
        ReseedWrapper() because otherwiswe the layout can change between episodes.
        But when we use that wrapper, it also makes random init selection impossible.
        As a hack, I stored some randomly generated (non-collision) locations to a
        file and that is the one we load here.
        """
        super().__init__(env)
        self.n_episodes = 0
        self.start_locations = pickle.load(open(start_loc_file, 'rb'))

        # TODO(ab): This assumes that the 2nd-to-last action is unused in the env
        # Not using the last action because that terminates the episode!
        self.no_op_action = env.action_space.n - 2

    def reset(self):
        super().reset()
        rand_pos = self.start_locations[self.n_episodes % len(
            self.start_locations)]
        self.n_episodes += 1
        return self.reset_to(rand_pos)

    def reset_to(self, rand_pos):
        new_pos = self.env.place_agent(
            top=rand_pos,
            size=(3, 3)
        )

        # Apply the no-op to get the observation image
        obs, _, _, info = self.env.step(self.no_op_action)

        info['player_x'] = new_pos[0]
        info['player_y'] = new_pos[1]
        info['player_pos'] = new_pos

        return obs, info


def determine_goal_pos(env):
    """Convinence hacky function to determine the goal location."""
    from minigrid.core.world_object import Goal
    for i in range(env.grid.width):
        for j in range(env.grid.height):
            tile = env.grid.get(i, j)
            if isinstance(tile, Goal):
                return i, j


def determine_is_door_open(env):
    """Convinence hacky function to determine the goal location."""
    from minigrid.core.world_object import Door
    for i in range(env.grid.width):
        for j in range(env.grid.height):
            tile = env.grid.get(i, j)
            if isinstance(tile, Door):
                return tile.is_open


def environment_builder(
    level_name='MiniGrid-Empty-8x8-v0',
    seed=42,
    use_img_obs=True,
    reward_fn='original',
    grayscale=True,
    max_steps=None,
    render_mode=None,
):
    if max_steps is not None and max_steps > 0:
        env = gym.make(level_name, max_steps=max_steps, render_mode=render_mode)
    else:
        env = gym.make(level_name, render_mode=render_mode)

    # make observation fully observable
    env = FullyObsWrapper(env)
    # force an environment to always keep the same configuration when reset
    env = ReseedWrapper(env, seeds=[seed])  # To fix the start-goal config
    
    # image observations
    if use_img_obs:
        env = RGBImgObsWrapper(env)  # Get pixel observations
        env = ImgObsWrapper(env)  # Get rid of the 'mission' field

    # sparse reward
    if reward_fn == 'sparse':
        env = SparseRewardWrapper(env)
    elif reward_fn == 'step_penalty':
        env = StepPenaltyRewardWrapper(env)
    else:
        assert reward_fn == 'original'

    # grayscale
    if grayscale:
        env = GrayscaleWrapper(env)

    # more information in the info dict
    env = MinigridInfoWrapper(env)

    return env
