import math
from abc import abstractmethod
from functools import cached_property

import numpy as np
from PIL import Image
import gymnasium as gym
from gymnasium.core import Wrapper, ObservationWrapper
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, ReseedWrapper, FullyObsWrapper

from reward.atari.reward_wrappers import get_k_tiered_reward
import pandas as pd

import threading
from collections import deque

global lock
lock = threading.Lock()

global num_goal_reaches
num_goal_reaches = pd.DataFrame(columns=['reached_goal'])


global df
df = pd.DataFrame(columns=['door_open', 'has_key','dist_to_key','dist_to_door','dist_to_goal','player_pos','key_pos','door_pos','goal_pos','tier', 'original_reward'])
#df = pd.DataFrame(columns=['player_pos','goal_pos','tier','original_reward', 'dist_to_goal'])
#df = pd.DataFrame(columns=['player_pos', 'goal_pos','tier', 'original_reward','dist_to_goal','max_dist'])

#python3 -m reward.minigrid.train --algo ppo --env MiniGrid-FourRooms-v0 -e MiniGrid-Multi-FourRooms --save-interval 10 --log-interval 1 --frames 200000 --reward-function tier --num-tiers 3 --seed 0 -n --gamma 0.5


def write_to_file(row_data):

    with lock:
        df.loc[len(df.index)] = row_data
        df.to_csv('./per_step_logging.csv')

def increment_goal_reaches():
    global num_goal_reaches
    with lock:
        num_goal_reaches.loc[len(num_goal_reaches.index)] = [True]


def get_num_goal_reaches():
    print(num_goal_reaches)
    return len(num_goal_reaches)

class MinigridInfoWrapper(Wrapper):
    """Include extra information in the info dict for debugging/visualizations."""

    def __init__(self, env):
        super().__init__(env)
        self._timestep = 0

        # Store the test-time start state when the environment is constructed
        self.official_start_obs, self.official_start_info = self.reset()
        #self.num_times_reach_goal = 0

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


class RewardWrapper(Wrapper):
    """
    change the reward, and log the original reward in a dictionary.
    
    NOTE: an environment should NOT use two wrappers that inherit this class.
    Because that will mess with logging info['original_reward']
    """
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        info['original_reward'] = reward
        r = self._modify_reward(reward, info)
        return obs, r, terminated, truncated, info
    
    @abstractmethod
    def _modify_reward(self, reward, info):
        raise NotImplementedError


class SparseRewardWrapper(RewardWrapper):
    """Return a reward of 1 when you reach the goal and 0 otherwise."""
    def initialize_num_times_reach_goal(self):
        self.num_times_reach_goal  = 0
    def _modify_reward(self, reward, info):
        #self.num_times_reach_goal += 1 if reward > 0 else 0
        if reward > 0:
            increment_goal_reaches()
            print('at goal')
        return float(reward > 0)


class StepPenaltyRewardWrapper(RewardWrapper):
    """Return a reward of -1 for each step, and 0 for you reach the goal."""
    def _modify_reward(self, reward, info):
        if reward > 0:
            increment_goal_reaches()
            print('at goal')
        return 0 if reward > 0 else -1


class NormalizeRewardWrapper(Wrapper):
    """
    Normalize rewrd so that its absoluate value is between [0, 1]
    For TieredReward that is designed to be negative, this wrapper should normalize
    reward values to be between [-1, 0]
    """
    def __init__(self, env, max_r_value):
        super().__init__(env)
        self.max_r_value = max_r_value

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        r = self._modify_reward(reward, info)
        return obs, r, terminated, truncated, info
    
    def _modify_reward(self, reward, info):
        new_r = reward / self.max_r_value
        assert -1 <= new_r <= 1, new_r
        return new_r


class TierRewardWrapper(Wrapper):
    """
    modify the reward() function to make the reward tiered
    
    Each class that follows this should have a _get_tier() method that determines
    what tier the agent is in.
    Then, it should implement the _modify_reward() method to return the tiered reward.
    """
    def __init__(self, env, num_tiers, gamma, delta):
        super().__init__(env)
        self.num_tiers = num_tiers
        self.gamma = gamma

        self.h = 1 / (1-gamma)
        self.delta = delta  # good tier r = H * prev tier + delta 
        self.tiers_hitting_count = np.zeros(num_tiers, dtype=np.int32)
        self.agent_start_pos = None

    def reset_hitting_count(self):
        """not used"""
        self.tiers_hitting_count = np.zeros(self.num_tiers, dtype=np.int32)

    def reset(self, reset_count=False, **kwargs):
        if reset_count:
            self.reset_hitting_count()
        
        out = self.env.reset(**kwargs)
        self.agent_start_pos = self.env.agent_pos
        self.auxilliary_reset()
        return out
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['original_reward'] = reward
        self.log_tier_hitting_count(info)
        r = self._modify_reward(reward, info)
        return obs, r, terminated, truncated, info
    
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
    
    @abstractmethod
    def auxilliary_reset(self):
        return None

    @abstractmethod
    def log_tier_hitting_count(self, info):
        """not used"""
        raise NotImplementedError


class TierBasedShapingReward(Wrapper):
    """
    Do reward shaping with a tiered reward as the shaping function
    
    NOTE: it must be wrapped around some tier reward
    NOTE: this assumes that we are shaping a step_penalty reward
    """
    def __init__(self, env: TierRewardWrapper, gamma):
        assert isinstance(env, TierRewardWrapper), env
        super().__init__(env)
        self.gamma = gamma
        
        # keep track of current obs so that we can shape reward
        self.current_obs = None
        self.current_tier = None
    
    def reset(self, **kwargs):
        self.current_obs, info = self.env.reset(**kwargs)
        self.current_tier = self.env._get_tier(info)
        return self.current_obs, info
    
    def step(self, action):
        """assumes that we are shaping a step_penalty reward"""
        next_obs, tier_r, terminated, truncated, info = self.env.step(action)
        assert 'original_reward' in info
        
        ns_tier = self.env._get_tier(info)
        step_penalty_r = 0 if info['original_reward'] > 0 else -1
        shaped_reward = step_penalty_r + self.gamma * self.env._get_tier_reward(ns_tier) - self.env._get_tier_reward(self.current_tier)
        
        self.current_obs = next_obs
        self.current_tier = ns_tier

        
        return next_obs, shaped_reward, terminated, truncated, info


class DoorKeyMiniGridTierReward(TierRewardWrapper):
    """
    Tier Reward for MiniGrid-DoorKey-nxn-v0
    
    Tiers are assigned based on teht agent's L1 distance to the goal
    
    The goal is always its own tier -- the highest tier.
    The rest of the tiers are spread out evenly according to the L1 distance.
    
    NOTE: this assumes there is only a single goal state
    """
    prev_has_key = False
    prev_door_open = False

    @cached_property
    def goal_pos(self):
        return determine_goal_pos(self.env)
    
    
    def key_pos(self):
        return determine_key_pos(self.env)
    
    @cached_property
    def door_pos(self):
        return determine_door_pos(self.env)
    
    
    def check_door_open(self):
        return determine_is_door_open(self.env)

    def check_has_key(self):
        return determine_has_key(self.env)


    @cached_property
    def max_dist(self):
        # two corner margin & -1 each side for the distance
        return self.env.grid.width + self.env.grid.height - 6
    
    

    def temp(self, info):
        
        def get_l1_distance(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        dist_to_goal = get_l1_distance(info['player_pos'], self.goal_pos)
        
        if dist_to_goal == 0:
            return self.num_tiers - 1
        else:
            return math.floor((self.num_tiers - 1) * (1 - dist_to_goal / self.max_dist))

    def _get_tier(self, info):
        def get_l1_distance(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def round_to_half(n):
            return round(n * 2) / 2
        
        key_pos = self.key_pos()
         
        dist_to_goal = get_l1_distance(info['player_pos'], self.goal_pos)
        dist_to_door = get_l1_distance(info['player_pos'], self.door_pos)
        dist_to_key = get_l1_distance(info['player_pos'], key_pos) if key_pos is not None else 0

        door_open = self.check_door_open()
        has_key = self.check_has_key()
    
        #print(has_key, self.prev_has_key, key_pos)    
        
        #tier 6: reach final goal
        #exactly T reward
        if dist_to_goal == 0:
            out  =  (self.num_tiers-1)
            #self.num_times_reach_goal += 1
            
            #increment_goal_reaches()
            #print(len(num_goal_reaches))
            print('at goal')
            
        #tier 5: walking to the goal after opening door
        #between 2T/3 < t < T
        elif door_open and has_key and self.prev_door_open:
            out =  math.floor((2*(self.num_tiers-1))/3 + ((self.num_tiers-1)/3)*(1-dist_to_goal/self.max_dist))

        #tier 3: obtain the key and walk to the door
        #between T/3 < t < 2T/3
        elif door_open and not self.prev_door_open:
            out = (2*(self.num_tiers-1))//3
        elif (has_key and not door_open and self.prev_has_key):
            out =  math.floor(((self.num_tiers-1)/3) + ((self.num_tiers-1)/3)*(1-dist_to_door/self.max_dist))
            
        #tier 1: searching for key
        #between 0 < t < T/3
        elif has_key and not self.prev_has_key:
            out =  (self.num_tiers-1)//3
        elif not has_key:
            out =  math.floor(((self.num_tiers-1)/3)*(1-dist_to_key/self.max_dist))
            
        else:
            #True False False False
            print(has_key, door_open, self.prev_has_key, self.prev_door_open)
            raise NotImplemented
        #'door_open', 'has_key','dist_to_goal','dist_to_key','dist_to_goal','tier'
        #df.loc[len(df.index)] = [door_open, has_key, dist_to_key,dist_to_door, dist_to_goal, info['player_pos'], self.key_pos(), self.door_pos, self.goal_pos, out]
        #df.to_csv('./temp_doorkey_tiers.csv')
        
        #write_to_file([door_open, has_key, dist_to_key, dist_to_door, dist_to_goal, info['player_pos'], self.key_pos(), self.door_pos, self.goal_pos, out, info['original_reward']])

        self.prev_has_key = has_key
        self.prev_door_open = door_open
        #print('NUM GOAL: ', self.num_times_reach_goal)
        return out
    def _modify_reward(self, reward, info):
        tier = self._get_tier(info)
        return self._get_tier_reward(tier)

    def log_tier_hitting_count(self, info):
        pass



class FourRoomsMiniGridTierReward(TierRewardWrapper):
    """
    Tier Reward for MiniGrid-FourRooms-v0
    
    Tiers are assigned based on teht agent's L1 distance to the goal
    
    The goal is always its own tier -- the highest tier.
    The rest of the tiers are spread out evenly according to the L1 distance.
    
    NOTE: this assumes there is only a single goal state
    """
    @cached_property
    def goal_pos(self):
        return determine_goal_pos(self.env)
    
    @cached_property
    def max_dist(self):
        # two corner margin & -1 each side for the distance
        #print(self.env.grid.width, self.env.grid.height)
        #print(self.env.grid.width + self.env.grid.height - 6)
        return self.env.grid.width + self.env.grid.height - 6
    
    def auxilliary_reset(self):
        
        from minigrid.core.world_object import Wall
        self.dist_to_goal = np.ones((self.env.grid.width, self.env.grid.height))
        self.dist_to_goal *= np.inf

        unvisited_locs = deque()
        unvisited_locs.append((self.goal_pos,0))

        while len(unvisited_locs) > 0:
            (curr_loc, dist) = unvisited_locs.popleft()
             
            if  dist < self.dist_to_goal[curr_loc[0]][curr_loc[1]] and not isinstance(self.env.grid.get(curr_loc[0], curr_loc[1]), Wall):

                self.dist_to_goal[curr_loc[0]][curr_loc[1]] = dist
                
                if curr_loc[0]+1 < self.dist_to_goal.shape[0]:
                    unvisited_locs.append( ( ( curr_loc[0]+1 , curr_loc[1]), dist+1 ) )
                if curr_loc[0]-1 >= 0:
                    unvisited_locs.append( ( ( curr_loc[0]-1, curr_loc[1]), dist+1 ) )
                if curr_loc[1] + 1 < self.dist_to_goal.shape[1]:
                    unvisited_locs.append( ( (curr_loc[0], curr_loc[1]+1), dist+1 ) )
                if curr_loc[1] - 1 >= 0:
                    unvisited_locs.append( ( (curr_loc[0], curr_loc[1]-1), dist+1 ) )

        self.dist_to_goal[np.where(self.dist_to_goal==np.inf)] = -1


    def _get_tier(self, info):
        def get_l1_distance(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        dist_goal = get_l1_distance(info['player_pos'], self.goal_pos) 
        #(player_c, player_r) = info['player_pos']
        

        #if self.dist_to_goal[player_c][player_r]== 0:
        #    out = self.num_tiers - 1
            
        #    increment_goal_reaches()
        #    print('goal')

        #    #self.num_times_reach_goal += 1
        #else:
        #    out = math.floor((self.num_tiers-1) *  ( 1 -  (self.dist_to_goal[player_c][player_r]/np.max(self.dist_to_goal)) )  )    
        
        if dist_goal == 0:
            print('goal')
            out = self.num_tiers-1
        else:
           out = math.floor((self.num_tiers-1) * (1 - (dist_goal/self.max_dist)))
        
        #'player_pos','goal_pos','tier', 'dist_to_goal'
        #df.loc[len(df.index)] = [info['player_pos'], self.goal_pos, out, dist_goal]
        #df.to_csv('./new_fourrooms_tiers.csv')
        #write_to_file([info['player_pos'], self.goal_pos, out, info['original_reward'], self.dist_to_goal[player_c][player_r], np.max(self.dist_to_goal)])
        
        #if out >= 2.0 and dist_goal > 0:

        return out
    def _modify_reward(self, reward, info):
        tier = self._get_tier(info)
        return self._get_tier_reward(tier)

    def log_tier_hitting_count(self, info):
        pass


class EmptyMiniGridTierReward(TierRewardWrapper):
    """
    Tier Reward for MiniGrid-Empty-nxn-v0
    
    Tiers are assigned based on teht agent's L1 distance to the goal
    
    The goal is always its own tier -- the highest tier.
    The rest of the tiers are spread out evenly according to the L1 distance.
    
    NOTE: this assumes there is only a single goal state
    """
    @cached_property
    def goal_pos(self):
        return determine_goal_pos(self.env)
    
    @cached_property
    def max_dist(self):
        # two corner margin & -1 each side for the distance
        return self.env.grid.width + self.env.grid.height - 6
    
    def _get_tier(self, info):
        def get_l1_distance(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        dist = get_l1_distance(info['player_pos'], self.goal_pos)
        if dist == 0:
            out = self.num_tiers - 1
            #self.num_times_reach_goal += 1
            #increment_goal_reaches()
            print('goal')
        else:
            out = math.floor((self.num_tiers - 1) * (1 - dist / self.max_dist))
        
        #write_to_file([info['player_pos'], self.goal_pos, out, info['original_reward'], dist])
        return out

    def _modify_reward(self, reward, info):
        tier = self._get_tier(info)
        return self._get_tier_reward(tier)

    def log_tier_hitting_count(self, info):
        pass


class GrayscaleWrapper(ObservationWrapper):
    def observation(self, observation):
        observation = observation.mean(axis=0)[np.newaxis, :, :]
        return observation.astype(np.uint8)


def determine_goal_pos(env):
    """Convinence hacky function to determine the goal location."""
    from minigrid.core.world_object import Goal
    for i in range(env.grid.width):
        for j in range(env.grid.height):
            tile = env.grid.get(i, j)
            if isinstance(tile, Goal):
                return i, j

def print_grid_display(env):

    from minigrid.core.world_object import Goal, Wall

    grid = np.empty((env.grid.width, env.grid.height))

    for i in range(env.grid.width):
        for j in range(env.grid.height):
            tile = env.grid.get(i,j)

            if isinstance(tile, Goal):
                grid[i][j] = 1
            elif isinstance(tile, Wall):
                grid[i][j] = 2
            else:
                grid[i][j] = 0

    for r in range(env.grid.width):
        print(grid[r])

def determine_is_door_open(env):
    """Convinence hacky function to determine the goal location."""
    from minigrid.core.world_object import Door
    for i in range(env.grid.width):
        for j in range(env.grid.height):
            tile = env.grid.get(i, j)
            if isinstance(tile, Door):
                return tile.is_open



def determine_door_pos(env):
    """Convenient hacky function to determine the door location"""
    from minigrid.core.world_object import Door

    for i in range(env.grid.width):
        for j in range(env.grid.height):
            tile = env.grid.get(i,j)
            if isinstance(tile, Door):
                return i,j

def determine_has_key(env):
    """Convenient hacky function to determine if agent has grabbed key"""
    from minigrid.core.world_object import Key

    return isinstance(env.carrying, Key)

def determine_key_pos(env):
    """Convenience hacky function to determine the key location. """
    from minigrid.core.world_object import Key
    for i in range(env.grid.width):
        for j in range(env.grid.height):
            tile = env.grid.get(i,j)
            if isinstance(tile, Key):
                return i, j

def determine_start_pos(env):
    """Convenient function to determine the start position of the agent"""
    None
def environment_builder(
    level_name='MiniGrid-Empty-8x8-v0',
    seed=42,
    gamma=0.99,
    delta=0.1,
    num_tiers=5,
    use_img_obs=True,
    reward_fn='original',
    normalize_reward=False,
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

    # grayscale
    if grayscale:
        env = GrayscaleWrapper(env)

    # more information in the info dict
    env = MinigridInfoWrapper(env)

    # different reward functions
    if reward_fn == 'sparse':
        env = SparseRewardWrapper(env)
    elif reward_fn == 'step_penalty':
        env = StepPenaltyRewardWrapper(env)
    
    elif reward_fn in ('tier', 'tier_based_shaping'):
        if 'empty' in level_name.lower():
            env = EmptyMiniGridTierReward(env, num_tiers=num_tiers, gamma=gamma, delta=delta)
        
        elif 'door' in level_name.lower():
            env = DoorKeyMiniGridTierReward(env, num_tiers=num_tiers, gamma=gamma, delta=delta)
        
        elif 'four' in level_name.lower():
            env = FourRoomsMiniGridTierReward(env, num_tiers=num_tiers, gamma=gamma, delta=delta)
            #env.store_start_pos()
        else:
            raise NotImplementedError('This environment does not yet support tiered rewards')
        
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
