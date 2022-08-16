import numpy as np
from msdm.algorithms.tdlearning import TDLearningEventListener


class EpisodeLength(TDLearningEventListener):
    """record the length of each episode"""
    def __init__(self):
        self.current_episode = 0
        self.cumulative_steps = 0
        self.episode_lengths = {}
    def end_of_timestep(self, local_vars):
        pass
    def end_of_episode(self, local_vars):
        if self.current_episode == 0:
            self.episode_lengths[self.current_episode] = local_vars['i_step']
        else:
            self.episode_lengths[self.current_episode] = local_vars['i_step'] - self.cumulative_steps
        self.cumulative_steps += self.episode_lengths[self.current_episode]
        self.current_episode += 1
    def results(self):
        return self.episode_lengths


class TimeAtGoal(TDLearningEventListener):
    """which timestep did the agent get to the terminal state"""
    def __init__(self):
        self.time_at_goal = np.inf
    def end_of_timestep(self, local_vars):
        pass
    def end_of_episode(self, local_vars):
        s = local_vars['s']
        if local_vars['mdp'].is_terminal(s) and local_vars['i_step'] < self.time_at_goal:
            # make sure the terminal is a goal, because it can also be a obstacle
            if local_vars['mdp'].location_features.get(s, '') == 'g':
                self.time_at_goal = local_vars['i_step']
    def results(self):
        return self.time_at_goal


class NumGoalsHit(TDLearningEventListener):
    """how many times did the agent hit the goal in total"""
    def __init__(self):
        self.num_goals_hit = 0
    def end_of_timestep(self, local_vars):
        pass
    def end_of_episode(self, local_vars):
        self.num_goals_hit += 1
    def results(self):
        return self.num_goals_hit


class EpisodicReward(TDLearningEventListener):
    """record the episodic reward from one training run"""
    def __init__(self):
        self.episode_rewards = {}
        self.curr_ep_rewards = 0
        self.log_freq = 100
    def end_of_timestep(self, local_vars):
        self.curr_ep_rewards += local_vars['r']
        if local_vars['i_step'] % self.log_freq == 0:
            self.episode_rewards[local_vars['i_step']] = self.curr_ep_rewards
    def end_of_episode(self, local_vars):
        self.curr_ep_rewards = 0
    def results(self):
        return self.episode_rewards


class Seed(TDLearningEventListener):
    """record the seed of the environment"""
    def __init__(self):
        self.seed = None
    def end_of_timestep(self, local_vars):
        pass
    def end_of_episode(self, local_vars):
        self.seed = local_vars['self'].seed
    def results(self):
        return self.seed