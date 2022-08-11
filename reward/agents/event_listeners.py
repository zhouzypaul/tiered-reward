import numpy as np
from msdm.algorithms.tdlearning import TDLearningEventListener


class TimeAtGoal(TDLearningEventListener):
    """which timestep did the agent get to the terminal state"""
    def __init__(self):
        self.time_at_goal = np.inf
    def end_of_timestep(self, local_vars):
        pass
    def end_of_episode(self, local_vars):
        if local_vars['mdp'].is_terminal(local_vars['s']) and local_vars['i_step'] < self.time_at_goal:
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
    def __init__(self):
        self.seed = None
    def end_of_timestep(self, local_vars):
        pass
    def end_of_episode(self, local_vars):
        self.seed = local_vars['self'].seed
    def results(self):
        return self.seed