import numpy as np
from pathos.pools import ProcessPool
from msdm.algorithms.tdlearning import TDLearningEventListener, epsilon_softmax_sample
from msdm.algorithms import QLearning as MSDMQLearning
from msdm.core.algorithmclasses import Result


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
    

class QLearning(MSDMQLearning):
    """
    basically the same as msdm.algorithms.QLearning
    except that:
        1. this agent trains for a fixed nubmer of steps instead of episodes
        2. fully greedy, and learning rate of 1
        3. optimistically initialize the Q values to be be a very large number
    """
    def __init__(
        self, 
        num_steps: int,
        rand_choose: float = 0,
        initial_q: float = 1e10,
        seed: int = 0,
    ):
        super().__init__(
            episodes=None, 
            step_size=1, 
            rand_choose=rand_choose, 
            initial_q=initial_q,
            seed=seed, 
        )
        self.num_steps = num_steps
        self.event_listeners = [
            EpisodicReward(),
            TimeAtGoal(),
            Seed(),
            NumGoalsHit(),
        ]

    # def _check_q_convergence(self, mdp, q):
    #     """
    #     iterate through all (s, a) and check if they satisfy the Bellman optimality equation
    #     """
    #     states = mdp.state_list()
    #     actions = mdp.action_list()
    #     for s in states:
    #         for a in actions:
    #             next_states = mdp.next_state_dist(s, a).support
    #             for ns in next_states:
    #                 r = mdp.reward(s, a, ns)
    
    def _training(self, mdp, rng, event_listeners):
        q = self._initial_q_table(mdp)
        # initial state
        s = mdp.initial_state_dist().sample(rng=rng)
        for i_step in range(self.num_steps):
            # select action
            a = epsilon_softmax_sample(q[s], self.rand_choose, self.softmax_temp, rng)
            # transition to next state
            ns = mdp.next_state_dist(s, a).sample(rng=rng)
            r = mdp.reward(s, a, ns)
            # update
            q[s][a] += self.step_size*(r + mdp.discount_rate*max(q[ns].values()) - q[s][a])
            # end of timestep
            for listener in event_listeners:
                listener.end_of_timestep(locals())
            s = ns
            # end of episode
            if mdp.is_terminal(s):
                for listener in event_listeners:
                    listener.end_of_episode(locals())
                s = mdp.initial_state_dist().sample(rng=rng)
        return q

    def train_on(self, mdp):
        rng = self._init_random_number_generator()
        q = self._training(mdp, rng, self.event_listeners)
        listener_results = {
            listener.__class__.__name__: listener.results() for listener in self.event_listeners
        }
        return Result(
            q_values=q,
            policy=self._create_policy(mdp, q),
            **listener_results,
        )



def run_q_learning(env, agent):
    """
    returns:
        results: {q_values: ..., policy: ..., event_listener_results: ...}
    """
    res = agent.train_on(env)
    return res


def run_multiprocessing_q_learning(env, rand_choose, initial_q, num_seeds=10, num_learning_steps=2000):
    """
    speed up run_q_learning with multiprocessing
    each process runs a different seed
    """
    seeds = list(range(num_seeds))

    # make agents
    agents = [
        QLearning(
            num_steps=num_learning_steps,
            rand_choose=rand_choose,
            initial_q=initial_q,
            seed=seeds[i],
        )
        for i in range(num_seeds)
    ]

    pool = ProcessPool(nodes=num_seeds)
    results = pool.amap(
        run_q_learning,
        [env]*num_seeds,
        [agents[i] for i in range(num_seeds)],
    )
    while not results.ready():
        pass
    results = results.get()

    return results
