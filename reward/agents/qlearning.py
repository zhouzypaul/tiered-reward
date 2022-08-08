from concurrent.futures import ProcessPoolExecutor

import numpy as np
from msdm.algorithms.tdlearning import TDLearningEventListener, epsilon_softmax_sample
from msdm.algorithms import QLearning as MSDMQLearning


class TimeAtGoalEventListener(TDLearningEventListener):
    """which timestep did the agent get to the terminal state"""
    def __init__(self):
        self.time_at_goal = np.inf
    def end_of_timestep(self, local_vars):
        if local_vars['mdp'].is_terminal(local_vars['s']) and local_vars['i_step'] < self.time_at_goal:
            self.time_at_goal = local_vars['i_step']
    def end_of_episode(self, local_vars):
        if local_vars['mdp'].is_terminal(local_vars['s']) and local_vars['i_step'] < self.time_at_goal:
            self.time_at_goal = local_vars['i_step']
    def results(self):
        return self.time_at_goal
    

class QLearning(MSDMQLearning):
    """
    basically the same as msdm.algorithms.QLearning
    except that:
        1. this agent trains for a fixed nubmer of steps instead of episodes
        2. fully greedy, and learning rate of 1
    """
    def __init__(
        self, 
        num_steps: int,
        rand_choose: float = 0.05,
        seed: int = 0,
        event_listener_class: TDLearningEventListener = TimeAtGoalEventListener,
    ):
        super().__init__(
            episodes=None, 
            step_size=1, 
            rand_choose=rand_choose, 
            initial_q=0., 
            seed=seed, 
            event_listener_class=event_listener_class,
        )
        self.num_steps = num_steps

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
    #                 # TODO
    
    def _training(self, mdp, rng, event_listener):
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
            q[s][a] += self.step_size*(r + mdp.discount_rate*max(q.get(ns, {0: 0}).values()) - q[s][a])
            # end of timestep
            event_listener.end_of_timestep(locals())
            s = ns
            # end of episode
            if mdp.is_terminal(s):
                event_listener.end_of_episode(locals())
                s = mdp.initial_state_dist().sample(rng=rng)
        return q



def run_q_learning(env, agent):
    """
    returns:
        results: {q_values: ..., policy: ..., event_listener_results: ...}
    """
    res = agent.train_on(env)
    return res


def run_multiprocessing_q_learning(env, num_seeds=10, num_learning_steps=2000):
    """
    speed up run_q_learning with multiprocessing
    each process runs a different seed
    """
    seeds = list(range(num_seeds))

    # make agents
    agents = [
        QLearning(
            num_steps=num_learning_steps,
            seed=seeds[i],
        )
        for i in range(num_seeds)
    ]

    with ProcessPoolExecutor(num_seeds) as executor:
        futures = []
        for i in range(num_seeds):
            future = executor.submit(
                run_q_learning,
                env,
                agents[i],
            )
            futures.append(future)
        
        # wait till all jobs is done
        while not all([f.done() for f in futures]):
            pass
        # check for exceptions
        for f in futures:
            if f.exception():
                raise f.exception()
        print("all qlearning jobs done")

        # collect results
        results = []
        for f in futures:
            results.append(f.result())
    
    return results 
