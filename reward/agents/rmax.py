import random
from collections import defaultdict

import numpy as np
from msdm.core.algorithmclasses import Learns, Result
from msdm.core.distributions import DictDistribution
from msdm.core.problemclasses.mdp import TabularMarkovDecisionProcess, TabularPolicy

from reward.agents.event_listeners import EpisodicReward, TimeAtGoal, NumGoalsHit, Seed


class RMax():
    '''
    Implementation for an R-Max Agent [Strehl, Li and Littman 2009]
    implementation from simple_rl by Dave Abel 
    '''

    def __init__(self, actions, gamma=0.95, s_a_threshold=2, epsilon_one=0.99, max_reward=1.0, custom_q_init=None):
        self.actions = list(actions) # Just in case we're given a numpy array (like from Atari).
        self.gamma = gamma
        self.epsilon_one = epsilon_one
        self.episode_number = 0
        self.prev_state = None
        self.prev_action = None

        self.rmax = max_reward
        self.s_a_threshold = s_a_threshold
        self.custom_q_init = custom_q_init 
        self.reset()

    def reset(self):
        '''
        Summary:
            Resets the agent back to its tabular asa config.
        '''
        self.rewards = defaultdict(lambda : defaultdict(list)) # S --> A --> reward
        self.transitions = defaultdict(lambda : defaultdict(lambda : defaultdict(int))) # S --> A --> S' --> counts
        self.r_s_a_counts = defaultdict(lambda : defaultdict(int)) # S --> A --> #rs
        self.t_s_a_counts = defaultdict(lambda : defaultdict(int)) # S --> A --> #ts
        self.prev_state = None
        self.prev_action = None

        if self.custom_q_init:
            self.q_func = self.custom_q_init
        else:
            self.q_func = defaultdict(lambda: defaultdict(lambda: self.rmax))

    def act(self, state, reward):
        # Update given s, a, r, s' : self.prev_state, self.prev_action, reward, state
        self.update(self.prev_state, self.prev_action, reward, state)

        # Compute best action by argmaxing over Q values of all possible s,a pairs
        action = self.get_max_q_action(state)

        # Update pointers.
        self.prev_action = action
        self.prev_state = state

        return action

    def update(self, state, action, reward, next_state):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)
        Summary:
            Updates T and R.
        '''
        if state != None and action != None:
            if self.r_s_a_counts[state][action] <= self.s_a_threshold or self.t_s_a_counts[state][action] <= self.s_a_threshold:
                # Add new data points if we haven't seen this s-a enough.
                self.rewards[state][action] += [reward]
                self.r_s_a_counts[state][action] += 1
                self.transitions[state][action][next_state] += 1
                self.t_s_a_counts[state][action] += 1

                if self.r_s_a_counts[state][action] == self.s_a_threshold:
                    # Start updating Q values for subsequent states
                    lim = int(np.log(1/(self.epsilon_one * (1 - self.gamma))) / (1 - self.gamma))
                    for i in range(1, lim):
                        for curr_state in self.rewards.keys():
                            for curr_action in self.actions:
                                if self.r_s_a_counts[curr_state][curr_action] >= self.s_a_threshold:
                                    self.q_func[curr_state][curr_action] = self._get_reward(curr_state, curr_action) + (self.gamma * self.get_transition_q_value(curr_state, curr_action))

    def get_transition_q_value(self, state, action):
        '''
        Args: 
            state
            action 
        Returns:
            empirical transition probability 
        '''
        return sum([(self._get_transition(state, action, next_state) * self.get_max_q_value(next_state)) for next_state in self.q_func.keys()])


    def get_value(self, state):
        '''
        Args:
            state (State)
        Returns:
            (float)
        '''
        return self.get_max_q_value(state)

    def _compute_max_qval_action_pair(self, state):
        '''
        Args:
            state (State)
        Returns:
            (tuple) --> (float, str): where the float is the Qval, str is the action.
        '''
        # Grab random initial action in case all equal
        best_action = random.choice(self.actions)
        max_q_val = self.get_q_value(state, best_action)

        # Find best action (action w/ current max predicted Q value) 
        for action in self.actions:
            q_s_a = self.get_q_value(state, action)
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = action
        return max_q_val, best_action

    def get_max_q_action(self, state):
        '''
        Args:
            state (State)
        Returns:
            (str): The string associated with the action with highest Q value.
        '''
        return self._compute_max_qval_action_pair(state)[1]

    def get_max_q_value(self, state):
        '''
        Args:
            state (State)
        Returns:
            (float): The Q value of the best action in this state.
        '''
        return self._compute_max_qval_action_pair(state)[0]

    def get_q_value(self, state, action):
        '''
        Args:
            state (State)
            action (str)
        
        Returns:
            (float)
        '''

        return self.q_func[state][action]

    def _get_reward(self, state, action):
        '''
        Args:
            state (State)
            action (str)
        Returns:
            Believed reward of executing @action in @state. If R(s,a) is unknown
            for this s,a pair, return self.rmax. Otherwise, return the MLE.
        '''

        if self.r_s_a_counts[state][action] >= self.s_a_threshold:
            # Compute MLE if we've seen this s,a pair enough.
            rewards_s_a = self.rewards[state][action]
            return float(sum(rewards_s_a)) / len(rewards_s_a)
        else:
            # Otherwise return rmax.
            return self.rmax
    
    def _get_transition(self, state, action, next_state):
        '''
        Args: 
            state (State)
            action (str)
            next_state (str)
            Returns:
                Empirical probability of transition n(s,a,s')/n(s,a) 
        '''

        return self.transitions[state][action][next_state] / self.t_s_a_counts[state][action]


class RMaxAgent(Learns):
    def __init__(
        self,
        num_steps : int,
        rmax : float,
        seed : int = 0,
    ):
        self.num_steps = num_steps
        self.rmax = rmax
        self.seed = seed

        self.event_listeners = [
            EpisodicReward(),
            TimeAtGoal(),
            Seed(),
            NumGoalsHit(),
        ]

    def _training(self, mdp, rng, event_listeners):
        """This is the main training loop. It should return
        a nested dictionary. Specifically, a dictionary with
        states as keys and action-value dictionaries as values."""
        # create agent
        agent = RMax(
            actions=range(len(mdp.action_list)),
            gamma=mdp.discount_rate,
            s_a_threshold=2,
            epsilon_one=0.99,
            max_reward=self.rmax,
            custom_q_init=None,
        )
        state_to_index = lambda s: mdp.state_index[s]
        index_to_action = lambda ai: {ai: a for a, ai in mdp.action_index.items()}[ai]

        # initital state
        s = mdp.initial_state_dist().sample(rng=rng)
        r = None
        for i_step in range(self.num_steps):
            # select action (including update)
            a = agent.act(state_to_index(s), r)
            # transition to next state
            ns = mdp.next_state_dist(s, index_to_action(a)).sample(rng=rng)
            r = mdp.reward(s, index_to_action(a), ns)
            # end of time step
            for listener in event_listeners:
                listener.end_of_timestep(locals())
            s = ns
            # end of episode
            if mdp.is_terminal(s):
                for listener in event_listeners:
                    listener.end_of_episode(locals())
                s = mdp.initial_state_dist().sample(rng=rng)
        
        return agent.q_func

    def _init_random_number_generator(self):
        if self.seed is not None:
            rng = random.Random(self.seed)
        else:
            rng = random
        return rng

    def _create_policy(self, mdp, q):
        policy = {}
        try:
            state_list = mdp.state_list
        except AttributeError:
            state_list = q.keys()
        for s in state_list:
            if s not in q:
                max_aa = mdp.actions(s)
            else:
                maxq = max(q[s].values())
                max_aa = [a for a in q[s].keys() if q[s][a] == maxq]
            policy[s] = DictDistribution({a: 1/len(max_aa) for a in max_aa})
        policy = TabularPolicy(policy)
        return policy

    def train_on(self, mdp: TabularMarkovDecisionProcess):
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


def run_rmax(env, agent):
    """
    returns:
        results: {q_values: ..., policy: ..., **event_listener_results}
    """
    res = agent.train_on(env)
    return res


if __name__ == "__main__":
    # testing this script
    from reward.environments import make_one_dim_chain, make_single_goal_square_grid

    gamma = 0.9
    # env = make_one_dim_chain(num_states=10, goal_reward=1, step_reward=-1, discount_rate=gamma, success_rate=0.8)
    mdp = make_single_goal_square_grid(num_side_states=4, discount_rate=gamma, success_prob=0.8, step_cost=-1, goal_reward=10)
    agent = RMax(actions=range(len(mdp.action_list)), gamma=gamma, max_reward=10)

    state_to_index = lambda s: mdp.state_index[s]
    index_to_action = lambda ai: {ai: a for a, ai in mdp.action_index.items()}[ai]

    # training loop 
    num_steps = 10000
    s = mdp.initial_state_dist().sample()
    r = None
    for i_step in range(num_steps):
        # select action
        a = agent.act(state_to_index(s), r)
        # transition to next state
        ns = mdp.next_state_dist(s, index_to_action(a)).sample()
        r = mdp.reward(s, index_to_action(a), ns)
        # update
        pass   
        s = ns
        # end of episode
        if mdp.is_terminal(s):
            s = mdp.initial_state_dist().sample()
    
    # check the policy
    s = mdp.initial_state_dist().sample()
    while True:
        a = agent.act(state_to_index(s), r)
        print(s, index_to_action(a))
        ns = mdp.next_state_dist(s, index_to_action(a)).sample()
        r = mdp.reward(s, index_to_action(a), ns)
        s = ns
        if mdp.is_terminal(s):
            break
    
    # check RMaxLearning
    learner = RMaxAgent(num_steps=num_steps, rmax=10, seed=0)
    result = learner.train_on(mdp)
    print(result)
