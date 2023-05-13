from pfrl import agents, explorers, replay_buffers
from pfrl.q_functions import DiscreteActionValueHead
import torch
import torch.nn as nn
import random
import numpy as np
from reward.agents.event_listeners import EpisodicReward, TimeAtGoal, NumGoalsHit, Seed, EpisodeLength
from msdm.core.algorithmclasses import Result
from tqdm import tqdm


class QFunc(nn.Module):
    def __init__(self, state_dim, hidden_dim, n_actions):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        self.linear_1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.linear_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_3 = nn.Linear(self.hidden_dim, self.n_actions)
        self.activation = nn.ReLU()
        self.action_head = DiscreteActionValueHead()
        
        if self.linear_3.bias is not None:
            nn.init.constant_(self.linear_3.bias, 1e3)

    def forward(self, state):
        x = self.activation(self.linear_1(state))
        x = self.activation(self.linear_2(x))
        x = self.linear_3(x)
        q = self.action_head(x)
        return q


class DQN:
    def __init__(
        self, 
        num_steps: int,
        learning_rate: float = 0.9,
        seed: int = 0,
        hidden_dim: int = 256,
        gpu: int = 7
    ):
        self.num_steps = num_steps
        self.event_listeners = [
            EpisodicReward(),
            TimeAtGoal(),
            Seed(),
            NumGoalsHit(),
            EpisodeLength(),
        ]
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.gpu = gpu
        self.seed = seed

    def _build_agent(self, mdp):
        state_dim = 2
        n_actions = len(mdp._actions)
        # q_func = nn.Sequential(
        #     nn.Linear(2, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     optimistic_init_chainer_default(nn.Linear(self.hidden_dim, n_actions)),
        #     DiscreteActionValueHead(),
        # )
        q_func = QFunc(state_dim, self.hidden_dim, n_actions)
        optimizer = torch.optim.Adam(q_func.parameters(), lr=self.learning_rate, eps=1e-2)
        gamma = mdp.discount_rate
        explorer = explorers.ConstantEpsilonGreedy(
            epsilon=0.3, random_action_func=lambda: random.choice(range(len(mdp._actions))))
        replay_buffer = replay_buffers.ReplayBuffer(capacity=10 ** 6)
        phi = lambda s: np.array([s['x'], s['y']], dtype=np.float32)

        self.agent = agents.DQN(
            q_func,
            optimizer,
            replay_buffer,
            gamma,
            explorer,
            replay_start_size=500,
            update_interval=1,
            target_update_interval=100,
            phi=phi,
            gpu=self.gpu,
        )

    def _training(self, mdp, rng, event_listeners):
        if not hasattr(self, 'agent'):
            self._build_agent(mdp)

        # initial state
        s = mdp.initial_state_dist().sample(rng=rng)
        for i_step in range(self.num_steps):
            # select action
            a = self.agent.act(s)
            # transition to next state
            ns = mdp.next_state_dist(s, mdp._actions[a]).sample(rng=rng)
            r = mdp.reward(s, a, ns)
            done = mdp.is_terminal(ns)
            # update
            self.agent.observe(ns, r, done, False)
            # end of timestep
            for listener in event_listeners:
                listener.end_of_timestep(locals())
            s = ns
            # end of episode
            if mdp.is_terminal(s):
                for listener in event_listeners:
                    listener.end_of_episode(locals())
                s = mdp.initial_state_dist().sample(rng=rng)
        
        print(f'Bias of last layer: {self.agent.model.linear_3.bias}')

    def train_on(self, mdp):
        rng = random.Random(self.seed)
        self._training(mdp, rng, self.event_listeners)
        listener_results = {
            listener.__class__.__name__: listener.results() for listener in self.event_listeners
        }
        return Result(
            agent=self.agent,
            **listener_results,
        )
    
    