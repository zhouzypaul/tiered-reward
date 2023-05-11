import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pfrl import agents, explorers
from pfrl import nn as pnn
from pfrl.initializers.lecun_normal import init_lecun_normal
from pfrl.q_functions import DiscreteActionValueHead, DuelingDQN
from pfrl import replay_buffers
from pfrl.initializers import init_chainer_default


def constant_bias_initializer(bias=0.0):
    @torch.no_grad()
    def init_bias(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.bias.fill_(bias)

    return init_bias


class SingleSharedBias(nn.Module):
    """Single shared bias used in the Double DQN paper.
    You can add this link after a Linear layer with nobias=True to implement a
    Linear layer with a single shared bias parameter.
    See http://arxiv.org/abs/1509.06461.
    """

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros([1], dtype=torch.float32))

    def __call__(self, x):
        return x + self.bias.expand_as(x)


@torch.no_grad()
def optimistic_init_chainer_default(layer):
    """
    difference from pfrl:
    initialize the bias term optimistically

    Initializes the layer with the chainer default.
    weights with LeCunNormal(scale=1.0) and zeros as biases
    """
    assert isinstance(layer, nn.Module)

    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        init_lecun_normal(layer.weight)
        if layer.bias is not None:
            # layer may be initialized with bias=False
            nn.init.constant_(layer.bias, 1e15)
    return layer


class LargeAtariCNN(nn.Module):
    """Large CNN module proposed for DQN in Nature, 2015.

    See: https://www.nature.com/articles/nature14236
    """

    def __init__(
        self, n_input_channels=4, n_output_channels=512, activation=F.relu, bias=0.1
    ):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, 32, 8, stride=4),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.Conv2d(64, 64, 3, stride=1),
            ]
        )
        
        self.apply(init_chainer_default)
        self.apply(constant_bias_initializer(bias=bias))
        
        self.output = nn.LazyLinear(n_output_channels)

    def forward(self, state):
        h = state
        for layer in self.layers:
            h = self.activation(layer(h))
        h_flat = h.view(h.size(0), -1)
        return self.activation(self.output(h_flat))


def parse_arch(arch, n_actions):
    if arch == "nature":
        return nn.Sequential(
            LargeAtariCNN(n_input_channels=3),
            optimistic_init_chainer_default(nn.Linear(512, n_actions)),
            DiscreteActionValueHead(),
        )
    elif arch == "doubledqn":
        # raise NotImplementedError("Single shared bias not implemented yet")
        return nn.Sequential(
            LargeAtariCNN(n_input_channels=3),
            optimistic_init_chainer_default(nn.Linear(512, n_actions, bias=False)),
            SingleSharedBias(),
            DiscreteActionValueHead(),
        )
    elif arch == "nips":
        return nn.Sequential(
            pnn.SmallAtariCNN(n_input_channels=3),
            optimistic_init_chainer_default(nn.Linear(256, n_actions)),
            DiscreteActionValueHead(),
        )
    elif arch == "dueling":
        return DuelingDQN(n_actions)
    else:
        raise RuntimeError("Not supported architecture: {}".format(arch))


def parse_agent(agent):
    return {"DQN": agents.DQN, "DoubleDQN": agents.DoubleDQN, "PAL": agents.PAL}[agent]


def make_dqn_agent(args, n_actions, gamma, preprocess_func=None):
    q_func = parse_arch(args.arch, n_actions)

    if args.noisy_net_sigma is not None:
        pnn.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
        # Turn off explorer
        explorer = explorers.Greedy()

    # Use the same hyper parameters as the Nature paper's
    opt = optim.RMSprop(
        q_func.parameters(),
        lr=args.lr,
        alpha=0.95,
        momentum=0.0,
        eps=1e-2,
        centered=True,
    )

    # Select a replay buffer to use
    if args.prioritized:
        # Anneal beta from beta0 to 1 throughout training
        betasteps = args.steps / args.update_interval
        rbuf = replay_buffers.PrioritizedReplayBuffer(
            10**6,
            alpha=0.6,
            beta0=0.4,
            betasteps=betasteps,
            num_steps=args.n_step_return,
        )
    else:
        rbuf = replay_buffers.ReplayBuffer(10**6, num_steps=args.n_step_return)

    explorer = explorers.Greedy()

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    Agent = parse_agent(args.agent)
    agent = Agent(
        q_func,
        opt,
        rbuf,
        gpu=args.gpu,
        gamma=gamma,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        target_update_interval=args.target_update_interval,
        clip_delta=args.clip_delta,
        update_interval=args.update_interval,
        batch_accumulator="sum",
        minibatch_size=args.batch_size,
        phi=phi if preprocess_func is None else preprocess_func,
    )

    return agent

