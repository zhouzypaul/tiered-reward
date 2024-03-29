import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pfrl import agents, explorers
from pfrl import nn as pnn
from pfrl.initializers.lecun_normal import init_lecun_normal
from pfrl.q_functions import DiscreteActionValueHead, DuelingDQN
from pfrl import replay_buffers


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


def parse_arch(arch, n_actions):
    if arch == "nature":
        return nn.Sequential(
            pnn.LargeAtariCNN(),
            optimistic_init_chainer_default(nn.Linear(512, n_actions)),
            DiscreteActionValueHead(),
        )
    elif arch == "doubledqn":
        # raise NotImplementedError("Single shared bias not implemented yet")
        return nn.Sequential(
            pnn.LargeAtariCNN(),
            optimistic_init_chainer_default(nn.Linear(512, n_actions, bias=False)),
            SingleSharedBias(),
            DiscreteActionValueHead(),
        )
    elif arch == "nips":
        return nn.Sequential(
            pnn.SmallAtariCNN(),
            optimistic_init_chainer_default(nn.Linear(256, n_actions)),
            DiscreteActionValueHead(),
        )
    elif arch == "dueling":
        return DuelingDQN(n_actions)
    else:
        raise RuntimeError("Not supported architecture: {}".format(arch))


def parse_agent(agent):
    return {"DQN": agents.DQN, "DoubleDQN": agents.DoubleDQN, "PAL": agents.PAL}[agent]


def make_agent(args, n_actions, gamma):
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
        phi=phi,
    )

    return agent

