import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import torch_ac


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
    
    
class GaussianHeadWithStateIndependentCovariance(nn.Module):
    """Taken from pfrl.policies.gaussian_policy
    
    Gaussian head with state-independent learned covariance.

    This link is intended to be attached to a neural network that outputs
    the mean of a Gaussian policy. The only learnable parameter this link has
    determines the variance in a state-independent way.

    State-independent parameterization of the variance of a Gaussian policy
    is often used with PPO and TRPO, e.g., in https://arxiv.org/abs/1709.06560.

    Args:
        action_size (int): Number of dimensions of the action space.
        var_type (str): Type of parameterization of variance. It must be
            'spherical' or 'diagonal'.
        var_func (callable): Callable that computes the variance from the var
            parameter. It should always return positive values.
        var_param_init (float): Initial value the var parameter.
    """
    def __init__(
        self,
        action_size,
        var_type="spherical",
        var_func=nn.functional.softplus,
        var_param_init=0,
    ):
        super().__init__()

        self.var_func = var_func
        var_size = {"spherical": 1, "diagonal": action_size}[var_type]

        self.var_param = nn.Parameter(
            torch.tensor(
                np.broadcast_to(var_param_init, var_size),
                dtype=torch.float,
            )
        )

    def forward(self, mean):
        """Return a Gaussian with given mean.

        Args:
            mean (torch.Tensor or ndarray): Mean of Gaussian.

        Returns:
            torch.distributions.Distribution: Gaussian whose mean is the
                mean argument and whose variance is computed from the parameter
                of this link.
        """
        var = self.var_func(self.var_param)
        return torch.distributions.Independent(
            torch.distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
        )


class StateSpaceACModel(nn.Module, torch_ac.RecurrentACModel):
    """This is for state space only, with continuous actions (e.g. D4RL envs)"""
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        """use_memory and use_text are not supported yet"""
        assert not use_memory
        assert not use_text
        super().__init__()

        self.obs_space = obs_space
        self.action_space = action_space
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.embedding = nn.Sequential(
            nn.Linear(obs_space.low.size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        ).double()

        # Resize image embedding
        self.embedding_size = 64
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.low.size),
            GaussianHeadWithStateIndependentCovariance(
                action_size=action_space.low.size,
                var_type="diagonal",
                var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
                var_param_init=0,  # log std = 0 => std = 1
            ),
        ).double()

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        ).double()

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.embedding_size

    def forward(self, obs, memory):
        embedding = self.embedding(obs)

        dist = self.actor(embedding)

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1)

    def forward(self, x):
        inputs = x
        x = torch.relu(x)
        x = self.conv0(x)
        x = torch.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):

    def __init__(self, input_shape, out_channels):
        super(ConvSequence, self).__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0],
                              out_channels=self._out_channels,
                              kernel_size=3,
                              padding=1)
        self.max_pool2d = nn.MaxPool2d(kernel_size=3,
                                       stride=2,
                                       padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool2d(x)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

    def get_output_shape(self):
        """I don't think this is correct. Need to do it step by step"""
        _c, h, w = self._input_shape
        conv_out_h = (h - 3 + 2 * 1) // 1 + 1
        conv_out_w = (w - 3 + 2 * 1) // 1 + 1
        return self._out_channels, (h + 1) // 2, (w + 1) // 2


class ImpalaCNN(nn.Module, torch_ac.RecurrentACModel):
    """Network from IMPALA paper, to work with pfrl."""

    def __init__(self, input_shape, num_outputs, use_memory=False, use_text=False):
        """
        NOTE: use_memory is not supported yet.
        currently not even going to bother with use_text
        """
        assert not use_memory
        assert not use_text

        super(ImpalaCNN, self).__init__()
        self.use_memory = use_memory
        assert not self.use_memory, "use_memory is not supported yet"
        self.use_text = use_text
        assert not self.use_text, "use_text is not supported yet"

        h, w, c = input_shape
        shape = (c, h, w)
        self.shape =shape

        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc = nn.LazyLinear(out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)
        # Initialize weights of logits_fc
        nn.init.orthogonal_(self.logits_fc.weight, gain=0.01)
        nn.init.zeros_(self.logits_fc.bias)

    @property
    def memory_size(self):
        return self.shape[0] * self.shape[1] * self.shape[2]

    def forward(self, obs, memory):
        x = obs.image
        assert x.ndim == 4
        # x = obs / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(x)
        x = self.hidden_fc(x)
        x = torch.relu(x)
        logits = self.logits_fc(x)
        if self.use_memory:
            memory = logits  # NOTE: this is probably not correct
        dist = torch.distributions.Categorical(logits=logits)
        value = self.value_fc(x).squeeze(1)
        return dist, value, memory

    def save_to_file(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load_from_file(self, model_path):
        self.load_state_dict(torch.load(model_path))
