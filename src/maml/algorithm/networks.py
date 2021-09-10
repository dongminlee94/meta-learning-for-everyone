"""
Various network architecture codes used in MAML algorithm
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class MLP(nn.Module):
    """Base MLP network class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        input_dim,
        output_dim,
        hidden_dim,
        hidden_activation=F.relu,
        init_w=3e-3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_activation = hidden_activation

        # Set fully connected layers
        self.fc_layers = nn.ModuleList()
        self.hidden_layers = [hidden_dim] * 2
        in_layer = input_dim

        for i, hidden_layer in enumerate(self.hidden_layers):
            fc_layer = nn.Linear(in_layer, hidden_layer)
            in_layer = hidden_layer
            self.__setattr__("fc_layer{}".format(i), fc_layer)
            self.fc_layers.append(fc_layer)

        # Set the output layer
        self.last_fc_layer = nn.Linear(hidden_dim, output_dim)
        self.last_fc_layer.weight.data.uniform_(-init_w, init_w)
        self.last_fc_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        """Get output when input is given"""
        for fc_layer in self.fc_layers:
            x = self.hidden_activation(fc_layer(x))
        x = self.last_fc_layer(x)
        return x


class LinearValue(nn.Module):
    """Linear observation-value function based on handcrafted features"""

    # Adapted from Tristan Deleu's implementation.
    # **References**
    # 1. Duan et al. 2016. “Benchmarking Deep Reinforcement Learning for Continuous Control.”
    # 2. https://github.com/tristandeleu/pytorch-maml-rl
    # 3. https://github.com/learnables/cherry/blob/master/cherry/models/robotics.py
    def __init__(self, input_dim, reg=1e-5):
        super().__init__()
        self.linear = nn.Linear(2 * input_dim + 4, 1, bias=False)
        self.reg = reg

    @staticmethod
    def features(obs):
        """Handcrafted Linear features"""
        length = obs.size(0)
        ones = torch.ones(length, 1)
        al = torch.arange(length, dtype=torch.float32).view(-1, 1) / 100.0
        return torch.cat([obs, obs ** 2, al, al ** 2, al ** 3, ones], dim=1)

    def fit(self, obs, returns):
        """Fit feature parameters to observation and returns by minimizing least-squares """
        features = LinearValue.features(obs)
        reg = self.reg * torch.eye(features.size(1))
        A = features.t() @ features + reg
        b = features.t() @ returns
        coeffs, _ = torch.lstsq(b, A)
        self.linear.weight.data = coeffs.data.t()

    def forward(self, obs):
        """Get value when observations are given"""
        features = LinearValue.features(obs)
        return self.linear(features)


class TanhGaussianPolicy(MLP):
    """Gaussian policy network class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        input_dim,
        output_dim,
        hidden_dim,
        is_deterministic=False,
        init_w=1e-3,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            init_w=init_w,
        )

        self.log_std = -0.5 * np.ones(output_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.Tensor(self.log_std))
        self.is_deterministic = is_deterministic

    def get_normal_dist(self, x):
        """Get Gaussian distribtion"""
        mean = super().forward(x)
        std = torch.exp(self.log_std)
        return Normal(mean, std), mean

    def get_log_prob(self, obs, action):
        """Get log probability of Gaussian distribution using obs and action"""
        normal, _ = self.get_normal_dist(obs)
        # Compute log prob from Gaussian,
        # and then apply correction for Tanh squashing.
        # NOTE: The correction formula is a little bit magic.
        # To get an understanding of where it comes from,
        # check out the original SAC paper
        # (https://arxiv.org/abs/1801.01290) and look in appendix C.
        # This is a more numerically-stable equivalent to Eq 21.
        # Derivation:
        #               log(1 - tanh(x)^2))
        #               = log(sech(x)^2))
        #               = 2 * log(sech(x)))
        #               = 2 * log(2e^-x / (e^-2x + 1)))
        #               = 2 * (log(2) - x - log(e^-2x + 1)))
        #               = 2 * (log(2) - x - softplus(-2x)))
        log_prob = normal.log_prob(action)
        log_prob -= 2 * (np.log(2) - action - F.softplus(-2 * action))
        log_prob = log_prob.sum(-1, keepdim=True)
        return log_prob

    def forward(self, x):
        normal, mean = self.get_normal_dist(x)

        if self.is_deterministic:
            log_prob = None
            action = torch.tanh(mean)
        else:
            action = normal.sample()
            log_prob = self.get_log_prob(x, action)
            action = torch.tanh(mean)
        return action, log_prob
