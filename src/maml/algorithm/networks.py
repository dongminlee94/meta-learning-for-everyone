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


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TanhGaussianPolicy(MLP):
    """Gaussian policy network class containing Value network"""

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

        self.is_deterministic = is_deterministic
        self.last_fc_log_std = nn.Linear(hidden_dim, output_dim)
        self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)

    def get_normal_dist(self, x):
        """Get Gaussian distribtion"""
        for fc_layer in self.fc_layers:
            x = self.hidden_activation(fc_layer(x))

        mean = self.last_fc_layer(x)
        log_std = self.last_fc_log_std(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        return Normal(mean, std), mean

    def get_log_prob(self, obs, action):
        """Get log probability of Gaussian distribution using obs and action"""
        normal, _ = self.get_normal_dist(obs)
        return normal.log_prob(action).sum(dim=-1)

    def forward(self, x):
        normal, mean = self.get_normal_dist(x)

        if self.is_deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            # If reparameterize, use reparameterization trick (mean + std * N(0,1))
            action = normal.rsample()

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

        action = torch.tanh(action)
        return action, log_prob


class Model(nn.Module):
    """Set of fully connected networks containing Policy and Value function"""

    def __init__(
        self,
        observ_dim,
        action_dim,
        hidden_dim,
    ):
        super().__init__()

        self.policy = TanhGaussianPolicy(
            input_dim=observ_dim,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
        )
        self.vf = MLP(
            input_dim=observ_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
        )

    def forward(self, obs):
        """Infer policy network"""
        return self.policy(obs)
