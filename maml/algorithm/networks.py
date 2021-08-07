"""
Various network architecture codes used in MAML algorithm
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal


class MLP(nn.Module):
    """Base MLP network class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        input_dim,
        output_dim,
        hidden_dims,
        hidden_activation=F.relu,
        init_w=3e-3,
    ):
        super().__init__()

        self.hidden_activation = hidden_activation

        # Set fully connected layers
        self.fc_layers = nn.ModuleList()
        in_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            fc_layer = nn.Linear(in_dim, hidden_dim)
            in_dim = hidden_dim
            self.__setattr__("fc_layer{}".format(i), fc_layer)
            self.fc_layers.append(fc_layer)

        # Set the output layer
        self.last_fc_layer = nn.Linear(in_dim, output_dim)
        self.last_fc_layer.weight.data.uniform_(-init_w, init_w)
        self.last_fc_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        """Get output when input is given"""
        for fc_layer in self.fc_layers:
            x = self.hidden_activation(fc_layer(x))
        x = self.last_fc_layer(x)
        return x


class GaussianPolicy(MLP):
    """Gaussian policy network class using MLP"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        input_dim,
        output_dim,
        hidden_dims,
        env_target,
        is_deterministic=False,
        init_w=1e-3,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            init_w=init_w,
        )

        if env_target == "cheetah-dir":
            self.log_std = -0.5 * np.ones(output_dim, dtype=np.float32)
        elif env_target == "cheetah-vel":
            self.log_std = -1.0 * np.ones(output_dim, dtype=np.float32)
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
        return normal.log_prob(action).sum(dim=-1)

    def forward(self, x):
        normal, mean = self.get_normal_dist(x)
        if self.is_deterministic:
            action = mean
            log_prob = None
        else:
            action = normal.sample()
            log_prob = normal.log_prob(action).sum(dim=-1)
        return action, log_prob
