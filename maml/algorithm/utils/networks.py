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
        hidden_units,
        hidden_activation=F.relu,
        init_w=3e-3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation

        # Set fully connected layers
        self.fc_layers = nn.ModuleList()
        in_dim = input_dim
        for i, hidden_unit in enumerate(hidden_units):
            fc_layer = nn.Linear(in_dim, hidden_unit)
            in_dim = hidden_unit
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
        hidden_units,
        env_target,
        init_w=1e-3,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_units=hidden_units,
            init_w=init_w,
        )

        if env_target == "cheetah-dir":
            self.log_std = -0.5 * np.ones(output_dim, dtype=np.float32)
        elif env_target == "cheetah-vel":
            self.log_std = -1.0 * np.ones(output_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.Tensor(self.log_std))

    def get_normal_dist(self, x):
        """Get Gaussian distribtion"""
        mean = super().forward(x)
        std = torch.exp(self.log_std)
        return Normal(mean, std), mean

    def get_log_prob(self, obs, action):
        """Get log probability of Gaussian distribution using obs and action"""
        normal, _ = self.get_normal_dist(obs)
        return normal.log_prob(action).sum(dim=-1)

    def forward(self, x, is_deterministic=False):
        normal, mean = self.get_normal_dist(x)
        if is_deterministic:
            policy = mean
            log_prob = None
        else:
            policy = normal.sample()
            log_prob = normal.log_prob(policy).sum(dim=-1)
        return policy, log_prob
