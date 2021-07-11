"""
Various network architecture codes used in PEARL algorithm
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class GRU(nn.Module):
    """Base GRU network class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        input_dim,
        output_dim,
        hidden_dim,
        hidden_activation=F.relu,
        init_w=3e-3,
    ):
        super().__init__()

        self.hidden_activation = hidden_activation

        # Set GRU layers
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim)

        # Set output layer
        self.last_fc_layer = nn.Linear(hidden_dim, output_dim)
        self.last_fc_layer.weight.data.uniform_(-init_w, init_w)
        self.last_fc_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, x, h):
        """Get output when input is given"""
        x, h = self.gru(x.unsqueeze(0), h.unsqueeze(0))
        x = self.hidden_activation(x)
        x = self.last_fc_layer(x)
        return x, h


class GaussianGRU(GRU):
    """Gaussian GRU network class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        input_dim,
        output_dim,
        hidden_dim,
        env_target,
        is_deterministic=False,
        init_w=1e-3,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            init_w=init_w,
        )

        if env_target == "cheetah-dir":
            self.log_std = -0.5 * np.ones(output_dim, dtype=np.float32)
        elif env_target == "cheetah-vel":
            self.log_std = -1.0 * np.ones(output_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.Tensor(self.log_std))
        self.is_deterministic = is_deterministic

    def get_normal_dist(self, x, h):
        """Get Gaussian distribtion"""
        mean, hidden = super().forward(x, h)
        std = torch.exp(self.log_std)
        return Normal(mean, std), mean, hidden

    def get_log_prob(self, trans, hidden, action):
        """Get log probability of Gaussian distribution using transtion and hidden"""
        normal, _, _ = self.get_normal_dist(trans, hidden)
        return normal.log_prob(action).sum(dim=-1)

    def forward(self, x, h):
        normal, mean, hidden = self.get_normal_dist(x, h)
        if self.is_deterministic:
            action = mean
            log_prob = None
        else:
            action = normal.sample()
            log_prob = normal.log_prob(action).sum(dim=-1)
        action = action.view(-1)
        return action, log_prob, hidden
