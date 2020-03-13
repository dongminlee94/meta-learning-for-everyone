import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from meta_rl.policies.base import Policy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GaussianPolicy(Policy):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes=(64,64),
                 activation=F.relu,
    ):
        super(GaussianPolicy, self).__init__(
            input_size=input_size,                                  
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

    def forward(self, x):
        mu = super(GaussianPolicy, self).forward(x)
        log_std = torch.zeros_like(mu)
        std = torch.exp(log_std)
        
        dist = Normal(mu, std)
        pi = dist.sample()
        return mu, std, dist, pi