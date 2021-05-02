import numpy as np
from typing import Callable, List

import torch
import torch.nn as nn
import torch.optim as optim
import algorithm.utils as ptu
from algorithm.networks import *


class SAC(object):
    def __init__(
            self,
            observ_dim,
            action_dim,
            latent_dim,
            hidden_units,
            encoder_input_dim,
            encoder_output_dim,
            device,
            **kwargs,
    ):

        self.gamma = kwargs['gamma']
        self.kl_lambda = kwargs['kl_lambda']
        self.batch_size = kwargs['batch_size']
        self.reward_scale = kwargs['reward_scale']
        
        # Instantiate networks
        self.qf1 = FlattenMLP(
            input_dim=observ_dim + action_dim + latent_dim,
            output_dim=1,
            hidden_units=hidden_units,
        ).to(device)
        self.qf2 = FlattenMLP(
            input_dim=observ_dim + action_dim + latent_dim,
            output_dim=1,
            hidden_units=hidden_units,
        ).to(device)
        self.target_qf1 = FlattenMLP(
            input_dim=observ_dim + action_dim + latent_dim,
            output_dim=1,
            hidden_units=hidden_units,
        ).to(device)
        self.target_qf2 = FlattenMLP(
            input_dim=observ_dim + action_dim + latent_dim,
            output_dim=1,
            hidden_units=hidden_units,
        ).to(device)
        self.encoder = MLPEncoder(
            input_dim=encoder_input_dim,
            output_dim=encoder_output_dim,
            hidden_units=hidden_units,
            latent_size=latent_dim,
        ).to(device)
        self.policy = TanhGaussianPolicy(
            input_dim=observ_dim + latent_dim,
            output_dim=action_dim,
            hidden_units=hidden_units,
        ).to(device)

        # Initialize target parameters to match main parameters
        ptu.hard_target_update(self.qf1, self.target_qf1)
        ptu.hard_target_update(self.qf2, self.target_qf2)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=kwargs['policy_lr'])
        self.context_optimizer = optim.Adam(self.encoder.parameters(), lr=kwargs['encoder_lr'])
        self.qf_parameters = list(self.qf1.parameters()) + list(self.qf2.parameters())
        self.qf_optimizer = optim.Adam(self.qf_parameters, lr=kwargs['qf_lr'])

        # If automatic entropy tuning is True, 
        # initialize a target entropy, a log alpha and an alpha optimizer
        if kwargs['automatic_entropy_tuning']:
            self.target_entropy = -np.prod((action_dim,)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=kwargs['policy_lr'])