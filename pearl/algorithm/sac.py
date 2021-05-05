import numpy as np
from typing import Callable, List

import torch
import torch.nn as nn
import torch.optim as optim
from algorithm.utils.util import *
from algorithm.utils.networks import *


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
        **config,
    ):

        self.device = device
        self.gamma = config['gamma']
        self.kl_lambda = config['kl_lambda']
        self.batch_size = config['batch_size']
        self.reward_scale = config['reward_scale']
        
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
            latent_dim=latent_dim,
            hidden_units=hidden_units,
        ).to(device)
        self.policy = TanhGaussianPolicy(
            input_dim=observ_dim + latent_dim,
            output_dim=action_dim,
            hidden_units=hidden_units,
        ).to(device)

        # Initialize target parameters to match main parameters
        hard_target_update(self.qf1, self.target_qf1)
        hard_target_update(self.qf2, self.target_qf2)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config['policy_lr'])
        self.context_optimizer = optim.Adam(self.encoder.parameters(), lr=config['encoder_lr'])
        self.qf_parameters = list(self.qf1.parameters()) + list(self.qf2.parameters())
        self.qf_optimizer = optim.Adam(self.qf_parameters, lr=config['qf_lr'])

        # If automatic entropy tuning is True, 
        # initialize a target entropy, a log alpha and an alpha optimizer
        if config['automatic_entropy_tuning']:
            self.target_entropy = -np.prod((action_dim,)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config['policy_lr'])

    def get_action(self, obs, deterministic=False):
        ''' Sample action from the policy, conditioned on the task embedding '''
        z = self.encoder.z.to(self.device)
        obs = torch.from_numpy(obs[None]).float().to(self.device)
        print(z.shape)
        print(obs.shape)
        inputs = torch.cat([obs, z], dim=-1).to(self.device)
        action, _ = self.policy(inputs, deterministic=deterministic)
        return action.detach().cpu().numpy()
