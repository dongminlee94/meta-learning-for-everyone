import numpy as np
from typing import Callable, List

import torch
import torch.nn as nn
import torch.optim as optim
from algorithm.utils.networks import MLP, FlattenMLP, MLPEncoder, TanhGaussianPolicy


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
        self.hard_target_update(self.qf1, self.target_qf1)
        self.hard_target_update(self.qf2, self.target_qf2)

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

    def hard_target_update(self, main, target):
        target.load_state_dict(main.state_dict())

    def soft_target_update(self, main, target, tau=0.005):
        for main_param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data.copy_(tau*main_param.data + (1.0-tau)*target_param.data)

    def get_action(self, obs, deterministic=False):
        ''' Sample action from the policy, conditioned on the task embedding '''
        z = self.encoder.z
        obs = torch.from_numpy(obs[None]).float()
        inputs = torch.cat([obs, z], dim=-1).to(self.device)
        action, _ = self.policy(inputs, deterministic=deterministic)
        return action.view(-1).detach().cpu().numpy()

    def train_model(self, num_tasks, context_batch, transition_batch):
        # Data is (task, batch, feature)
        obs, action, reward, next_obs, done = transition_batch

        # Flattens out the task dimension
        tensor_dim, matrix_dim, _ = obs.size()                  # torch.Size([4, 256, 26])
        obs = obs.view(tensor_dim * matrix_dim, -1)             # torch.Size([1024, 26])
        action = action.view(tensor_dim * matrix_dim, -1)       # torch.Size([1024, 6])
        next_obs = next_obs.view(tensor_dim * matrix_dim, -1)   # torch.Size([1024, 26])

        # Given context c, sample context variable z ~ posterior q(z|c)
        self.encoder.infer_posterior(context_batch)
        self.encoder.sample_z()
        task_z = self.encoder.z
        print('task_z_1', task_z)
        task_z = [z.repeat(matrix_dim, 1) for z in task_z]
        print('task_z_2', task_z)
        task_z = torch.cat(task_z, dim=0)
        print('task_z_3', task_z)
        print(dones)

