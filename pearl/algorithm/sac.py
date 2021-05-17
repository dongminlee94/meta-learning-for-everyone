import numpy as np
from typing import Callable, List

import torch
import torch.optim as optim
import torch.nn.functional as F
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
        self.policy = TanhGaussianPolicy(
            input_dim=observ_dim + latent_dim,
            output_dim=action_dim,
            hidden_units=hidden_units,
        ).to(device)
        self.encoder = MLPEncoder(
            input_dim=encoder_input_dim,
            output_dim=encoder_output_dim,
            latent_dim=latent_dim,
            hidden_units=hidden_units,
            device=device,
        ).to(device)
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

        # Initialize target parameters to match main parameters
        self.hard_target_update(self.qf1, self.target_qf1)
        self.hard_target_update(self.qf2, self.target_qf2)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config['policy_lr'])
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=config['encoder_lr'])
        self.qf_parameters = list(self.qf1.parameters()) + list(self.qf2.parameters())
        self.qf_optimizer = optim.Adam(self.qf_parameters, lr=config['qf_lr'])

        # Initialize target entropy, log alpha, and alpha optimizer
        self.target_entropy = -np.prod((action_dim,)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config['policy_lr'])

        self.net_dict = {
            'actor': self.policy,
            'encoder': self.encoder,
            'qf1': self.qf1,
            'qf2': self.qf2,
            'target_qf1': self.target_qf1,
            'target_qf2': self.target_qf2,
        }

    def hard_target_update(self, main, target):
        target.load_state_dict(main.state_dict())

    def soft_target_update(self, main, target, tau=0.005):
        for main_param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data.copy_(tau*main_param.data + (1.0-tau)*target_param.data)

    def get_action(self, obs, deterministic=False):
        ''' Sample action from the policy, conditioned on the task embedding '''
        task_z = self.encoder.z
        obs = torch.from_numpy(obs[None]).float().to(self.device)
        inputs = torch.cat([obs, task_z], dim=-1).to(self.device)
        action, _ = self.policy(inputs, deterministic=deterministic)
        return action.view(-1).detach().cpu().numpy()

    def train_model(self, meta_batch_size, batch_size, context_batch, transition_batch):
        # Data is (meta-batch, batch, feature)
        obs, action, reward, next_obs, done = transition_batch

        # Flattens out the transition batch dimension
        obs = obs.view(meta_batch_size * batch_size, -1)            # torch.Size([1024, 26])
        action = action.view(meta_batch_size * batch_size, -1)      # torch.Size([1024, 6])
        reward = reward.view(meta_batch_size * batch_size, -1)      # torch.Size([1024, 1])
        next_obs = next_obs.view(meta_batch_size * batch_size, -1)  # torch.Size([1024, 26])
        done = done.view(meta_batch_size * batch_size, -1)          # torch.Size([1024, 1])

        # Given context c, sample context variable z ~ posterior q(z|c)
        self.encoder.infer_posterior(context_batch)
        self.encoder.sample_z()

        # Flattens out the context batch dimension
        task_z = self.encoder.z                                     # torch.Size([4, 5])
        task_z = [z.repeat(batch_size, 1) for z in task_z]          # [torch.Size([256, 5]), 
                                                                    #  torch.Size([256, 5]),
                                                                    #  torch.Size([256, 5]), 
                                                                    #  torch.Size([256, 5])]
        task_z = torch.cat(task_z, dim=0)                           # torch.Size([1024, 5])

        # Target for Q regression
        with torch.no_grad():
            next_inputs = torch.cat([next_obs, task_z], dim=-1)     # torch.Size([1024, 31])
            next_pi, next_log_pi = self.policy(next_inputs)         # torch.Size([1024, 6])
                                                                    # torch.Size([1024, 1])
            min_target_q = torch.min(                               # torch.Size([1024, 1])
                self.target_qf1(next_obs, next_pi, task_z), 
                self.target_qf2(next_obs, next_pi, task_z)
            )
            target_v = min_target_q - self.alpha * next_log_pi      # torch.Size([1024, 1])
            target_q = reward + self.gamma * (1-done) * target_v    # torch.Size([1024, 1])

        # Q-functions losses
        q1 = self.qf1(obs, action, task_z)                          # torch.Size([1024, 1])
        q2 = self.qf2(obs, action, task_z)                          # torch.Size([1024, 1])
        qf1_loss = F.mse_loss(q1, target_q)
        qf2_loss = F.mse_loss(q2, target_q)
        qf_loss = qf1_loss + qf2_loss
        
        # Two Q-networks update
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        # Encoder loss using KL divergence on z
        kl_div = self.encoder.compute_kl_div()                      # torch.Size([4, 1])
        encoder_loss = self.kl_lambda * kl_div
        
        # Encoder network update
        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

        # Policy loss
        inputs = torch.cat([obs, task_z.detach()], dim=-1)          # torch.Size([1024, 31])
        pi, log_pi = self.policy(inputs)                            # torch.Size([1024, 6])
                                                                    # torch.Size([1024, 1])
        min_pi_q = torch.min(                                       # torch.Size([1024, 1])
                self.qf1(obs, pi, task_z.detach()), 
                self.qf2(obs, pi, task_z.detach())
        )
        policy_loss = (self.alpha * log_pi - min_pi_q).mean()

        # Policy network update
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Temperature parameter alpha update
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Polyak averaging for target parameter
        self.soft_target_update(self.qf1, self.target_qf1)
        self.soft_target_update(self.qf2, self.target_qf2)
    
    def save(self, path, net_dict=None):
        if net_dict is None:
            net_dict = self.net_dict

        state_dict = {name: net.state_dict() for name, net in net_dict.items()}
        state_dict['alpha'] = self.log_alpha
        torch.save(state_dict, path)

    def load(self, path, net_dict=None):
        if net_dict is None:
            net_dict = self.net_dict
        
        checkpoint = torch.load(path)
        for name, net in net_dict.items():
            if name == 'alpha':
                self.log_alpha.load(net)
            else:
                net.load_state_dict(checkpoint[name])
