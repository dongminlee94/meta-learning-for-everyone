"""
Soft Actor-Critic algorithm implementation for training when meta-train
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from meta_rl.pearl.algorithm.networks import FlattenMLP, MLPEncoder, TanhGaussianPolicy


class SAC:  # pylint: disable=too-many-instance-attributes
    """Soft Actor-Critic class with context"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        observ_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int,
        encoder_input_dim: int,
        encoder_output_dim: int,
        device: torch.device,
        **config,
    ) -> None:

        self.device = device
        self.gamma: float = config["gamma"]
        self.kl_lambda: float = config["kl_lambda"]
        self.batch_size: int = config["batch_size"]

        # Instantiate networks
        self.policy = TanhGaussianPolicy(
            input_dim=observ_dim + latent_dim,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            is_deterministic=False,
        ).to(device)
        self.encoder = MLPEncoder(
            input_dim=encoder_input_dim,
            output_dim=encoder_output_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            device=device,
        ).to(device)
        self.qf1 = FlattenMLP(
            input_dim=observ_dim + action_dim + latent_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
        ).to(device)
        self.qf2 = FlattenMLP(
            input_dim=observ_dim + action_dim + latent_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
        ).to(device)
        self.target_qf1 = FlattenMLP(
            input_dim=observ_dim + action_dim + latent_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
        ).to(device)
        self.target_qf2 = FlattenMLP(
            input_dim=observ_dim + action_dim + latent_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
        ).to(device)

        # Initialize target parameters to match main parameters
        self.hard_target_update(self.qf1, self.target_qf1)
        self.hard_target_update(self.qf2, self.target_qf2)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config["policy_lr"])
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=config["encoder_lr"])
        self.qf_parameters: List[torch.nn.parameter.Parameter] = list(self.qf1.parameters()) + list(
            self.qf2.parameters(),
        )
        self.qf_optimizer = optim.Adam(self.qf_parameters, lr=config["qf_lr"])

        # Initialize target entropy, log alpha, and alpha optimizer
        self.target_entropy: int = -np.prod((action_dim,)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config["policy_lr"])

    @classmethod
    def hard_target_update(cls, main: FlattenMLP, target: FlattenMLP) -> None:
        """Update target network to be the same as main network"""
        target.load_state_dict(main.state_dict())

    @classmethod
    def soft_target_update(cls, main: FlattenMLP, target: FlattenMLP, tau: float = 0.005):
        """Update target network by polyak averaging."""
        for main_param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Sample action from the policy"""
        task_z = self.encoder.task_z
        obs = torch.Tensor(obs).view(1, -1).to(self.device)
        inputs = torch.cat([obs, task_z], dim=-1).to(self.device)
        action, _ = self.policy(inputs)
        return action.view(-1).detach().cpu().numpy()

    # pylint: disable=too-many-locals
    def train_model(
        self,
        meta_batch_size: int,
        batch_size: int,
        context_batch: torch.Tensor,
        transition_batch: List[torch.Tensor],
    ) -> Dict[str, float]:
        """Train models according to training method of SAC algorithm"""
        # Data is (meta-batch, batch, feature)
        cur_obs, actions, rewards, next_obs, dones = transition_batch

        # Flattens out the transition batch dimension
        cur_obs = cur_obs.view(meta_batch_size * batch_size, -1)
        actions = actions.view(meta_batch_size * batch_size, -1)
        rewards = rewards.view(meta_batch_size * batch_size, -1)
        next_obs = next_obs.view(meta_batch_size * batch_size, -1)
        dones = dones.view(meta_batch_size * batch_size, -1)

        # Given context c, sample context variable z ~ posterior q(z|c)
        self.encoder.infer_posterior(context_batch)
        task_z = self.encoder.task_z

        # Flattens out the context batch dimension
        task_z = [z.repeat(batch_size, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)

        # Encoder loss using KL divergence on z
        kl_div = self.encoder.compute_kl_div()
        encoder_loss = self.kl_lambda * kl_div
        self.encoder_optimizer.zero_grad()
        encoder_loss.backward(retain_graph=True)

        # Target for Q regression
        with torch.no_grad():
            next_inputs = torch.cat([next_obs, task_z], dim=-1)
            next_policy, next_log_policy = self.policy(next_inputs)
            min_target_q = torch.min(
                self.target_qf1(next_obs, next_policy, task_z),
                self.target_qf2(next_obs, next_policy, task_z),
            )
            target_v = min_target_q - self.alpha * next_log_policy
            target_q = rewards + self.gamma * (1 - dones) * target_v

        # Q-function loss
        pred_q1 = self.qf1(cur_obs, actions, task_z)
        pred_q2 = self.qf2(cur_obs, actions, task_z)
        qf1_loss = F.mse_loss(pred_q1, target_q)
        qf2_loss = F.mse_loss(pred_q2, target_q)
        qf_loss = qf1_loss + qf2_loss
        self.qf_optimizer.zero_grad()
        qf_loss.backward()

        # Update two Q-network parameters and encoder network parameters
        self.qf_optimizer.step()
        self.encoder_optimizer.step()

        # Policy loss
        inputs = torch.cat([cur_obs, task_z.detach()], dim=-1)
        policy, log_policy = self.policy(inputs)
        min_q = torch.min(
            self.qf1(cur_obs, policy, task_z.detach()),
            self.qf2(cur_obs, policy, task_z.detach()),
        )
        policy_loss = (self.alpha * log_policy - min_q).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Temperature parameter alpha update
        alpha_loss = -(self.log_alpha * (log_policy + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Polyak averaging for target parameter
        self.soft_target_update(self.qf1, self.target_qf1)
        self.soft_target_update(self.qf2, self.target_qf2)
        return dict(
            policy_loss=policy_loss.item(),
            qf1_loss=qf1_loss.item(),
            qf2_loss=qf2_loss.item(),
            encoder_loss=encoder_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=self.alpha.item(),
            z_mean=self.encoder.z_mean.detach().cpu().numpy().mean().item(),
            z_var=self.encoder.z_var.detach().cpu().numpy().mean().item(),
        )
