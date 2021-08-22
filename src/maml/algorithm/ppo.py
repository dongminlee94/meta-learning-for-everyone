"""
Proximal Policy Optimization algorithm implementation for training
"""

import torch
import torch.nn.functional as F
import torch.optim as optim

from maml.algorithm.networks import MLP, GaussianPolicy


class PPO:  # pylint: disable=too-many-instance-attributes
    """Proximal Policy Optimization class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        observ_dim,
        action_dim,
        hidden_dims,
        env_target,
        device,
        **config,
    ):

        self.device = device
        self.clip_param = config["clip_param"]

        # Instantiate networks
        self.policy = GaussianPolicy(
            input_dim=observ_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            env_target=env_target,
        ).to(device)
        self.vf = MLP(
            input_dim=observ_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
        ).to(device)
        # self.vf.to(device)
        self.vf_optimizer = optim.Adam(
            list(self.vf.parameters()),
            lr=config["vf_learning_rate"],
        )

    def get_value(self, obs):
        """Get an value from the value network"""
        value = self.vf(torch.Tensor(obs).to(self.device))
        return value.detach().cpu().numpy()

    def compute_losses(self, policy, batch):
        """Compute loesses according to method of PPO algorithm"""
        obs_batch = batch["obs"]
        action_batch = batch["actions"]
        return_batch = batch["returns"]
        advant_batch = batch["advants"]
        log_prob_batch = batch["log_probs"]

        # Update Value function
        value_batch = self.vf(obs_batch)
        value_loss = F.mse_loss(value_batch, return_batch)

        self.vf_optimizer.zero_grad()
        value_loss.backward()
        self.vf_optimizer.step()

        # Compute Policy loss
        new_log_prob_batch = policy.get_log_prob(obs_batch, action_batch)
        ratio = torch.exp(new_log_prob_batch - log_prob_batch)

        policy_loss = ratio * advant_batch
        clipped_loss = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advant_batch
        return -torch.min(policy_loss, clipped_loss).mean()
