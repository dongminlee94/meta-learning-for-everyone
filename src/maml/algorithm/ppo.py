"""
Proximal Policy Optimization algorithm implementation for training
"""

import torch
import torch.nn.functional as F

from src.maml.algorithm.networks import Model


class PPO:  # pylint: disable=too-many-instance-attributes
    """Proximal Policy Optimization class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        observ_dim,
        action_dim,
        hidden_dim,
        device,
        **config,
    ):

        self.device = device
        self.clip_param = config["clip_param"]

        # Instantiate networks
        self.model = Model(
            observ_dim,
            action_dim,
            hidden_dim,
        ).to(device)

    @staticmethod
    def get_action(model, obs, device):
        """Get an action from the policy"""
        action, log_prob = model(torch.Tensor(obs).to(device))
        if log_prob:
            log_prob = log_prob.detach().cpu().numpy()
        return action.detach().cpu().numpy(), log_prob

    def get_value(self, obs):
        """Get an value from the value network"""
        value = self.model.vf(torch.Tensor(obs).to(self.device))
        return value.detach().cpu().numpy()

    def compute_losses(self, model, batch):
        """Compute loesses according to method of PPO algorithm"""
        obs_batch = batch["obs"]
        action_batch = batch["actions"]
        return_batch = batch["returns"]
        advant_batch = batch["advants"]
        log_prob_batch = batch["log_probs"]

        # Compute Value function loss
        value_batch = self.model.vf(obs_batch)
        value_loss = F.mse_loss(value_batch, return_batch)

        # Compute Policy loss
        new_log_prob_batch = model.policy.get_log_prob(obs_batch, action_batch)
        ratio = torch.exp(new_log_prob_batch - log_prob_batch)

        policy_loss = ratio * advant_batch
        clipped_loss = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advant_batch

        policy_loss = -torch.min(policy_loss, clipped_loss).mean()

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss

        return total_loss, policy_loss, value_loss
