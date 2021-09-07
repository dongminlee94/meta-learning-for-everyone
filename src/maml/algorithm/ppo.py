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
        gamma=0.99,
        lamda=0.97,
        **config,
    ):

        self.device = device
        self.gamma = gamma
        self.lamda = lamda
        self.clip_param = config["clip_param"]
        self.vf_clip_param = config["vf_clip_param"]

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

    # pylint: disable=too-many-locals
    def compute_losses(self, model, batch, clip_loss=True):
        """Compute model losses according to PPO algorithm"""
        obs_batch = batch["obs"]
        action_batch = batch["actions"]
        return_batch = batch["returns"]
        advant_batch = batch["advants"]
        log_prob_batch = batch["log_probs"]

        # Compute GAE Value function loss
        value_batch = self.model.vf(obs_batch)
        value_loss = F.mse_loss(value_batch, return_batch)

        # Compute Policy loss
        new_log_prob_batch = model.policy.get_log_prob(obs_batch, action_batch)
        ratio = torch.exp(new_log_prob_batch - log_prob_batch)

        surr_loss = ratio * advant_batch
        if clip_loss:
            clipped_loss = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advant_batch
            surr_loss = torch.min(surr_loss, clipped_loss).mean()
        else:
            surr_loss = surr_loss.mean()

        # Total loss
        total_loss = -surr_loss + 0.5 * value_loss

        # print(policy_loss)
        return total_loss, surr_loss, value_loss
