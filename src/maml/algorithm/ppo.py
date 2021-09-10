"""
Proximal Policy Optimization algorithm implementation for training
"""

import torch

from src.maml.algorithm.buffer import Buffer
from src.maml.algorithm.networks import LinearValue, TanhGaussianPolicy


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

        self.policy = TanhGaussianPolicy(
            input_dim=observ_dim, output_dim=action_dim, hidden_dim=hidden_dim
        ).to(device)

        self.baseline = LinearValue(
            input_dim=observ_dim,
        ).to(device)

    @staticmethod
    def get_action(policy, obs, device):
        """Get an action from the policy"""
        action, log_prob = policy(torch.Tensor(obs).to(device))
        if log_prob:
            log_prob = log_prob.detach().cpu().numpy()
        return action.detach().cpu().numpy(), log_prob

    def get_value(self, obs):
        """Get an value from the value network"""
        value = self.baseline(torch.Tensor(obs).to(self.device))
        return value.detach().cpu().numpy()

    # pylint: disable=too-many-locals
    def compute_losses(self, policy, batch, a2c_loss=False, clip_loss=True):
        """Compute policy losses according to PPO algorithm"""
        obs_batch = batch["obs"]
        action_batch = batch["actions"]
        return_batch = batch["returns"]
        reward_batch = batch["rewards"]
        done_batch = batch["dones"]
        log_prob_batch = batch["log_probs"]

        # Compute advantage with standard linear feature baseline (Daun el al. 2016)
        self.baseline.fit(obs_batch, return_batch)
        value_batch = self.baseline(obs_batch).detach()
        advant_batch = Buffer.compute_gae(reward_batch, done_batch, value_batch).to(self.device)

        # Compute Policy loss
        new_log_prob_batch = policy.get_log_prob(obs_batch, action_batch)
        if a2c_loss:
            policy_loss = new_log_prob_batch * advant_batch
            policy_loss = -policy_loss.mean()
        else:
            ratio = torch.exp(new_log_prob_batch - log_prob_batch)
            surr_loss = ratio * advant_batch
            if clip_loss:
                clipped_loss = (
                    torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advant_batch
                )
                policy_loss = -torch.min(surr_loss, clipped_loss).mean()
            else:
                policy_loss = -surr_loss.mean()

        # print(policy_loss)
        return policy_loss
