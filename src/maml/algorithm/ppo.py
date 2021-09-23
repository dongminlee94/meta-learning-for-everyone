"""
Proximal Policy Optimization algorithm implementation for training
"""

import torch

from src.maml.algorithm.networks import MLP, GaussianPolicy


class PPO:  # pylint: disable=too-many-instance-attributes
    """Proximal Policy Optimization class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        observ_dim,
        action_dim,
        policy_hidden_dim,
        vf_hidden_dim,
        device,
        gamma=0.99,
        lamda=0.97,
        **config,
    ):

        self.device = device
        self.gamma = gamma
        self.lamda = lamda
        self.clip_param = config["clip_param"]

        self.policy = GaussianPolicy(
            input_dim=observ_dim, output_dim=action_dim, hidden_dim=policy_hidden_dim
        ).to(device)

        self.vf = MLP(input_dim=observ_dim, output_dim=1, hidden_dim=vf_hidden_dim).to(device)

        self.vf_optimizer = torch.optim.Adam(
            list(self.vf.parameters()),
            lr=config["vf_learning_rate"],
        )
        self.initial_vf_state = self.vf.state_dict()

    def reset_vf(self):
        """Reset value fuction"""
        self.vf.load_state_dict(self.initial_vf_state)

    # pylint: disable=too-many-locals
    def compute_loss(self, new_policy, batch, meta_loss=False):
        """Compute policy losses according to PPO algorithm"""
        obs_batch = batch["obs"]
        action_batch = batch["actions"]
        advant_batch = batch["advants"]

        # Policy loss
        if meta_loss:
            old_log_prob_batch = batch["log_probs"]
        else:
            old_log_prob_batch = self.policy.get_log_prob(obs_batch, action_batch)
        new_log_prob_batch = new_policy.get_log_prob(obs_batch, action_batch)

        ratio = torch.exp(new_log_prob_batch.view(-1, 1) - old_log_prob_batch)
        policy_loss = ratio * advant_batch
        clipped_loss = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advant_batch
        policy_loss = -torch.min(policy_loss, clipped_loss).mean()

        return policy_loss
