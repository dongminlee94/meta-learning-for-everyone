"""
Proximal Policy Optimization algorithm implementation for training
"""

import torch
import torch.nn.functional as F

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

    def compute_value(self, batch):
        """Train value network and infer value for a current batch"""
        obs_batch = batch["obs"]
        rewards_batch = batch["rewards"]
        dones_batch = batch["dones"]
        returns_batch = torch.zeros_like(rewards_batch)
        running_return = 0

        for t in reversed(range(len(rewards_batch))):
            # Compute return
            running_return = rewards_batch[t] + self.gamma * (1 - dones_batch[t]) * running_return
            returns_batch[t] = running_return

        # Reset the value fuction to its initial state
        self.reset_vf()
        self.vf_optimizer.zero_grad()

        # Value function loss
        value_batch = self.vf(obs_batch.to(self.device))
        value_loss = F.mse_loss(value_batch.view(-1, 1), returns_batch.to(self.device))

        value_loss.backward()
        self.vf_optimizer.step()

        return self.vf(obs_batch).detach().cpu().numpy()

    def compute_gae(self, batch):
        """Compute return and GAE"""
        rewards_batch = batch["rewards"]
        dones_batch = batch["dones"]
        advants_batch = torch.zeros_like(rewards_batch)
        prev_value = 0
        running_advant = 0

        values_batch = self.compute_value(batch)

        for t in reversed(range(len(rewards_batch))):
            # Compute GAE
            running_tderror = (
                rewards_batch[t] + self.gamma * (1 - dones_batch[t]) * prev_value - values_batch[t]
            )
            running_advant = (
                running_tderror + self.gamma * self.lamda * (1 - dones_batch[t]) * running_advant
            )
            advants_batch[t] = running_advant
            prev_value = values_batch[t]

        # Normalize advantage
        advants_batch = (advants_batch - advants_batch.mean()) / advants_batch.std()

        return advants_batch

    # pylint: disable=too-many-locals
    def compute_loss(self, new_policy, batch, is_metaloss=False):
        """Compute policy losses according to PPO algorithm"""
        obs_batch = batch["obs"]
        action_batch = batch["actions"]

        advant_batch = self.compute_gae(batch)

        # Policy loss
        if is_metaloss:
            # Outer-loop
            # Set the adapted policy (theta') as an old policy for a surrogate advantage
            old_log_prob_batch = batch["log_probs"]
        else:
            # Inner-loop
            # Set the meta policy (theta) as an old policy for a surrogate adavantage
            old_log_prob_batch = self.policy.get_log_prob(obs_batch, action_batch)
        new_log_prob_batch = new_policy.get_log_prob(obs_batch, action_batch)

        ratio = torch.exp(new_log_prob_batch.view(-1, 1) - old_log_prob_batch)
        policy_loss = ratio * advant_batch
        clipped_loss = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advant_batch
        policy_loss = -torch.min(policy_loss, clipped_loss).mean()

        return policy_loss
