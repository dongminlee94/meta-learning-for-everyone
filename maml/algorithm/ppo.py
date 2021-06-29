"""
Proximal Policy Optimization algorithm implementation for training
"""

import torch
import torch.nn.functional as F
import torch.optim as optim

from maml.algorithm.utils.networks import MLP, GaussianPolicy


class PPO:  # pylint: disable=too-many-instance-attributes
    """Proximal Policy Optimization class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        observ_dim,
        action_dim,
        hidden_units,
        env_target,
        device,
        **config,
    ):

        self.device = device
        self.num_train_iters = config["num_train_iters"]
        self.train_batch_size = config["train_batch_size"]
        self.train_minibatch_size = config["train_minibatch_size"]
        self.clip_param = config["clip_param"]

        # Instantiate networks
        self.policy = GaussianPolicy(
            input_dim=observ_dim,
            output_dim=action_dim,
            hidden_units=hidden_units,
            env_target=env_target,
        ).to(device)
        self.vf = MLP(
            input_dim=observ_dim,
            output_dim=1,
            hidden_units=hidden_units,
        ).to(device)

        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.vf.parameters()),
            lr=config["vf_lr"],
        )

        self.net_dict = {
            "policy": self.policy,
            "vf": self.vf,
        }

    def get_action(self, obs):
        """Get an action from the policy"""
        action, _ = self.policy(torch.Tensor(obs).to(self.device))
        return action.detach().cpu().numpy()

    # pylint: disable=too-many-locals
    def train(self, batch):
        """Train models according to training method of PPO algorithm"""
        obs = batch["obs"]
        actions = batch["actions"]
        returns = batch["returns"]
        advants = batch["advants"]
        log_probs = batch["log_probs"]

        num_mini_batch = int(self.train_batch_size / self.train_minibatch_size)

        obs_batches = torch.chunk(obs, num_mini_batch)
        action_batches = torch.chunk(actions, num_mini_batch)
        return_batches = torch.chunk(returns, num_mini_batch)
        advant_batches = torch.chunk(advants, num_mini_batch)
        log_prob_batches = torch.chunk(log_probs, num_mini_batch)

        for _ in range(self.num_train_iters):
            for (
                obs_batch,
                action_batch,
                return_batch,
                advant_batch,
                log_prob_batch,
            ) in zip(
                obs_batches,
                action_batches,
                return_batches,
                advant_batches,
                log_prob_batches,
            ):
                # Value function loss
                value_batch = self.vf(obs_batch)
                value_loss = F.mse_loss(value_batch, return_batch)

                # Policy loss
                new_log_prob_batch = self.policy.get_log_prob(obs_batch, action_batch)
                ratio = torch.exp(new_log_prob_batch - log_prob_batch)

                policy_loss = ratio * advant_batch
                clipped_loss = (
                    torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
                    * advant_batch
                )

                policy_loss = -torch.min(policy_loss, clipped_loss).mean()

                # Total loss
                total_loss = policy_loss + 0.5 * value_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
