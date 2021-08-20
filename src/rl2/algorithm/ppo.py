"""
Proximal Policy Optimization algorithm code for training when meta-train
"""

import torch
import torch.nn.functional as F
import torch.optim as optim

from src.rl2.algorithm.networks import GRU, GaussianGRU


class PPO:  # pylint: disable=too-many-instance-attributes
    """Proximal Policy Optimization class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        trans_dim,
        action_dim,
        hidden_dim,
        device,
        **config,
    ):

        self.device = device
        self.num_epochs = config["num_epochs"]
        self.mini_batch_size = config["mini_batch_size"]
        self.clip_param = config["clip_param"]

        # Instantiate networks
        self.policy = GaussianGRU(
            input_dim=trans_dim,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(device)
        self.vf = GRU(
            input_dim=trans_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
        ).to(device)

        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.vf.parameters()),
            lr=config["learning_rate"],
        )

        self.net_dict = {
            "policy": self.policy,
            "vf": self.vf,
        }

    def get_action(self, trans, hidden):
        """Get an action from the policy network"""
        action, log_prob, hidden = self.policy(
            torch.Tensor(trans).to(self.device), torch.Tensor(hidden).to(self.device)
        )
        if log_prob:
            log_prob = log_prob.detach().cpu().numpy()
        return action.detach().cpu().numpy(), log_prob, hidden.detach().cpu().numpy()

    def get_value(self, trans, hidden):
        """Get an value from the value network"""
        value, hidden = self.vf(
            torch.Tensor(trans).to(self.device), torch.Tensor(hidden).to(self.device)
        )
        return value.detach().cpu().numpy(), hidden.detach().cpu().numpy()

    def train_model(self, batch_size, batch):  # pylint: disable=too-many-locals
        """Train models according to training method of PPO algorithm"""
        trans = batch["trans"]
        pi_hiddens = batch["pi_hiddens"]
        v_hiddens = batch["v_hiddens"]
        actions = batch["actions"]
        returns = batch["returns"]
        advants = batch["advants"]
        log_probs = batch["log_probs"]

        num_mini_batch = int(batch_size / self.mini_batch_size)

        trans_batches = torch.chunk(trans, num_mini_batch)
        pi_hidden_batches = torch.chunk(pi_hiddens, num_mini_batch)
        v_hidden_batches = torch.chunk(v_hiddens, num_mini_batch)
        action_batches = torch.chunk(actions, num_mini_batch)
        return_batches = torch.chunk(returns, num_mini_batch)
        advant_batches = torch.chunk(advants, num_mini_batch)
        log_prob_batches = torch.chunk(log_probs, num_mini_batch)

        total_loss_sum = 0
        policy_loss_sum = 0
        value_loss_sum = 0

        for _ in range(self.num_epochs):
            total_loss_mini_batch_sum = 0
            policy_loss_mini_batch_sum = 0
            value_loss_mini_batch_sum = 0

            for (
                trans_batch,
                pi_hidden_batch,
                v_hidden_batch,
                action_batch,
                return_batch,
                advant_batch,
                log_prob_batch,
            ) in zip(
                trans_batches,
                pi_hidden_batches,
                v_hidden_batches,
                action_batches,
                return_batches,
                advant_batches,
                log_prob_batches,
            ):
                # Value function loss
                value_batch, _ = self.vf(trans_batch, v_hidden_batch)
                value_loss = F.mse_loss(value_batch.view(-1, 1), return_batch)

                # Policy loss
                new_log_prob_batch = self.policy.get_log_prob(
                    trans_batch, pi_hidden_batch, action_batch
                )
                ratio = torch.exp(new_log_prob_batch.view(-1, 1) - log_prob_batch)

                policy_loss = ratio * advant_batch
                clipped_loss = (
                    torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advant_batch
                )

                policy_loss = -torch.min(policy_loss, clipped_loss).mean()

                # Total loss
                total_loss = policy_loss + 0.5 * value_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                total_loss_mini_batch_sum += total_loss
                policy_loss_mini_batch_sum += policy_loss
                value_loss_mini_batch_sum += value_loss

            total_loss_sum += total_loss_mini_batch_sum / num_mini_batch
            policy_loss_sum += policy_loss_mini_batch_sum / num_mini_batch
            value_loss_sum += value_loss_mini_batch_sum / num_mini_batch

        total_loss_mean = total_loss_sum / self.num_epochs
        policy_loss_mean = policy_loss_sum / self.num_epochs
        value_loss_mean = value_loss_sum / self.num_epochs

        return dict(
            total_loss=total_loss_mean.item(),
            policy_loss=policy_loss_mean.item(),
            value_loss=value_loss_mean.item(),
        )

    def save(self, path, net_dict=None):
        """Save data related to models in path"""
        if net_dict is None:
            net_dict = self.net_dict

        state_dict = {name: net.state_dict() for name, net in net_dict.items()}
        state_dict["alpha"] = self.log_alpha
        torch.save(state_dict, path)

    def load(self, path, net_dict=None):
        """Load data stored as check point in models"""
        if net_dict is None:
            net_dict = self.net_dict

        checkpoint = torch.load(path)
        for name, net in net_dict.items():
            if name == "alpha":
                self.log_alpha.load(net)
            else:
                net.load_state_dict(checkpoint[name])
