from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from meta_rl.rl2.algorithm.networks import GRU, GaussianGRU


class PPO:
    def __init__(
        self,
        trans_dim: int,
        action_dim: int,
        hidden_dim: int,
        device: torch.device,
        **config,
    ) -> None:
        self.device = device
        self.num_epochs = config["num_epochs"]
        self.mini_batch_size = config["mini_batch_size"]
        self.clip_param = config["clip_param"]

        # 네트워크 초기화
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

    def get_action(
        self,
        trans: np.ndarray,
        hidden: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 주어진 관측 상태와 은닉 상태에 따른 현재 정책의 action 얻기
        action, log_prob, hidden_out = self.policy(
            torch.Tensor(trans).to(self.device),
            torch.Tensor(hidden).to(self.device),
        )
        return (
            action.detach().cpu().numpy(),
            log_prob.detach().cpu().numpy(),
            hidden_out.detach().cpu().numpy(),
        )

    def get_value(self, trans: np.ndarray, hidden: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 상태 가치 함수 추론
        value, hidden_out = self.vf(
            torch.Tensor(trans).to(self.device),
            torch.Tensor(hidden).to(self.device),
        )
        return value.detach().cpu().numpy(), hidden_out.detach().cpu().numpy()

    def train_model(self, batch_size: int, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        # PPO 알고리즘에 따른 네트워크 학습
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

        sum_total_loss: float = 0
        sum_policy_loss: float = 0
        sum_value_loss: float = 0

        for _ in range(self.num_epochs):
            sum_total_loss_mini_batch = 0
            sum_policy_loss_mini_batch = 0
            sum_value_loss_mini_batch = 0

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
                # 상태 가치 함수 손실 계산
                value_batch, _ = self.vf(trans_batch, v_hidden_batch)
                value_loss = F.mse_loss(value_batch.view(-1, 1), return_batch)

                # 정책 손실 계산
                new_log_prob_batch = self.policy.get_log_prob(
                    trans_batch,
                    pi_hidden_batch,
                    action_batch,
                )
                ratio = torch.exp(new_log_prob_batch.view(-1, 1) - log_prob_batch)

                policy_loss = ratio * advant_batch
                clipped_loss = (
                    torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advant_batch
                )

                policy_loss = -torch.min(policy_loss, clipped_loss).mean()

                # 손실 합 계산
                total_loss = policy_loss + 0.5 * value_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                sum_total_loss_mini_batch += total_loss
                sum_policy_loss_mini_batch += policy_loss
                sum_value_loss_mini_batch += value_loss

            sum_total_loss += sum_total_loss_mini_batch / num_mini_batch
            sum_policy_loss += sum_policy_loss_mini_batch / num_mini_batch
            sum_value_loss += sum_value_loss_mini_batch / num_mini_batch

        mean_total_loss = sum_total_loss / self.num_epochs
        mean_policy_loss = sum_policy_loss / self.num_epochs
        mean_value_loss = sum_value_loss / self.num_epochs
        return dict(
            total_loss=mean_total_loss.item(),
            policy_loss=mean_policy_loss.item(),
            value_loss=mean_value_loss.item(),
        )
