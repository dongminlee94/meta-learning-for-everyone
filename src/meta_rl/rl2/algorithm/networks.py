from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class GRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_activation: torch.nn.functional = F.relu,
        init_w: float = 3e-3,
    ) -> None:
        super().__init__()
        self.hidden_activation = hidden_activation

        # GRU 레이어 생성
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim)

        # 출력 레이어 생성
        self.last_fc_layer = nn.Linear(hidden_dim, output_dim)
        self.last_fc_layer.weight.data.uniform_(-init_w, init_w)
        self.last_fc_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, h = self.gru(x.unsqueeze(0), h.unsqueeze(0))
        x = self.hidden_activation(x)
        x = self.last_fc_layer(x)
        return x, h


class GaussianGRU(GRU):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        is_deterministic: bool = False,
        init_w: float = 1e-3,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            init_w=init_w,
        )

        self.log_std = -0.5 * np.ones(output_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.Tensor(self.log_std))
        self.is_deterministic = is_deterministic

    def get_normal_dist(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.distributions.normal.Normal, torch.Tensor, torch.Tensor]:
        mean, hidden = super().forward(x, h)
        std = torch.exp(self.log_std)
        return Normal(mean, std), mean, hidden

    def get_log_prob(
        self,
        trans: torch.Tensor,
        hidden: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        normal, _, _ = self.get_normal_dist(trans, hidden)
        return normal.log_prob(action).sum(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normal, mean, hidden = self.get_normal_dist(x, h)
        if self.is_deterministic:
            action = mean
            log_prob = torch.zeros(1)
        else:
            action = normal.sample()
            log_prob = normal.log_prob(action).sum(dim=-1)
        action = action.view(-1)
        return action, log_prob, hidden
