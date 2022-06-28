from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_activation: torch.nn.functional = F.relu,
        init_w: float = 3e-3,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_activation = hidden_activation

        # Fully connected 레이어 생성
        self.fc_layers = nn.ModuleList()
        self.hidden_layers = [hidden_dim] * 3
        in_layer = input_dim

        for i, hidden_layer in enumerate(self.hidden_layers):
            fc_layer = nn.Linear(in_layer, hidden_layer)
            in_layer = hidden_layer
            self.__setattr__("fc_layer{}".format(i), fc_layer)
            self.fc_layers.append(fc_layer)

        # 출력 레이어 생성
        self.last_fc_layer = nn.Linear(hidden_dim, output_dim)
        self.last_fc_layer.weight.data.uniform_(-init_w, init_w)
        self.last_fc_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc_layer in self.fc_layers:
            x = self.hidden_activation(fc_layer(x))
        x = self.last_fc_layer(x)
        return x


class FlattenMLP(MLP):
    def forward(self, *x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = torch.cat(x, dim=-1)
        return super().forward(x)


class MLPEncoder(FlattenMLP):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int,
        hidden_dim: int,
        device: torch.device,
    ) -> None:
        super().__init__(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)

        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.device = device

        self.z_mean = None
        self.z_var = None
        self.task_z = None
        self.clear_z()

    def clear_z(self, num_tasks: int = 1) -> None:
        # q(z|c)를 prior r(z)로 초기화
        self.z_mean = torch.zeros(num_tasks, self.latent_dim).to(self.device)
        self.z_var = torch.ones(num_tasks, self.latent_dim).to(self.device)

        # Prior r(z)에서 새로운 z를 생성
        self.sample_z()

        # 지금까지 모은 context 초기화
        self.context = None

    def sample_z(self) -> None:
        # z ~ r(z) 또는 z ~ q(z|c) 생성
        dists = []
        for mean, var in zip(torch.unbind(self.z_mean), torch.unbind(self.z_var)):
            dist = torch.distributions.Normal(mean, torch.sqrt(var))
            dists.append(dist)
        sampled_z = [dist.rsample() for dist in dists]
        self.task_z = torch.stack(sampled_z).to(self.device)

    @classmethod
    def product_of_gaussians(
        cls,
        mean: torch.Tensor,
        var: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Product of gaussians (POG)의 평균과 표준편차 계산
        var = torch.clamp(var, min=1e-7)
        pog_var = 1.0 / torch.sum(torch.reciprocal(var), dim=0)
        pog_mean = pog_var * torch.sum(mean / var, dim=0)
        return pog_mean, pog_var

    def infer_posterior(self, context: torch.Tensor) -> None:
        params = self.forward(context)
        params = params.view(context.size(0), -1, self.output_dim).to(self.device)

        # q(z|c)의 평균과 분산 계산
        z_mean = torch.unbind(params[..., : self.latent_dim])
        z_var = torch.unbind(F.softplus(params[..., self.latent_dim :]))
        z_params = [self.product_of_gaussians(mu, var) for mu, var in zip(z_mean, z_var)]

        self.z_mean = torch.stack([z_param[0] for z_param in z_params]).to(self.device)
        self.z_var = torch.stack([z_param[1] for z_param in z_params]).to(self.device)
        self.sample_z()

    def compute_kl_div(self) -> torch.Tensor:
        # KL( q(z|c) || r(z) ) 계산
        prior = torch.distributions.Normal(
            torch.zeros(self.latent_dim).to(self.device),
            torch.ones(self.latent_dim).to(self.device),
        )

        posteriors = []
        for mean, var in zip(torch.unbind(self.z_mean), torch.unbind(self.z_var)):
            dist = torch.distributions.Normal(mean, torch.sqrt(var))
            posteriors.append(dist)

        kl_div = [torch.distributions.kl.kl_divergence(posterior, prior) for posterior in posteriors]
        kl_div = torch.stack(kl_div).sum().to(self.device)
        return kl_div


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TanhGaussianPolicy(MLP):
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

        self.is_deterministic = is_deterministic
        self.last_fc_log_std = nn.Linear(hidden_dim, output_dim)
        self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for fc_layer in self.fc_layers:
            x = self.hidden_activation(fc_layer(x))

        mean = self.last_fc_layer(x)
        log_std = self.last_fc_log_std(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        if self.is_deterministic:
            action = mean
            log_prob = None
        else:
            normal = Normal(mean, std)
            action = normal.rsample()

            # 가우시안 분포에 대한 로그 확률값 계산
            log_prob = normal.log_prob(action)
            log_prob -= 2 * (np.log(2) - action - F.softplus(-2 * action))
            log_prob = log_prob.sum(-1, keepdim=True)

        action = torch.tanh(action)
        return action, log_prob
