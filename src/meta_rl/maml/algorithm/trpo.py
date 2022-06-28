from copy import deepcopy
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from meta_rl.maml.algorithm.networks import MLP, GaussianPolicy


class TRPO:
    def __init__(
        self,
        observ_dim,
        action_dim,
        policy_hidden_dim,
        vf_hidden_dim,
        device,
        **config,
    ):
        self.device = device
        self.gamma = config["gamma"]
        self.lamda = config["lamda"]

        self.policy = GaussianPolicy(
            input_dim=observ_dim,
            output_dim=action_dim,
            hidden_dim=policy_hidden_dim,
        ).to(device)

        # 이전 정책 및 상태 가치 함수 초기화
        self.old_policy = deepcopy(self.policy)
        self.vf = MLP(input_dim=observ_dim, output_dim=1, hidden_dim=vf_hidden_dim).to(device)

        self.vf_optimizer = optim.Adam(
            list(self.vf.parameters()),
            lr=config["vf_learning_rate"],
        )
        self.vf_learning_iters = config["vf_learning_iters"]
        self.initial_vf_params = deepcopy(self.vf.state_dict())

    @classmethod
    def flat_grad(cls, gradients: Tuple[torch.Tensor, ...], is_hessian: bool = False) -> torch.Tensor:
        # 그래디언트 벡터화
        if is_hessian:
            grads = [g.contiguous() for g in gradients]
            return parameters_to_vector(grads)
        grads = [g.detach() for g in gradients]
        return parameters_to_vector(grads)

    @classmethod
    def update_model(cls, module: nn.Module, new_params: Dict[str, torch.nn.parameter.Parameter]):
        # 현재 모델의 파라미터를 새로운 파라미터로 대체
        # [출처] (https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/_functions.py)
        named_modules = dict(module.named_modules())

        def update(m, name, param):
            del m._parameters[name]
            setattr(m, name, param)
            m._parameters[name] = param

        for name, new_param in new_params.items():
            if "." in name:
                module_name, param_name = tuple(name.rsplit(".", 1))
                if module_name in named_modules:
                    update(named_modules[module_name], param_name, new_param)
            else:
                update(module, name, new_param)

        for param in module.parameters():
            param.grad = None

    @classmethod
    def hessian_vector_product(
        cls,
        kl: torch.Tensor,
        parameters: torch.nn.parameter.Parameter,
        hvp_reg_coeff: float = 1e-5,
    ) -> Callable:
        # 벡터 입력에 대해 Hessian-vector product를 계산하는 callable 객체 반환
        parameters = list(parameters)
        kl_grad = torch.autograd.grad(kl, parameters, create_graph=True)
        kl_grad = cls.flat_grad(kl_grad, is_hessian=True)

        def hvp(vector):
            # Hessian-vector product 계산
            kl_grad_prod = torch.dot(kl_grad, vector)
            kl_hessian_prod = torch.autograd.grad(kl_grad_prod, parameters, retain_graph=True)
            kl_hessian_prod = cls.flat_grad(kl_hessian_prod, is_hessian=True)
            kl_hessian_prod = kl_hessian_prod + hvp_reg_coeff * vector
            return kl_hessian_prod

        return hvp

    @classmethod
    def conjugate_gradient(
        cls,
        fnc_Ax: Callable,
        b: torch.Tensor,
        num_iters: int = 10,
        residual_tol: float = 1e-10,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        # 켤레 기울기 계산
        x = torch.zeros_like(b)
        r = b
        p = r
        rdotr_old = torch.dot(r, r)

        for _ in range(num_iters):
            Ap = fnc_Ax(p)
            alpha = rdotr_old / (torch.dot(p, Ap) + eps)
            x += alpha * p
            r -= alpha * Ap
            rdotr_new = torch.dot(r, r)
            p = r + (rdotr_new / rdotr_old) * p
            rdotr_old = rdotr_new

            if rdotr_new.item() < residual_tol:
                break
        return x

    def compute_descent_step(
        self,
        Hvp: Callable,
        search_dir: torch.Tensor,
        max_kl: float,
    ) -> torch.nn.parameter.Parameter:
        # Line search를 시작하기 위한 첫 경사하강 스텝 계산
        sHs = torch.dot(search_dir, Hvp(search_dir))
        lagrange_multiplier = torch.sqrt(sHs / (2 * max_kl))

        step = search_dir / lagrange_multiplier
        step_param = [torch.zeros_like(params.data) for params in self.policy.parameters()]

        vector_to_parameters(step, step_param)
        return step_param

    def infer_baselines(self, batch: Dict[str, torch.Tensor]):
        # 상태 가치 함수 학습 및 baseline 추론
        obs_batch = batch["obs"]
        rewards_batch = batch["rewards"]
        dones_batch = batch["dones"]
        returns_batch = torch.zeros_like(rewards_batch)
        running_return = 0

        for t in reversed(range(len(rewards_batch))):
            # 보상합 계산
            running_return = rewards_batch[t] + self.gamma * (1 - dones_batch[t]) * running_return
            returns_batch[t] = running_return

        # 상태 가치 함수 초기화
        self.vf.load_state_dict(self.initial_vf_params)

        # 상태 가치 함수 업데이트
        for _ in range(self.vf_learning_iters):
            self.vf_optimizer.zero_grad()
            value_batch = self.vf(obs_batch.to(self.device))
            value_loss = F.mse_loss(value_batch.view(-1, 1), returns_batch.to(self.device))
            value_loss.backward()
            self.vf_optimizer.step()

        # 상태 가치 함수로부터 baseline 추론
        with torch.no_grad():
            baselines = self.vf(obs_batch)
        return baselines.cpu().numpy()

    def compute_gae(self, batch: Dict[str, torch.Tensor]):
        # 보상합 및 GAE 계산
        rewards_batch = batch["rewards"]
        dones_batch = batch["dones"]
        values_batch = batch["baselines"]
        advants_batch = torch.zeros_like(rewards_batch)
        prev_value = 0
        running_advant = 0

        for t in reversed(range(len(rewards_batch))):
            # GAE 계산
            running_tderror = (
                rewards_batch[t] + self.gamma * (1 - dones_batch[t]) * prev_value - values_batch[t]
            )
            running_advant = (
                running_tderror + self.gamma * self.lamda * (1 - dones_batch[t]) * running_advant
            )
            advants_batch[t] = running_advant
            prev_value = values_batch[t]

        # 어드밴티지 정규화
        advants_batch = (advants_batch - advants_batch.mean()) / (advants_batch.std() + 1e-8)
        return advants_batch

    def kl_divergence(self, batch: Dict[str, torch.Tensor]):
        # 이전 정책과 업데이트된 정책 사이의 KL divergence 계산
        obs_batch = batch["obs"]

        with torch.no_grad():
            old_dist, _ = self.old_policy.get_normal_dist(obs_batch)

        new_dist, _ = self.policy.get_normal_dist(obs_batch)
        kl_constraint = torch.distributions.kl.kl_divergence(old_dist, new_dist)

        return kl_constraint.mean()

    def compute_policy_entropy(self, batch: Dict[str, torch.Tensor]):
        # 정책 엔트로피 계산
        obs_batch = batch["obs"]

        with torch.no_grad():
            dist, _ = self.policy.get_normal_dist(obs_batch)
            policy_entropy = dist.entropy().sum(dim=-1)

        return policy_entropy.mean()

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # 주어진 관측 상태에 따른 현재 정책의 action 얻기
        action, _ = self.policy(torch.Tensor(obs).to(self.device))

        return action.detach().cpu().numpy()

    def policy_loss(self, batch: Dict[str, torch.Tensor], is_meta_loss: bool = False):
        # TRPO 및 Policy Gradient 알고리즘의 정책 손실 계산
        obs_batch = batch["obs"]
        action_batch = batch["actions"]
        advant_batch = self.compute_gae(batch).detach()

        if is_meta_loss:
            # Surrogate 손실
            old_log_prob_batch = self.old_policy.get_log_prob(obs_batch, action_batch).view(-1, 1)
            new_log_prob_batch = self.policy.get_log_prob(obs_batch, action_batch).view(-1, 1)
            ratio = torch.exp(new_log_prob_batch - old_log_prob_batch.detach())
            surrogate_loss = ratio * advant_batch
            loss = -surrogate_loss.mean()
        else:
            # 액터-크리틱 손실
            log_prob_batch = self.policy.get_log_prob(obs_batch, action_batch).view(-1, 1)
            loss = -torch.mean(log_prob_batch * advant_batch)
        return loss
