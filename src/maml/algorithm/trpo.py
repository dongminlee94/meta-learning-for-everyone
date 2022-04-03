"""
Trust Region Policy Optimization algorithm implementation for training
"""


from copy import deepcopy
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from src.maml.algorithm.networks import MLP, GaussianPolicy


class TRPO:  # pylint: disable=too-many-instance-attributes
    """Policy gradient based agent"""

    def __init__(  # pylint: disable=too-many-arguments
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
            input_dim=observ_dim, output_dim=action_dim, hidden_dim=policy_hidden_dim
        ).to(device)
        self.old_policy = deepcopy(self.policy)
        self.vf = MLP(input_dim=observ_dim, output_dim=1, hidden_dim=vf_hidden_dim).to(device)

        self.vf_optimizer = optim.Adam(
            list(self.vf.parameters()),
            lr=config["vf_learning_rate"],
        )
        self.vf_learning_iters = config["vf_learning_iters"]
        self.initial_vf_params = deepcopy(self.vf.state_dict())

    # methods for TRPO
    @classmethod
    def flat_grad(cls, gradients: Tuple[torch.Tensor, ...], is_hessian: bool = False) -> torch.Tensor:
        """Flat gradients"""
        if is_hessian:
            grads = [g.contiguous() for g in gradients]
            return parameters_to_vector(grads)
        grads = [g.detach() for g in gradients]
        return parameters_to_vector(grads)

    @classmethod
    def update_model(cls, module: nn.Module, new_params: Dict[str, torch.nn.parameter.Parameter]):
        """
        Replace model's parameters with new parameters
        [source](https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/_functions.py)
        """
        named_modules = dict(module.named_modules())

        # pylint: disable=protected-access
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
        cls, kl: torch.Tensor, parameters: torch.nn.parameter.Parameter, hvp_reg_coeff: float = 1e-5
    ) -> Callable:
        """Returns a callable that computes Hessian-vector product"""
        parameters = list(parameters)
        kl_grad = torch.autograd.grad(kl, parameters, create_graph=True)
        kl_grad = cls.flat_grad(kl_grad, is_hessian=True)

        def hvp(vector):
            """Product of Hessian and vector"""
            kl_grad_prod = torch.dot(kl_grad, vector)
            kl_hessian_prod = torch.autograd.grad(kl_grad_prod, parameters, retain_graph=True)
            kl_hessian_prod = cls.flat_grad(kl_hessian_prod, is_hessian=True)
            kl_hessian_prod = kl_hessian_prod + hvp_reg_coeff * vector

            return kl_hessian_prod

        return hvp

    @classmethod
    def conjugate_gradient(  # pylint: disable=too-many-arguments
        cls,
        fnc_Ax: Callable,
        b: torch.Tensor,
        num_iters: int = 10,
        residual_tol: float = 1e-10,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Conjugate gradient algorithm"""
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
        self, Hvp: Callable, search_dir: torch.Tensor, max_kl: float
    ) -> torch.nn.parameter.Parameter:
        """Calculate descent step for backtracking line search according to kl constraint"""
        sHs = torch.dot(search_dir, Hvp(search_dir))
        lagrange_multiplier = torch.sqrt(sHs / (2 * max_kl))
        step = search_dir / lagrange_multiplier
        step_param = [torch.zeros_like(params.data) for params in self.policy.parameters()]
        vector_to_parameters(step, step_param)

        return step_param

    def infer_baselines(self, batch: Dict[str, torch.Tensor]):
        """Train value function and infer values as baselines"""
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
        self.vf.load_state_dict(self.initial_vf_params)

        # Update value function
        for _ in range(self.vf_learning_iters):
            self.vf_optimizer.zero_grad()
            value_batch = self.vf(obs_batch.to(self.device))
            value_loss = F.mse_loss(value_batch.view(-1, 1), returns_batch.to(self.device))
            value_loss.backward()
            self.vf_optimizer.step()

        # Infer baseline with the updated value function
        with torch.no_grad():
            baselines = self.vf(obs_batch)

        return baselines.cpu().numpy()

    def compute_gae(self, batch: Dict[str, torch.Tensor]):
        """Compute return and GAE"""
        rewards_batch = batch["rewards"]
        dones_batch = batch["dones"]
        values_batch = batch["baselines"]
        advants_batch = torch.zeros_like(rewards_batch)
        prev_value = 0
        running_advant = 0

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
        advants_batch = (advants_batch - advants_batch.mean()) / (advants_batch.std() + 1e-8)

        return advants_batch

    def kl_divergence(self, batch: Dict[str, torch.Tensor]):
        """Compute KL divergence between old policy and new policy"""
        obs_batch = batch["obs"]

        with torch.no_grad():
            old_dist, _ = self.old_policy.get_normal_dist(obs_batch)

        new_dist, _ = self.policy.get_normal_dist(obs_batch)
        kl_constraint = torch.distributions.kl.kl_divergence(old_dist, new_dist)

        return kl_constraint.mean()

    def compute_policy_entropy(self, batch: Dict[str, torch.Tensor]):
        """Compute policy entropy"""
        obs_batch = batch["obs"]

        with torch.no_grad():
            dist, _ = self.policy.get_normal_dist(obs_batch)
            policy_entropy = dist.entropy().sum(dim=-1)

        return policy_entropy.mean()

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Sample action from the policy"""
        action, _ = self.policy(torch.Tensor(obs).to(self.device))

        return action.detach().cpu().numpy()

    # pylint: disable=too-many-locals
    def policy_loss(self, batch: Dict[str, torch.Tensor], is_meta_loss: bool = False):
        """Compute policy losses according to TRPO algorithm"""
        obs_batch = batch["obs"]
        action_batch = batch["actions"]
        advant_batch = self.compute_gae(batch).detach()

        if is_meta_loss:
            # Surrogate loss
            old_log_prob_batch = self.old_policy.get_log_prob(obs_batch, action_batch).view(-1, 1)
            new_log_prob_batch = self.policy.get_log_prob(obs_batch, action_batch).view(-1, 1)
            ratio = torch.exp(new_log_prob_batch - old_log_prob_batch.detach())
            surrogate_loss = ratio * advant_batch
            loss = -surrogate_loss.mean()
        else:
            # A2C
            log_prob_batch = self.policy.get_log_prob(obs_batch, action_batch).view(-1, 1)
            loss = -torch.mean(log_prob_batch * advant_batch)
        return loss
