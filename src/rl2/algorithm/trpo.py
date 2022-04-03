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

from src.maml.algorithm.networks import GRU, GaussianGRU


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
        
        self.policy = GaussianGRU(
            input_dim=observ_dim,
            output_dim=action_dim,
            hidden_dim=policy_hidden_dim,
        ).to(device)
        self.old_policy = deepcopy(self.policy)
        self.vf = GRU(
            input_dim=observ_dim,
            output_dim=1,
            hidden_dim=vf_hidden_dim,
        ).to(device)
        self.vf_optimizer = optim.Adam(
            list(self.vf.parameters()),
            lr=config["vf_learning_rate"],
        )
        self.vf_learning_iters = config["vf_learning_iters"]
        self.initial_vf_params = deepcopy(self.vf.state_dict())
        
        self.backtrack_iters = config["backtrack_iters"]
        self.backtrack_coeff = config["backtrack_coeff"]
        self.max_kl = config["max_kl"]

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

    def kl_divergence(self, obs_batch):
        """Compute KL divergence between old policy and new policy"""

        with torch.no_grad():
            old_dist, _ = self.old_policy.get_normal_dist(obs_batch)

        new_dist, _ = self.policy.get_normal_dist(obs_batch)
        kl_constraint = torch.distributions.kl.kl_divergence(old_dist, new_dist)

        return kl_constraint.mean()


    def compute_policy_entropy(self, obs_batch):
        """Compute policy entropy"""

        with torch.no_grad():
            dist, _ = self.policy.get_normal_dist(obs_batch)
            policy_entropy = dist.entropy().sum(dim=-1)

        return policy_entropy.mean()


    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Sample action from the policy"""
        action, _ = self.policy(torch.Tensor(obs).to(self.device))

        return action.detach().cpu().numpy()


    # pylint: disable=too-many-locals
    def policy_loss(self, trans_batch, pi_hidden_batch, action_batch, advant_batch, is_meta_loss=False):
        """Compute policy losses according to TRPO algorithm"""

        if is_meta_loss:
            # Surrogate loss
            old_log_prob_batch = self.old_policy.get_log_prob(trans_batch, pi_hidden_batch, action_batch).view(-1, 1)
            new_log_prob_batch = self.policy.get_log_prob(trans_batch, pi_hidden_batch, action_batch).view(-1, 1)
            ratio = torch.exp(new_log_prob_batch - old_log_prob_batch.detach())
            surrogate_loss = ratio * advant_batch
            loss = -surrogate_loss.mean()
        else:
            # A2C
            log_prob_batch = self.policy.get_log_prob(trans_batch, pi_hidden_batch, action_batch).view(-1, 1)
            loss = -torch.mean(log_prob_batch * advant_batch)
        return loss
    
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

        sum_total_loss = 0
        sum_policy_loss = 0
        sum_value_loss = 0
        
        # calculate the initial loss and KL
        # Value function loss
        value_batch, _ = self.vf(trans, v_hiddens)
        value_loss = F.mse_loss(value_batch.view(-1, 1), returns)
        # Policy loss
        policy_loss = self.policy_loss(trans, pi_hiddens, actions, advant_batch) + self.compute_policy_entropy(trans)
        # Total loss
        loss_before = policy_loss + 0.5 * value_loss
        
        kl_before = self.kl_divergence(trans)
        
        gradient = torch.autograd.grad(loss_before, self.policy.parameters(), retain_graph=True)
        gradient = self.flat_grad(gradient)
        Hvp = self.hessian_vector_product(kl_before, self.policy.parameters())
        search_dir = self.conjugate_gradient(Hvp, gradient)
        descent_step = self.compute_descent_step(Hvp, search_dir, self.max_kl)
        loss_before.detach_()
        
        backup_params = deepcopy(dict(self.policy.named_parameters()))

        for i in range(self.backtrack_iters):
            ratio = self.backtrack_coeff ** i

            for params, step in zip(self.agent.policy.parameters(), descent_step):
                params.data.add_(step, alpha=-ratio)
            
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
                # Value function loss
                value_batch, _ = self.vf(trans_batch, v_hidden_batch)
                value_loss = F.mse_loss(value_batch.view(-1, 1), return_batch)

                # Policy loss
                policy_loss = self.policy_loss(trans_batch, pi_hidden_batch, action_batch, advant_batch)
                
                # Total loss
                total_loss = policy_loss + 0.5 * value_loss
                
                kl = self.kl_divergence(trans_batch)
                
                # Update the policy, when the KL constraint is satisfied
                is_improved = total_loss < loss_before
                is_constrained = kl <= self.max_kl
                if is_improved and is_constrained:
                    print(f"Update meta-policy through {i+1} backtracking line search step(s)")
                    break
                
                self.update_model(self.agent.policy, backup_params)

                sum_total_loss_mini_batch += total_loss
                sum_policy_loss_mini_batch += policy_loss
                sum_value_loss_mini_batch += value_loss
            
            if i == self.backtrack_iters - 1:
                print("Keep current meta-policy skipping meta-update")

            sum_total_loss += sum_total_loss_mini_batch / num_mini_batch
            sum_policy_loss += sum_policy_loss_mini_batch / num_mini_batch
            sum_value_loss += sum_value_loss_mini_batch / num_mini_batch

        mean_total_loss = sum_total_loss / self.backtrack_iters
        mean_policy_loss = sum_policy_loss / self.backtrack_iters
        mean_value_loss = sum_value_loss / self.backtrack_iters
        
        return dict(
            total_loss=mean_total_loss.item(),
            policy_loss=mean_policy_loss.item(),
            value_loss=mean_value_loss.item(),
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
