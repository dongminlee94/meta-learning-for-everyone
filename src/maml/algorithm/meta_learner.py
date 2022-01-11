"""
Meta-train and meta-test codes with MAML algorithm
"""


import datetime
import os
import time
from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from torch.utils.tensorboard import SummaryWriter

from src.maml.algorithm.buffer import MultiTaskBuffer
from src.maml.algorithm.optimizer import DifferentiableSGD
from src.maml.algorithm.sampler import Sampler
from src.maml.algorithm.trpo import PolicyGradient


class MetaLearner:  # pylint: disable=too-many-instance-attributes
    """MAML meta-learner class"""

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        env: HalfCheetahEnv,
        env_name: str,
        agent: PolicyGradient,
        observ_dim: int,
        action_dim: int,
        train_tasks: List[int],
        test_tasks: List[int],
        save_exp_name: str,
        save_file_name: str,
        load_exp_name: str,
        load_file_name: str,
        load_ckpt_num: int,
        device: torch.device,
        **config,
    ) -> None:

        self.env = env
        self.env_name = env_name
        self.agent = agent
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks

        self.num_iterations = config["num_iterations"]
        self.num_sample_tasks = config["num_sample_tasks"]
        self.num_samples = config["num_samples"]
        self.max_steps = config["max_steps"]

        self.num_adapt_epochs = config["num_adapt_epochs"]
        self.backtrack_iters = config["backtrack_iters"]
        self.backtrack_coeff = config["backtrack_coeff"]
        self.max_kl = config["max_kl"]

        self.sampler = Sampler(
            env=env,
            agent=agent,
            action_dim=action_dim,
            max_step=config["max_steps"],
            device=device,
        )

        self.buffer = MultiTaskBuffer(
            observ_dim=observ_dim,
            action_dim=action_dim,
            agent=agent,
            num_tasks=max(self.num_sample_tasks, len(test_tasks)),
            num_episodes=(self.num_adapt_epochs + 1),  # [num of adapatation for train] + [validation]
            max_size=self.num_samples,
            device=device,
        )

        self.inner_optimizer = DifferentiableSGD(
            self.agent.policy,
            lr=config["inner_learning_rate"],
        )

        if not save_file_name:
            save_file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.result_path = os.path.join("results", save_exp_name, save_file_name)
        self.writer = SummaryWriter(log_dir=self.result_path)

        if load_exp_name and load_file_name:
            ckpt_path = os.path.join(
                "results", load_exp_name, load_file_name, "checkpoint_" + str(load_ckpt_num) + ".pt"
            )
            ckpt = torch.load(ckpt_path)

            self.agent.policy.load_state_dict(ckpt["policy"])

        # Set up early stopping condition
        self.dq: deque = deque(maxlen=config["num_stop_conditions"])
        self.num_stop_conditions: int = config["num_stop_conditions"]
        self.stop_goal: int = config["stop_goal"]
        self.is_early_stopping = False

    def collect_train_data(self, indices: List[int], is_eval: bool = False) -> None:
        """Collect data before & after gradient for each task batch"""
        losses = []
        backup_params = dict(self.agent.policy.named_parameters())

        for cur_task, task_index in enumerate(indices):

            mode = "test" if is_eval else "train"
            print(f"[{cur_task + 1}/{len(indices)}] collecting samples for {mode}-task batch")

            self.env.reset_task(task_index)
            # Adapt policy to each task through few grandient steps
            for cur_adapt in range(self.num_adapt_epochs + 1):

                if cur_adapt == self.num_adapt_epochs:
                    self.agent.policy.is_deterministic = is_eval
                else:
                    self.agent.policy.is_deterministic = False

                # Sample trajectory while adaptating steps and trajectory after adaptation
                trajs = self.sampler.obtain_samples(max_samples=self.num_samples)
                self.buffer.add_trajs(cur_task, cur_adapt, trajs)

                if cur_adapt < self.num_adapt_epochs:
                    # Update policy except validation episode
                    # Get adaptation trajectory for the current task and adaptation step
                    train_batch = self.buffer.get_samples(cur_task, cur_adapt)

                    # Adapt the inner-policy
                    inner_loss = self.agent.policy_loss(train_batch)
                    self.inner_optimizer.zero_grad(set_to_none=True)
                    require_grad = cur_adapt < self.num_adapt_epochs - 1
                    inner_loss.backward(create_graph=require_grad)
                    losses.append(inner_loss.item())

                    with torch.set_grad_enabled(require_grad):
                        self.inner_optimizer.step()

            # Save validation policy
            self.buffer.add_params(
                cur_task, self.num_adapt_epochs, dict(self.agent.policy.named_parameters())
            )
            # Restore to pre-updated policy
            self.agent.update_model(self.agent.policy, backup_params)

    # pylint: disable=too-many-locals
    def meta_surrogate_loss(self, set_grad: bool = True) -> Tuple[torch.Tensor, ...]:
        """Compute meta-surrogate loss across batch tasks"""
        losses, kls, entropies = [], [], []
        backup_params = dict(self.agent.policy.named_parameters())

        # Compute loss for each sampled task
        for cur_task in range(self.num_sample_tasks):
            # Adapt policy to each task through few grandient steps
            for cur_adapt in range(self.num_adapt_epochs):

                require_grad = cur_adapt < self.num_adapt_epochs - 1 or set_grad

                # Get adaptation trajectory
                train_batch = self.buffer.get_samples(cur_task, cur_adapt)

                # Adapt the inner-policy by A2C
                inner_loss = self.agent.policy_loss(train_batch)
                self.inner_optimizer.zero_grad(set_to_none=True)
                inner_loss.backward(create_graph=require_grad)

                with torch.set_grad_enabled(require_grad):
                    self.inner_optimizer.step()

            # Get validation trajectory and policy
            valid_batch = self.buffer.get_samples(cur_task, self.num_adapt_epochs)
            valid_params = self.buffer.get_params(cur_task, self.num_adapt_epochs)
            self.agent.update_model(self.agent.old_policy, valid_params)

            # Compute average of surrogate loss across batch tasks
            loss = self.agent.policy_loss(valid_batch, is_meta_loss=True)
            losses.append(loss)

            # Compute average of KL divergence across batch tasks
            kl = self.agent.kl_divergence(valid_batch)
            kls.append(kl)

            # Compute average of policy entropy across batch tasks
            entropy = self.agent.compute_policy_entropy(valid_batch)
            entropies.append(entropy)

            self.agent.update_model(self.agent.policy, backup_params)

        return torch.stack(losses).mean(), torch.stack(kls).mean(), torch.stack(entropies).mean()

    def meta_update(self) -> Dict[str, float]:  # pylint: disable=too-many-locals
        """Update meta-policy using TRPO algorithm"""

        # Compute initial descent steps of line search
        loss_before, kl_before, _ = self.meta_surrogate_loss()
        gradient = torch.autograd.grad(loss_before, self.agent.policy.parameters(), retain_graph=True)
        gradient = self.agent.flat_grad(gradient)
        Hvp = self.agent.hessian_vector_product(kl_before, self.agent.policy.parameters())
        search_dir = self.agent.conjugate_gradient(Hvp, gradient)
        descent_step = self.agent.compute_descent_step(Hvp, search_dir, self.max_kl)

        assert len(descent_step) == len(list(self.agent.policy.parameters()))
        loss_before.detach_()
        del Hvp, gradient

        # Backtracking line search
        backup_params = deepcopy(dict(self.agent.policy.named_parameters()))

        for i in range(self.backtrack_iters):
            ratio = self.backtrack_coeff ** i

            for params, step in zip(self.agent.policy.parameters(), descent_step):
                params.data.add_(step, alpha=-ratio)

            loss_new, kl_new, _ = self.meta_surrogate_loss(set_grad=False)

            # Update the policy, when the KL constraint is satisfied
            is_improved = loss_new < loss_before
            is_constrained = kl_new <= self.max_kl
            print(f"{i}-Backtracks | Loss {loss_new:.4f} < Loss_old {loss_before:.4f} : ", end="")
            print(f"{is_improved} | KL {kl_new:.4f} <= maxKL {self.max_kl:.4f} : {is_constrained}")

            if is_improved and is_constrained:
                print(f"Update meta-policy through {i+1} backtracking line search step(s)")
                break

            self.agent.update_model(self.agent.policy, backup_params)

            if i == self.backtrack_iters - 1:
                print("Keep current meta-policy skipping meta-update")

        loss_after, kl_after, policy_entropy = self.meta_surrogate_loss(set_grad=False)
        self.buffer.clear()

        return dict(
            loss_before=loss_before.item(),
            loss_after=loss_after.item(),
            kl_before=kl_before.item(),
            kl_after=kl_after.item(),
            policy_entropy=policy_entropy.item(),
        )

    def meta_train(self) -> None:  # pylint: disable=too-many-locals
        """MAML meta-training"""
        total_start_time = time.time()
        for iteration in range(self.num_iterations):
            start_time = time.time()

            print(f"\n=============== Iteration {iteration} ===============")
            # Sample batch of tasks randomly from train task distribution and
            # optain adaptating data for each batch task
            indices = np.random.randint(len(self.train_tasks), size=self.num_sample_tasks)
            self.collect_train_data(indices)

            # Meta update
            log_values = self.meta_update()

            # Evaluate on test tasks
            self.meta_test(iteration, total_start_time, start_time, log_values)

            if self.is_early_stopping:
                print(
                    f"\n==================================================\n"
                    f"The last {self.num_stop_conditions} meta-testing results are {self.dq}.\n"
                    f"And early stopping condition is {self.is_early_stopping}.\n"
                    f"Therefore, meta-training is terminated."
                )
                break

    def visualize_within_tensorboard(self, test_results: Dict[str, Any], iteration: int) -> None:
        """Tensorboard visualization"""
        self.writer.add_scalar("test/return_before_grad", test_results["return_before_grad"], iteration)
        self.writer.add_scalar("test/return_after_grad", test_results["return_after_grad"], iteration)
        if self.env_name == "vel":
            self.writer.add_scalar(
                "test/sum_run_cost_before_grad",
                test_results["sum_run_cost_before_grad"],
                iteration,
            )
            self.writer.add_scalar(
                "test/sum_run_cost_after_grad", test_results["sum_run_cost_after_grad"], iteration
            )
            for step in range(len(test_results["run_cost_before_grad"])):
                self.writer.add_scalar(
                    "run_cost_before_grad/iteration_" + str(iteration),
                    test_results["run_cost_before_grad"][step],
                    step,
                )
                self.writer.add_scalar(
                    "run_cost_after_grad/iteration_" + str(iteration),
                    test_results["run_cost_after_grad"][step],
                    step,
                )
        self.writer.add_scalar("train/loss_before", test_results["loss_before"], iteration)
        self.writer.add_scalar("train/loss_after", test_results["loss_after"], iteration)
        self.writer.add_scalar("train/loss_diff", test_results["loss_diff"], iteration)
        self.writer.add_scalar("train/kl_before", test_results["kl_before"], iteration)
        self.writer.add_scalar("train/kl_after", test_results["kl_after"], iteration)
        self.writer.add_scalar("train/policy_entropy", test_results["policy_entropy"], iteration)

        self.writer.add_scalar("time/total_time", test_results["total_time"], iteration)
        self.writer.add_scalar("time/time_per_iter", test_results["time_per_iter"], iteration)

    # pylint: disable=too-many-locals, disable=too-many-statements
    def meta_test(
        self, iteration: int, total_start_time: float, start_time: float, log_values: Dict[str, float]
    ) -> None:
        """MAML meta-testing"""
        test_results = {}
        returns_before_grad = []
        returns_after_grad = []
        run_costs_before_grad = []
        run_costs_after_grad = []

        self.collect_train_data(self.test_tasks, is_eval=True)

        for task in range(len(self.test_tasks)):
            batch_before_grad = self.buffer.get_samples(task, 0)
            batch_after_grad = self.buffer.get_samples(task, self.num_adapt_epochs)

            rewards_before_grad = batch_before_grad["rewards"][: self.max_steps]
            rewards_after_grad = batch_after_grad["rewards"][: self.max_steps]
            returns_before_grad.append(torch.sum(rewards_before_grad).item())
            returns_after_grad.append(torch.sum(rewards_after_grad).item())

            if self.env_name == "vel":
                run_costs_before_grad.append(batch_before_grad["infos"][: self.max_steps].cpu().numpy())
                run_costs_after_grad.append(batch_after_grad["infos"][: self.max_steps].cpu().numpy())

        run_cost_before_grad = np.sum(run_costs_before_grad, axis=0)
        run_cost_after_grad = np.sum(run_costs_after_grad, axis=0)

        self.buffer.clear()

        # Collect meta-test results
        test_results["return_before_grad"] = sum(returns_before_grad) / len(self.test_tasks)
        test_results["return_after_grad"] = sum(returns_after_grad) / len(self.test_tasks)
        if self.env_name == "vel":
            test_results["run_cost_before_grad"] = run_cost_before_grad / len(self.test_tasks)
            test_results["run_cost_after_grad"] = run_cost_after_grad / len(self.test_tasks)
            test_results["sum_run_cost_before_grad"] = sum(
                abs(run_cost_before_grad / len(self.test_tasks))
            )
            test_results["sum_run_cost_after_grad"] = sum(
                abs(run_cost_after_grad / len(self.test_tasks))
            )
        test_results["loss_before"] = log_values["loss_before"]
        test_results["loss_after"] = log_values["loss_after"]
        test_results["loss_diff"] = log_values["loss_before"] - log_values["loss_after"]
        test_results["kl_before"] = log_values["kl_before"]
        test_results["kl_after"] = log_values["kl_after"]
        test_results["policy_entropy"] = log_values["policy_entropy"]

        test_results["total_time"] = time.time() - total_start_time
        test_results["time_per_iter"] = time.time() - start_time

        self.visualize_within_tensorboard(test_results, iteration)

        # Check if each element of self.dq satisfies early stopping condition
        if self.env_name == "dir":
            self.dq.append(test_results["return_after_grad"])
            if all(list(map((lambda x: x >= self.stop_goal), self.dq))):
                self.is_early_stopping = True
        elif self.env_name == "vel":
            self.dq.append(test_results["sum_run_cost_after_grad"])
            if all(list(map((lambda x: x <= self.stop_goal), self.dq))):
                self.is_early_stopping = True

        # Save the trained models
        if self.is_early_stopping:
            ckpt_path = os.path.join(self.result_path, "checkpoint_" + str(iteration) + ".pt")
            torch.save({"policy": self.agent.policy.state_dict()}, ckpt_path)
