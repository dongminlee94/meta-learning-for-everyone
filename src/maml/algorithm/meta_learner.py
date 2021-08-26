"""
Meta-train and meta-test codes with MAML algorithm
"""


import datetime
import os
import time

import higher
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.maml.algorithm.buffer import Buffer
from src.maml.algorithm.sampler import Sampler


class MetaLearner:  # pylint: disable=too-many-instance-attributes
    """MAML meta-learner class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env,
        env_name,
        agent,
        observ_dim,
        action_dim,
        train_tasks,
        test_tasks,
        exp_name,
        file_name,
        device,
        **config,
    ):

        self.env = env
        self.env_name = env_name
        self.agent = agent
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks

        self.train_iters = config["train_iters"]
        self.num_sample_tasks = config["num_sample_tasks"]
        self.train_samples = config["train_samples"]
        self.max_step = config["max_step"]
        self.test_samples = config["test_samples"]

        self.inner_grad_iters = config["inner_grad_iters"]
        self.outer_grad_iters = config["outer_grad_iters"]

        self.sampler = Sampler(
            env=env,
            agent=agent,
            action_dim=action_dim,
            device=device,
        )

        self.outer_buffer = Buffer(
            observ_dim=observ_dim,
            action_dim=action_dim,
            max_size=self.train_samples * self.num_sample_tasks,
            device=device,
        )
        self.inner_buffer = Buffer(
            observ_dim=observ_dim,
            action_dim=action_dim,
            max_size=self.train_samples,
            device=device,
        )

        self.outer_optimizer = torch.optim.Adam(
            list(self.agent.policy.parameters()),
            lr=config["learning_rate"],
        )
        self.inner_optimizer_base = torch.optim.SGD(
            list(self.agent.policy.parameters()),
            lr=config["inner_learning_rate"],
        )

        if file_name is None:
            file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.writer = SummaryWriter(
            log_dir=os.path.join(
                ".",
                "results",
                exp_name,
                file_name,
            )
        )

    # pylint: disable=too-many-locals
    def meta_train(self):
        """MAML meta-training"""
        total_start_time = time.time()
        for iteration in range(self.train_iters):
            start_time = time.time()
            inner_policy_loss_sum = 0
            outer_policy_loss_sum = 0
            value_loss_sum = 0

            print(f"=============== Iteration {iteration} ===============")
            # Sample tasks randomly from train tasks distribution.
            for i in range(self.num_sample_tasks):
                index = np.random.randint(len(self.train_tasks))
                self.env.reset_task(index)
                self.agent.policy.is_deterministic = False
                print(f"[{i+1}/{self.num_sample_tasks}] collecting inner-loop losses")

                # Branch policy network and its optimizer by Higher
                with higher.innerloop_ctx(
                    self.agent.policy,
                    self.inner_optimizer_base,
                    copy_initial_weights=False,
                ) as (inner_policy, inner_optimizer_branch):

                    inner_policy.is_deterministic = False
                    # Adapt meta-policy to each task with n-steps grandients
                    for _ in range(self.inner_grad_iters):
                        # Sample trajectory with inner policy
                        trajs = self.sampler.obtain_trajs(
                            inner_policy,
                            max_samples=self.train_samples,
                            max_step=self.max_step,
                        )
                        self.inner_buffer.add_trajs(trajs)
                        batch = self.inner_buffer.get_samples()
                        # Inner update
                        inner_policy_loss, value_loss = self.agent.compute_losses(inner_policy, batch)
                        inner_optimizer_branch.step(inner_policy_loss)

                        inner_policy_loss_sum += inner_policy_loss
                        value_loss_sum += value_loss

                    # Sample trajectory with adapted policy
                    trajs = self.sampler.obtain_trajs(
                        inner_policy,
                        max_samples=self.train_samples,
                        max_step=self.max_step,
                    )
                    self.outer_buffer.add_trajs(trajs)

            # Meta update
            print("Meta-updating outer-loop policy")
            batch = self.outer_buffer.get_samples()
            for _ in range(self.outer_grad_iters):
                outer_policy_loss, value_loss = self.agent.compute_losses(self.agent.policy, batch)
                self.outer_optimizer.zero_grad()
                outer_policy_loss.backward()
                self.outer_optimizer.step()

                outer_policy_loss_sum += outer_policy_loss
                value_loss_sum += value_loss

            inner_policy_loss_mean = inner_policy_loss_sum / (
                self.inner_grad_iters * self.num_sample_tasks
            )
            outer_policy_loss_mean = outer_policy_loss_sum / (
                self.inner_grad_iters * self.num_sample_tasks
            )
            value_loss_mean = value_loss_sum / (
                self.inner_grad_iters * self.num_sample_tasks + self.outer_grad_iters
            )

            log_values = dict(
                inner_policy_loss=inner_policy_loss_mean.item(),
                outer_policy_loss=outer_policy_loss_mean.item(),
                value_loss=value_loss_mean.item(),
            )

            # Evaluate on test tasks
            self.meta_test(iteration, total_start_time, start_time, log_values)

    # pylint: disable=too-many-locals
    def meta_test(self, iteration, total_start_time, start_time, log_values):
        """MAML meta-testing"""
        test_results = {}
        return_before_grad = 0
        return_after_grad = 0
        run_cost_before_grad = np.zeros(self.max_step)
        run_cost_after_grad = np.zeros(self.max_step)

        for index in self.test_tasks:
            self.env.reset_task(index)
            self.agent.policy.is_deterministic = True

            # Branch policy network and its optimizer by Higher
            with higher.innerloop_ctx(
                self.agent.policy,
                self.inner_optimizer_base,
                copy_initial_weights=False,
            ) as (inner_policy, inner_optimizer_branch):

                inner_policy.is_deterministic = False
                # Adapt meta-policy to each task with n-steps grandients
                for _ in range(self.inner_grad_iters):
                    # Sample trajectory with inner policy
                    trajs = self.sampler.obtain_trajs(
                        inner_policy,
                        max_samples=self.train_samples,
                        max_step=self.max_step,
                    )
                    self.inner_buffer.add_trajs(trajs)
                    batch = self.inner_buffer.get_samples()
                    # Inner update
                    ppo_loss = self.agent.compute_losses(inner_policy, batch)
                    inner_optimizer_branch.step(ppo_loss)

                # Sample trajectory with adapted policy
                trajs = self.sampler.obtain_trajs(
                    inner_policy,
                    max_samples=self.train_samples,
                    max_step=self.max_step,
                )

                return_before_grad += sum(trajs[0]["rewards"])[0]
                return_after_grad += sum(trajs[1]["rewards"])[0]
                if self.env_name == "cheetah-vel":
                    for i in range(self.max_step):
                        run_cost_before_grad[i] += trajs[0]["infos"][i]
                        run_cost_after_grad[i] += trajs[1]["infos"][i]

        # Collect meta-test results
        test_results["return_before_grad"] = return_before_grad / len(self.test_tasks)
        test_results["return_after_grad"] = return_after_grad / len(self.test_tasks)
        if self.env_name == "cheetah-vel":
            test_results["run_cost_before_grad"] = run_cost_before_grad / len(self.test_tasks)
            test_results["run_cost_after_grad"] = run_cost_after_grad / len(self.test_tasks)
            test_results["total_run_cost_before_grad"] = sum(test_results["run_cost_before_grad"])
            test_results["total_run_cost_after_grad"] = sum(test_results["run_cost_after_grad"])
        test_results["inner_policy_loss"] = log_values["inner_policy_loss"]
        test_results["outer_policy_loss"] = log_values["outer_policy_loss"]
        test_results["value_loss"] = log_values["value_loss"]
        test_results["total_time"] = time.time() - total_start_time
        test_results["time_per_iter"] = time.time() - start_time

        # Tensorboard
        self.writer.add_scalar("test/return_before_grad", test_results["return_before_grad"], iteration)
        self.writer.add_scalar("test/return_after_grad", test_results["return_after_grad"], iteration)
        if self.env_name == "cheetah-vel":
            self.writer.add_scalar(
                "test/total_run_cost_before_grad",
                test_results["total_run_cost_before_grad"],
                iteration,
            )
            self.writer.add_scalar(
                "test/total_run_cost_after_grad", test_results["total_run_cost_after_grad"], iteration
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
        self.writer.add_scalar("train/inner_policy_loss", test_results["inner_policy_loss"], iteration)
        self.writer.add_scalar("train/outer_policy_loss", test_results["outer_policy_loss"], iteration)
        self.writer.add_scalar("train/value_loss", test_results["value_loss"], iteration)
        self.writer.add_scalar("time/total_time", test_results["total_time"], iteration)
        self.writer.add_scalar("time/time_per_iter", test_results["time_per_iter"], iteration)

        # Logging
        print(
            f"--------------------------------------- \n"
            f'return: {round(test_results["return_after_grad"], 2)} \n'
            f'inner_policy_loss: {round(test_results["inner_policy_loss"], 2)} \n'
            f'outer_policy_loss: {round(test_results["outer_policy_loss"], 2)} \n'
            f'value_loss: {round(test_results["value_loss"], 2)} \n'
            f'time_per_iter: {round(test_results["time_per_iter"], 2)} \n'
            f'total_time: {round(test_results["total_time"], 2)} \n'
            f"--------------------------------------- \n"
        )

        # Save the trained model
        # TBU
        # TEST
