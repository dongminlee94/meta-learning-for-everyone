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

        self.inner_buffer = Buffer(
            observ_dim=observ_dim,
            action_dim=action_dim,
            max_size=self.train_samples * self.inner_grad_iters,
            device=device,
        )
        self.outer_buffer = Buffer(
            observ_dim=observ_dim,
            action_dim=action_dim,
            max_size=self.train_samples * self.inner_grad_iters * self.num_sample_tasks,
            device=device,
        )

        self.inner_optimizer_base = torch.optim.SGD(
            list(self.agent.model.parameters()),
            lr=config["inner_learning_rate"],
        )
        self.outer_optimizer = torch.optim.Adam(
            list(self.agent.model.parameters()),
            lr=config["outer_learning_rate"],
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
            inner_vf_loss_sum = 0
            outer_policy_loss_sum = 0
            outer_vf_loss_sum = 0

            print(f"=============== Iteration {iteration} ===============")
            # Sample tasks randomly from train tasks distribution.
            for i in range(self.num_sample_tasks):
                index = np.random.randint(len(self.train_tasks))
                self.env.reset_task(index)
                self.agent.model.policy.is_deterministic = False
                print(f"[{i+1}/{self.num_sample_tasks}] collecting inner-loop losses")

                # Branch policy network and its optimizer by Higher
                with higher.innerloop_ctx(
                    self.agent.model,
                    self.inner_optimizer_base,
                    copy_initial_weights=False,
                ) as (inner_model, inner_optimizer_branch):

                    inner_model.policy.is_deterministic = False
                    # Adapt meta-policy to each task with n-steps grandients
                    for _ in range(self.inner_grad_iters):
                        # Sample trajectory with inner policy
                        trajs = self.sampler.obtain_trajs(
                            inner_model,
                            max_samples=self.train_samples,
                            max_step=self.max_step,
                        )
                        self.inner_buffer.add_trajs(trajs)

                        # Inner update
                        batch = self.inner_buffer.get_samples()
                        total_loss, inner_policy_loss, inner_vf_loss = self.agent.compute_losses(
                            inner_model, batch
                        )
                        inner_optimizer_branch.step(total_loss)

                        inner_policy_loss_sum += inner_policy_loss
                        inner_vf_loss_sum += inner_vf_loss

                    inner_model.policy.is_deterministic = False
                    # Sample trajectory with adapted policy
                    trajs = self.sampler.obtain_trajs(
                        inner_model,
                        max_samples=self.train_samples,
                        max_step=self.max_step,
                    )
                    self.outer_buffer.add_trajs(trajs)

            # Meta update
            print("Meta-updating outer-loop policy")
            batch = self.outer_buffer.get_samples()
            for _ in range(self.outer_grad_iters):
                meta_loss, outer_policy_loss, outer_vf_loss = self.agent.compute_losses(
                    self.agent.model, batch
                )

                self.outer_optimizer.zero_grad()
                meta_loss.backward()
                self.outer_optimizer.step()

                outer_policy_loss_sum += outer_policy_loss
                outer_vf_loss_sum += outer_vf_loss

            inner_policy_loss_mean = inner_policy_loss_sum / (
                self.inner_grad_iters * self.num_sample_tasks
            )
            inner_vf_loss_mean = inner_vf_loss_sum / (self.inner_grad_iters * self.num_sample_tasks)
            outer_policy_loss_mean = outer_policy_loss_sum / (
                self.outer_grad_iters * self.num_sample_tasks
            )
            outer_vf_loss_mean = outer_vf_loss_sum / (self.outer_grad_iters * self.num_sample_tasks)

            log_values = dict(
                inner_policy_loss=inner_policy_loss_mean.item(),
                inner_vf_loss=inner_vf_loss_mean.item(),
                outer_policy_loss=outer_policy_loss_mean.item(),
                outer_vf_loss=outer_vf_loss_mean.item(),
            )

            # Evaluate on test tasks
            self.meta_test(iteration, total_start_time, start_time, log_values)

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    def meta_test(self, iteration, total_start_time, start_time, log_values):
        """MAML meta-testing"""
        test_results = {}
        return_before_grad = 0
        return_after_grad = 0
        run_cost_before_grad = np.zeros(self.max_step)
        run_cost_after_grad = np.zeros(self.max_step)

        for index in self.test_tasks:
            trajs = []
            self.env.reset_task(index)

            # Branch policy network and its optimizer by Higher
            with higher.innerloop_ctx(
                self.agent.model,
                self.inner_optimizer_base,
                copy_initial_weights=False,
            ) as (inner_model, inner_optimizer_branch):

                inner_model.policy.is_deterministic = False
                # Adapt meta-policy to each task with n-steps grandients
                for _ in range(self.inner_grad_iters):
                    # Sample trajectory with inner policy
                    pre_adapted_trajs = self.sampler.obtain_trajs(
                        inner_model,
                        max_samples=self.test_samples,
                        max_step=self.max_step,
                    )
                    self.inner_buffer.add_trajs(pre_adapted_trajs)
                    trajs.append(pre_adapted_trajs[0])

                    # Inner update
                    batch = self.inner_buffer.get_samples()
                    total_loss, _, _ = self.agent.compute_losses(inner_model, batch)
                    inner_optimizer_branch.step(total_loss)

                inner_model.policy.is_deterministic = True
                # Sample trajectory with adapted policy
                post_adapted_trajs = self.sampler.obtain_trajs(
                    inner_model,
                    max_samples=self.test_samples,
                    max_step=self.max_step,
                )
                trajs.append(post_adapted_trajs[0])

                return_before_grad += sum(trajs[0]["rewards"])[0]
                return_after_grad += sum(trajs[-1]["rewards"])[0]

                if self.env_name == "cheetah-vel":
                    for i in range(self.max_step):
                        run_cost_before_grad[i] += trajs[0]["infos"][i]
                        run_cost_after_grad[i] += trajs[-1]["infos"][i]

        # Collect meta-test results
        test_results["return_before_grad"] = return_before_grad / len(self.test_tasks)
        test_results["return_after_grad"] = return_after_grad / len(self.test_tasks)
        if self.env_name == "cheetah-vel":
            test_results["run_cost_before_grad"] = run_cost_before_grad / len(self.test_tasks)
            test_results["run_cost_after_grad"] = run_cost_after_grad / len(self.test_tasks)
            test_results["total_run_cost_before_grad"] = sum(test_results["run_cost_before_grad"])
            test_results["total_run_cost_after_grad"] = sum(test_results["run_cost_after_grad"])
        test_results["inner_policy_loss"] = log_values["inner_policy_loss"]
        test_results["inner_vf_loss"] = log_values["inner_vf_loss"]
        test_results["outer_policy_loss"] = log_values["outer_policy_loss"]
        test_results["outer_vf_loss"] = log_values["outer_vf_loss"]

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
        self.writer.add_scalar("train/inner_vf_loss", test_results["inner_vf_loss"], iteration)
        self.writer.add_scalar("train/outer_policy_loss", test_results["outer_policy_loss"], iteration)
        self.writer.add_scalar("train/outer_vf_loss", test_results["outer_vf_loss"], iteration)
        self.writer.add_scalar("time/total_time", test_results["total_time"], iteration)
        self.writer.add_scalar("time/time_per_iter", test_results["time_per_iter"], iteration)

        # Logging
        print(
            f"--------------------------------------- \n"
            f'return_before_grad: {round(test_results["return_before_grad"], 2)} \n'
            f'return_after_grad: {round(test_results["return_after_grad"], 2)} \n'
            f'inner_policy_loss: {round(test_results["inner_policy_loss"], 2)} \n'
            f'inner_vf_loss: {round(test_results["inner_vf_loss"], 2)} \n'
            f'outer_policy_loss: {round(test_results["outer_policy_loss"], 2)} \n'
            f'outer_vf_loss: {round(test_results["outer_vf_loss"], 2)} \n'
            f'time_per_iter: {round(test_results["time_per_iter"], 2)} \n'
            f'total_time: {round(test_results["total_time"], 2)} \n'
            f"--------------------------------------- \n"
        )

        # Save the trained model
        # TBU
