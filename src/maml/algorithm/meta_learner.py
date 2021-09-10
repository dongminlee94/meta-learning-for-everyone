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

from src.maml.algorithm.buffer import Buffer, MultiBuffer
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

        self.num_iterations = config["num_iterations"]
        self.num_sample_tasks = config["num_sample_tasks"]
        self.num_samples = config["num_samples"]
        self.max_steps = config["max_steps"]

        self.num_adapt_epochs = config["num_adapt_epochs"]
        self.num_maml_epochs = config["num_maml_epochs"]

        self.sampler = Sampler(
            env=env,
            agent=agent,
            action_dim=action_dim,
            device=device,
        )

        self.train_buffer = MultiBuffer(
            observ_dim=observ_dim,
            action_dim=action_dim,
            num_tasks=self.num_sample_tasks,
            num_episodes=(self.num_adapt_epochs + 1),  # [num of adapatation for train] + [validation]
            max_size=self.num_samples,
            device=device,
        )

        self.test_buffer = Buffer(
            observ_dim=observ_dim, action_dim=action_dim, max_size=self.num_samples, device=device
        )

        self.inner_optimizer = torch.optim.SGD(
            list(self.agent.policy.parameters()),
            lr=config["inner_learning_rate"],
        )
        self.outer_optimizer = torch.optim.Adam(
            list(self.agent.policy.parameters()),
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

    def meta_update(self):
        """Update meta-policy using PPO algorithm"""
        sum_policy_loss = 0
        # Repeat meta-update as PPO steps
        for _ in range(self.num_maml_epochs):

            # Compute loss for each sampled task
            self.outer_optimizer.zero_grad()
            for cur_task in range(self.num_sample_tasks):

                # Get branches of meta-poicy as inner-policy
                with higher.innerloop_ctx(
                    self.agent.policy,
                    self.inner_optimizer,
                    copy_initial_weights=False,
                ) as (inner_policy, inner_optimizer_branch):

                    # Adapt inner-policy to each task with n-steps grandients
                    inner_policy.is_deterministic = False
                    for cur_adapt in range(self.num_adapt_epochs):

                        # Get pre-adapted samples for the current task
                        pre_adapted_batch = self.train_buffer.get_samples(cur_task, cur_adapt)

                        # Adapt the inner-policy
                        inner_policy_loss = self.agent.compute_losses(
                            inner_policy, pre_adapted_batch, a2c_loss=True, clip_loss=False
                        )
                        inner_optimizer_branch.step(inner_policy_loss)

                    # Get post-adapted samples for the current task
                    post_adapted_batch = self.train_buffer.get_samples(cur_task, self.num_adapt_epochs)

                    # Compute PPO loss and backpropagate it through the meta-policy
                    policy_loss = self.agent.compute_losses(
                        inner_policy, post_adapted_batch, a2c_loss=False, clip_loss=True
                    )
                    policy_loss.backward()
                    sum_policy_loss += policy_loss.item() / self.num_sample_tasks

            self.outer_optimizer.step()
            policy_loss_mean = sum_policy_loss / self.num_maml_epochs

        log_values = dict(
            policy_loss=policy_loss_mean,
        )
        return log_values

    # pylint: disable=too-many-locals
    def meta_train(self):
        """MAML meta-training"""
        total_start_time = time.time()
        for iteration in range(self.num_iterations):
            start_time = time.time()

            print(f"=============== Iteration {iteration} ===============")
            # Sample tasks randomly from train tasks distribution.
            indices = np.random.randint(len(self.train_tasks), size=self.num_sample_tasks)
            for cur_task, index in enumerate(indices):
                self.env.reset_task(index)
                self.agent.policy.is_deterministic = False

                print(f"[{cur_task + 1}/{self.num_sample_tasks}] collecting task adapation samples")
                # Get branches of meta-poicy as inner-policy
                with higher.innerloop_ctx(
                    self.agent.policy,
                    self.inner_optimizer,
                    copy_initial_weights=False,
                ) as (inner_policy, inner_optimizer_branch):

                    # Adapt meta-policy to each task with n-steps grandients
                    inner_policy.is_deterministic = False
                    for cur_adapt in range(self.num_adapt_epochs):

                        # Sample pre-adapted trajectory with branched policy of the meta-policy
                        pre_adapted_trajs = self.sampler.obtain_trajs(
                            inner_policy,
                            max_samples=self.num_samples,
                            max_steps=self.max_steps,
                        )
                        self.train_buffer.add_trajs(cur_task, cur_adapt, pre_adapted_trajs)

                        # Get pre-adapted samples for the current task
                        pre_adapted_batch = self.train_buffer.get_samples(cur_task, cur_adapt)

                        # Adapt the inner-policy
                        inner_policy_loss = self.agent.compute_losses(
                            inner_policy, pre_adapted_batch, a2c_loss=True, clip_loss=False
                        )
                        inner_optimizer_branch.step(inner_policy_loss)

                    # Sample post-adapted trajectory with adapted policy
                    inner_policy.is_deterministic = False
                    post_adapted_trajs = self.sampler.obtain_trajs(
                        inner_policy,
                        max_samples=self.num_samples,
                        max_steps=self.max_steps,
                    )
                    self.train_buffer.add_trajs(cur_task, self.num_adapt_epochs, post_adapted_trajs)

            # Meta update
            log_values = self.meta_update()
            self.train_buffer.clear()

            # Evaluate on test tasks
            self.meta_test(iteration, total_start_time, start_time, log_values)

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    def meta_test(self, iteration, total_start_time, start_time, log_values):
        """MAML meta-testing"""
        test_results = {}
        return_before_grad = 0
        return_after_grad = 0
        run_cost_before_grad = np.zeros(self.max_steps)
        run_cost_after_grad = np.zeros(self.max_steps)

        for index in self.test_tasks:
            test_trajs = []
            self.env.reset_task(index)

            # Get branches of meta-poicy as inner-policy
            with higher.innerloop_ctx(
                self.agent.policy,
                self.inner_optimizer,
                copy_initial_weights=False,
            ) as (inner_policy, inner_optimizer_branch):

                # Adapt meta-policy to each task with n-steps grandients
                inner_policy.is_deterministic = False
                for _ in range(self.num_adapt_epochs):

                    # Sample pre-adapted trajectory with branched policy of the meta-policy
                    pre_adapted_trajs = self.sampler.obtain_trajs(
                        inner_policy,
                        max_samples=self.num_samples,
                        max_steps=self.max_steps,
                    )
                    self.test_buffer.add_trajs(pre_adapted_trajs)
                    test_trajs.append(pre_adapted_trajs[0])

                    # Get pre-adapted samples for the current task
                    pre_adapted_batch = self.test_buffer.get_samples()
                    self.test_buffer.clear()

                    # Adapt the inner-policy
                    inner_policy_loss = self.agent.compute_losses(
                        inner_policy, pre_adapted_batch, a2c_loss=True, clip_loss=False
                    )
                    inner_optimizer_branch.step(inner_policy_loss)

                # Sample post-adapted trajectory with adapted policy
                inner_policy.is_deterministic = True
                post_adapted_trajs = self.sampler.obtain_trajs(
                    inner_policy,
                    max_samples=self.num_samples,
                    max_steps=self.max_steps,
                )
                test_trajs.append(post_adapted_trajs[0])

                return_before_grad += sum(test_trajs[0]["rewards"])[0]
                return_after_grad += sum(test_trajs[-1]["rewards"])[0]

                if self.env_name == "cheetah-vel":
                    for i in range(self.max_steps):
                        run_cost_before_grad[i] += test_trajs[0]["infos"][i]
                        run_cost_after_grad[i] += test_trajs[-1]["infos"][i]

        # Collect meta-test results
        test_results["return_before_grad"] = return_before_grad / len(self.test_tasks)
        test_results["return_after_grad"] = return_after_grad / len(self.test_tasks)
        if self.env_name == "cheetah-vel":
            test_results["run_cost_before_grad"] = run_cost_before_grad / len(self.test_tasks)
            test_results["run_cost_after_grad"] = run_cost_after_grad / len(self.test_tasks)
            test_results["total_run_cost_before_grad"] = sum(test_results["run_cost_before_grad"])
            test_results["total_run_cost_after_grad"] = sum(test_results["run_cost_after_grad"])
        test_results["policy_loss"] = log_values["policy_loss"]

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
        self.writer.add_scalar("train/policy_loss", test_results["policy_loss"], iteration)
        self.writer.add_scalar("time/total_time", test_results["total_time"], iteration)
        self.writer.add_scalar("time/time_per_iter", test_results["time_per_iter"], iteration)

        # Logging
        print(
            f"--------------------------------------- \n"
            f'return_before_grad: {round(test_results["return_before_grad"], 2)} \n'
            f'return_after_grad: {round(test_results["return_after_grad"], 2)} \n'
            f'policy_loss: {round(test_results["policy_loss"], 2)} \n'
            f'time_per_iter: {round(test_results["time_per_iter"], 2)} \n'
            f'total_time: {round(test_results["total_time"], 2)} \n'
            f"--------------------------------------- \n"
        )

        # Save the trained model
        # TBU
