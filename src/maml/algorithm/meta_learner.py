"""
Meta-train and meta-test codes with MAML algorithm
"""


import copy
import datetime
import os
import time

import higher
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.maml.algorithm.buffer import MultiTaskBuffer
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

        self.sampler = Sampler(
            env=env,
            agent=agent,
            action_dim=action_dim,
            device=device,
        )

        self.buffer = MultiTaskBuffer(
            observ_dim=observ_dim,
            action_dim=action_dim,
            agent=agent,
            num_tasks=self.num_sample_tasks,
            num_episodes=(self.num_adapt_epochs + 1),  # [num of adapatation for train] + [validation]
            max_size=self.num_samples,
            device=device,
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

    def collect_train_data(self, indices, eval_mode=False):
        """Collect data before & after gradient for task batch"""

        for cur_task, task_index in enumerate(indices):
            self.env.reset_task(task_index)
            if not eval_mode:
                print(f"[{cur_task + 1}/{len(indices)}] collecting data for task batch")

            # Get branches of outer-poicy as inner-policy
            with higher.innerloop_ctx(
                self.agent.policy, self.inner_optimizer, track_higher_grads=False
            ) as (inner_policy, inner_optimizer):

                # Adapt policy to each task through few grandient steps
                for cur_adapt in range(self.num_adapt_epochs + 1):

                    # Sample trajectories while adaptating
                    if cur_adapt == self.num_adapt_epochs:
                        inner_policy.is_deterministic = eval_mode
                    trajs = self.sampler.obtain_samples(
                        inner_policy,
                        max_samples=self.num_samples,
                        max_steps=self.max_steps,
                    )
                    self.buffer.add_trajs(cur_task, cur_adapt, trajs)

                    # Update policy except validation episode
                    if cur_adapt < self.num_adapt_epochs:
                        # Get preprocessed batch samples for the current task and adaptation step
                        batch_samples = self.buffer.get_samples(cur_task, cur_adapt)

                        # Adapt the inner-policy
                        inner_policy_loss = self.agent.compute_loss(inner_policy, batch_samples)
                        inner_optimizer.step(inner_policy_loss)

    def meta_update(self):
        """Update meta-policy using PPO algorithm"""
        self.outer_optimizer.zero_grad()
        policy_loss_mean = 0

        # Compute loss for each sampled task
        for cur_task in range(self.num_sample_tasks):

            # Get branches of outer-poicy as inner-policy
            with higher.innerloop_ctx(
                self.agent.policy,
                self.inner_optimizer,
                copy_initial_weights=False,
            ) as (inner_policy, inner_optimizer):

                # Adapt policy to each task through few grandient steps
                for cur_adapt in range(self.num_adapt_epochs):

                    # Get preprocessed batch samples for the current task and adaptation step
                    batch_samples = self.buffer.get_samples(cur_task, cur_adapt)

                    # Adapt the inner-policy
                    inner_policy_loss = self.agent.compute_loss(inner_policy, batch_samples)
                    inner_optimizer.step(inner_policy_loss)

                # Get validation batch samples for the current task
                validation_batch = self.buffer.get_samples(cur_task, self.num_adapt_epochs)

                # Compute Meta-loss and backpropagate it through the gradient steps.
                # Losses across all of the sampled-batch tasks are cumulated
                # until `self.outer_optimizer.step()`
                policy_loss = self.agent.compute_loss(inner_policy, validation_batch, valid=True)
                policy_loss.backward()

                policy_loss_mean += policy_loss.item() / self.num_sample_tasks

        self.outer_optimizer.step()
        self.buffer.clear()

        # Replace the old_policy with the updated meta-policy
        self.agent.old_policy = copy.deepcopy(self.agent.policy)

        return dict(policy_loss=policy_loss_mean)

    def meta_train(self):  # pylint: disable=too-many-locals
        """MAML meta-training"""
        total_start_time = time.time()
        for iteration in range(self.num_iterations):
            start_time = time.time()

            print(f"=============== Iteration {iteration} ===============")
            # Sample batch of tasks randomly from train task distribution and
            # Optain adaptating samples for the batch tasks
            indices = np.random.randint(len(self.train_tasks), size=self.num_sample_tasks)
            self.collect_train_data(indices)

            # Meta update
            log_values = self.meta_update()

            # Evaluate on test tasks
            self.meta_test(iteration, total_start_time, start_time, log_values)

    # pylint: disable=too-many-locals, disable=too-many-statements
    def meta_test(self, iteration, total_start_time, start_time, log_values):
        """MAML meta-testing"""

        test_results = {}
        return_before_grad = 0
        return_after_grad = 0
        run_cost_before_grad = np.zeros(self.max_steps)
        run_cost_after_grad = np.zeros(self.max_steps)

        self.collect_train_data(self.test_tasks, eval_mode=True)

        returns_before_grad = [
            torch.sum(self.buffer.get_samples(task, 0, log=True)["rewards"][: self.max_steps]).item()
            for task in range(len(self.test_tasks))
        ]
        returns_after_grad = [
            torch.sum(
                self.buffer.get_samples(task, self.num_adapt_epochs, log=True)["rewards"][
                    : self.max_steps
                ]
            ).item()
            for task in range(len(self.test_tasks))
        ]
        return_before_grad = sum(returns_before_grad)
        return_after_grad = sum(returns_after_grad)

        if self.env_name == "cheetah-vel":
            run_cost_before_grad = [
                self.buffer.get_samples(task, 0, log=True)["infos"][: self.max_steps]
                for task in range(len(self.test_tasks))
            ]
            run_cost_before_grad = torch.sum(torch.cat(run_cost_before_grad, dim=1), 1).numpy()

            run_cost_after_grad = [
                self.buffer.get_samples(task, self.num_adapt_epochs, log=True)["infos"][
                    : self.max_steps
                ]
                for task in range(len(self.test_tasks))
            ]
            run_cost_after_grad = torch.sum(torch.cat(run_cost_after_grad, dim=1), 1).numpy()

        self.buffer.clear()

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
