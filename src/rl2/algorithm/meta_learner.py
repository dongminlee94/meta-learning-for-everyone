"""
Meta-train and meta-test codes with RL^2 algorithm
"""

import datetime
import os
import time
from collections import deque

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.rl2.algorithm.buffer import Buffer
from src.rl2.algorithm.sampler import Sampler


class MetaLearner:  # pylint: disable=too-many-instance-attributes
    """RL^2 meta-learner class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env,
        env_name,
        agent,
        trans_dim,
        action_dim,
        hidden_dim,
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
        self.num_samples = config["num_samples"]
        self.batch_size = len(train_tasks) * config["num_samples"]
        self.max_step = config["max_step"]

        self.sampler = Sampler(
            env=env,
            agent=agent,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_step=config["max_step"],
        )

        self.buffer = Buffer(
            trans_dim=trans_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_size=self.batch_size,
            device=device,
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

        # Set up early stopping condition
        self.dq = deque(maxlen=config["num_stopping_conditions"])
        self.stopping_goal_mean = config["stopping_goal_mean"]
        self.early_stopping = False

    def meta_train(self):
        """RL^2 meta-training"""
        total_start_time = time.time()
        for iteration in range(self.num_iterations):
            start_time = time.time()

            print(f"=============== Iteration {iteration} ===============")
            # Sample data randomly from train tasks.
            for index in range(len(self.train_tasks)):
                self.env.reset_task(index)
                self.agent.policy.is_deterministic = False

                print(f"[{index + 1}/{len(self.train_tasks)}] collecting samples")
                trajs = self.sampler.obtain_trajs(max_samples=self.num_samples)
                self.buffer.add_trajs(trajs)

            # Get all samples for the train tasks
            batch = self.buffer.get_samples()

            # Train the policy and the value function
            print(f"Start the meta-gradient update of iteration {iteration}")
            log_values = self.agent.train_model(self.batch_size, batch)

            # Evaluate on test tasks
            self.meta_test(iteration, total_start_time, start_time, log_values)

            if self.early_stopping:
                print(f"End meta-training because early stopping condition is {self.early_stopping}")
                break

    def visualize_within_tensorboard(self, test_results, iteration):
        """Tensorboard visualization"""
        self.writer.add_scalar("test/return", test_results["return"], iteration)
        if self.env_name == "cheetah-vel":
            self.writer.add_scalar("test/sum_run_cost", test_results["sum_run_cost"], iteration)
            for step in range(len(test_results["run_cost"])):
                self.writer.add_scalar(
                    "run_cost/iteration_" + str(iteration),
                    test_results["run_cost"][step],
                    step,
                )
        self.writer.add_scalar("train/total_loss", test_results["total_loss"], iteration)
        self.writer.add_scalar("train/policy_loss", test_results["policy_loss"], iteration)
        self.writer.add_scalar("train/value_loss", test_results["value_loss"], iteration)
        self.writer.add_scalar("time/total_time", test_results["total_time"], iteration)
        self.writer.add_scalar("time/time_per_iter", test_results["time_per_iter"], iteration)

    def meta_test(self, iteration, total_start_time, start_time, log_values):
        """RL^2 meta-testing"""
        test_results = {}
        test_return = 0
        test_run_cost = np.zeros(self.max_step)

        for index in self.test_tasks:
            self.env.reset_task(index)
            self.agent.policy.is_deterministic = True

            trajs = self.sampler.obtain_trajs(max_samples=self.max_step)
            test_return += sum(trajs[0]["rewards"])[0]
            if self.env_name == "cheetah-vel":
                for i in range(self.max_step):
                    test_run_cost[i] += trajs[0]["infos"][i]

        # Collect meta-test results
        test_results["return"] = test_return / len(self.test_tasks)
        if self.env_name == "cheetah-vel":
            test_results["run_cost"] = test_run_cost / len(self.test_tasks)
            test_results["sum_run_cost"] = sum(abs(test_results["run_cost"]))
        test_results["total_loss"] = log_values["total_loss"]
        test_results["policy_loss"] = log_values["policy_loss"]
        test_results["value_loss"] = log_values["value_loss"]
        test_results["total_time"] = time.time() - total_start_time
        test_results["time_per_iter"] = time.time() - start_time

        self.visualize_within_tensorboard(test_results, iteration)

        # Check if np.mean(self.dq) satisfies early stopping condition
        if self.env_name == "cheetah-dir":
            self.dq.append(test_results["return"])
            if np.mean(self.dq) >= self.stopping_goal_mean:
                self.early_stopping = True
        elif self.env_name == "cheetah-vel":
            self.dq.append(test_results["sum_run_cost"])
            if np.mean(self.dq) <= self.stopping_goal_mean:
                self.early_stopping = True

        # Save the trained models
        if self.early_stopping:
            pass
