"""
Meta-train and meta-test codes with RL^2 algorithm
"""


import datetime
import os
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from rl2.algorithm.utils.buffer import Buffer
from rl2.algorithm.utils.sampler import Sampler


class RL2:  # pylint: disable=too-many-instance-attributes
    """RL^2 algorithm class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env,
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
        self.agent = agent
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks

        self.train_iters = config["train_iters"]
        self.train_samples = config["train_samples"]
        self.train_grad_iters = config["train_grad_iters"]
        self.batch_size = config["batch_size"]

        self.max_step = config["max_step"]
        self.test_samples = config["test_samples"]

        self.sampler = Sampler(
            env=env,
            agent=agent,
            max_step=config["max_step"],
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

        self.buffer = Buffer(
            trans_dim=trans_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_size=config["batch_size"],
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

    def meta_train(self):
        """RL^2 meta-training"""
        total_start_time = time.time()
        for iteration in range(self.train_iters):
            start_time = time.time()

            print("=============== Iteration {} ===============".format(iteration))
            # Sample data randomly from train tasks.
            for index in range(len(self.train_tasks)):
                self.env.reset_task(index)
                self.agent.policy.is_deterministic = False

                print(
                    "[{0}/{1}] collecting samples".format(
                        index + 1, len(self.train_tasks)
                    )
                )
                trajs = self.sampler.obtain_trajs(max_samples=self.train_samples)
                self.buffer.add_trajs(trajs)

            # Get all samples for the train tasks
            batch = self.buffer.get_samples()

            # Train the policy and the value function
            print("Start the meta-gradient update of iteration {}".format(iteration))
            log_values = self.agent.train_model(
                self.train_grad_iters, self.batch_size, batch
            )

            # Clear the collected batch
            self.buffer.clear()

            # Evaluate on test tasks
            self.meta_test(iteration, total_start_time, start_time, log_values)

    def meta_test(self, iteration, total_start_time, start_time, log_values):
        """PEARL meta-testing"""
        test_results = {}
        test_return = 0
        test_run_cost = np.zeros(self.max_step)

        for index in self.test_tasks:
            self.env.reset_task(index)
            self.agent.policy.is_deterministic = True

            trajs = self.sampler.obtain_trajs(max_samples=self.test_samples)
            test_return += sum(trajs[0]["rewards"])[0] + sum(trajs[1]["rewards"])[0]
            for i in range(self.max_step):
                test_run_cost[i] += trajs[0]["infos"][i] + trajs[1]["infos"][i]

        # Collect meta-test results
        test_results["return"] = test_return / len(self.test_tasks)
        test_results["run_cost"] = test_run_cost / len(self.test_tasks)
        test_results["total_loss"] = log_values["total_loss"]
        test_results["policy_loss"] = log_values["policy_loss"]
        test_results["value_loss"] = log_values["value_loss"]
        test_results["total_time"] = time.time() - total_start_time
        test_results["time_per_iter"] = time.time() - start_time

        # Tensorboard
        self.writer.add_scalar("test/return", test_results["return"], iteration)
        for step in range(len(test_results["run_cost"])):
            self.writer.add_scalar(
                "test/run_cost", test_results["run_cost"][step], step
            )
        self.writer.add_scalar(
            "train/total_loss", test_results["total_loss"], iteration
        )
        self.writer.add_scalar(
            "train/policy_loss", test_results["policy_loss"], iteration
        )
        self.writer.add_scalar(
            "train/value_loss", test_results["value_loss"], iteration
        )
        self.writer.add_scalar("time/total_time", test_results["total_time"], iteration)
        self.writer.add_scalar(
            "time/time_per_iter", test_results["time_per_iter"], iteration
        )

        # Logging
        print(
            f"--------------------------------------- \n"
            f'return: {round(test_results["return"], 2)} \n'
            f'total_loss: {round(test_results["total_loss"], 2)} \n'
            f'policy_loss: {round(test_results["policy_loss"], 2)} \n'
            f'value_loss: {round(test_results["value_loss"], 2)} \n'
            f'time_per_iter: {round(test_results["time_per_iter"], 2)} \n'
            f'total_time: {round(test_results["total_time"], 2)} \n'
            f"--------------------------------------- \n"
        )

        # Save the trained model
        # TBU
