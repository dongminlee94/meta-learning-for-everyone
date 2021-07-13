"""
Meta-train and meta-test codes with RL^2 algorithm
"""


# import datetime
# import os
# import time

from rl2.algorithm.utils.buffer import Buffer
from rl2.algorithm.utils.sampler import Sampler

# import numpy as np
# from torch.utils.tensorboard import SummaryWriter


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
        # exp_name,
        # file_name,
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
            max_size=config["max_buffer_size"],
            device=device,
        )

    def meta_train(self):
        """RL^2 meta-training"""
        # total_start_time = time.time()
        for iteration in range(self.train_iters):
            # start_time = time.time()

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
            log_values = self.agent.train_model(self.train_grad_iters, batch)
            print(log_values)

            # Evaluate on test tasks
            # self.meta_test(iteration, total_start_time, start_time, log_values)

    def meta_test(self):
        """RL^2 meta-testing"""
        for _ in range(self.train_tasks):
            pass
        return NotImplementedError
