"""
Meta-train and meta-test codes with RL^2 algorithm
"""


# import datetime
# import os
# import time

# import numpy as np
# import torch
# from torch.utils.tensorboard import SummaryWriter

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
        eval_tasks,
        device,
        **config,
    ):

        self.agent = agent
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks

        self.train_iters = config["train_iters"]
        self.train_max_samples = config["train_max_samples"]

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
                        index + 1, self.train_task_iters
                    )
                )
                trajs = self.sampler.obtain_trajs(max_samples=self.train_max_samples)
                self.buffer.add_trajs(index, trajs)

            # # Sample train tasks and compute gradient updates on parameters.
            # print("Start meta-gradient updates of iteration {}".format(iteration))
            # for i in range(self.meta_grad_iters):
            #     indices = np.random.choice(self.train_tasks, self.meta_batch_size)

            #     # Zero out context and hidden encoder state
            #     self.agent.encoder.clear_z(num_tasks=len(indices))

            #     # Sample context batch
            #     context_batch = self.sample_context(indices)

            #     # Sample transition batch
            #     transition_batch = self.sample_transition(indices)

            #     # Train the policy, Q-functions and the encoder
            #     log_values = self.agent.train_model(
            #         meta_batch_size=self.meta_batch_size,
            #         batch_size=self.batch_size,
            #         context_batch=context_batch,
            #         transition_batch=transition_batch,
            #     )

            #     # Stop backprop
            #     self.agent.encoder.task_z.detach()

            # Evaluate on test tasks
            # self.meta_test(iteration, total_start_time, start_time, log_values)

    def meta_test(self):
        """Meta-test loop"""
        for _ in range(self.train_tasks):
            pass
        return NotImplementedError
