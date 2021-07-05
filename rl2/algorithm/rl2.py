"""
Meta-train and meta-test codes with RL^2 algorithm
"""

# import numpy as np
# import torch
from rl2.algorithm.utils.buffer import Buffer
from rl2.algorithm.utils.sampler import Sampler


class RL2:  # pylint: disable=too-many-instance-attributes
    """RL^2 algorithm class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env,
        agent,
        observ_dim,
        action_dim,
        train_tasks,
        eval_tasks,
        device,
        **config,
    ):

        self.agent = agent
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks

        self.num_iterations = config["num_iterations"]
        self.num_outer_optim = config["num_outer_optim"]
        self.num_inner_adapt = config["num_inner_adapt"]

        self.sampler = Sampler(env=env, agent=agent, max_step=config["max_step"])

        self.buffer = Buffer(
            observ_dim=observ_dim,
            action_dim=action_dim,
            max_size=config["max_buffer_size"],
            device=device,
        )

        self._total_samples = 0
        self._total_train_steps = 0

    # def compute_loss(self):
    #     pass

    def meta_train(self):
        """Meta-train loop"""
        for _ in range(self.train_tasks):
            for _ in range(self.num_inner_adapt):
                pass
        return NotImplementedError

    def meta_test(self):
        """Meta-test loop"""
        for _ in range(self.train_tasks):
            for _ in range(self.num_inner_adapt):
                pass
        return NotImplementedError
