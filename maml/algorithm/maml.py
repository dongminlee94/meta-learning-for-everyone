"""
Meta-train and meta-test codes with MAML algorithm
"""


import datetime
import os

# import numpy as np
# import torch
from torch.utils.tensorboard import SummaryWriter

from maml.algorithm.utils.buffer import Buffer
from maml.algorithm.utils.sampler import Sampler

# import time


class MAML:  # pylint: disable=too-many-instance-attributes
    """MAML algorithm class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env,
        agent,
        observ_dim,
        action_dim,
        train_tasks,
        eval_tasks,
        exp_name,
        file_name,
        device,
        **config,
    ):

        self.env = env
        self.agent = agent
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.device = device

        self.num_iterations = config["num_iterations"]
        self.num_outer_optim = config["num_outer_optim"]
        self.num_inner_adapt = config["num_inner_adapt"]
        self.max_step = config["max_step"]

        self.sampler = Sampler(env=env, agent=agent, max_step=config["max_step"])

        self.buffer = Buffer(
            observ_dim=observ_dim,
            action_dim=action_dim,
            max_size=config["max_buffer_size"],
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
