"""
Meta-train and meta-test codes with MAML algorithm
"""


import datetime
import os

import higher
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from maml.algorithm.buffer import Buffer
from maml.algorithm.sampler import Sampler


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

        self.inner_adaptation_steps = config["inner_adaptation_steps"]
        self.meta_grad_iters = config["maml_optimizer_steps"]

        self.sampler = Sampler(
            env=env,
            agent=agent,
            action_dim=action_dim,
            device=device,
        )

        self.meta_buffer = Buffer(
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

        self.meta_optimizer = torch.optim.Adam(
            list(self.agent.policy.parameters()),
            lr=config["learning_rate"],
        )
        self.inner_optimizer = torch.optim.SGD(
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

    def meta_train(self):
        """MAML meta-training"""
        for iteration in range(self.train_iters):

            print("=============== Iteration {} ===============".format(iteration))
            # Sample tasks randomly from train tasks distribution.
            for _ in range(self.num_sample_tasks):
                index = np.random.randint(len(self.train_tasks))
                self.env.reset_task(index)
                self.agent.policy.is_deterministic = False
                print("[{0}/{1}] collecting losses".format(index + 1, len(self.train_tasks)))

                # Branch policy and optimizer
                with higher.innerloop_ctx(
                    self.agent.policy,
                    self.inner_optimizer,
                    copy_initial_weights=False,
                ) as (inner_policy, branch_optimizer):

                    for _ in range(self.inner_adaptation_steps):
                        # Sample trajectory with inner policy
                        trajs = self.sampler.obtain_trajs(
                            inner_policy,
                            max_samples=self.train_samples,
                            max_step=self.max_step,
                        )
                        self.inner_buffer.add_trajs(trajs)
                        print("inner_buffer")
                        batch = self.inner_buffer.get_samples()
                        # Inner update
                        ppo_loss = self.agent.compute_losses(inner_policy, batch)
                        branch_optimizer.step(ppo_loss)

                    # Sample trajectory with adapted policy
                    inner_policy.is_deterministic = False
                    trajs = self.sampler.obtain_trajs(
                        inner_policy,
                        max_samples=self.train_samples,
                        max_step=self.max_step,
                    )
                    self.meta_buffer.add_trajs(trajs)

            # Meta update
            print("meta_buffer")
            batch = self.meta_buffer.get_samples()
            for _ in range(self.meta_grad_iters):
                meta_loss = self.agent.compute_losses(self.agent.policy, batch)
                self.meta_optimizer.zero_grad()
                meta_loss.backward()
                self.meta_optimizer.step()

            # Evaluate on test tasks
            self.meta_test()
        return NotImplementedError

    def meta_test(self):
        """MAML meta-testing"""
        for index in self.test_tasks:
            self.env.reset_task(index)
            self.agent.policy.is_deterministic = True
        return NotImplementedError
