"""
Meta-train and meta-test codes with PEARL algorithm
"""


import os
import time

import numpy as np
import torch
from algorithm.utils.buffers import MultiTaskReplayBuffer
from algorithm.utils.sampler import Sampler
from algorithm.utils.torch_utils import np_to_torch_batch, unpack_batch
from torch.utils.tensorboard import SummaryWriter


class PEARL:  # pylint: disable=too-many-instance-attributes
    """PEARL algorithm class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env,
        agent,
        observ_dim,
        action_dim,
        train_tasks,
        test_tasks,
        filename,
        device,
        **config,
    ):

        self.env = env
        self.agent = agent
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        self.filename = filename
        self.device = device

        self.train_iters = config["train_iters"]
        self.train_task_iters = config["train_task_iters"]

        self.train_init_samples = config["train_init_samples"]
        self.train_prior_samples = config["train_prior_samples"]
        self.train_posterior_samples = config["train_posterior_samples"]

        self.meta_grad_iters = config["meta_grad_iters"]
        self.meta_batch_size = config["meta_batch_size"]
        self.batch_size = config["batch_size"]

        self.test_iters = config["test_iters"]
        self.test_samples = config["test_samples"]

        self.sampler = Sampler(
            env=env, agent=agent, max_step=config["max_step"], device=device
        )

        # separate replay buffers for
        # - training RL update
        # - training encoder update
        self.rl_replay_buffer = MultiTaskReplayBuffer(
            observ_dim=observ_dim,
            action_dim=action_dim,
            tasks=train_tasks,
            max_size=config["max_buffer_size"],
        )
        self.encoder_replay_buffer = MultiTaskReplayBuffer(
            observ_dim=observ_dim,
            action_dim=action_dim,
            tasks=train_tasks,
            max_size=config["max_buffer_size"],
        )

        self.writer = SummaryWriter(log_dir=os.path.join(".", "results", filename))

        self.train_total_samples = 0
        self.train_total_steps = 0
        self.test_results = {}

    def meat_train(self):
        """PEARL meta-training"""
        total_start_time = time.time()
        for iteration in range(self.train_iters):
            start_time = time.time()
            if iteration == 0:
                print(
                    "[{0}/{1}] collecting initial samples with prior".format(
                        len(self.train_tasks), len(self.train_tasks)
                    )
                )
                for index in self.train_tasks:
                    self.env.reset_task(index)
                    self.collect_data(
                        task_index=index,
                        max_samples=self.train_init_samples,
                        update_posterior=False,
                        add_to_enc_buffer=True,
                    )

            print("=============== Iteration {} ===============".format(iteration))
            # Sample data randomly from train tasks.
            for i in range(self.train_task_iters):
                index = np.random.randint(len(self.train_tasks))
                self.env.reset_task(index)
                self.encoder_replay_buffer.task_buffers[index].clear()

                # Collect some trajectories with z ~ prior r(z)
                if self.train_prior_samples > 0:
                    print(
                        "[{0}/{1}] collecting samples with prior".format(
                            i + 1, self.train_task_iters
                        )
                    )
                    self.collect_data(
                        task_index=index,
                        max_samples=self.train_prior_samples,
                        update_posterior=False,
                        add_to_enc_buffer=True,
                    )

                # Even if encoder is trained only on samples from the prior r(z),
                # the policy needs to learn to handle z ~ posterior q(z|c)
                if self.train_posterior_samples > 0:
                    print(
                        "[{0}/{1}] collecting samples with posterior".format(
                            i + 1, self.train_task_iters
                        )
                    )
                    self.collect_data(
                        task_index=index,
                        max_samples=self.train_posterior_samples,
                        update_posterior=True,
                        add_to_enc_buffer=False,
                    )

            # Sample train tasks and compute gradient updates on parameters.
            print("Start meta-gradient of iteration {}".format(iteration))
            for i in range(self.meta_grad_iters):
                indices = np.random.choice(self.train_tasks, self.meta_batch_size)

                # Zero out context and hidden encoder state
                self.agent.encoder.clear_z(num_tasks=len(indices))

                # Sample context batch
                context_batch = self.sample_context(indices)  # torch.Size([4, 256, 33])

                # Sample transition batch
                transition_batch = self.sample_transition(indices)

                # Train the policy, Q-functions and the encoder
                log_values = self.agent.train_model(
                    meta_batch_size=self.meta_batch_size,
                    batch_size=self.batch_size,
                    context_batch=context_batch,
                    transition_batch=transition_batch,
                )

                # Stop backprop
                self.agent.encoder.task_z.detach()

                self.train_total_steps += 1

            # Evaluate on test tasks
            self.meta_test(iteration, total_start_time, start_time, log_values)

    def collect_data(
        self, task_index, max_samples, update_posterior, add_to_enc_buffer
    ):
        """Data collecting for training"""
        self.agent.encoder.clear_z()

        curr_samples = 0
        while curr_samples < max_samples:
            trajs, num_samples = self.sampler.obtain_samples(
                max_samples=max_samples - curr_samples,
                min_trajs=int(update_posterior),
                accum_context=False,
            )
            curr_samples += num_samples

            self.rl_replay_buffer.add_trajs(task_index, trajs)
            if add_to_enc_buffer:
                self.encoder_replay_buffer.add_trajs(task_index, trajs)

            if update_posterior:
                context_batch = self.sample_context([task_index])
                self.agent.encoder.infer_posterior(context_batch)
        self.train_total_samples += curr_samples

    def sample_context(self, indices):
        """Sample batch of context from a list of tasks from the replay buffer"""
        # This batch consists of context sampled randomly from encoder's replay buffer
        context_batch = []
        for index in indices:
            batch = self.encoder_replay_buffer.sample_batch(
                task=index, batch_size=self.batch_size
            )
            batch = np_to_torch_batch(batch)
            batch = unpack_batch(batch)
            context_batch.append(batch)

        # Group like elements together
        context_batch = [
            [context[i] for context in context_batch]
            for i in range(len(context_batch[0]))
        ]
        context_batch = [torch.cat(context, dim=0) for context in context_batch]

        # Full context consists of [obs, action, reward, next_obs, done]
        # If dynamics don't change across tasks,
        # don't include next_obs and done in context
        context_batch = torch.cat(context_batch[:-2], dim=2)
        return context_batch.to(self.device)

    def sample_transition(self, indices):
        """
        Sample batch of transitions from a list of tasks for training the actor-critic
        """
        # This batch consists of transitions sampled randomly from RL's replay buffer
        transition_batch = []
        for index in indices:
            batch = self.rl_replay_buffer.sample_batch(
                task=index, batch_size=self.batch_size
            )
            batch = np_to_torch_batch(batch)
            batch = unpack_batch(batch)
            transition_batch.append(batch)

        # Group like elements together
        transition_batch = [
            [transition[i] for transition in transition_batch]
            for i in range(len(transition_batch[0]))
        ]
        transition_batch = [
            torch.cat(transition, dim=0).to(self.device)
            for transition in transition_batch
        ]
        return transition_batch

    def collect_trajs(self, index):
        """Data collecting for meta-test"""
        self.env.reset_task(index)
        self.agent.encoder.clear_z()

        traj_batch = []
        curr_samples = 0
        while curr_samples < self.test_samples:
            trajs, num_samples = self.sampler.obtain_samples(
                max_samples=self.test_samples - curr_samples,
                min_trajs=1,
                accum_context=True,
            )
            traj_batch += trajs
            curr_samples += num_samples
            self.agent.encoder.infer_posterior(self.agent.encoder.context)
        return traj_batch

    def meta_test(self, iteration, total_start_time, start_time, log_values):
        """PEARL meta-testing"""
        self.test_results = {}

        print("Evaluating on {} test tasks".format(len(self.test_tasks)))
        test_tasks_return = 0
        for index in self.test_tasks:
            test_iters_return = 0
            for _ in range(self.test_iters):
                trajs = self.collect_trajs(index)
                test_iters_return += np.mean([sum(traj["rewards"]) for traj in trajs])

            test_tasks_return += test_iters_return / self.test_iters

        self.test_results["return"] = test_tasks_return / len(self.test_tasks)
        self.test_results["policy_loss"] = log_values["policy_loss"]
        self.test_results["qf1_loss"] = log_values["qf1_loss"]
        self.test_results["qf2_loss"] = log_values["qf2_loss"]
        self.test_results["alpha_loss"] = log_values["alpha_loss"]
        self.test_results["z_mean"] = log_values["z_mean"]
        self.test_results["z_var"] = log_values["z_var"]
        self.test_results["total_time"] = time.time() - total_start_time
        self.test_results["time_per_iter"] = time.time() - start_time

        # Tensorboard
        self.writer.add_scalar("eval/return", self.test_results["return"], iteration)
        self.writer.add_scalar(
            "train/policy_loss", self.test_results["policy_loss"], iteration
        )
        self.writer.add_scalar(
            "train/qf1_loss", self.test_results["qf1_loss"], iteration
        )
        self.writer.add_scalar(
            "train/qf2_loss", self.test_results["qf2_loss"], iteration
        )
        self.writer.add_scalar(
            "train/alpha_loss", self.test_results["alpha_loss"], iteration
        )
        self.writer.add_scalar("train/z_mean", self.test_results["z_mean"], iteration)
        self.writer.add_scalar("train/z_var", self.test_results["z_var"], iteration)
        self.writer.add_scalar(
            "time/total_time", self.test_results["total_time"], iteration
        )
        self.writer.add_scalar(
            "time/time_per_iter", self.test_results["time_per_iter"], iteration
        )

        # Logging
        print(
            f"--------------------------------------- \n"
            f'return: {round(self.test_results["return"], 2)} \n'
            f'policy_loss: {round(self.test_results["policy_loss"], 2)} \n'
            f'qf1_loss: {round(self.test_results["qf1_loss"], 2)} \n'
            f'qf2_loss: {round(self.test_results["qf2_loss"], 2)} \n'
            f'alpha_loss: {round(self.test_results["alpha_loss"], 2)} \n'
            f'z_mean: {self.test_results["z_mean"]} \n'
            f'z_var: {self.test_results["z_var"]} \n'
            f'time_per_iter: {round(self.test_results["time_per_iter"], 2)} \n'
            f'total_time: {round(self.test_results["total_time"], 2)} \n'
            f"--------------------------------------- \n"
        )

        # Save the trained model
        # TBU
