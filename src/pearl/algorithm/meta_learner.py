"""
Meta-train and meta-test implementations with PEARL algorithm
"""


import datetime
import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.pearl.algorithm.buffers import MultiTaskReplayBuffer
from src.pearl.algorithm.sampler import Sampler
from src.pearl.algorithm.torch_utils import np_to_torch_batch, unpack_batch


class MetaLearner:  # pylint: disable=too-many-instance-attributes
    """PEARL meta-learner class"""

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
        self.device = device

        self.num_iterations = config["num_iterations"]
        self.num_sample_tasks = config["num_sample_tasks"]

        self.num_init_samples = config["num_init_samples"]
        self.num_prior_samples = config["num_prior_samples"]
        self.num_posterior_samples = config["num_posterior_samples"]

        self.num_meta_grads = config["num_meta_grads"]
        self.meta_batch_size = config["meta_batch_size"]
        self.batch_size = config["batch_size"]
        self.max_step = config["max_step"]

        self.sampler = Sampler(env=env, agent=agent, max_step=config["max_step"], device=device)

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

    def collect_train_data(self, task_index, max_samples, update_posterior, add_to_enc_buffer):
        """Data collecting for meta-train"""
        self.agent.encoder.clear_z()
        self.agent.policy.is_deterministic = False

        cur_samples = 0
        while cur_samples < max_samples:
            trajs, num_samples = self.sampler.obtain_samples(
                max_samples=max_samples - cur_samples,
                update_posterior=update_posterior,
                accum_context=False,
            )
            cur_samples += num_samples

            self.rl_replay_buffer.add_trajs(task_index, trajs)
            if add_to_enc_buffer:
                self.encoder_replay_buffer.add_trajs(task_index, trajs)

            if update_posterior:
                context_batch = self.sample_context([task_index])
                self.agent.encoder.infer_posterior(context_batch)

    def sample_context(self, indices):
        """Sample batch of context from a list of tasks from the replay buffer"""
        # This batch consists of context sampled randomly from encoder's replay buffer
        context_batch = []
        for index in indices:
            batch = self.encoder_replay_buffer.sample_batch(task=index, batch_size=self.batch_size)
            batch = np_to_torch_batch(batch)
            batch = unpack_batch(batch)
            context_batch.append(batch)

        # Group like elements together
        context_batch = [
            [context[i] for context in context_batch] for i in range(len(context_batch[0]))
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
            batch = self.rl_replay_buffer.sample_batch(task=index, batch_size=self.batch_size)
            batch = np_to_torch_batch(batch)
            batch = unpack_batch(batch)
            transition_batch.append(batch)

        # Group like elements together
        transition_batch = [
            [transition[i] for transition in transition_batch] for i in range(len(transition_batch[0]))
        ]
        transition_batch = [
            torch.cat(transition, dim=0).to(self.device) for transition in transition_batch
        ]
        return transition_batch

    def meta_train(self):
        """PEARL meta-training"""
        total_start_time = time.time()
        for iteration in range(self.num_iterations):
            start_time = time.time()

            if iteration == 0:
                for index in self.train_tasks:
                    self.env.reset_task(index)
                    self.collect_train_data(
                        task_index=index,
                        max_samples=self.num_init_samples,
                        update_posterior=False,
                        add_to_enc_buffer=True,
                    )

            print(f"=============== Iteration {iteration} ===============")
            # Sample data randomly from train tasks.
            for i in range(self.num_sample_tasks):
                index = np.random.randint(len(self.train_tasks))
                self.env.reset_task(index)
                self.encoder_replay_buffer.task_buffers[index].clear()

                # Collect some trajectories with z ~ prior r(z)
                if self.num_prior_samples > 0:
                    print(f"[{i + 1}/{self.num_sample_tasks}] collecting samples with prior")
                    self.collect_train_data(
                        task_index=index,
                        max_samples=self.num_prior_samples,
                        update_posterior=False,
                        add_to_enc_buffer=True,
                    )

                # Even if encoder is trained only on samples from the prior r(z),
                # the policy needs to learn to handle z ~ posterior q(z|c)
                if self.num_posterior_samples > 0:
                    print(f"[{i + 1}/{self.num_sample_tasks}] collecting samples with posterior")
                    self.collect_train_data(
                        task_index=index,
                        max_samples=self.num_posterior_samples,
                        update_posterior=True,
                        add_to_enc_buffer=False,
                    )

            # Sample train tasks and compute gradient updates on parameters.
            print(f"Start meta-gradient updates of iteration {iteration}")
            for i in range(self.num_meta_grads):
                indices = np.random.choice(self.train_tasks, self.meta_batch_size)

                # Zero out context and hidden encoder state
                self.agent.encoder.clear_z(num_tasks=len(indices))

                # Sample context batch
                context_batch = self.sample_context(indices)

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

            # Evaluate on test tasks
            self.meta_test(iteration, total_start_time, start_time, log_values)

    def collect_test_data(self, max_samples, update_posterior):
        """Data collecting for meta-test"""
        self.agent.encoder.clear_z()
        self.agent.policy.is_deterministic = True

        cur_trajs = []
        cur_samples = 0
        while cur_samples < max_samples:
            trajs, num_samples = self.sampler.obtain_samples(
                max_samples=max_samples - cur_samples,
                update_posterior=update_posterior,
                accum_context=True,
            )
            cur_trajs += trajs
            cur_samples += num_samples
            self.agent.encoder.infer_posterior(self.agent.encoder.context)
        return cur_trajs

    # pylint: disable=too-many-locals
    def meta_test(self, iteration, total_start_time, start_time, log_values):
        """PEARL meta-testing"""
        test_results = {}
        return_before_infer = 0
        return_after_infer = 0
        run_cost_before_infer = np.zeros(self.max_step)
        run_cost_after_infer = np.zeros(self.max_step)

        for index in self.test_tasks:
            self.env.reset_task(index)
            trajs = self.collect_test_data(
                max_samples=self.max_step * 2,
                update_posterior=True,
            )
            return_before_infer += sum(trajs[0]["rewards"])[0]
            return_after_infer += sum(trajs[1]["rewards"])[0]
            if self.env_name == "cheetah-vel":
                for i in range(self.max_step):
                    run_cost_before_infer[i] += trajs[0]["infos"][i]
                    run_cost_after_infer[i] += trajs[1]["infos"][i]

        # Collect meta-test results
        test_results["return_before_infer"] = return_before_infer / len(self.test_tasks)
        test_results["return_after_infer"] = return_after_infer / len(self.test_tasks)
        if self.env_name == "cheetah-vel":
            test_results["run_cost_before_infer"] = run_cost_before_infer / len(self.test_tasks)
            test_results["run_cost_after_infer"] = run_cost_after_infer / len(self.test_tasks)
            test_results["total_run_cost_before_infer"] = sum(
                abs(test_results["run_cost_before_infer"])
            )
            test_results["total_run_cost_after_infer"] = sum(abs(test_results["run_cost_after_infer"]))
        test_results["policy_loss"] = log_values["policy_loss"]
        test_results["qf1_loss"] = log_values["qf1_loss"]
        test_results["qf2_loss"] = log_values["qf2_loss"]
        test_results["encoder_loss"] = log_values["encoder_loss"]
        test_results["alpha_loss"] = log_values["alpha_loss"]
        test_results["alpha"] = log_values["alpha"]
        test_results["z_mean"] = log_values["z_mean"]
        test_results["z_var"] = log_values["z_var"]
        test_results["total_time"] = time.time() - total_start_time
        test_results["time_per_iter"] = time.time() - start_time

        # Tensorboard
        self.writer.add_scalar(
            "test/return_before_infer", test_results["return_before_infer"], iteration
        )
        self.writer.add_scalar("test/return_after_infer", test_results["return_after_infer"], iteration)
        if self.env_name == "cheetah-vel":
            self.writer.add_scalar(
                "test/total_run_cost_before_infer",
                test_results["total_run_cost_before_infer"],
                iteration,
            )
            self.writer.add_scalar(
                "test/total_run_cost_after_infer", test_results["total_run_cost_after_infer"], iteration
            )
            for step in range(len(test_results["run_cost_before_infer"])):
                self.writer.add_scalar(
                    "run_cost_before_infer/iteration_" + str(iteration),
                    test_results["run_cost_before_infer"][step],
                    step,
                )
                self.writer.add_scalar(
                    "run_cost_after_infer/iteration_" + str(iteration),
                    test_results["run_cost_after_infer"][step],
                    step,
                )
        self.writer.add_scalar("train/policy_loss", test_results["policy_loss"], iteration)
        self.writer.add_scalar("train/qf1_loss", test_results["qf1_loss"], iteration)
        self.writer.add_scalar("train/qf2_loss", test_results["qf2_loss"], iteration)
        self.writer.add_scalar("train/encoder_loss", test_results["encoder_loss"], iteration)
        self.writer.add_scalar("train/alpha_loss", test_results["alpha_loss"], iteration)
        self.writer.add_scalar("train/alpha", test_results["alpha"], iteration)
        self.writer.add_scalar("train/z_mean", test_results["z_mean"], iteration)
        self.writer.add_scalar("train/z_var", test_results["z_var"], iteration)
        self.writer.add_scalar("time/total_time", test_results["total_time"], iteration)
        self.writer.add_scalar("time/time_per_iter", test_results["time_per_iter"], iteration)

        # Logging
        print(
            f"--------------------------------------- \n"
            f'return: {round(test_results["return_after_infer"], 2)} \n'
            f'policy_loss: {round(test_results["policy_loss"], 2)} \n'
            f'qf1_loss: {round(test_results["qf1_loss"], 2)} \n'
            f'qf2_loss: {round(test_results["qf2_loss"], 2)} \n'
            f'encoder_loss: {round(test_results["encoder_loss"], 2)} \n'
            f'alpha_loss: {round(test_results["alpha_loss"], 2)} \n'
            f'alpha: {round(test_results["alpha"], 2)} \n'
            f'z_mean: {test_results["z_mean"]} \n'
            f'z_var: {test_results["z_var"]} \n'
            f'time_per_iter: {round(test_results["time_per_iter"], 2)} \n'
            f'total_time: {round(test_results["total_time"], 2)} \n'
            f"--------------------------------------- \n"
        )

        # Save the trained model
        # TBU
