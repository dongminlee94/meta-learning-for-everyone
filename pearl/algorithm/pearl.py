"""
Meta-train and meta-test codes with PEARL algorithm
"""


import numpy as np
import torch
from algorithm.utils.buffers import MultiTaskReplayBuffer
from algorithm.utils.sampler import Sampler
from algorithm.utils.torch_utils import np_to_torch_batch, unpack_batch

# from torch.utils.tensorboard import SummaryWriter


class PEARL:  # pylint: disable=too-many-instance-attributes
    """PEARL algorithm class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env,
        agent,
        train_tasks,
        eval_tasks,
        device,
        **config,
    ):

        self.env = env
        self.agent = agent
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.device = device

        self.num_iterations = config["num_iterations"]
        self.num_init_samples = config["num_init_samples"]
        self.num_task_samples = config["num_task_samples"]
        self.num_prior_samples = config["num_prior_samples"]
        self.num_posterior_samples = config["num_posterior_samples"]

        self.num_meta_training = config["num_meta_training"]
        self.meta_batch_size = config["meta_batch_size"]
        self.batch_size = config["batch_size"]

        self.sampler = Sampler(
            env=env, agent=agent, max_step=config["max_step"], device=device
        )

        # separate replay buffers for
        # - training RL update
        # - training encoder update
        self.rl_replay_buffer = MultiTaskReplayBuffer(
            env=env,
            tasks=train_tasks,
            max_size=config["max_buffer_size"],
        )
        self.encoder_replay_buffer = MultiTaskReplayBuffer(
            env=env,
            tasks=train_tasks,
            max_size=config["max_buffer_size"],
        )

        self._total_samples = 0
        self._total_train_steps = 0

    def meta_train(self):
        """Meta-training loop"""
        for iteration in range(self.num_iterations):
            if iteration == 0:
                print(
                    "[{0} training tasks] collecting initial samples with prior".format(
                        len(self.train_tasks)
                    )
                )
                for index in self.train_tasks:
                    self.env.reset_task(index)
                    self.collect_data(
                        task_index=index,
                        num_samples=self.num_init_samples,
                        update_posterior=False,
                        add_to_enc_buffer=True,
                    )

            print("=============== Iteration {0} ===============".format(iteration))
            # Sample data randomly from train tasks.
            for i in range(self.num_task_samples):
                index = np.random.randint(len(self.train_tasks))
                self.env.reset_task(index)
                self.encoder_replay_buffer.task_buffers[index].clear()

                # Collect some trajectories with z ~ prior r(z)
                if self.num_prior_samples > 0:
                    print(
                        "[{0}/{1} sampled tasks] \
                            collecting samples with prior".format(
                            i + 1, self.num_task_samples
                        )
                    )
                    self.collect_data(
                        task_index=index,
                        num_samples=self.num_prior_samples,
                        update_posterior=False,
                        add_to_enc_buffer=True,
                    )

                # Even if encoder is trained only on samples from the prior r(z),
                # the policy needs to learn to handle z ~ posterior q(z|c)
                if self.num_posterior_samples > 0:
                    print(
                        "[{0}/{1} sampled tasks] \
                            collecting samples with posterior".format(
                            i + 1, self.num_task_samples
                        )
                    )
                    self.collect_data(
                        task_index=index,
                        num_samples=self.num_posterior_samples,
                        update_posterior=True,
                        add_to_enc_buffer=False,
                    )

            # Sample train tasks and compute gradient updates on parameters.
            print("Start meta-training of iteration {0}".format(iteration))
            for i in range(self.num_meta_training):
                indices = np.random.choice(self.train_tasks, self.meta_batch_size)

                # Zero out context and hidden encoder state
                self.agent.encoder.clear_z(num_tasks=len(indices))

                # Sample context batch
                context_batch = self.sample_context(indices)  # torch.Size([4, 256, 33])

                # Sample transition batch
                transition_batch = self.sample_transition(indices)

                # Train the policy, Q-functions and the encoder
                self.agent.train_model(
                    meta_batch_size=self.meta_batch_size,
                    batch_size=self.batch_size,
                    context_batch=context_batch,
                    transition_batch=transition_batch,
                )

                # Stop backprop
                self.agent.encoder.z.detach()

                self._total_train_steps += 1
        return dict(
            total_samples=self._total_samples,
            total_train_steps=self._total_train_steps,
        )

    def collect_data(
        self, task_index, num_samples, update_posterior, add_to_enc_buffer
    ):
        """Data collecting for training"""
        self.agent.encoder.clear_z()

        cur_samples = 0
        while cur_samples < num_samples:
            trajs, samples = self.sampler.obtain_samples(
                max_samples=num_samples - cur_samples,
                min_trajs=int(update_posterior),
                accum_context=False,
            )
            cur_samples += samples

            self.rl_replay_buffer.add_trajs(task_index, trajs)
            if add_to_enc_buffer:
                self.encoder_replay_buffer.add_trajs(task_index, trajs)

            if update_posterior:
                context_batch = self.sample_context([task_index])
                self.agent.encoder.infer_posterior(context_batch)
        self._total_samples += cur_samples

    def sample_context(self, indices):
        """Sample batch of context from a list of tasks from the replay buffer"""
        # This batch consists of context sampled randomly from encoder's replay buffer
        context_batch = []
        for index in indices:
            batch = self.encoder_replay_buffer.sample(
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
            batch = self.rl_replay_buffer.sample(task=index, batch_size=self.batch_size)
            batch = np_to_torch_batch(batch)
            batch = unpack_batch(batch)
            transition_batch.append(batch)

        # Group like elements together
        transition_batch = [
            [transition[i] for transition in transition_batch]
            for i in range(len(transition_batch[0]))
        ]
        transition_batch = [
            torch.cat(transition, dim=0) for transition in transition_batch
        ]
        return transition_batch.to(self.device)
