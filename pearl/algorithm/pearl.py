import os
import datetime
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.tensorboard import SummaryWriter

import algorithm.utils.util as util
from algorithm.utils.sampler import Sampler
from algorithm.utils.buffers import MultiTaskReplayBuffer


class PEARL(object):
    def __init__(
        self,
        env,
        agent,
        train_tasks,
        eval_tasks,
        **config,
    ):

        self.env = env
        self.agent = agent
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks

        self.num_iterations = config['num_iterations']
        self.num_init_samples = config['num_init_samples']
        self.num_task_samples = config['num_task_samples']
        self.num_prior_samples = config['num_prior_samples']
        self.num_posterior_samples = config['num_posterior_samples']

        self.num_meta_gradient = config['num_meta_gradient']
        self.meta_batch = config['meta_batch']
        self.batch_size = config['batch_size']

        self.sampler = Sampler(
            env=env,
            agent=agent,
            max_step=config['max_step'],
        )

        # separate replay buffers for
        # - training RL update
        # - training encoder update
        self.rl_replay_buffer = MultiTaskReplayBuffer(
            env=env,
            tasks=train_tasks,
            max_size=config['max_buffer_size'],
        )
        self.encoder_replay_buffer = MultiTaskReplayBuffer(
            env=env,
            tasks=train_tasks,
            max_size=config['max_buffer_size'],
        )

        self._total_train_steps = 0
        self._total_samples = 0
    
    def meta_train(self):
        ''' 
        Meta-training loop 
        
        At each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        '''
        for i in range(self.num_iterations):
            if i == 0:
                for index in self.train_tasks:
                    self.env.reset_task(index)
                    self.collect_data(
                        task_index=index, 
                        num_samples=self.num_init_samples, 
                        update_posterior=False,
                        add_to_enc_buffer=True,
                    )
            
            # Sample data randomly from train tasks.
            for i in range(self.num_task_samples):
                index = np.random.randint(len(self.train_tasks))
                self.env.reset_task(index)
                self.encoder_replay_buffer.task_buffers[index].clear()

                # Collect some trajectories with z ~ prior r(z)
                if self.num_prior_samples > 0:
                    print("[{0}/{1} sampled tasks] collecting with Prior".format(
                        i+1, self.num_task_samples)
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
                    print("[{0}/{1} sampled tasks] collecting with Posterior updates".format(
                        i+1, self.num_task_samples)
                    )
                    self.collect_data(
                        task_index=index,
                        num_samples=self.num_posterior_samples, 
                        update_posterior=True, 
                        add_to_enc_buffer=False,
                    )

            # Sample train tasks and compute gradient updates on parameters.
            for i in range(self.num_meta_gradient):
                indices = np.random.choice(self.train_tasks, self.meta_batch)

                # Zero out context and hidden encoder state
                self.agent.encoder.clear_z(num_tasks=len(indices))
                
                # Sample context batch
                context_batch = self.sample_context(indices)
                context = context_batch[:, :self.batch_size, :]

                # Train the policy, Q-functions and the encoder
                # self.agent.train_model(indices, context)

                # Stop backprop
                self.agent.encoder.detach_z()
                
                self._total_train_steps += 1

    def collect_data(self, task_index, num_samples, update_posterior, add_to_enc_buffer):
        ''' Data collecting for training '''
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
        ''' Sample batch of context from a list of tasks from the replay buffer '''
        context_batch = []
        for index in indices:
            batch = self.encoder_replay_buffer.sample(
                task=index, 
                batch_size=self.batch_size, 
            )
            batch = util.np_to_pytorch_batch(batch)
            batch = util.unpack_batch(batch)
            context_batch.append(batch)    
        
        # Group like elements together
        context_batch = [[x[i] for x in context_batch] for i in range(len(context_batch[0]))]
        context_batch = [torch.cat(x, dim=0) for x in context_batch]
        
        # Full context consists of [observs, actions, rewards, next_observs, dones]
        # If dynamics don't change across tasks, don't include next_obs
        # Don't include dones in context
        context_batch = torch.cat(context_batch[:-2], dim=2)
        return context_batch