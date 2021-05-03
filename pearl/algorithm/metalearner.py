import os
import datetime
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.tensorboard import SummaryWriter

from algorithm.utils.util import *
from algorithm.utils.sampler import Sampler
from algorithm.utils.buffers import MultiTaskReplayBuffer


class MetaLearner(object):
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
        self.num_initial_steps = config['num_initial_steps']
        self.num_random_sample = config['num_random_sample']
        self.num_prior_sample = config['num_prior_sample']
        self.num_posterior_sample = config['num_posterior_sample']
        self.num_meta_gradient = config['num_meta_gradient']
        self.meta_batch = config['meta_batch']

        self.num_evals = config['num_evals']
        self.num_steps_per_eval = config['num_steps_per_eval']
        self.embedding_batch_size = config['embedding_batch_size']
        self.embedding_mini_batch_size = config['embedding_mini_batch_size']
        self.replay_buffer_size = config['replay_buffer_size']
        self.num_exp_traj_eval = config['num_exp_traj_eval']
        
        self.sampler = Sampler(
            env=env,
            agent=agent,
            max_step=config['max_step'],
        )

        # separate replay buffers for
        # - training RL update
        # - training encoder update
        self.replay_buffer = MultiTaskReplayBuffer(
            env=env,
            tasks=train_tasks,
            max_replay_buffer_size=config['replay_buffer_size'],
        )
        self.enc_replay_buffer = MultiTaskReplayBuffer(
            env=env,
            tasks=train_tasks,
            max_replay_buffer_size=config['replay_buffer_size'],
        )

        self._total_train_steps = 0
        self._total_env_steps = 0
    
    def meta_train(self):
        ''' 
        Meta-training loop 
        At each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        '''
        for i in range(self.num_iterations):
            if i == 0:
                # Create a SummaryWriter object by TensorBoard
                # dir_name = 'runs/' + self.env._wrapped_env.env_name + '_200_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                # writer = SummaryWriter(log_dir=dir_name)

                for task_index in self.train_tasks:
                    self.env.reset_task(task_index)
                    self.collect_data(
                        task_index=task_index, 
                        num_samples=self.num_initial_steps, 
                        update_posterior=False,
                        add_to_enc_buffer=True,
                    )
            
            # Sample data from train tasks.
            for i in range(self.num_random_sample):
                index = np.random.randint(len(self.train_tasks))
                self.env.reset_task(index)
                self.enc_replay_buffer.task_buffers[index].clear()

                # Collect some trajectories with z ~ prior r(z)
                if self.num_prior_sample > 0:
                    print("[{0}/{1} randomly sampled tasks] collecting with Prior".format(
                        i+1, self.num_random_sample)
                    )
                    self.collect_data(
                        task_index=index,
                        num_samples=self.num_prior_sample, 
                        update_posterior=False,
                        add_to_enc_buffer=True,
                    )
                
                # Even if encoder is trained only on samples from the prior r(z), 
                # the policy needs to learn to handle z ~ posterior q(z|c)
                if self.num_posterior_sample > 0:
                    print("[{0}/{1} randomly sampled tasks] collecting with Posterior updates".format(
                        i+1, self.num_random_sample)
                    )
                    self.collect_data(
                        task_index=index,
                        num_samples=self.num_posterior_sample, 
                        update_posterior=True, 
                        add_to_enc_buffer=False,
                    )

            # Sample train tasks and compute gradient updates on parameters.
            for i in range(self.num_meta_gradient):
                indices = np.random.choice(self.train_tasks, self.meta_batch)
                
                mb_size = self.embedding_mini_batch_size
                num_updates = self.embedding_batch_size // mb_size

                # sample context batch
                context_batch = self.sample_context(indices)

                # zero out context and hidden encoder state
                self.clear_z(num_tasks=len(indices))

                # do this in a loop so we can truncate backprop in the recurrent encoder
                for i in range(num_updates):
                    context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
                    # self.agent.train_model(indices, context)

                    # stop backprop
                    self.agent.detach_z()
                
                self._total_train_steps_+= 1
                if (i+1) % 500 == 0:
                    print("[{0}/{1} meta-gradients]".format(i+1,self.num_meta_gradient))

            print("End Training")

    
    def collect_data(self, task_index, num_samples, update_posterior, add_to_enc_buffer):
        ''' Data collecting for training '''
        self.clear_z()

        cur_samples = 0
        while cur_samples < num_samples:
            trajs, samples = self.sampler.obtain_samples(
                max_samples=num_samples - cur_samples,
                min_trajs=int(update_posterior),
                accum_context=False,
            )
            cur_samples += samples

            self.replay_buffer.add_paths(task_index, trajs)
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(task_index, trajs)
            
            if update_posterior:
                context = self.sample_context(task_index)
                self.agent.encoder.infer_posterior(context)
        self._total_env_steps += cur_samples

    def sample_context(self, indices):
        ''' Sample batch of context from a list of tasks from the replay buffer '''
        # Make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        
        batches = [np_to_pytorch_batch(self.enc_replay_buffer.random_batch(index, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for index in indices]
        context = [unpack_batch(batch) for batch in batches]
        
        # Group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        
        # Full context consists of [observs, actions, rewards, next_observs, dones]
        # If dynamics don't change across tasks, don't include next_obs
        # Don't include dones in context
        context = torch.cat(context[:-2], dim=2)
        return context