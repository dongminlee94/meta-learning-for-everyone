import os
import datetime
import numpy as np
from collections import OrderedDict

import torch
from torch.utils.tensorboard import SummaryWriter

from algorithm.utils.util import *
from algorithm.utils.sampler import InPlacePathSampler
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
        self.meta_batch = config['meta_batch']
        self.num_iterations = config['num_iterations']
        self.num_train_steps_per_itr = config['num_train_steps_per_itr']
        self.num_initial_steps = config['num_initial_steps']
        self.num_tasks_sample = config['num_tasks_sample']
        self.num_steps_prior = config['num_steps_prior']
        self.num_extra_rl_steps_posterior = config['num_extra_rl_steps_posterior']
        self.num_evals = config['num_evals']
        self.num_steps_per_eval = config['num_evals']
        self.embedding_batch_size = config['embedding_batch_size']
        self.embedding_mini_batch_size = config['embedding_mini_batch_size']
        self.replay_buffer_size = config['replay_buffer_size']
        self.num_exp_traj_eval = config['num_exp_traj_eval']
        
        self.sampler = InPlacePathSampler(
            env=env,
            policy=agent.policy,
            encoder=agent.encoder,
            max_path_length=config['max_path_length'],
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

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0