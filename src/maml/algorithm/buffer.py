"""
Simple buffer code
"""

import numpy as np
import torch


class MultiBuffer:
    """Multi-buffer class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        observ_dim,
        action_dim,
        num_tasks,
        num_episodes,
        max_size,
        device,
    ):
        self.num_episodes = num_episodes
        self.num_tasks = num_tasks
        self.num_buffers = num_tasks * num_episodes
        self.multi_buffers = {}
        for i in range(self.num_buffers):
            self.multi_buffers[i] = Buffer(
                observ_dim=observ_dim,
                action_dim=action_dim,
                max_size=max_size,
                device=device,
            )

    def add_trajs(self, cur_task, cur_adapt, trajs):
        """Add trajectories to the sub-buffer of multi-buffer"""
        buffer_index = self.num_tasks * cur_adapt + cur_task
        self.multi_buffers[buffer_index].add_trajs(trajs)

    def get_samples(self, cur_task, cur_adapt):
        """Sample batch of the sub-buffer in multi-buffer"""
        buffer_index = self.num_tasks * cur_adapt + cur_task
        batch = self.multi_buffers[buffer_index].get_samples()
        return batch

    def clear(self):
        """Clear variables of all sub-buffers"""
        for buffer_index in range(self.num_buffers):
            self.multi_buffers[buffer_index].clear()


class Buffer:  # pylint: disable=too-many-instance-attributes
    """Simple buffer class that includes computing return and gae"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        observ_dim,
        action_dim,
        max_size,
        device,
        gamma=0.99,
    ):

        self._cur_obs = np.zeros((max_size, observ_dim))
        self._actions = np.zeros((max_size, action_dim))
        self._rewards = np.zeros((max_size, 1))
        self._next_obs = np.zeros((max_size, observ_dim))
        self._dones = np.zeros((max_size, 1), dtype="uint8")
        self._returns = np.zeros((max_size, 1))
        self._advants = np.zeros((max_size, 1))
        self._log_probs = np.zeros((max_size, 1))
        self.device = device
        self.gamma = gamma

        self._max_size = max_size
        self._top = 0

    # pylint: disable=too-many-arguments
    def add(self, obs, action, reward, next_obs, done, log_prob):
        """Add transition, value, and log_prob to buffer"""
        assert self._top < self._max_size
        self._cur_obs[self._top] = obs
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._next_obs[self._top] = next_obs
        self._dones[self._top] = done
        self._log_probs[self._top] = log_prob
        self._top += 1

    def add_trajs(self, trajs):
        """Add trajectories to the buffer"""
        for traj in trajs:
            for (obs, action, reward, next_obs, done, log_prob) in zip(
                traj["cur_obs"],
                traj["actions"],
                traj["rewards"],
                traj["next_obs"],
                traj["dones"],
                traj["log_probs"],
            ):
                self.add(obs, action, reward, next_obs, done, log_prob)

    def compute_return(self):
        """Compute return"""
        running_return = 0

        for t in reversed(range(0, len(self._rewards))):
            # Compute return
            running_return = self._rewards[t] + self.gamma * (1 - self._dones[t]) * running_return
            self._returns[t] = running_return

    @staticmethod
    def compute_gae(rewards, dones, values):
        """Compute GAE"""
        gamma = 0.99
        lamda = 1
        prev_value = 0
        running_advant = 0
        rewards = rewards.numpy()
        dones = dones.numpy()
        values = values.numpy()
        advants = np.zeros_like(rewards)

        for t in reversed(range(0, len(rewards))):
            # Compute GAE
            running_tderror = rewards[t] + gamma * (1 - dones[t]) * prev_value - values[t]
            running_advant = running_tderror + gamma * lamda * (1 - dones[t]) * running_advant
            advants[t] = running_advant
            prev_value = values[t]

        # Normalize advantage
        advants = (advants - advants.mean()) / advants.std()
        return torch.Tensor(advants)

    def get_samples(self):
        """Get sample batch in buffer"""
        assert self._top == self._max_size

        self.compute_return()
        batch = dict(
            obs=self._cur_obs,
            actions=self._actions,
            rewards=self._rewards,
            dones=self._dones,
            returns=self._returns,
            log_probs=self._log_probs,
        )
        return {key: torch.Tensor(value).to(self.device) for key, value in batch.items()}

    def clear(self):
        """Clear variables of replay buffer"""
        self._top = 0
