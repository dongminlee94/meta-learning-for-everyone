"""
Simple buffer code
"""

import numpy as np
import torch


class Buffer:  # pylint: disable=too-many-instance-attributes
    """Simple buffer class that includes computing return and gae"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        observ_dim,
        action_dim,
        max_size,
        device,
        gamma=0.99,
        lamda=0.97,
    ):

        self._cur_obs = np.zeros((max_size, observ_dim))
        self._actions = np.zeros((max_size, action_dim))
        self._rewards = np.zeros((max_size, 1))
        self._next_obs = np.zeros((max_size, observ_dim))
        self._dones = np.zeros((max_size, 1), dtype="uint8")
        self._returns = np.zeros((max_size, 1))
        self._advants = np.zeros((max_size, 1))
        self._values = np.zeros((max_size, 1))
        self._log_probs = np.zeros((max_size, 1))
        self.device = device
        self.gamma = gamma
        self.lamda = lamda

        self._max_size = max_size
        self._top = 0

    # pylint: disable=too-many-arguments
    def add(self, obs, action, reward, next_obs, done, value, log_prob):
        """Add transition, value, and log_prob to buffer"""
        assert self._top < self._max_size
        self._cur_obs[self._top] = obs
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._next_obs[self._top] = next_obs
        self._dones[self._top] = done
        self._values[self._top] = value
        self._log_probs[self._top] = log_prob
        self._top += 1

    def add_trajs(self, trajs):
        """Add trajectories to buffer"""
        for traj in trajs:
            for (obs, action, reward, next_obs, done, value, log_prob) in zip(
                traj["cur_obs"],
                traj["actions"],
                traj["rewards"],
                traj["next_obs"],
                traj["dones"],
                traj["values"],
                traj["log_probs"],
            ):
                self.add(obs, action, reward, next_obs, done, value, log_prob)

    def compute_gae(self):
        """Compute return and GAE"""
        prev_value = 0
        running_return = 0
        running_advant = 0

        for t in reversed(range(0, len(self._rewards))):
            # Compute return
            running_return = (
                self._rewards[t] + self.gamma * (1 - self._dones[t]) * running_return
            )
            self._returns[t] = running_return

            # Compute GAE
            running_tderror = (
                self._rewards[t]
                + self.gamma * (1 - self._dones[t]) * prev_value
                - self._values[t]
            )
            running_advant = (
                running_tderror
                + self.gamma * self.lamda * (1 - self._dones[t]) * running_advant
            )
            self._advants[t] = running_advant
            prev_value = self._values[t]

        # Normalize advantage
        self._advants = (self._advants - self._advants.mean()) / self._advants.std()

    def get_samples(self):
        """Get sample batch in buffer"""
        assert self._top == self._max_size
        self._top = 0

        self.compute_gae()
        batch = dict(
            obs=self._cur_obs,
            actions=self._actions,
            returns=self._returns,
            advants=self._advants,
            log_probs=self._log_probs,
        )
        return {
            key: torch.Tensor(value).to(self.device) for key, value in batch.items()
        }
