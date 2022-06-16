"""
Multi-task replay buffer implementation
"""


from typing import Dict, List

import numpy as np


class MultiTaskReplayBuffer:
    """Multi-task replay buffer class"""

    def __init__(
        self,
        observ_dim: int,
        action_dim: int,
        tasks: List[int],
        max_size: int,
    ) -> None:

        self.task_buffers = {}
        for i in tasks:
            self.task_buffers[i] = SimpleReplayBuffer(
                observ_dim=observ_dim,
                action_dim=action_dim,
                max_size=max_size,
            )

    def add_trajs(self, task: int, trajs: List[Dict[str, np.ndarray]]) -> None:
        """Add trajectories of the task to multi-task replay buffer"""
        for traj in trajs:
            self.task_buffers[task].add_traj(traj)

    def sample_batch(self, task: int, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch of the task in multi-task replay buffer"""
        return self.task_buffers[task].sample(batch_size)


class SimpleReplayBuffer:
    """Single task replay buffer code"""

    def __init__(
        self,
        observ_dim: int,
        action_dim: int,
        max_size: int,
    ) -> None:

        self._cur_obs = np.zeros((max_size, observ_dim))
        self._actions = np.zeros((max_size, action_dim))
        self._rewards = np.zeros((max_size, 1))
        self._next_obs = np.zeros((max_size, observ_dim))
        self._dones = np.zeros((max_size, 1), dtype="uint8")
        self._max_size = max_size
        self._top = 0
        self._size = 0

    def clear(self) -> None:
        """Clear variables of replay buffer"""
        self._top = 0
        self._size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
    ) -> None:
        """Add transition to replay buffer"""
        self._cur_obs[self._top] = obs
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._next_obs[self._top] = next_obs
        self._dones[self._top] = done

        self._top = (self._top + 1) % self._max_size
        if self._size < self._max_size:
            self._size += 1

    def add_traj(self, traj: Dict[str, np.ndarray]) -> None:
        """Add trajectory to replay buffer"""
        for (obs, action, reward, next_obs, done) in zip(
            traj["cur_obs"],
            traj["actions"],
            traj["rewards"],
            traj["next_obs"],
            traj["dones"],
        ):
            self.add(obs, action, reward, next_obs, done)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch in replay buffer"""
        indices: np.ndarray = np.random.randint(0, self._size, batch_size)
        return dict(
            cur_obs=self._cur_obs[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_obs=self._next_obs[indices],
            dones=self._dones[indices],
        )
