"""
Multi-task replay buffer code
"""


import numpy as np


class MultiTaskReplayBuffer:
    """Multi-task replay buffer class"""

    def __init__(
        self,
        env,
        tasks,
        max_size,
    ):

        self.env = env
        self.task_buffers = {}

        for i in tasks:
            self.task_buffers[i] = SimpleReplayBuffer(
                observ_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                max_size=max_size,
            )

    def add_trajs(self, task, trajs):
        """Add trajectories of the task to multi-task replay buffer"""
        for traj in trajs:
            self.task_buffers[task].add_traj(traj)

    def sample(self, task, batch_size):
        """Sample batch of the task in multi-task replay buffer"""
        batch = self.task_buffers[task].sample(batch_size)
        return batch


class SimpleReplayBuffer:  # pylint: disable=too-many-instance-attributes
    """Single task replay buffer code"""

    def __init__(
        self,
        observ_dim,
        action_dim,
        max_size,
    ):

        self._curr_obs = np.zeros((max_size, observ_dim))
        self._actions = np.zeros((max_size, action_dim))
        self._rewards = np.zeros((max_size, 1))
        self._next_obs = np.zeros((max_size, observ_dim))
        self._dones = np.zeros((max_size, 1), dtype="uint8")
        self._max_size = max_size

        self._top = None
        self._size = None
        self._episode_starts = None
        self._cur_episode_start = None
        self.clear()

    def clear(self):
        """Clear variables of replay buffer"""
        self._top = 0
        self._size = 0
        self._episode_starts = []
        self._cur_episode_start = 0

    def add(
        self, obs, action, reward, next_obs, done
    ):  # pylint: disable=too-many-arguments
        """Add transition to replay buffer"""
        self._curr_obs[self._top] = obs
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._next_obs[self._top] = next_obs
        self._dones[self._top] = done

        self._top = (self._top + 1) % self._max_size
        if self._size < self._max_size:
            self._size += 1

    def termination(self):
        """Store episode beginning once episode is over"""
        self._episode_starts.append(self._cur_episode_start)
        self._cur_episode_start = self._top

    def add_traj(self, traj):
        """Add trajectory to replay buffer"""
        for (obs, action, reward, next_obs, done) in zip(
            traj["curr_obs"],
            traj["actions"],
            traj["rewards"],
            traj["next_obs"],
            traj["dones"],
        ):
            self.add(obs, action, reward, next_obs, done)
        self.termination()

    def sample(self, batch_size):
        """Sample batch in replay buffer"""
        indices = np.random.randint(0, self._size, batch_size)
        return dict(
            curr_obs=self._curr_obs[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_obs=self._next_obs[indices],
            dones=self._dones[indices],
        )
