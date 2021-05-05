import numpy as np
from gym.spaces import Box, Discrete, Tuple


class MultiTaskReplayBuffer(object):
    def __init__(
        self,
        env,
        tasks,
        max_size,
    ):

        self.env = env
        self.task_buffers = dict([(index, SimpleReplayBuffer(
            observ_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            max_size=max_size,
        )) for index in tasks])

    def add_trajs(self, task, trajs):
        for traj in trajs:
            self.task_buffers[task].add_traj(traj)
    
    def sample(self, task, batch_size):
        batch = self.task_buffers[task].sample(batch_size)
        return batch


class SimpleReplayBuffer(object):
    def __init__(
        self, 
        observ_dim, 
        action_dim,
        max_size, 
    ):

        self._obs = np.zeros((max_size, observ_dim))
        self._next_obs = np.zeros((max_size, observ_dim))
        self._action = np.zeros((max_size, action_dim))
        self._reward = np.zeros((max_size, 1))
        self._done = np.zeros((max_size, 1), dtype='uint8')
        self._max_size = max_size
        self.clear()

    def clear(self):
        self._top = 0
        self._size = 0
        self._episode_starts = []
        self._cur_episode_start = 0

    def add(self, obs, action, reward, next_obs, done):
        self._obs[self._top] = obs
        self._action[self._top] = action
        self._reward[self._top] = reward
        self._next_obs[self._top] = next_obs
        self._done[self._top] = done
        
        self._top = (self._top + 1) % self._max_size
        if self._size < self._max_size:
            self._size += 1

    def termination(self):
        # Store the episode beginning once the episode is over
        # n.b. allows last episode to loop but whatever
        self._episode_starts.append(self._cur_episode_start)
        self._cur_episode_start = self._top

    def add_traj(self, traj):
        """
        Add a trajectory to the replay buffer
        
        This default implementation naively goes through every step, 
        but you may want to optimize this
        """
        for i, (obs, action, reward, next_obs, done) in enumerate(zip(
            traj["obs"], traj["action"], traj["reward"], traj["next_obs"], traj["done"])):
            self.add(obs, action, reward, next_obs, done)
        self.termination()

    def sample(self, batch_size):
        ''' Batch of unordered transitions '''
        indices = np.random.randint(0, self._size, batch_size)
        return dict(
            obs=self._obs[indices],
            action=self._action[indices],
            reward=self._reward[indices],
            next_obs=self._next_obs[indices],
            done=self._done[indices],
        )
