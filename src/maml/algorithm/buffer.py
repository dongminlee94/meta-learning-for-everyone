"""
Simple buffer code
"""

import numpy as np
import torch
import torch.nn.functional as F


class MultiTaskBuffer:
    """Multiple-task buffer class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        observ_dim,
        action_dim,
        agent,
        num_tasks,
        num_episodes,
        max_size,
        device,
    ):
        self.num_tasks = num_tasks
        self.num_buffers = num_tasks * num_episodes
        self.task_buffers = {}
        for i in range(self.num_buffers):
            self.task_buffers[i] = Buffer(
                agent=agent,
                observ_dim=observ_dim,
                action_dim=action_dim,
                max_size=max_size,
                device=device,
            )

    def assign_index(self, task_index, adapt_index):
        """Assign buffer index according to current task and adapation"""
        return self.num_tasks * adapt_index + task_index

    def add_trajs(self, cur_task, cur_adapt, trajs):
        """Add trajectories to the assigned task buffer"""
        self.task_buffers[self.assign_index(cur_task, cur_adapt)].add_trajs(trajs)

    def get_samples(self, cur_task, cur_adapt, log=False):
        """Sample batch of the sassigned task buffer"""
        return self.task_buffers[self.assign_index(cur_task, cur_adapt)].get_samples(log=log)

    def clear(self):
        """Clear variables of all task buffers"""
        for buffer_index in range(self.num_buffers):
            self.task_buffers[buffer_index].clear()


class Buffer:  # pylint: disable=too-many-instance-attributes
    """Simple buffer class that includes computing return and gae"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        agent,
        observ_dim,
        action_dim,
        max_size,
        device,
        gamma=0.99,
        lamda=0.97,
    ):
        self.agent = agent
        self._cur_obs = np.zeros((max_size, observ_dim))
        self._actions = np.zeros((max_size, action_dim))
        self._rewards = np.zeros((max_size, 1))
        self._next_obs = np.zeros((max_size, observ_dim))
        self._dones = np.zeros((max_size, 1), dtype="uint8")
        self._returns = np.zeros((max_size, 1))
        self._advants = np.zeros((max_size, 1))
        self._values = np.zeros((max_size, 1))
        self._infos = np.zeros((max_size, 1))
        self._log_probs = np.zeros((max_size, 1))
        self.device = device
        self.gamma = gamma
        self.lamda = lamda

        self._max_size = max_size
        self._top = 0

    # pylint: disable=too-many-arguments
    def add(self, obs, action, reward, next_obs, done, info, log_prob):
        """Add transition and log_prob to the buffer"""
        assert self._top < self._max_size
        self._cur_obs[self._top] = obs
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._next_obs[self._top] = next_obs
        self._dones[self._top] = done
        self._infos[self._top] = info
        self._log_probs[self._top] = log_prob
        self._top += 1

    def add_trajs(self, trajs):
        """Add trajectories to the buffer"""
        for traj in trajs:
            for (obs, action, reward, next_obs, done, info, log_prob) in zip(
                traj["cur_obs"],
                traj["actions"],
                traj["rewards"],
                traj["next_obs"],
                traj["dones"],
                traj["infos"],
                traj["log_probs"],
            ):
                self.add(obs, action, reward, next_obs, done, info, log_prob)

    def compute_value(self):
        """Train value network and infer value for current batch"""
        running_return = 0

        for t in reversed(range(len(self._rewards))):
            # Compute return
            running_return = self._rewards[t] + self.gamma * (1 - self._dones[t]) * running_return
            self._returns[t] = running_return

        # Reset the value fuction to its initial state
        self.agent.reset_vf()
        self.agent.vf_optimizer.zero_grad()

        # Value function loss
        obs_batch = torch.Tensor(self._cur_obs).to(self.device)
        value_batch = self.agent.vf(obs_batch)
        value_loss = F.mse_loss(value_batch.view(-1, 1), torch.Tensor(self._returns).to(self.device))

        value_loss.backward()
        self.agent.vf_optimizer.step()

        self._values = self.agent.vf(obs_batch).detach().cpu().numpy()

    def compute_gae(self):
        """Compute return and GAE"""
        prev_value = 0
        running_advant = 0

        self.compute_value()

        for t in reversed(range(len(self._rewards))):
            # Compute GAE
            running_tderror = (
                self._rewards[t] + self.gamma * (1 - self._dones[t]) * prev_value - self._values[t]
            )
            running_advant = (
                running_tderror + self.gamma * self.lamda * (1 - self._dones[t]) * running_advant
            )
            self._advants[t] = running_advant
            prev_value = self._values[t]

        # Normalize advantage
        self._advants = (self._advants - self._advants.mean()) / self._advants.std()

    def get_samples(self, log_mode=False):
        """Get sample batch in buffer"""
        assert self._top == self._max_size

        if not log_mode:
            self.compute_gae()

        batch = dict(
            obs=self._cur_obs,
            actions=self._actions,
            rewards=self._rewards,
            advants=self._advants,
            infos=self._infos,
            log_probs=self._log_probs,
        )
        return {key: torch.Tensor(value).to(self.device) for key, value in batch.items()}

    def clear(self):
        """Clear variables of replay buffer"""
        self._top = 0
