"""
Sample collection code through interaction between agent and environment
"""

from typing import Dict, List

import numpy as np
import torch
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv

from meta_rl.maml.algorithm.trpo import TRPO


class Sampler:
    """Data sampling class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env: HalfCheetahEnv,
        agent: TRPO,
        action_dim: int,
        max_step: int,
        device: torch.device,
    ) -> None:

        self.env = env
        self.agent = agent
        self.action_dim = action_dim
        self.max_step = max_step
        self.device = device
        self.cur_samples = 0

    def obtain_samples(self, max_samples: int) -> List[Dict[str, np.ndarray]]:
        """Obtain samples up to the number of maximum samples"""
        trajs = []
        cur_samples = 0

        while cur_samples < max_samples:
            traj = self.rollout()
            trajs.append(traj)
            cur_samples += len(traj["cur_obs"])

        return trajs

    def rollout(self) -> Dict[str, np.ndarray]:  # pylint: disable=too-many-locals
        """Rollout up to maximum trajectory length"""
        _cur_obs = []
        _actions = []
        _rewards = []
        _dones = []
        _infos = []

        cur_step = 0
        obs = self.env.reset()
        action = np.zeros(self.action_dim)
        reward = np.zeros(1)
        done = np.zeros(1)

        while not (done or cur_step == self.max_step):
            # Get action
            action = self.agent.get_action(obs)

            next_obs, reward, done, info = self.env.step(action)
            reward = np.array(reward)
            done = np.array(int(done))
            _cur_obs.append(obs)
            _actions.append(action)
            _rewards.append(reward)
            _dones.append(done)
            _infos.append(info["run_cost"])

            obs = next_obs
            cur_step += 1

        return dict(
            cur_obs=np.array(_cur_obs),
            actions=np.array(_actions),
            rewards=np.array(_rewards),
            dones=np.array(_dones),
            infos=np.array(_infos),
        )
