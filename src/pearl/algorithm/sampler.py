"""
Sample collection implementation through interaction between agent and environment
"""


from typing import Dict, List, Tuple

import numpy as np
import torch
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv

from src.pearl.algorithm.sac import SAC


class Sampler:
    """Data sampling class"""

    def __init__(
        self,
        env: HalfCheetahEnv,
        agent: SAC,
        max_step: int,
        device: torch.device,
    ) -> None:

        self.env = env
        self.agent = agent
        self.max_step = max_step
        self.device = device

    def obtain_samples(
        self, max_samples: int, update_posterior: bool, accum_context: bool = True
    ) -> Tuple[List[Dict[str, np.ndarray]], int]:
        """Obtain samples up to the number of maximum samples"""
        trajs = []
        cur_samples = 0

        while cur_samples < max_samples:
            traj = self.rollout(accum_context=accum_context)
            trajs.append(traj)
            cur_samples += len(traj["cur_obs"])
            self.agent.encoder.sample_z()

            if update_posterior:
                break
        return trajs, cur_samples

    def rollout(self, accum_context: bool = True) -> Dict[str, np.ndarray]:
        """Rollout up to maximum trajectory length"""
        _cur_obs = []
        _actions = []
        _rewards = []
        _next_obs = []
        _dones = []
        _infos = []

        obs = self.env.reset()
        done = False
        cur_step = 0

        while not (done or cur_step == self.max_step):
            action = self.agent.get_action(obs)
            next_obs, reward, done, info = self.env.step(action)

            # Update the agent's current context
            if accum_context:
                self.update_context(obs, action, reward)

            _cur_obs.append(obs)
            _actions.append(action)
            _rewards.append(reward)
            _next_obs.append(next_obs)
            _dones.append(done)
            _infos.append(info["run_cost"])

            cur_step += 1
            obs = next_obs
        return dict(
            cur_obs=np.array(_cur_obs),
            actions=np.array(_actions),
            rewards=np.array(_rewards).reshape(-1, 1),
            next_obs=np.array(_next_obs),
            dones=np.array(_dones).reshape(-1, 1),
            infos=np.array(_infos),
        )

    def update_context(self, obs: np.ndarray, action: np.ndarray, reward: float) -> None:
        """Append single transition to the current context"""
        obs = torch.from_numpy(obs[None, None, ...]).float().to(self.device)
        action = torch.from_numpy(action[None, None, ...]).float().to(self.device)
        reward = torch.from_numpy(np.array([reward])[None, None, ...]).float().to(self.device)
        transition = torch.cat([obs, action, reward], dim=-1).to(self.device)

        if self.agent.encoder.context is None:
            self.agent.encoder.context = transition
        else:
            self.agent.encoder.context = torch.cat([self.agent.encoder.context, transition], dim=1).to(
                self.device
            )
