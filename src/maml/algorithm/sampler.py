"""
Sample collection code through interaction between agent and environment
"""

import numpy as np
import torch


class Sampler:
    """Data sampling class"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env,
        agent,
        action_dim,
        max_step,
        device,
    ):

        self.env = env
        self.agent = agent
        self.action_dim = action_dim
        self.max_step = max_step
        self.device = device
        self.cur_samples = 0

    def obtain_samples(self, policy, max_samples):
        """Obtain samples up to the number of maximum samples"""
        trajs = []
        while not self.cur_samples == max_samples:
            traj = self.rollout(policy, max_samples)
            trajs.append(traj)
        self.cur_samples = 0
        return trajs

    def rollout(self, policy, max_samples):  # pylint: disable=too-many-locals
        """Rollout up to maximum trajectory length"""
        cur_obs = []
        actions = []
        rewards = []
        dones = []
        infos = []
        log_probs = []

        cur_step = 0
        obs = self.env.reset()
        action = np.zeros(self.action_dim)
        reward = np.zeros(1)
        done = np.zeros(1)

        while not (done or cur_step == self.max_step or self.cur_samples == max_samples):
            # Get action
            action, log_prob = policy(torch.Tensor(obs).to(self.device))
            action = action.detach().cpu().numpy()
            log_prob = log_prob.detach().cpu().numpy().reshape(-1)

            next_obs, reward, done, info = self.env.step(action)
            reward = np.array(reward)
            done = np.array(int(done))
            cur_obs.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            infos.append(info["run_cost"])
            log_probs.append(log_prob)

            obs = next_obs
            cur_step += 1
            self.cur_samples += 1
        return dict(
            cur_obs=np.array(cur_obs),
            actions=np.array(actions),
            rewards=np.array(rewards),
            dones=np.array(dones),
            infos=np.array(infos),
            log_probs=np.array(log_probs),
        )
