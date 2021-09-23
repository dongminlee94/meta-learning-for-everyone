"""
Sample collection code through interaction between agent and environment
"""

import numpy as np
import torch


class Sampler:
    """Data sampling class"""

    def __init__(
        self,
        env,
        agent,
        action_dim,
        device,
    ):

        self.env = env
        self.agent = agent
        self.policy = agent.policy
        self.action_dim = action_dim
        self.device = device

    def obtain_samples(self, policy, max_samples, max_steps, use_rendering=False):
        """Obtain samples up to the number of maximum samples"""
        self.policy = policy
        trajs = []
        cur_samples = 0

        while cur_samples < max_samples:
            if max_steps > max_samples - cur_samples:
                max_steps = max_samples - cur_samples
            traj = self.rollout(max_steps=max_steps, use_rendering=use_rendering)
            trajs.append(traj)

            cur_samples += len(traj["cur_obs"])
        return trajs

    # pylint: disable=too-many-locals
    def rollout(self, max_steps, use_rendering=False):
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

        if use_rendering:
            self.env.render()

        while cur_step < max_steps:
            action, log_prob = self.policy(torch.Tensor(obs).to(self.device))
            action = action.detach().cpu().numpy()
            if log_prob:
                log_prob = log_prob.detach().cpu().numpy()

            next_obs, reward, done, info = self.env.step(action)
            reward = np.array(reward).reshape(-1)
            done = np.array(int(done)).reshape(-1)

            cur_obs.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            infos.append(info["run_cost"])
            if log_prob:
                log_probs.append(log_prob.reshape(-1))
            else:
                log_probs.append(None)

            obs = next_obs
            cur_step += 1

            if done:
                break

        return dict(
            cur_obs=np.array(cur_obs),
            actions=np.array(actions),
            rewards=np.array(rewards),
            next_obs=np.vstack((cur_obs[1:], np.expand_dims(next_obs, 0))),
            dones=np.array(dones),
            infos=np.array(infos),
            log_probs=np.array(log_probs),
        )
