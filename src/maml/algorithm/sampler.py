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
        self.sampling_policy = agent.policy
        self.action_dim = action_dim
        self.max_step = max_step
        self.device = device
        self.cur_samples = 0

    def obtain_samples(self, policy, max_samples):
        """Obtain samples up to the number of maximum samples"""
        self.sampling_policy = policy
        trajs = []
        while not self.cur_samples == max_samples:
            traj = self.rollout(max_samples)
            trajs.append(traj)
        self.cur_samples = 0
        return trajs

    def rollout(self, max_samples):  # pylint: disable=too-many-locals
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
            action, log_prob = self.sampling_policy(torch.Tensor(obs).to(self.device))
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
