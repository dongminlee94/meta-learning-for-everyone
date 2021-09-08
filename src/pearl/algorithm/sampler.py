"""
Sample collection implementation through interaction between agent and environment
"""


import numpy as np
import torch


class Sampler:
    """Data sampling class"""

    def __init__(
        self,
        env,
        agent,
        max_step,
        device,
    ):

        self.env = env
        self.agent = agent
        self.max_step = max_step
        self.device = device

    def obtain_samples(self, max_samples, update_posterior, accum_context=True):
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

    def rollout(self, accum_context=True):
        """Rollout up to maximum trajectory length"""
        cur_obs = []
        actions = []
        rewards = []
        dones = []
        infos = []

        obs = self.env.reset()
        done = False
        cur_step = 0

        while not (done or cur_step == self.max_step):
            action = self.agent.get_action(obs)
            next_obs, reward, done, info = self.env.step(action)

            # Update the agent's current context
            if accum_context:
                self.update_context(obs, action, reward)

            cur_obs.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            infos.append(info["run_cost"])

            cur_step += 1
            obs = next_obs

        cur_obs = np.array(cur_obs)
        return dict(
            cur_obs=cur_obs,
            actions=np.array(actions),
            rewards=np.array(rewards).reshape(-1, 1),
            next_obs=np.vstack((cur_obs[1:, :], np.expand_dims(next_obs, 0))),
            dones=np.array(dones).reshape(-1, 1),
            infos=np.array(infos),
        )

    def update_context(self, obs, action, reward):
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
