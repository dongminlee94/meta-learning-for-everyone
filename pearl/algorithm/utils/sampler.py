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
        max_step,
        device,
    ):

        self.env = env
        self.agent = agent
        self.max_step = max_step
        self.device = device

    def obtain_samples(
        self, max_samples, min_trajs, accum_context=True, use_rendering=False
    ):
        """Obtain samples up to the number of maximum samples"""
        trajs = []
        num_samples = 0
        num_trajs = 0

        while num_samples < max_samples:
            traj = self.rollout(
                accum_context=accum_context, use_rendering=use_rendering
            )

            trajs.append(traj)
            num_samples += len(traj["curr_obs"])
            num_trajs += 1

            self.agent.encoder.sample_z()

            if min_trajs == 1:
                break
        return trajs, num_samples

    def rollout(self, accum_context=True, use_rendering=False):
        """Rollout up to maximum trajectory length"""
        curr_obs = []
        actions = []
        rewards = []
        dones = []

        obs = self.env.reset()
        cur_step = 0

        if use_rendering:
            self.env.render()

        while cur_step < self.max_step:
            action = self.agent.get_action(obs)
            next_obs, reward, done, _ = self.env.step(action)

            # Update the agent's current context
            if accum_context:
                self.update_context(obs, action, reward)

            curr_obs.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            cur_step += 1
            obs = next_obs
            if done:
                break

        curr_obs = np.array(curr_obs)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape(-1, 1)
        next_obs = np.vstack((curr_obs[1:, :], np.expand_dims(next_obs, 0)))
        dones = np.array(dones).reshape(-1, 1)
        return dict(
            curr_obs=curr_obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
        )

    def update_context(self, obs, action, reward):
        """Append single transition to the current context"""
        obs = torch.from_numpy(obs[None, None, ...]).float().to(self.device)
        action = torch.from_numpy(action[None, None, ...]).float().to(self.device)
        reward = (
            torch.from_numpy(np.array([reward])[None, None, ...])
            .float()
            .to(self.device)
        )
        transition = torch.cat([obs, action, reward], dim=-1).to(self.device)

        if self.agent.encoder.context is None:
            self.agent.encoder.context = transition
        else:
            self.agent.encoder.context = torch.cat(
                [self.agent.encoder.context, transition], dim=1
            ).to(self.device)