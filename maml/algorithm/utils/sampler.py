"""
Sample collection code through interaction between agent and environment
"""

import numpy as np


class Sampler:
    """Data sampling class"""

    def __init__(
        self,
        env,
        agent,
        max_step,
    ):

        self.env = env
        self.agent = agent
        self.max_step = max_step

    def obtain_trajs(self, max_samples, use_rendering=False):
        """Obtain samples up to the number of maximum samples"""
        trajs = []
        num_samples = 0

        while num_samples < max_samples:
            traj = self.rollout(use_rendering=use_rendering)
            trajs.append(traj)

            num_samples += len(traj["curr_obs"])
        return trajs, num_samples

    # pylint: disable=too-many-locals
    def rollout(self, use_rendering=False):
        """Rollout up to maximum trajectory length"""
        curr_obs = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []

        curr_step = 0
        obs = self.env.reset()

        if use_rendering:
            self.env.render()

        while curr_step < self.max_step:
            action, log_prob = self.agent.get_action(obs)
            next_obs, reward, done, _ = self.env.step(action)
            value = self.agent.vf(obs)

            curr_obs.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            log_probs.append(log_prob)

            obs = next_obs
            curr_step += 1

            if done:
                break

        curr_obs = np.array(curr_obs)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape(-1, 1)
        next_obs = np.vstack((curr_obs[1:, :], np.expand_dims(next_obs, 0)))
        dones = np.array(dones).reshape(-1, 1)
        values = np.array(values).reshape(-1, 1)
        log_probs = np.array(log_probs).reshape(-1, 1)
        return dict(
            curr_obs=curr_obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            values=values,
            log_probs=log_probs,
        )
