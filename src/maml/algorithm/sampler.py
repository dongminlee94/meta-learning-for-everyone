"""
Sample collection code through interaction between agent and environment
"""

import numpy as np


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

    def obtain_samples(self, max_samples):
        """Obtain samples up to the number of maximum samples"""
        trajs = []
        cur_samples = 0

        while cur_samples < max_samples:
            traj = self.rollout()
            trajs.append(traj)
            cur_samples += len(traj["cur_obs"])

        return trajs

    def rollout(self):  # pylint: disable=too-many-locals
        """Rollout up to maximum trajectory length"""
        cur_obs = []
        actions = []
        rewards = []
        dones = []
        infos = []

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
            cur_obs.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            infos.append(info["run_cost"])

            obs = next_obs
            cur_step += 1

        return dict(
            cur_obs=np.array(cur_obs),
            actions=np.array(actions),
            rewards=np.array(rewards),
            dones=np.array(dones),
            infos=np.array(infos),
        )
