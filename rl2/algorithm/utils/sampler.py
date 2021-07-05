"""
Sample collection code through interaction between agent and environment
"""

import numpy as np


class Sampler:
    """Data sampling class"""

    def __init__(   # pylint: disable=too-many-arguments
        self,
        env,
        agent,
        max_step,
        action_dim,
        hidden_dim,
    ):

        self.env = env
        self.agent = agent
        self.max_step = max_step
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

    def obtain_trajs(self, max_samples, use_rendering=False):
        """Obtain samples up to the number of maximum samples"""
        trajs = []
        num_samples = 0

        while num_samples < max_samples:
            traj = self.rollout(use_rendering=use_rendering)
            trajs.append(traj)

            num_samples += len(traj["cur_obs"])
        return trajs, num_samples

    # pylint: disable=too-many-locals
    def rollout(self, use_rendering=False):
        """Rollout up to maximum trajectory length"""
        cur_obs = []
        actions = []
        rewards = []
        dones = []
        pi_hiddens = []
        v_hiddens = []
        values = []
        log_probs = []

        cur_step = 0
        obs = self.env.reset()
        action = np.zeros(self.action_dim)
        reward = np.zeros(1)
        done = np.zeros(1)
        pi_hidden = np.zeros((1, 1, self.action_dim))
        v_hidden = np.zeros((1, 1, self.action_dim))

        if use_rendering:
            self.env.render()

        while cur_step < self.max_step:
            trans = np.concatenate((obs, action, reward, done), axis=-1)
            trans = np.reshape(trans, (1, 1, -1))
            action, log_prob, pi_hidden = self.agent.get_action(trans, pi_hidden)
            next_obs, reward, done, _ = self.env.step(action)
            value, v_hidden = self.agent.vf(trans, v_hidden)

            cur_obs.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))
            pi_hiddens.append(pi_hidden)
            v_hiddens.append(v_hidden)
            values.append(value)
            log_probs.append(log_prob)

            obs = next_obs
            cur_step += 1

            if done:
                break

        cur_obs = np.array(cur_obs)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape(-1, 1)
        dones = np.array(dones).reshape(-1, 1)
        pi_hiddens = np.array(pi_hiddens)
        v_hiddens = np.array(v_hiddens)
        values = np.array(values).reshape(-1, 1)
        log_probs = np.array(log_probs).reshape(-1, 1)
        return dict(
            cur_obs=cur_obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            pi_hiddens=pi_hiddens,
            v_hiddens=v_hiddens,
            values=values,
            log_probs=log_probs,
        )
