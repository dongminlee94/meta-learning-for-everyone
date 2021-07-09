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
        cur_samples = 0

        while cur_samples < max_samples:
            if self.max_step > max_samples - cur_samples:
                self.max_step = max_samples - cur_samples

            traj = self.rollout(
                max_step=self.max_step,
                use_rendering=use_rendering,
            )
            trajs.append(traj)

            cur_samples += len(traj["trans"])
        return trajs

    # pylint: disable=too-many-locals
    def rollout(self, max_step, use_rendering=False):
        """Rollout up to maximum trajectory length"""
        trans = []
        pi_hiddens = []
        v_hiddens = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []

        cur_step = 0
        obs = self.env.reset()
        action = np.zeros(self.action_dim)
        reward = np.zeros(1)
        done = np.zeros(1)
        pi_hidden = np.zeros((1, self.hidden_dim))
        v_hidden = np.zeros((1, self.hidden_dim))

        if use_rendering:
            self.env.render()

        while cur_step < max_step:
            tran = np.concatenate((obs, action, reward, done), axis=-1)
            action, log_prob, next_pi_hidden = self.agent.get_action(tran, pi_hidden)
            next_obs, reward, done, _ = self.env.step(action)
            value, next_v_hidden = self.agent.vf(tran, v_hidden)

            trans.append(tran)
            pi_hiddens.append(pi_hidden)
            v_hiddens.append(v_hidden)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))
            values.append(value)
            log_probs.append(log_prob)

            obs = next_obs
            pi_hidden = next_pi_hidden[0]
            v_hidden = next_v_hidden[0]
            cur_step += 1

            if done:
                break

        trans = np.array(trans)
        pi_hiddens = np.array(pi_hiddens)
        v_hiddens = np.array(v_hiddens)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape(-1, 1)
        dones = np.array(dones).reshape(-1, 1)
        values = np.array(values).reshape(-1, 1)
        log_probs = np.array(log_probs).reshape(-1, 1)
        return dict(
            trans=trans,
            pi_hiddens=pi_hiddens,
            v_hiddens=v_hiddens,
            actions=actions,
            rewards=rewards,
            dones=dones,
            values=values,
            log_probs=log_probs,
        )
