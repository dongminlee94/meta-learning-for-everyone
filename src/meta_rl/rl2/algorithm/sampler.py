from typing import Dict, List

import numpy as np
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv

from meta_rl.rl2.algorithm.ppo import PPO


class Sampler:
    def __init__(
        self,
        env: HalfCheetahEnv,
        agent: PPO,
        action_dim: int,
        hidden_dim: int,
        max_step: int,
    ) -> None:

        self.env = env
        self.agent = agent
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_step = max_step
        self.cur_samples = 0
        self.pi_hidden = None
        self.v_hidden = None

    def obtain_samples(self, max_samples: int) -> List[Dict[str, np.ndarray]]:
        # 은닉 상태 유지를 위한 변수들 생성
        self.pi_hidden = np.zeros((1, self.hidden_dim))
        self.v_hidden = np.zeros((1, self.hidden_dim))

        # 최대 샘플량의 수까지 샘플들 얻기
        trajs = []
        while not self.cur_samples == max_samples:
            traj = self.rollout(max_samples)
            trajs.append(traj)

        self.cur_samples = 0
        return trajs

    def rollout(self, max_samples: int) -> Dict[str, np.ndarray]:
        # 최대 경로 길이까지 경로 생성
        trans = []
        pi_hiddens = []
        v_hiddens = []
        actions = []
        rewards = []
        dones = []
        infos = []
        values = []
        log_probs = []

        cur_step = 0
        obs = self.env.reset()
        action = np.zeros(self.action_dim)
        reward = np.zeros(1)
        done = np.zeros(1)

        while not (done or cur_step == self.max_step or self.cur_samples == max_samples):
            tran = np.concatenate((obs, action, reward, done), axis=-1).reshape(1, -1)
            action, log_prob, next_pi_hidden = self.agent.get_action(tran, self.pi_hidden)
            value, next_v_hidden = self.agent.get_value(tran, self.v_hidden)

            next_obs, reward, done, info = self.env.step(action)
            reward = np.array(reward).reshape(-1)
            done = np.array(int(done)).reshape(-1)

            trans.append(tran.reshape(-1))
            pi_hiddens.append(self.pi_hidden.reshape(-1))
            v_hiddens.append(self.v_hidden.reshape(-1))
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            infos.append(info["run_cost"])
            values.append(value.reshape(-1))
            log_probs.append(log_prob.reshape(-1))

            obs = next_obs.reshape(-1)
            self.pi_hidden = next_pi_hidden[0]
            self.v_hidden = next_v_hidden[0]
            cur_step += 1
            self.cur_samples += 1
        return dict(
            trans=np.array(trans),
            pi_hiddens=np.array(pi_hiddens),
            v_hiddens=np.array(v_hiddens),
            actions=np.array(actions),
            rewards=np.array(rewards),
            dones=np.array(dones),
            infos=np.array(infos),
            values=np.array(values),
            log_probs=np.array(log_probs),
        )
