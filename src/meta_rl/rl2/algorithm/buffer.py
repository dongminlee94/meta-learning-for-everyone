from typing import Dict, List

import numpy as np
import torch


class Buffer:
    def __init__(
        self,
        trans_dim: int,
        action_dim: int,
        hidden_dim: int,
        max_size: int,
        device: torch.device,
        gamma: float = 0.99,
        lamda: float = 0.97,
    ) -> None:

        self._trans = np.zeros((max_size, trans_dim))
        self._pi_hiddens = np.zeros((max_size, hidden_dim))
        self._v_hiddens = np.zeros((max_size, hidden_dim))
        self._actions = np.zeros((max_size, action_dim))
        self._rewards = np.zeros((max_size, 1))
        self._dones = np.zeros((max_size, 1), dtype="uint8")
        self._returns = np.zeros((max_size, 1))
        self._advants = np.zeros((max_size, 1))
        self._values = np.zeros((max_size, 1))
        self._log_probs = np.zeros((max_size, 1))
        self.device = device
        self.gamma = gamma
        self.lamda = lamda
        self._max_size = max_size
        self._top = 0

    def add(
        self,
        tran: np.ndarray,
        pi_hidden: np.ndarray,
        v_hidden: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
    ) -> None:
        # 버퍼에 변수들 추가
        assert self._top < self._max_size
        self._trans[self._top] = tran
        self._pi_hiddens[self._top] = pi_hidden
        self._v_hiddens[self._top] = v_hidden
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._dones[self._top] = done
        self._values[self._top] = value
        self._log_probs[self._top] = log_prob
        self._top += 1

    def add_trajs(self, trajs: List[Dict[str, np.ndarray]]) -> None:
        # 버퍼에 경로들 추가
        for traj in trajs:
            for (tran, pi_hidden, v_hidden, action, reward, done, value, log_prob) in zip(
                traj["trans"],
                traj["pi_hiddens"],
                traj["v_hiddens"],
                traj["actions"],
                traj["rewards"],
                traj["dones"],
                traj["values"],
                traj["log_probs"],
            ):
                self.add(
                    tran=tran,
                    pi_hidden=pi_hidden,
                    v_hidden=v_hidden,
                    action=action,
                    reward=reward,
                    done=done,
                    value=value,
                    log_prob=log_prob,
                )

    def compute_gae(self) -> None:
        # 보상합과 GAE 값 계산
        prev_value = 0
        running_return = 0
        running_advant = 0

        for t in reversed(range(len(self._rewards))):
            # 보상합 계산
            running_return = self._rewards[t] + self.gamma * (1 - self._dones[t]) * running_return
            self._returns[t] = running_return

            # GAE 값 계산
            running_tderror = (
                self._rewards[t] + self.gamma * (1 - self._dones[t]) * prev_value - self._values[t]
            )
            running_advant = (
                running_tderror + self.gamma * self.lamda * (1 - self._dones[t]) * running_advant
            )
            self._advants[t] = running_advant
            prev_value = self._values[t]

        # 어드밴티지 정규화
        self._advants = (self._advants - self._advants.mean()) / self._advants.std()

    def sample_batch(self) -> Dict[str, torch.Tensor]:
        # 버퍼에서 배치 생성
        assert self._top == self._max_size
        self._top = 0

        self.compute_gae()
        samples = dict(
            trans=self._trans,
            pi_hiddens=self._pi_hiddens,
            v_hiddens=self._v_hiddens,
            actions=self._actions,
            returns=self._returns,
            advants=self._advants,
            log_probs=self._log_probs,
        )
        return {key: torch.Tensor(value).to(self.device) for key, value in samples.items()}
