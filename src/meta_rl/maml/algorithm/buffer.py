from typing import Dict, List, Optional

import numpy as np
import torch

from meta_rl.maml.algorithm.trpo import TRPO


class MultiTaskBuffer:
    def __init__(
        self,
        observ_dim: int,
        action_dim: int,
        agent: TRPO,
        num_tasks: int,
        num_episodes: int,
        max_size: int,
        device: torch.device,
    ) -> None:

        self.num_tasks = num_tasks
        self.num_buffers = num_tasks * num_episodes
        self.task_buffers = {}
        for i in range(self.num_buffers):
            self.task_buffers[i] = Buffer(
                agent=agent,
                observ_dim=observ_dim,
                action_dim=action_dim,
                max_size=max_size,
                device=device,
            )

    def assign_index(self, task_index: int, adapt_index: int) -> int:
        # 태스크 인덱스와 적응 인덱스에 따라 버퍼 인덱스 할당
        return self.num_tasks * adapt_index + task_index

    def add_trajs(self, cur_task: int, cur_adapt: int, trajs: List[Dict[str, np.ndarray]]) -> None:
        # 할당된 태스크 버퍼에 이동한 경로들 추가
        self.task_buffers[self.assign_index(cur_task, cur_adapt)].add_task_trajs(trajs)

    def add_params(
        self,
        cur_task: int,
        cur_adapt: int,
        params: Dict[str, torch.nn.parameter.Parameter],
    ) -> None:
        # 할당된 태스크 버퍼에 태스크 파라미터 추가
        self.task_buffers[self.assign_index(cur_task, cur_adapt)].add_task_params(params)

    def get_trajs(self, cur_task: int, cur_adapt: int) -> Dict[str, torch.Tensor]:
        # 할당된 태스크 버퍼의 배치 얻기
        return self.task_buffers[self.assign_index(cur_task, cur_adapt)].get_task_trajs()

    def get_params(
        self,
        cur_task: int,
        cur_adapt: int,
    ) -> Optional[Dict[str, torch.nn.parameter.Parameter]]:
        # 할당된 태스크에서 정책 네트워크의 파라미터 얻기
        return self.task_buffers[self.assign_index(cur_task, cur_adapt)].get_task_params()

    def clear(self) -> None:
        # 모든 태스크 버퍼들의 변수들 초기화
        for buffer_index in range(self.num_buffers):
            self.task_buffers[buffer_index].clear_task()


class Buffer:
    def __init__(
        self,
        agent: TRPO,
        observ_dim: int,
        action_dim: int,
        max_size: int,
        device: torch.device,
    ) -> None:
        self.agent = agent
        self._cur_obs = np.zeros((max_size, observ_dim))
        self._actions = np.zeros((max_size, action_dim))
        self._rewards = np.zeros((max_size, 1))
        self._baselines = np.zeros((max_size, 1))
        self._dones = np.zeros((max_size, 1), dtype="uint8")
        self._infos = np.zeros((max_size, 1))
        self._params = None
        self.device = device

        self._max_size = max_size
        self._top = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        info: np.ndarray,
    ) -> None:
        assert self._top < self._max_size
        self._cur_obs[self._top] = obs
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._dones[self._top] = done
        self._infos[self._top] = info
        self._top += 1

    def add_task_trajs(self, trajs: List[Dict[str, np.ndarray]]) -> None:
        for traj in trajs:
            for (obs, action, reward, done, info) in zip(
                traj["cur_obs"],
                traj["actions"],
                traj["rewards"],
                traj["dones"],
                traj["infos"],
            ):
                self.add(obs, action, reward, done, info)

        batch = self.get_task_trajs()
        self._baselines = self.agent.infer_baselines(batch)

    def add_task_params(self, params: Dict[str, torch.nn.parameter.Parameter]) -> None:
        self._params = params

    def get_task_trajs(self) -> Dict[str, torch.Tensor]:
        assert self._top == self._max_size and len(self._baselines) == self._max_size

        batch = dict(
            obs=self._cur_obs,
            actions=self._actions,
            rewards=self._rewards,
            dones=self._dones,
            baselines=self._baselines,
            infos=self._infos,
        )
        return {key: torch.Tensor(value).to(self.device) for key, value in batch.items()}

    def get_task_params(self) -> Optional[Dict[str, torch.nn.parameter.Parameter]]:
        assert self._top is not None
        return self._params

    def clear_task(self) -> None:
        self._top = 0
        self._params = None
