from typing import Any, Dict, List, Tuple

import numpy as np

from meta_rl.envs import register_env
from meta_rl.envs.half_cheetah import HalfCheetahEnv


@register_env("cheetah-dir")
class HalfCheetahDirEnv(HalfCheetahEnv):
    def __init__(self, num_tasks: int) -> None:
        directions = [-1, 1, -1, 1]
        self.tasks = [{"direction": direction} for direction in directions]
        assert num_tasks == len(self.tasks)
        self._task = self.tasks[0]
        self._goal_dir = self._task["direction"]
        super().__init__()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, Dict[str, Any]]:
        xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]

        progress = (xposafter - xposbefore) / self.dt
        run_cost = self._goal_dir * progress
        control_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = run_cost - control_cost
        done = False
        info = dict(run_cost=run_cost, control_cost=-control_cost, task=self._task)
        return observation, reward, done, info

    def get_all_task_idx(self) -> List[int]:
        return list(range(len(self.tasks)))

    def reset_task(self, idx: int) -> None:
        self._task = self.tasks[idx]
        self._goal_dir = self._task["direction"]
        self.reset()
