"""
Half-cheetah environment with direction target reward

Reference:
    https://github.com/katerakelly/oyster/blob/master/rlkit/envs/half_cheetah_dir.py
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from src.envs import register_env
from src.envs.half_cheetah import HalfCheetahEnv


@register_env("cheetah-dir")
class HalfCheetahDirEnv(HalfCheetahEnv):
    """
    Half-cheetah environment class with direction target reward, as described in [1].

    The code is adapted from
    https://github.com/cbfinn/maml_rl/blob/master/rllab/envs/mujoco/half_cheetah_env_rand_direc.py

    The half-cheetah follows the dynamics from MuJoCo [2], and receives at each
    time step a reward composed of a control cost and a reward equal to its
    velocity in the target direction.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for
        model-based control", 2012
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """

    def __init__(self, num_tasks: int) -> None:
        directions = [-1, 1, -1, 1]
        self.tasks = [{"direction": direction} for direction in directions]
        assert num_tasks == len(self.tasks)
        self._task = self.tasks[0]
        self._goal_dir = self._task["direction"]
        super().__init__()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, Dict[str, Any]]:
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

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
