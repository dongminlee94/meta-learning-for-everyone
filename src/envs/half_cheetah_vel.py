"""
Half-cheetah environment code with velocity target reward
"""

import numpy as np
from pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv

from . import register_env


@register_env("cheetah-vel")
class HalfCheetahVelEnv(HalfCheetahBulletEnv):  # pylint: disable=too-many-instance-attributes
    """Half-cheetah environment class with velocity target reward"""

    def __init__(self, num_tasks=2, seed=0):
        super().__init__(render=False)
        self.tasks = self.sample_tasks(num_tasks)
        self._goal_vel = self.tasks[0].get("velocity", 0.0)
        self._goal = self._goal_vel
        self._task = None
        self._alive = None
        self.rewards = None
        self.potential = None
        self.seed(seed)

    def step(self, a):
        # If multiplayer, action first applied to all robots,
        # then global step() called, then _step() for all robots with the same actions
        if not self.scene.multiplayer:
            self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()

        self._alive = float(
            self.robot.alive_bonus(
                # state[0] is body height above ground, body_rpy[1] is pitch
                state[0] + self.robot.initial_z,
                self.robot.body_rpy[1],
            )
        )
        done = self._isDone()
        if not np.isfinite(state).all():
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)
        run_cost = progress - self._goal_vel
        scaled_run_cost = -5.0 * abs(run_cost)

        feet_collision_cost = 0.0
        for i, feet in enumerate(self.robot.feet):
            contact_ids = set((x[2], x[4]) for x in feet.contact_list())
            if self.ground_ids & contact_ids:
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        electricity_cost = self.electricity_cost * float(np.abs(a * self.robot.joint_speeds).mean())
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)

        self.rewards = [
            self._alive,
            scaled_run_cost,
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost,
        ]

        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        info = {}
        info["run_cost"] = run_cost

        return state, sum(self.rewards), bool(done), info

    @classmethod
    def sample_tasks(cls, num_tasks):
        """Sample tasks as many as num_tasks"""
        velocities = np.random.uniform(0.5, 1.5, size=(num_tasks,))
        tasks = [{"velocity": velocity} for velocity in velocities]
        return tasks

    def get_all_task_idx(self):
        """Get index of all the tasks"""
        return list(range(len(self.tasks)))

    def reset_task(self, index):
        """Reset velocity target to index of task"""
        self._task = self.tasks[index]
        self._goal_vel = self._task["velocity"]
        self._goal = self._goal_vel
        self.reset()
