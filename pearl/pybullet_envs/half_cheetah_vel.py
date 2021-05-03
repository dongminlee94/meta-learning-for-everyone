import numpy as np

from .gym_locomotion_envs import HalfCheetahBulletEnv
from .robot_locomotors import HalfCheetah
from . import register_env


@register_env('cheetah-vel')
class HalfCheetahVelBulletEnv(HalfCheetahBulletEnv):
  def __init__(self, task={}, n_tasks=2, randomize_tasks=True):
    self.env_name = 'cheetah-vel'
    self.robot = HalfCheetah()
    self._task = task
    self.tasks = self.sample_tasks(n_tasks)
    self._goal_vel = self.tasks[0].get('velocity', 0.0)
    self._goal = self._goal_vel
    super(HalfCheetahBulletEnv, self).__init__(self.robot)

  def step(self, action):
    if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
      self.robot.apply_action(action)
      self.scene.global_step()

    state = self.robot.calc_state()  # also calculates self.joints_at_limit

    self._alive = float(
        self.robot.alive_bonus(
            state[0] + self.robot.initial_z,
            self.robot.body_rpy[1]))  # state[0] is body height above ground, body_rpy[1] is pitch
    done = self._isDone()
    if not np.isfinite(state).all():
      print("~INF~", state)
      done = True

    potential_old = self.potential
    self.potential = self.robot.calc_potential()
    progress = float(self.potential - potential_old)
    run_cost = -5.0 * abs(progress - self._goal_vel)

    feet_collision_cost = 0.0
    for i, f in enumerate(
        self.robot.feet
    ):  # TODO: Maybe calculating feet contacts could be done within the robot code
      contact_ids = set((x[2], x[4]) for x in f.contact_list())
      #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
      if (self.ground_ids & contact_ids):
        #see Issue 63: https://github.com/openai/roboschool/issues/63
        #feet_collision_cost += self.foot_collision_cost
        self.robot.feet_contact[i] = 1.0
      else:
        self.robot.feet_contact[i] = 0.0

    electricity_cost = self.electricity_cost * float(np.abs(action * self.robot.joint_speeds).mean(
    ))  # let's assume we have DC motor with controller, and reverse current braking
    electricity_cost += self.stall_torque_cost * float(np.square(action).mean())

    joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)
    # print(self._goal_vel, progress)
    debugmode = 0
    if (debugmode):
      print("alive=")
      print(self._alive)
      print("progress")
      print(progress)
      print("electricity_cost")
      print(electricity_cost)
      print("joints_at_limit_cost")
      print(joints_at_limit_cost)
      print("feet_collision_cost")
      print(feet_collision_cost)

    self.rewards = [
        self._alive, run_cost, electricity_cost, joints_at_limit_cost, feet_collision_cost
    ]
    if (debugmode):
      print("rewards=")
      print(self.rewards)
      print("sum rewards")
      print(sum(self.rewards))
    self.HUD(state, action, done)
    self.reward += sum(self.rewards)

    return state, sum(self.rewards), bool(done), {}

  def sample_tasks(self, num_tasks):
    np.random.seed(1337)
    velocities = np.random.uniform(0.0, 1.5, size=(num_tasks,))
    tasks = [{'velocity': velocity} for velocity in velocities]
    return tasks

  def get_all_task_idx(self):
      return range(len(self.tasks))

  def reset_task(self, idx):
      self._task = self.tasks[idx]
      self._goal_vel = self._task['velocity']
      self._goal = self._goal_vel
      self.reset()

  def set_task(self, goal_coeff):
      self._task = {'velocity':goal_coeff}
      self._goal_vel = self._task['velocity']
      self._goal = self._goal_vel