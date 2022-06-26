from typing import List, Union

import numpy as np
from gym import utils
from gym.envs.mujoco import HalfCheetahEnv as HalfCheetahEnv_
from gym.envs.mujoco import mujoco_env


class HalfCheetahEnv(HalfCheetahEnv_):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self) -> np.ndarray:
        return (
            np.concatenate(
                [
                    self.data.qpos.flat[1:],
                    self.data.qvel.flat,
                    self.get_body_com("torso").flat,
                ],
            )
            .astype(np.float32)
            .flatten()
        )

    def viewer_setup(self) -> None:
        camera_id = self.model.camera_name2id("track")
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        self.viewer._hide_overlay = True

    def render(self, mode: str = "human") -> Union[List[float], None]:
        if mode == "rgb_array":
            self._get_viewer().render()
            width, height = 500, 500
            data = self._get_viewer().read_pixels(width, height, depth=False)
            return data
        elif mode == "human":
            self._get_viewer().render()
