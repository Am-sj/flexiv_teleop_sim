from typing import Dict

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("Empty-v1", max_episode_steps=200000)
class EmptyEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["none"]
    """
    This is just a dummy environment for showcasing robots in a empty scene
    """

    def __init__(self, *args, robot_uids="panda", **kwargs):
        self.dual_flexiv_assembly = None
        if isinstance(robot_uids, (tuple, list)) and tuple(robot_uids) == ("flexiv_rizon4", "flexiv_rizon4"):
            from mani_skill.envs.tasks.flexiv_dual_arm import load_assembly
            self.dual_flexiv_assembly = load_assembly()
            robot_uids = tuple(robot_uids)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([1.25, -1.25, 1.5], [0.0, 0.0, 0.2])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.25, -1.25, 1.5], [0.0, 0.0, 0.2])
        if self.dual_flexiv_assembly is not None:
            pose = sapien_utils.look_at([2.8, -2.3, 2.0], [0.12, 0., 0.85])
        return CameraConfig("render_camera", pose, 2048, 2048, 1, 0.01, 100)

    @property
    def _default_viewer_camera_configs(self):
        if self.dual_flexiv_assembly is not None:
            return CameraConfig("viewer", sapien_utils.look_at([2.8, -2.3, 2.0], [0.12, 0., 0.85]),
                                1280, 960, 0.85, 0.01, 100, shader_pack="default")
        return super()._default_viewer_camera_configs

    def _load_agent(self, options: dict):
        if self.dual_flexiv_assembly is not None:
            from mani_skill.envs.tasks.flexiv_dual_arm import mount_pose
            mounts = self.dual_flexiv_assembly["mounts"]
            super()._load_agent(options, [mount_pose(mounts[side]) for side in ("left", "right")])
        else:
            super()._load_agent(options, sapien.Pose())

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene)
        self.ground.set_collision_group_bit(group=2, bit_idx=30, bit=1)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if self.dual_flexiv_assembly is not None:
            from mani_skill.envs.tasks.flexiv_dual_arm import mount_pose
            for agent, side in zip(self.agent.agents, ("left", "right")):
                mount = self.dual_flexiv_assembly["mounts"][side]
                qpos = agent.keyframes["rest"].qpos.copy()
                for name, angle in zip(agent.arm_joint_names, mount["arm_qpos"]):
                    qpos[agent.robot.active_joints_map[name].active_index[0].item()] = angle
                agent.reset(qpos)
                agent.robot.set_pose(mount_pose(mount))

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: Dict):
        return dict()
