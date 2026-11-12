"""Two Rizon4 + Grav arms mounted on the supplied Flexiv STEP stand."""

import json

import numpy as np
import sapien

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env


ASSEMBLY_DIR = PACKAGE_ASSET_DIR / "scenes/flexiv_dual_arm"


def load_assembly():
    return json.loads((ASSEMBLY_DIR / "assembly.json").read_text())


def mount_pose(mount):
    return sapien.Pose(mount["position"], mount["quaternion_wxyz"])


@register_env("FlexivDualArm-v1", max_episode_steps=200000)
class FlexivDualArmEnv(BaseEnv):
    """Fixed stand with independent left/right Rizon4 articulations.

    World +X points forward, +Y is the robot's left, and +Z points up.
    The left connector points down, and the right connector points up.
    Action dict keys: flexiv_rizon4-0 (left), flexiv_rizon4-1 (right).
    """

    SUPPORTED_ROBOTS = [("flexiv_rizon4", "flexiv_rizon4")]
    SUPPORTED_REWARD_MODES = ["none"]

    def __init__(self, *args, robot_uids=("flexiv_rizon4", "flexiv_rizon4"), **kwargs):
        if tuple(robot_uids) != self.SUPPORTED_ROBOTS[0]:
            raise ValueError("FlexivDualArm-v1 requires two flexiv_rizon4 agents")
        self.assembly = load_assembly()
        super().__init__(*args, robot_uids=tuple(robot_uids), **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([2.5, -2.5, 2.0], [0.15, 0., 0.85])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([2.8, -2.3, 2.0], [0.12, 0., 0.85])
        return CameraConfig("render_camera", pose, 1400, 1200, 0.85, 0.01, 100)

    @property
    def _default_viewer_camera_configs(self):
        return CameraConfig("viewer", sapien_utils.look_at([2.8, -2.3, 2.0], [0.12, 0., 0.85]),
                            1280, 960, 0.85, 0.01, 100, shader_pack="default")

    def _load_agent(self, options):
        poses = [mount_pose(self.assembly["mounts"][side]) for side in ("left", "right")]
        super()._load_agent(options, poses)

    def _load_scene(self, options):
        self.ground = build_ground(self.scene, floor_width=10)
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        builder.add_visual_from_file(str(ASSEMBLY_DIR / "stand.glb"))
        # A single convex hull would incorrectly fill the space around the column
        # and between its plates. Static triangle collision keeps the CAD shape.
        builder.add_nonconvex_collision_from_file(str(ASSEMBLY_DIR / "stand_collision.stl"))
        self.stand = builder.build_static(name="flexiv_dual_arm_stand")
        for agent, side in zip(self.agent.agents, ("left", "right")):
            home = agent.keyframes["rest"].qpos.copy()
            for name, angle in zip(agent.arm_joint_names, self.assembly["mounts"][side]["arm_qpos"]):
                home[agent.robot.active_joints_map[name].active_index[0].item()] = angle
            for keyframe in agent.keyframes.values():
                keyframe.qpos = home.copy()

    def _initialize_episode(self, env_idx, options):
        for agent, side in zip(self.agent.agents, ("left", "right")):
            agent.reset(agent.keyframes["rest"].qpos)
            agent.robot.set_pose(mount_pose(self.assembly["mounts"][side]))

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info):
        return {}
