"""Flexiv Rizon4 with the official Grav (Flexiv-GN01) parallel gripper."""

from copy import deepcopy

import numpy as np
import sapien
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (
    PDEEPoseControllerConfig,
    PDJointPosControllerConfig,
    PDJointPosMimicControllerConfig,
)
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils


@register_agent()
class FlexivRizon4(BaseAgent):
    uid = "flexiv_rizon4"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/flexiv_rizon4/rizon4.urdf"
    arm_joint_names = [f"joint{i}" for i in range(1, 8)]
    ee_link_name = "grav_tcp"
    gripper_joint_names = [
        "finger_width_joint", "left_outer_knuckle_joint", "left_inner_knuckle_joint",
        "left_inner_finger_joint", "right_outer_knuckle_joint",
        "right_inner_knuckle_joint", "right_inner_finger_joint",
    ]
    urdf_config = dict(
        _materials=dict(gripper=dict(static_friction=2., dynamic_friction=2., restitution=0.)),
        link={name: dict(material="gripper", patch_radius=0.02, min_patch_radius=0.005)
              for name in ("left_finger_tip", "right_finger_tip")},
    )

    def _after_loading_articulation(self):
        # PhysX orders the branched finger joints differently from the XML.
        # Resolve by name, including the virtual width joint (0=closed, 0.1=open).
        positions = dict(zip(self.arm_joint_names, [0., -0.69813, 0., 1.570796, 0., 0.69813, 0.]))
        positions.update(finger_width_joint=0.1)
        for name in self.gripper_joint_names[1:]:
            positions[name] = -0.7854 if "inner_finger" in name else 0.7854
        qpos = np.array([positions[j.name] for j in self.robot.active_joints])
        self.keyframes = {name: Keyframe(qpos=qpos.copy(), pose=sapien.Pose()) for name in ("rest", "home")}
        for link in self.robot.get_links():
            if link.name not in {"base_link", "flange", *[f"link{i}" for i in range(1, 8)]}:
                # Give the two arms different bits: suppress internal linkage
                # collisions while retaining collisions between their grippers.
                link.set_collision_group_bit(group=2, bit_idx=29 - (self._agent_idx or 0), bit=1)

    @property
    def _controller_configs(self):
        gains = dict(stiffness=1000, damping=100, force_limit=[123, 123, 64, 64, 39, 39, 39])
        configs = dict(
            pd_joint_pos=dict(arm=PDJointPosControllerConfig(
                self.arm_joint_names, lower=None, upper=None, normalize_action=False, **gains)),
            pd_joint_delta_pos=dict(arm=PDJointPosControllerConfig(
                self.arm_joint_names, lower=-0.1, upper=0.1, use_delta=True, **gains)),
            pd_ee_delta_pose=dict(arm=PDEEPoseControllerConfig(
                joint_names=self.arm_joint_names, pos_lower=-0.1, pos_upper=0.1,
                rot_lower=-0.1, rot_upper=0.1, ee_link=self.ee_link_name,
                urdf_path=self.urdf_path, **gains)),
        )
        mimic = {}
        for name in self.gripper_joint_names[1:]:
            sign = -1 if "inner_finger" in name else 1
            mimic[name] = dict(joint="finger_width_joint", multiplier=sign * 9.404, offset=-sign * 0.155)
        gripper = PDJointPosMimicControllerConfig(
            self.gripper_joint_names, lower=0., upper=0.1,
            stiffness=1000, damping=20, force_limit=5, mimic=mimic,
        )
        for config in configs.values():
            config["gripper"] = gripper
        return deepcopy(configs)

    def _after_init(self):
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)
        self.finger1_link = self.robot.links_map["left_finger_tip"]
        self.finger2_link = self.robot.links_map["right_finger_tip"]

    def is_grasping(self, object, min_force=0.5, max_angle=85):
        flags = []
        for link, sign in ((self.finger1_link, -1), (self.finger2_link, -1)):
            force = self.scene.get_pairwise_contact_forces(link, object)
            direction = sign * link.pose.to_transformation_matrix()[..., :3, 1]
            angle = common.compute_angle_between(direction, force)
            flags.append((torch.linalg.norm(force, dim=1) >= min_force) & (torch.rad2deg(angle) <= max_angle))
        return flags[0] & flags[1]

    def reset(self, init_qpos=None):
        super().reset(self.keyframes["rest"].qpos if init_qpos is None else init_qpos)

    def is_static(self, threshold=0.2):
        indices = self.controller.controllers["arm"].active_joint_indices
        return torch.max(torch.abs(self.robot.get_qvel()[..., indices]), dim=1).values <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose
