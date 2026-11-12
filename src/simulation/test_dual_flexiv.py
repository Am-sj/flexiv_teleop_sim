"""CPU assembly checks: python -m unittest test_dual_flexiv."""

import unittest

import gymnasium as gym
import mani_skill.envs
import numpy as np
import torch
import trimesh

from dual_flexiv_viewer import hold_actions
from mani_skill.envs.tasks.flexiv_dual_arm import ASSEMBLY_DIR, load_assembly


class DualFlexivTest(unittest.TestCase):
    def make_env(self):
        env = gym.make("FlexivDualArm-v1", obs_mode="state", control_mode="pd_joint_pos",
                       sim_backend="cpu", render_backend="none")
        self.addCleanup(env.close)
        env.reset(seed=0)
        return env

    def test_cad_scale_and_mounting_surfaces(self):
        data = load_assembly()
        self.assertEqual(data["solid_count"], 36)
        mesh = trimesh.load(ASSEMBLY_DIR / "stand_collision.stl", force="mesh")
        np.testing.assert_allclose(mesh.bounds, data["bounds_m"], atol=1e-6)
        np.testing.assert_allclose(mesh.extents, [0.86, 0.86, 1.395], atol=1e-5)
        left, right = (data["mounts"][side] for side in ("left", "right"))
        np.testing.assert_allclose(left["position"], [0.115, 0.25, 1.305])
        np.testing.assert_allclose(right["position"], [0.115, -0.25, 1.305])

    def test_actual_mount_axes_and_connector_directions(self):
        env = self.make_env()
        self.assertEqual(list(env.action_space.spaces), ["flexiv_rizon4-0", "flexiv_rizon4-1"])
        for agent, side, connector_z in zip(env.unwrapped.agent.agents, ("left", "right"), (-1., 1.)):
            pose = agent.robot.pose
            rotation = pose.to_transformation_matrix()[0, :3, :3].numpy()
            np.testing.assert_allclose(rotation @ [0, 1, 0], [0, 0, connector_z], atol=1e-6)
            np.testing.assert_allclose(rotation @ [0, 0, 1], [1, 0, 0], atol=1e-6)
            np.testing.assert_allclose(pose.p[0].numpy(), env.unwrapped.assembly["mounts"][side]["position"], atol=1e-6)
            self.assertEqual(agent.robot.max_dof, 14)
            # Reference-photo posture: elbows spread beside the column and
            # grippers below the wrists, rather than both arms hanging down.
            elbow = agent.robot.links_map["link4"].pose.p[0].numpy()
            side_sign = 1 if side == "left" else -1
            self.assertGreater(side_sign * (elbow[1] - pose.p[0, 1].item()), 0.25)
            self.assertGreater(elbow[2], pose.p[0, 2].item() - 0.2)
            wrist = agent.robot.links_map["link7"].pose.p[0].numpy()
            self.assertLess(agent.tcp_pos[0, 2].item(), wrist[2] - 0.1)
            if side == "right":
                # Check the visible wrist axes against the reference photo,
                # not just whether the gripper generally points downward.
                camera = env.unwrapped._default_human_render_camera_configs.pose
                transform = camera.to_transformation_matrix()[0].numpy()
                cap = agent.robot.links_map["link6"].pose.p[0].numpy()
                flange = agent.robot.links_map["flange"].pose.p[0].numpy()
                points = np.array([cap, wrist, flange])
                local = (points - transform[:3, 3]) @ transform[:3, :3]
                pixels = -local[:, 1:] / local[:, :1]
                raised_wrist = pixels[1] - pixels[0]
                tilted_flange = pixels[2] - pixels[1]
                self.assertGreater(raised_wrist[0], 0)
                self.assertLess(raised_wrist[1], 0)
                self.assertGreater(tilted_flange[0], 0)
                self.assertGreater(tilted_flange[1], 0)
                cap_axis = agent.robot.links_map["link6"].pose.to_transformation_matrix()[0, :3, 1].numpy()
                toward_camera = transform[:3, 3] - cap
                toward_camera /= np.linalg.norm(toward_camera)
                self.assertGreater(np.dot(cap_axis, toward_camera), 0.8)
        self.assertTrue(any("TriangleMesh" in type(shape).__name__
                            for shape in env.unwrapped.stand._bodies[0].get_collision_shapes()))

    def test_independent_arm_and_gripper_control_then_reset(self):
        env = self.make_env()
        left, right = env.unwrapped.agent.agents
        actions = hold_actions(env)
        left_initial = actions["flexiv_rizon4-0"].copy()
        right_initial = actions["flexiv_rizon4-1"].copy()
        actions["flexiv_rizon4-0"][0] += 0.1
        actions["flexiv_rizon4-0"][7] = -1.
        for _ in range(40):
            env.step(actions)
        np.testing.assert_allclose(left.controller.controllers["arm"].qpos[0].numpy(), actions["flexiv_rizon4-0"][:7], atol=0.015)
        np.testing.assert_allclose(right.controller.controllers["arm"].qpos[0].numpy(), right_initial[:7], atol=0.015)
        self.assertLess(left.controller.controllers["gripper"].qpos[0, 0].item(), 0.005)
        self.assertGreater(right.controller.controllers["gripper"].qpos[0, 0].item(), 0.095)
        for agent in (left, right):
            self.assertTrue(torch.isfinite(agent.robot.qpos).all())
        # The mounted assembly should not contain initial arm/stand or arm/arm
        # penetrations. The rigid wrist seam is filtered separately in SRDF.
        # PhysX also returns speculative contacts with positive separation.
        # Check actual penetration rather than treating every query as a hit.
        for contact in env.unwrapped.scene.get_contacts():
            self.assertGreaterEqual(min(point.separation for point in contact.points), -1e-4)
        env.reset(seed=0)
        reset_actions = hold_actions(env)
        np.testing.assert_allclose(reset_actions["flexiv_rizon4-0"], left_initial, atol=1e-6)
        np.testing.assert_allclose(reset_actions["flexiv_rizon4-1"], right_initial, atol=1e-6)

    def test_internal_gripper_filter_does_not_disable_cross_arm_collisions(self):
        env = self.make_env()
        left, right = env.unwrapped.agent.agents
        left_bits = left.finger1_link._bodies[0].get_collision_shapes()[0].get_collision_groups()[2]
        right_bits = right.finger1_link._bodies[0].get_collision_shapes()[0].get_collision_groups()[2]
        self.assertTrue(left_bits & (1 << 29))
        self.assertTrue(right_bits & (1 << 28))
        self.assertEqual(left_bits & right_bits, 0)


if __name__ == "__main__":
    unittest.main()
