"""Run from src/simulation: python -m unittest test_flexiv_rizon4."""

import unittest

import gymnasium as gym
import mani_skill.envs
import numpy as np
import sapien
import torch
from mani_skill.envs.tasks.empty_env import EmptyEnv

import test_teleop_sim


class FlexivServoMappingTest(unittest.TestCase):
    def test_eight_channels_map_independently_and_clip(self):
        sim = test_teleop_sim.ServoMappingTest().make_sim("flexiv_rizon4")
        self.assertEqual(sim.num_servos, 8)
        baseline = sim.convert_pose_to_action([0.] * 8)
        np.testing.assert_allclose(baseline[:7], sim.sim_init_angles)
        self.assertEqual(baseline[7], 1.)
        for joint in range(7):
            pose = np.zeros(8)
            pose[joint] = 0.1
            expected = baseline.copy()
            expected[joint] += -0.1 if joint == 1 else 0.1
            np.testing.assert_allclose(sim.convert_pose_to_action(pose), expected, atol=1e-6)
        for offset in (100., -100.):
            expected = (sim.arm_action_high if offset > 0 else sim.arm_action_low).copy()
            expected[1] = sim.arm_action_low[1] if offset > 0 else sim.arm_action_high[1]
            np.testing.assert_allclose(sim.convert_pose_to_action([offset] * 8)[:7], expected)
        for count in (6, 7, 9):
            with self.assertRaisesRegex(ValueError, "8 servo angles"):
                sim.convert_pose_to_action([0.] * count)

    def test_grav_trigger_toggles_without_moving_arm(self):
        sim = test_teleop_sim.ServoMappingTest().make_sim("flexiv_rizon4")
        for angle, expected in [(0, 1), (-39, 1), (-40, -1), (-55, -1), (-31, -1), (-30, -1), (-40, 1)]:
            action = sim.convert_pose_to_action([0.] * 7 + [np.deg2rad(angle)])
            np.testing.assert_allclose(action[:7], sim.sim_init_angles)
            self.assertEqual(action[7], expected)


class FlexivSimulationTest(unittest.TestCase):
    def make_env(self, control_mode="pd_joint_pos"):
        env = gym.make("Empty-v1", robot_uids="flexiv_rizon4", obs_mode="state",
                       control_mode=control_mode, sim_backend="cpu", render_backend="none")
        self.addCleanup(env.close)
        env.reset(seed=0)
        return env

    def test_load_move_and_reset(self):
        env = self.make_env()
        agent = env.unwrapped.agent
        self.assertEqual([joint.name for joint in agent.robot.active_joints][:7], agent.arm_joint_names)
        self.assertEqual(len(agent.robot.active_joints), 14)
        self.assertEqual(env.action_space.shape, (8,))
        np.testing.assert_allclose(agent.robot.qpos[0].numpy(), agent.keyframes["rest"].qpos, atol=1e-6)
        initial_tcp = agent.tcp_pos.clone()
        target = agent.controller.controllers["arm"].qpos[0].numpy().copy()
        target[0] += 0.2
        for _ in range(30):
            env.step(np.r_[target, 1.].astype(np.float32))
        self.assertTrue(torch.isfinite(agent.robot.qpos).all())
        np.testing.assert_allclose(agent.robot.qpos[0, :7].numpy(), target, atol=0.02)
        self.assertGreater(torch.linalg.norm(agent.tcp_pos - initial_tcp).item(), 0.01)
        env.reset(seed=0)
        np.testing.assert_allclose(agent.robot.qpos[0].numpy(), agent.keyframes["rest"].qpos, atol=1e-6)
        custom_qpos = agent.keyframes["rest"].qpos.copy()
        custom_qpos[:7] = target
        agent.reset(custom_qpos)
        np.testing.assert_allclose(agent.robot.qpos[0].numpy(), custom_qpos, atol=1e-6)

    def test_delta_control(self):
        env = self.make_env("pd_joint_delta_pos")
        for _ in range(5):
            env.step(np.r_[np.zeros(7), 1.].astype(np.float32))
        self.assertTrue(torch.isfinite(env.unwrapped.agent.robot.qpos).all())

    def test_grav_open_close_and_mimic(self):
        env = self.make_env()
        agent = env.unwrapped.agent
        arm = agent.controller.controllers["arm"].qpos[0].numpy().copy()
        gripper = agent.controller.controllers["gripper"]
        gaps = []
        for command, width in [(1., 0.1), (-1., 0.), (1., 0.1)]:
            for _ in range(50):
                env.step(np.r_[arm, command].astype(np.float32))
            expected = [width] + [
                (-1 if "inner_finger" in name else 1) * (9.404 * width - 0.155)
                for name in agent.gripper_joint_names[1:]
            ]
            np.testing.assert_allclose(gripper.qpos[0].numpy(), expected, atol=0.005)
            gaps.append(torch.linalg.norm(agent.finger1_link.pose.p - agent.finger2_link.pose.p).item())
            np.testing.assert_allclose(agent.controller.controllers["arm"].qpos[0].numpy(), arm, atol=0.01)
        self.assertGreater(gaps[0] - gaps[1], 0.07)
        self.assertAlmostEqual(gaps[0], gaps[2], delta=0.002)

    def test_end_effector_control(self):
        env = self.make_env("pd_ee_delta_pose")
        initial_tcp = env.unwrapped.agent.tcp_pos.clone()
        for _ in range(5):
            env.step(np.r_[np.zeros(6), 1.].astype(np.float32))
        self.assertTrue(torch.isfinite(env.unwrapped.agent.robot.qpos).all())
        torch.testing.assert_close(env.unwrapped.agent.tcp_pos, initial_tcp, atol=0.005, rtol=0)

    def test_grav_finger_contact_and_release(self):
        class ContactEnv(EmptyEnv):
            def _load_scene(self, options):
                super()._load_scene(options)
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(half_size=[0.012, 0.015, 0.012])
                builder.initial_pose = sapien.Pose([0, 0, 0.3])
                self.cube = builder.build_kinematic(name="grasp_test_cube")

            def _initialize_episode(self, env_idx, options):
                pose = self.agent.robot.links_map["grav_base_link"].pose.sp
                self.cube.set_pose(pose * sapien.Pose([0, 0, 0.19]))

        env = ContactEnv(robot_uids="flexiv_rizon4", control_mode="pd_joint_pos",
                         obs_mode="state", sim_backend="cpu", render_backend="none")
        self.addCleanup(env.close)
        env.reset(seed=0)
        arm = env.agent.controller.controllers["arm"].qpos[0].numpy().copy()
        for command, grasping in [(1., False), (-1., True), (1., False)]:
            for _ in range(60):
                env.step(np.r_[arm, command].astype(np.float32))
            self.assertEqual(env.agent.is_grasping(env.cube).item(), grasping)


if __name__ == "__main__":
    unittest.main()
