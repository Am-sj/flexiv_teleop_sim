"""Hardware-free checks for servo channel routing and Panda joint targets.

Run from this directory with: python -m unittest test_teleop_sim
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch

from teleop_sim import ServoTeleoperatorSim


class ServoMappingTest(unittest.TestCase):
    def make_sim(self, robot="panda"):
        env = MagicMock()
        env.reset.return_value = ({}, {})
        arm = env.unwrapped.agent.controller.controllers["arm"]
        arm.qpos = torch.tensor([[0., 0.3, 0., -1.9, 0., 2.3, 0.7]])
        arm.single_action_space.low = np.array([-2.9, -1.7, -2.9, -3., -2.9, -0.01, -2.9])
        arm.single_action_space.high = np.array([2.9, 1.7, 2.9, -0.06, 2.9, 3.7, 2.9])
        with patch("teleop_sim.serial.Serial"), patch("teleop_sim.gym.make", return_value=env), \
                patch.object(ServoTeleoperatorSim, "_setup_camera_pose"), \
                patch.object(ServoTeleoperatorSim, "send_command", return_value="#000P1500!"):
            return ServoTeleoperatorSim("Empty-v1", robot)

    def test_each_panda_servo_moves_only_its_joint(self):
        sim = self.make_sim()
        baseline = sim.convert_pose_to_action([0.] * 8)
        np.testing.assert_allclose(baseline[:7], sim.sim_init_angles)
        for joint in range(7):
            pose = np.zeros(8)
            pose[joint] = 0.1
            expected = baseline.copy()
            expected[joint] += 0.1
            np.testing.assert_allclose(sim.convert_pose_to_action(pose), expected, atol=1e-6)

    def test_eighth_servo_controls_only_gripper(self):
        sim = self.make_sim()
        # Idle jitter, press, hold, threshold jitter, release, press again.
        sequence = [(0, 1), (-30, 1), (10, 1), (-39, 1), (-40, -1),
                    (-55, -1), (-39, -1), (-41, -1), (-31, -1),
                    (-30, -1), (10, -1), (-40, 1), (0, 1)]
        for angle, expected in sequence:
            action = sim.convert_pose_to_action([0.] * 7 + [np.deg2rad(angle)])
            np.testing.assert_allclose(action[:7], sim.sim_init_angles)
            self.assertAlmostEqual(action[7], expected)

    def test_positive_trigger_travel_does_not_toggle(self):
        sim = self.make_sim()
        for angle in [0, 10, 45, 55, 90, 0]:
            action = sim.convert_pose_to_action([0.] * 7 + [np.deg2rad(angle)])
            self.assertEqual(action[7], 1.)

    def test_panda_limits_and_channel_count(self):
        sim = self.make_sim()
        for offset, expected in [(100., sim.arm_action_high), (-100., sim.arm_action_low)]:
            action = sim.convert_pose_to_action([offset] * 8)
            np.testing.assert_allclose(action[:7], expected)
        with self.assertRaisesRegex(ValueError, "8 servo angles"):
            sim.convert_pose_to_action([0.] * 7)

    def test_calibration_and_stream_read_every_channel(self):
        for robot, count in [("panda", 8), ("flexiv_rizon4", 8), ("xarm6_robotiq", 7)]:
            with self.subTest(robot=robot):
                sim = self.make_sim(robot)
                self.assertEqual(sim.num_servos, count)
                self.assertEqual(len(sim.zero_angles), count)

                def response(command):
                    return command[:4] + "P1500!"

                sim.send_command = Mock(side_effect=response)
                sim._calibrate_servos()
                reads = [call.args[0] for call in sim.send_command.call_args_list
                         if call.args[0].endswith("PRAD!")]
                expected = [f"#{i:03d}PRAD!" for i in range(count)]
                self.assertEqual(reads, expected)
                np.testing.assert_allclose(sim.zero_angles, [135.] * count)

                sim.send_command.reset_mock()
                published = []

                def on_send(pose):
                    published.append(pose)
                    sim.stop_event.set()

                sim.angle_stream_loop(on_send)
                self.assertEqual([c.args[0] for c in sim.send_command.call_args_list], expected)
                np.testing.assert_allclose(published, [[0.] * count])

    def test_startup_unlocks_without_calibrating(self):
        sim = self.make_sim("flexiv_rizon4")
        self.assertFalse(sim.calibrated.is_set())
        self.assertFalse(sim.calibration_requested.is_set())
        sim.send_command = Mock()
        sim._init_servos()
        commands = [call.args[0] for call in sim.send_command.call_args_list]
        self.assertFalse(any(command.endswith("PRAD!") for command in commands))
        self.assertEqual([command for command in commands if command.endswith("PULK!")],
                         [f"#{i:03d}PULK!" for i in range(8)])

    def test_waiting_for_calibration_keeps_rendering_without_actions(self):
        sim = self.make_sim()
        sim.publish_arm_pos([1.] * 8)
        sim.env.render.side_effect = sim.stop_event.set
        on_pose = Mock()
        sim.pose_consumer_loop(on_pose)
        sim.env.render.assert_called_once()
        sim.env.step.assert_not_called()
        on_pose.assert_not_called()

    def test_serial_reader_waits_for_manual_request(self):
        sim = self.make_sim()
        sim.send_command = Mock()

        def stop_waiting(timeout):
            sim.stop_event.set()
            return False

        with patch.object(sim.calibration_requested, "wait", side_effect=stop_waiting):
            sim.angle_stream_loop(Mock())
        sim.send_command.assert_not_called()
        self.assertFalse(sim.calibrated.is_set())

    def test_manual_request_calibrates_before_first_pose(self):
        sim = self.make_sim("flexiv_rizon4")
        sim.send_command = Mock(side_effect=lambda command: command[:4] + "P1500!")
        sim.request_calibration()
        published = []

        def on_send(pose):
            published.append(pose)
            self.assertTrue(sim.calibrated.is_set())
            sim.stop_event.set()

        sim.angle_stream_loop(on_send)
        expected_reads = [f"#{i:03d}PRAD!" for i in range(8)] * 2
        self.assertEqual([c.args[0] for c in sim.send_command.call_args_list], expected_reads)
        np.testing.assert_allclose(published, [[0.] * 8])
        action = sim.convert_pose_to_action(published[0])
        np.testing.assert_allclose(action[:7], sim.sim_init_angles)
        self.assertEqual(action[7], 1.)

    def test_failed_calibration_is_atomic_and_can_retry(self):
        sim = self.make_sim()
        original = sim.zero_angles.copy()
        sim.send_command = Mock(side_effect=lambda command:
                                "" if command == "#007PRAD!" else command[:4] + "P1500!")
        with self.assertRaisesRegex(ValueError, "Servo 7 calibration response error"):
            sim._calibrate_servos()
        self.assertEqual(sim.zero_angles, original)
        self.assertFalse(sim.calibrated.is_set())
        self.assertTrue(sim.arm_pos_queue.empty())
        sim.send_command = Mock(side_effect=lambda command: command[:4] + "P1500!")
        sim._calibrate_servos()
        self.assertTrue(sim.calibrated.is_set())
        np.testing.assert_allclose(sim.zero_angles, [135.] * 8)

    def test_six_axis_mapping_keeps_seventh_channel_as_gripper(self):
        sim = self.make_sim("xarm6_robotiq")
        action = sim.convert_pose_to_action([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.])
        np.testing.assert_allclose(action, [0.1, 0.2, 0.3, 0.5, -0.4, 0.6, 0.])


if __name__ == "__main__":
    unittest.main()
