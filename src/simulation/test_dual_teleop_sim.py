"""Hardware-free dual serial routing and CPU physics checks."""

import io
from queue import Queue
from threading import Event
import time
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from dual_teleop_sim import (AGENT_IDS, ARM_MAPPING, ArmMapping, DualServoTeleoperatorSim,
                             SerialLeader, SingleArmWorldMapping)


class FakeSerial:
    def __init__(self, *args, **kwargs):
        self.pwm = [1500] * 8
        self.command = ""
        self.commands = []
        self.closed = False
        self.invalid = None

    def write(self, value):
        self.command = value.decode("ascii")
        self.commands.append(self.command)

    def read_all(self):
        if self.command.endswith("PRAD!"):
            channel = int(self.command[1:4])
            if channel == self.invalid:
                return b""
            return f"#{channel:03d}P{self.pwm[channel]:04d}!".encode()
        return b""

    def close(self):
        self.closed = True


class DualTeleopTest(unittest.TestCase):
    def make_sim(self, scene="FlexivDualArm-v1", mapping_mode="joint"):
        ports = {}

        def open_port(port, *args, **kwargs):
            ports[port] = FakeSerial()
            return ports[port]

        with patch("dual_teleop_sim.serial.Serial", side_effect=open_port):
            sim = DualServoTeleoperatorSim(scene=scene, left_port="/dev/fake-left", right_port="/dev/fake-right", headless=True, mapping_mode=mapping_mode)
        self.addCleanup(sim.close)
        return sim, ports

    def calibrate(self, sim):
        for leader in sim.leaders.values():
            leader.zero_angles = leader.read_angles()
            leader.calibrated.set()

    def test_serial_samples_drive_independent_agents_and_grippers(self):
        sim, ports = self.make_sim()
        initial = {side: mapper.initial.copy() for side, mapper in sim.mappings.items()}
        self.assertFalse(sim.step_if_ready())
        self.assertEqual(sim.env.unwrapped._elapsed_steps.item(), 0)
        self.calibrate(sim)
        left = sim.leaders["left"]
        right = sim.leaders["right"]
        # +0.094 rad on left joint1; -54 degrees on left trigger.
        ports["/dev/fake-left"].pwm[0] += 40
        ports["/dev/fake-left"].pwm[7] -= 400
        # +5.4 degrees on right servo001 maps negatively to joint2.
        ports["/dev/fake-right"].pwm[1] += 40
        for _ in range(40):
            for leader in (left, right):
                leader.publish_arm_pos(np.radians(leader.read_angles() - leader.zero_angles))
            self.assertTrue(sim.step_if_ready())
        for side, mapper in sim.mappings.items():
            expected = initial[side].copy()
            expected[0 if side == "left" else 1] += np.deg2rad(5.4) * (1 if side == "left" else -1)
            np.testing.assert_allclose(mapper.action[:7], expected, atol=1e-6)
            agent = sim.env.unwrapped.agent.agents_dict[AGENT_IDS[side]]
            np.testing.assert_allclose(agent.controller.controllers["arm"].qpos[0].numpy(), expected, atol=0.02)
            width = agent.controller.controllers["gripper"].qpos[0, 0].item()
            self.assertLess(width, 0.005) if side == "left" else self.assertGreater(width, 0.095)
        # Release and press left again; right trigger state remains independent.
        for angle in (0., -45.):
            left.publish_arm_pos(np.radians([0.] * 7 + [angle]))
            sim.step_if_ready()
        self.assertEqual(sim.mappings["left"].action[7], 1.)
        self.assertEqual(sim.mappings["right"].action[7], 1.)
        # A single reference is insufficient even when that side has a pose.
        right.calibrated.clear()
        elapsed = sim.env.unwrapped._elapsed_steps.clone()
        self.assertFalse(sim.step_if_ready())
        self.assertEqual(sim.env.unwrapped._elapsed_steps.item(), elapsed.item())

    def test_only_fresh_samples_step_and_other_side_holds_target(self):
        sim, _ = self.make_sim(scene="Empty-v1")
        self.calibrate(sim)
        self.assertFalse(sim.step_if_ready())
        self.assertEqual(sim.env.unwrapped._elapsed_steps.item(), 0)
        right_initial = sim.mappings["right"].action.copy()
        left = sim.leaders["left"]
        left.publish_arm_pos([.01] + [0.] * 7)
        left.publish_arm_pos([.02] + [0.] * 7)
        self.assertTrue(sim.step_if_ready())
        self.assertAlmostEqual(sim.mappings["left"].action[0],
                               sim.mappings["left"].initial[0] + .02, places=6)
        np.testing.assert_array_equal(sim.mappings["right"].action, right_initial)
        self.assertFalse(sim.step_if_ready())
        self.assertEqual(sim.env.unwrapped._elapsed_steps.item(), 1)

    def test_run_stops_on_worker_error_and_releases_resources(self):
        sim, ports = self.make_sim()
        started = time.monotonic()
        first_poll = True

        def poll(*args):
            nonlocal first_poll
            if time.monotonic() - started > 5.:
                raise AssertionError("run did not react to serial failure")
            if first_poll:
                first_poll = False
                return ([None], [], [])
            if sim.env.unwrapped._elapsed_steps.item() >= 3:
                ports["/dev/fake-left"].invalid = 2
            return ([], [], [])

        with patch("dual_teleop_sim.sys.stdin", io.StringIO("\n")), \
                patch("dual_teleop_sim.select.select", side_effect=poll):
            with self.assertRaisesRegex(RuntimeError, "left serial acquisition stopped"):
                sim.run()
        self.assertTrue(all(device.closed for device in ports.values()))
        self.assertIsNone(sim.env)

    def test_empty_scene_dual_mounts_physics_and_reset(self):
        from mani_skill.envs.tasks.flexiv_dual_arm import load_assembly
        sim, _ = self.make_sim(scene="Empty-v1")
        self.assertFalse(hasattr(sim.env.unwrapped, "stand"))
        assembly = load_assembly()
        self.calibrate(sim)
        for _ in range(20):
            for leader in sim.leaders.values():
                leader.publish_arm_pos([0.] * 8)
            self.assertTrue(sim.step_if_ready())
        for side, mapper in sim.mappings.items():
            agent = sim.env.unwrapped.agent.agents_dict[AGENT_IDS[side]]
            np.testing.assert_allclose(agent.controller.controllers["arm"].qpos[0].numpy(), mapper.initial, atol=0.02)
        sim.env.reset(seed=0)
        for side, uid in AGENT_IDS.items():
            agent = sim.env.unwrapped.agent.agents_dict[uid]
            mount = assembly["mounts"][side]
            np.testing.assert_allclose(agent.robot.pose.p[0].numpy(), mount["position"], atol=1e-6)
            np.testing.assert_allclose(agent.robot.pose.q[0].numpy(), mount["quaternion_wxyz"], atol=1e-6)
            np.testing.assert_allclose(agent.controller.controllers["arm"].qpos[0].numpy(), mount["arm_qpos"], atol=1e-6)

    def test_empty_scene_single_arm_stays_at_origin(self):
        env = gym.make("Empty-v1", robot_uids="flexiv_rizon4", obs_mode="state",
                       control_mode="pd_joint_pos", sim_backend="cpu", render_backend="none")
        self.addCleanup(env.close)
        env.reset(seed=0)
        self.assertIsNone(env.unwrapped.dual_flexiv_assembly)
        np.testing.assert_allclose(env.unwrapped.agent.robot.pose.p[0].numpy(), [0., 0., 0.], atol=1e-7)

    def test_custom_scene_is_forwarded_and_ports_close(self):
        make_env = gym.make
        calls = []

        def make_scene(scene, **kwargs):
            calls.append((scene, kwargs))
            return make_env("FlexivDualArm-v1", **kwargs)

        serials = [FakeSerial(), FakeSerial()]
        with patch("dual_teleop_sim.gym.make", side_effect=make_scene), \
                patch("dual_teleop_sim.serial.Serial", side_effect=serials):
            sim = DualServoTeleoperatorSim(scene="CustomDualScene-v1", headless=True)
        try:
            self.assertEqual(calls[0][0], "CustomDualScene-v1")
            self.assertEqual(calls[0][1]["robot_uids"], ("flexiv_rizon4", "flexiv_rizon4"))
            for device in serials:
                self.assertEqual([cmd for cmd in device.commands if cmd.endswith("PULK!")],
                                 [f"#{i:03d}PULK!" for i in range(8)])
                self.assertFalse(any(cmd.endswith("PRAD!") for cmd in device.commands))
        finally:
            sim.close()
        self.assertTrue(all(device.closed for device in serials))

    def test_default_entry_uses_single_arm_joint_strategy(self):
        from teleop_sim import ServoTeleoperatorSim
        with patch("dual_teleop_sim.serial.Serial", side_effect=[FakeSerial(), FakeSerial()]):
            sim = DualServoTeleoperatorSim(scene="Empty-v1", headless=True)
        self.addCleanup(sim.close)
        for side, mapper in sim.mappings.items():
            self.assertIs(type(mapper), ArmMapping)
            # Same input, initial pose and limits must give exactly the same
            # arm/gripper targets as the existing single-arm implementation.
            single = ServoTeleoperatorSim.__new__(ServoTeleoperatorSim)
            single.robot_uids = "flexiv_rizon4"
            single.sim_init_angles = mapper.initial.copy()
            single.arm_action_low = mapper.low.copy()
            single.arm_action_high = mapper.high.copy()
            single.gripper_closed = False
            single.trigger_pressed = False
            for trigger in (0., -50., -50., 0., -50.):
                pose = [.02, -.03, .04, -.05, .06, -.07, .08, np.deg2rad(trigger)]
                np.testing.assert_allclose(mapper.convert(pose), single.convert_pose_to_action(pose), atol=1e-6)

    def test_mapping_permutation_limits_and_invalid_samples(self):
        sim, _ = self.make_sim()
        agent = sim.env.unwrapped.agent.agents_dict[AGENT_IDS["left"]]
        mapper = ArmMapping(agent, dict(servo_indices=[1, 0, 2, 3, 4, 5, 6],
                                       joint_signs=[1] * 7, joint_scales=[0.5] * 7))
        action = mapper.convert([0., 0.2] + [0.] * 6)
        expected = mapper.initial.copy()
        expected[0] += 0.1
        np.testing.assert_allclose(action[:7], expected, atol=1e-6)
        np.testing.assert_allclose(mapper.convert([100.] * 7 + [0.])[:7], mapper.high)
        for invalid in ([0.] * 7, [float("nan")] * 8):
            with self.assertRaises(ValueError):
                mapper.convert(invalid)

    def test_partial_initialization_closes_first_port(self):
        device = FakeSerial()
        with patch("dual_teleop_sim.serial.Serial", side_effect=[device, OSError("missing USB")]):
            with self.assertRaisesRegex(OSError, "missing USB"):
                DualServoTeleoperatorSim(headless=True)
        self.assertTrue(device.closed)

    def test_invalid_ports_and_rate(self):
        for kwargs in (dict(left_port="/dev/same", right_port="/dev/same"),
                       dict(rate=0), dict(rate=float("nan"))):
            with self.assertRaises(ValueError):
                DualServoTeleoperatorSim(**kwargs)


class WorldMappingTest(unittest.TestCase):
    def setUp(self):
        from teleop_sim import ServoTeleoperatorSim
        self.single_env = gym.make("Empty-v1", robot_uids="flexiv_rizon4", obs_mode="state",
                                   control_mode="pd_joint_pos", sim_backend="cpu", render_backend="none")
        self.addCleanup(self.single_env.close)
        self.single_env.reset(seed=0)
        agent = self.single_env.unwrapped.agent
        arm = agent.controller.controllers["arm"]
        self.single = ServoTeleoperatorSim.__new__(ServoTeleoperatorSim)
        self.single.robot_uids = "flexiv_rizon4"
        self.single.sim_init_angles = arm.qpos[0].numpy().copy()
        self.single.arm_action_low = arm.single_action_space.low.copy()
        self.single.arm_action_high = arm.single_action_space.high.copy()
        self.single.gripper_closed = False
        self.single.trigger_pressed = False
        self.single_model = agent.robot.create_pinocchio_model()
        self.single_qpos = agent.robot.qpos[0].numpy().copy()
        self.single_indices = arm.active_joint_indices.cpu().numpy()
        self.single_link = agent.tcp.index
        self.single_model.compute_forward_kinematics(self.single_qpos)
        self.single_initial = self.single_model.get_link_pose(self.single_link)
        self.dual_env = gym.make("Empty-v1", robot_uids=("flexiv_rizon4", "flexiv_rizon4"),
                                 obs_mode="state", control_mode="pd_joint_pos",
                                 sim_backend="cpu", render_backend="none")
        self.addCleanup(self.dual_env.close)
        self.dual_env.reset(seed=0)
        self.mappers = {
            side: SingleArmWorldMapping(agent, ARM_MAPPING[side])
            for side, agent in zip(("left", "right"), self.dual_env.unwrapped.agent.agents)
        }

    def single_target(self, pose):
        action = self.single.convert_pose_to_action(list(pose))
        qpos = self.single_qpos.copy()
        qpos[self.single_indices] = action[:7]
        self.single_model.compute_forward_kinematics(qpos)
        return self.single_model.get_link_pose(self.single_link)

    def test_all_channels_match_actual_single_arm_world_motion(self):
        for channel in range(7):
            for sign in (-1, 1):
                pose = np.zeros(8)
                pose[channel] = sign * .1
                for side, mapper in self.mappers.items():
                    reference = self.single_target(pose)
                    single_delta_rotation = (reference.to_transformation_matrix()[:3, :3]
                                             @ self.single_initial.to_transformation_matrix()[:3, :3].T)
                    with self.subTest(side=side, channel=channel, sign=sign):
                        action = mapper.convert(pose)
                        self.assertTrue(mapper.last_ik_success)
                        qpos = mapper.seed.copy()
                        qpos[mapper.arm_indices] = action[:7]
                        world = mapper.root * mapper.fk(qpos)
                        np.testing.assert_allclose(world.p - mapper.world_initial.p,
                                                   reference.p - self.single_initial.p, atol=2e-5)
                        expected_rotation = single_delta_rotation @ mapper.world_initial.to_transformation_matrix()[:3, :3]
                        np.testing.assert_allclose(world.to_transformation_matrix()[:3, :3],
                                                   expected_rotation, atol=3e-5)
                        self.assertTrue(np.all(action[:7] >= mapper.low))
                        self.assertTrue(np.all(action[:7] <= mapper.high))

    def test_lift_moves_both_actual_tcps_up_and_grippers_independently(self):
        pose = np.zeros(8)
        pose[1] = -.1  # The working single-arm mapping produces an upward TCP motion.
        single = self.single_target(pose)
        delta = single.p - self.single_initial.p
        self.assertGreater(delta[2], .06)
        actions = {}
        for side, mapper in self.mappers.items():
            local_pose = pose.copy()
            if side == "left":
                local_pose[7] = np.deg2rad(-50.)
            actions[AGENT_IDS[side]] = mapper.convert(local_pose)
        for _ in range(40):
            self.dual_env.step(actions)
        for side, mapper in self.mappers.items():
            agent = self.dual_env.unwrapped.agent.agents_dict[AGENT_IDS[side]]
            measured = agent.tcp.pose.p[0].numpy() - mapper.world_initial.p
            np.testing.assert_allclose(measured, delta, atol=.01)
            self.assertGreater(measured[2], .05)
            self.assertEqual(actions[AGENT_IDS[side]][7], -1. if side == "left" else 1.)

    def test_failed_ik_holds_previous_target_but_trigger_still_works(self):
        mapper = self.mappers["left"]
        previous = mapper.action.copy()
        class FailedModel:
            def compute_inverse_kinematics(self, *args, **kwargs):
                return mapper.seed.copy(), False, np.ones(6)
        with patch.object(mapper, "target_world", return_value=mapper.world_initial), \
                patch.object(mapper, "model", FailedModel()):
            action = mapper.convert([0.] * 7 + [np.deg2rad(-50.)])
        self.assertFalse(mapper.last_ik_success)
        np.testing.assert_array_equal(action[:7], previous[:7])
        self.assertEqual(action[7], -1.)

    def test_explicit_world_mapping(self):
        devices = [FakeSerial(), FakeSerial()]
        with patch("dual_teleop_sim.serial.Serial", side_effect=devices):
            sim = DualServoTeleoperatorSim(scene="Empty-v1", headless=True, mapping_mode="single-world")
        self.addCleanup(sim.close)
        self.assertTrue(all(isinstance(mapper, SingleArmWorldMapping) for mapper in sim.mappings.values()))


class SerialWorkerTest(unittest.TestCase):
    def test_default_angle_output_matches_single_arm_with_side_label(self):
        from teleop_sim import ServoTeleoperatorSim
        stop, errors = Event(), Queue()
        with patch("dual_teleop_sim.serial.Serial", return_value=FakeSerial()):
            leader = SerialLeader("right", "/dev/fake", 50., stop, errors)
        self.addCleanup(leader.close)
        single = ServoTeleoperatorSim.__new__(ServoTeleoperatorSim)
        pose = [0.01 * i for i in range(8)]
        with patch("builtins.print") as output:
            single.default_sender(pose)
            single_line = output.call_args.args[0]
            leader.default_sender(pose)
            self.assertEqual(output.call_args.args[0], f"[right] {single_line}")
            count = output.call_count
            leader.default_sender(pose)
            self.assertEqual(output.call_count, count + 1)
            leader.debug_angles = False
            leader.default_sender(pose)
            self.assertEqual(output.call_count, count + 1)

    def test_failed_reference_retry_and_stream_failure(self):
        device = FakeSerial()
        stop, errors = Event(), Queue()
        with patch("dual_teleop_sim.serial.Serial", return_value=device):
            leader = SerialLeader("left", "/dev/fake", 50, stop, errors)
        self.addCleanup(leader.close)
        self.addCleanup(stop.set)
        device.invalid = 7
        failed = Event()
        read_angles = leader.read_angles

        def read():
            try:
                return read_angles()
            except ValueError:
                failed.set()
                raise

        leader.read_angles = read
        leader.thread.start()
        leader.request_calibration()
        self.assertTrue(failed.wait(2.))
        self.assertFalse(leader.calibrated.is_set())
        self.assertEqual(leader.zero_angles, [0.] * 8)
        device.invalid = None
        leader.request_calibration()
        self.assertTrue(leader.calibrated.wait(2.))
        self.assertEqual(len(leader.zero_angles), 8)
        # A malformed frame after calibration must signal the main loop to stop.
        device.invalid = 3
        side, error = errors.get(timeout=2.)
        self.assertEqual(side, "left")
        self.assertIsInstance(error, ValueError)
        self.assertTrue(stop.is_set())


if __name__ == "__main__":
    unittest.main()
