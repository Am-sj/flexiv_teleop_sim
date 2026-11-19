"""Hardware-free checks for the dual real teleoperation entrypoint."""

import types
import unittest
from unittest.mock import patch

import numpy as np

import dual_teleop_real as teleop


class FakeSerial:
    def __init__(self, *args, **kwargs):
        self.command = ""
        self.closed = False
        self.is_open = True

    def write(self, data):
        self.command = data.decode("ascii")

    def read_all(self):
        match = self.command[1:4]
        if self.command.endswith("PRAD!"):
            return f"#{match}P1500!".encode("ascii")
        return b""

    def cancel_read(self):
        pass

    def cancel_write(self):
        pass

    def close(self):
        self.closed = True
        self.is_open = False


class FakeRobot:
    def __init__(self, sn):
        self.sn = sn
        self.q = [0.1 * i for i in range(7)]
        self.calls = []
        self.is_fault = False
        self.is_operational = True
        self.current_mode = "IDLE"

    def fault(self):
        return self.is_fault

    def operational(self):
        return self.is_operational

    def states(self):
        return types.SimpleNamespace(q=self.q)

    def info(self):
        return types.SimpleNamespace(
            DoF=7,
            q_min=teleop.RIZON4_Q_MIN.tolist(),
            q_max=teleop.RIZON4_Q_MAX.tolist(),
            dq_max=teleop.RIZON4_QD_MAX.tolist(),
        )

    def mode(self):
        return self.current_mode

    def SwitchMode(self, mode):
        self.calls.append(("SwitchMode", mode))
        self.current_mode = mode

    def SendJointPosition(self, *args):
        self.calls.append(("SendJointPosition", args))

    def Stop(self):
        self.calls.append(("Stop",))

    def Enable(self):
        self.calls.append(("Enable",))

    def ClearFault(self, *args):
        self.calls.append(("ClearFault",))


class FakeGripper:
    def __init__(self, robot):
        self.calls = []
        self._params = types.SimpleNamespace(
            min_width=0.0, max_width=0.1, min_vel=0.01, max_vel=0.2,
            min_force=5.0, max_force=40.0,
        )

    def Enable(self, name):
        self.calls.append(("Enable", name))

    def params(self):
        return self._params

    def Move(self, *args):
        self.calls.append(("Move", args))

    def Init(self):
        self.calls.append(("Init",))

    def Stop(self):
        self.calls.append(("Stop",))


class FakeTool:
    def __init__(self, robot):
        self.calls = []

    def Switch(self, name):
        self.calls.append(("Switch", name))


class FakeRdk:
    Mode = types.SimpleNamespace(IDLE="IDLE", NRT_JOINT_POSITION="NRT_JOINT_POSITION")
    Robot = FakeRobot
    Gripper = FakeGripper
    Tool = FakeTool


class DualRealTeleopTests(unittest.TestCase):
    def make_teleop(self, **kwargs):
        devices = []

        def open_serial(*args, **serial_kwargs):
            device = FakeSerial()
            devices.append(device)
            return device

        with patch.object(teleop.serial, "Serial", side_effect=open_serial), \
                patch.object(teleop.time, "sleep", return_value=None):
            kwargs.setdefault("left_gripper_name", "Left Grav")
            kwargs.setdefault("right_gripper_name", "Right Grav")
            instance = teleop.DualRealTeleoperator(
                "left-sn", "right-sn", "/dev/fake-left", "/dev/fake-right",
                rdk=FakeRdk, **kwargs,
            )
        self.addCleanup(instance.close)
        return instance, devices

    def test_joint_mapping_sign_clamp_and_trigger_hysteresis(self):
        mapper = teleop.ArmMapping(np.zeros(7), max_offset=0.5)
        target, edge = mapper.convert([0.2, 0.3, 0, 0, 0, 0, 0,
                                      np.deg2rad(-41)])
        np.testing.assert_allclose(target[:2], [0.2, -0.3])
        self.assertTrue(edge)
        self.assertTrue(mapper.closed)
        _, edge = mapper.convert([0] * 7 + [np.deg2rad(-35)])
        self.assertFalse(edge)
        _, edge = mapper.convert([0] * 7 + [np.deg2rad(-29)])
        self.assertFalse(edge)
        self.assertFalse(mapper.pressed)
        _, edge = mapper.convert([0] * 7 + [np.deg2rad(-41)])
        self.assertTrue(edge)
        self.assertFalse(mapper.closed)
        target, _ = mapper.convert([2.0] + [0] * 7)
        self.assertEqual(target[0], 0.5)

    def test_joint_target_clips_to_absolute_rizon_hard_limits(self):
        initial = teleop.RIZON4_Q_MAX.copy()
        mapper = teleop.ArmMapping(initial, max_offset=1.0)
        target, _ = mapper.convert([1.0] + [0] * 7)
        self.assertEqual(target[0], teleop.RIZON4_Q_MAX[0])

    def test_velocity_limits_default_conservative_and_cap_at_model_limits(self):
        instance, _ = self.make_teleop()
        for values in instance.max_vel.values():
            np.testing.assert_allclose(values, [0.5] * 7)
        np.testing.assert_allclose(instance.max_acc, [1.0] * 7)
        instance, _ = self.make_teleop(max_joint_velocity=10.0)
        for values in instance.max_vel.values():
            np.testing.assert_allclose(values, teleop.RIZON4_QD_MAX)

    def test_control_sends_live_relative_targets_and_velocity_constraints(self):
        instance, _ = self.make_teleop(max_joint_velocity=0.4)
        offsets = np.radians([10, 10, 0, 0, 0, 0, 0, 0])
        for leader in instance.leaders.values():
            leader.samples.put_nowait(offsets.copy())
        instance.stop_event.wait = lambda timeout: instance.stop_event.set()
        instance.run_control()
        for side in teleop.SIDES:
            robot = instance.robots[side]
            command = next(call[1] for call in robot.calls
                           if call[0] == "SendJointPosition")
            target, velocity, max_velocity, max_acceleration = command
            expected = np.asarray(robot.q) + np.deg2rad([10, -10, 0, 0, 0, 0, 0])
            np.testing.assert_allclose(target, expected)
            np.testing.assert_allclose(velocity, np.zeros(7))
            np.testing.assert_allclose(max_velocity, [0.4] * 7)
            np.testing.assert_allclose(max_acceleration, [1.0] * 7)

    def test_confirmation_setup_uses_live_baseline_without_enable_or_open(self):
        instance, _ = self.make_teleop()
        instance._arm_after_confirmation()
        for side in teleop.SIDES:
            robot = instance.robots[side]
            np.testing.assert_array_equal(instance.mappings[side].initial, robot.q)
            self.assertFalse(any(call[0] in {"Enable", "ClearFault"}
                                 for call in robot.calls))
            self.assertFalse(any(call[0] == "SendJointPosition" for call in robot.calls))
            self.assertEqual(instance.grippers[side].calls,
                             [("Enable", f"{'Left' if side == 'left' else 'Right'} Grav")])

    def test_calibration_is_not_committed_when_either_read_fails(self):
        instance, _ = self.make_teleop()
        left, right = instance.leaders.values()
        left.calibration_candidate = np.zeros(8)
        left.calibration_ready.set()
        instance.errors.put(("right", ValueError("invalid serial response")))
        instance.stop_event.set()
        with self.assertRaisesRegex(RuntimeError, "right UArm serial acquisition failed"):
            instance._wait_for_calibration()
        self.assertFalse(left.calibrated.is_set())
        self.assertFalse(right.calibrated.is_set())

    def test_gripper_moves_only_after_trigger_edge_and_cleanup_stops_both(self):
        instance, _ = self.make_teleop()
        instance._arm_after_confirmation()
        for side in teleop.SIDES:
            target, edge = instance.mappings[side].convert(
                [0.0] * 7 + [np.deg2rad(-41.0)]
            )
            self.assertEqual(target.shape, (7,))
            self.assertTrue(edge)
            instance._command_gripper(side)
            moves = [call for call in instance.grippers[side].calls if call[0] == "Move"]
            self.assertEqual(moves, [("Move", (0.0, 0.05, 20.0))])
        robots = list(instance.robots.values())
        grippers = list(instance.grippers.values())
        instance.close()
        self.assertTrue(all(any(call[0] == "Stop" for call in robot.calls)
                            for robot in robots))
        self.assertTrue(all(any(call[0] == "Stop" for call in gripper.calls)
                            for gripper in grippers))

    def test_invalid_robot_state_fails_and_cleans_connected_robot(self):
        invalid_rdk = type("InvalidRdk", (FakeRdk,), {})
        created = []

        class BadStateRobot(FakeRobot):
            def __init__(self, sn):
                super().__init__(sn)
                created.append(self)
                self.q = [float("nan")] * 7

        invalid_rdk.Robot = BadStateRobot
        with patch.object(teleop.serial, "Serial", side_effect=FakeSerial), \
                patch.object(teleop.time, "sleep", return_value=None):
            with self.assertRaisesRegex(ValueError, "finite joint positions"):
                teleop.DualRealTeleoperator(
                    "left-sn", "right-sn", "/dev/fake-left", "/dev/fake-right",
                    rdk=invalid_rdk, left_gripper_name="Left Grav",
                    right_gripper_name="Right Grav",
                )
        self.assertFalse(any(call[0] == "Stop" for call in created[0].calls))

    def test_cli_keeps_grippers_enabled_by_default_and_requires_robot_sns(self):
        args = teleop.parse_args([
            "--left-robot-sn", "L", "--right-robot-sn", "R",
            "--left-gripper-name", "Left Grav", "--right-gripper-name", "Right Grav",
        ])
        self.assertFalse(args.no_grippers)
        self.assertEqual(args.left_gripper_name, "Left Grav")
        with self.assertRaises(SystemExit):
            teleop.parse_args([])
        with self.assertRaises(SystemExit):
            teleop.parse_args(["--left-robot-sn", "L", "--right-robot-sn", "R"])
        disabled = teleop.parse_args([
            "--left-robot-sn", "L", "--right-robot-sn", "R", "--no-grippers",
        ])
        self.assertTrue(disabled.no_grippers)

    def test_rdk_version_gate_rejects_other_api_versions(self):
        with patch.object(teleop, "distribution_version", return_value="2.0.0"):
            with self.assertRaisesRegex(RuntimeError, "RDK 2.x is not supported"):
                teleop.load_flexiv_rdk()


if __name__ == "__main__":
    unittest.main()
