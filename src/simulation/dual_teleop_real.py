"""Control two Flexiv Rizon4 robots from two UArm leaders.

This entrypoint targets flexivrdk 1.9.4.1 (robot software v3.11.3).
No robot is enabled and no fault is cleared by this program.
"""

import argparse
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from queue import Empty, Queue
import re
import select
import sys
from threading import Event, Thread
import time

import numpy as np
import serial


SIDES = ("left", "right")
REQUIRED_RDK_VERSION = "1.9.4.1"
BAUDRATE = 115200
NUM_SERVOS = 8
PWM_MIN = 500
PWM_MAX = 2500
ANGLE_RANGE_DEG = 270
TRIGGER_PRESS_ANGLE = np.deg2rad(-40.0)
TRIGGER_RELEASE_ANGLE = np.deg2rad(-30.0)
JOINT_SIGNS = np.array([1, -1, 1, 1, 1, 1, 1], dtype=np.float64)
# Hard joint stops copied from mani_skill/assets/robots/flexiv_rizon4/rizon4.urdf.
RIZON4_Q_MIN = np.array(
    [-2.8798, -2.3562, -3.0543, -1.9548, -3.0543, -1.4835, -3.0543],
    dtype=np.float64,
)
RIZON4_Q_MAX = np.array(
    [2.8798, 2.3562, 3.0543, 2.7751, 3.0543, 4.6251, 3.0543],
    dtype=np.float64,
)
RIZON4_QD_MAX = np.array(
    [2.0944, 2.0944, 2.4435, 2.4435, 4.8869, 4.8869, 4.8869],
    dtype=np.float64,
)
MAX_JOINT_OFFSET = 1.0
DEFAULT_MAX_JOINT_VELOCITY = 0.5
DEFAULT_MAX_JOINT_ACCELERATION = 1.0
ZERO_VEL = [0.0] * 7
GRIPPER_VELOCITY = 0.05
GRIPPER_FORCE = 20.0


def load_flexiv_rdk():
    """Load only the user-approved Flexiv RDK API line."""
    try:
        installed_version = distribution_version("flexivrdk")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"Install flexivrdk=={REQUIRED_RDK_VERSION} in this Python environment"
        ) from exc
    if installed_version != REQUIRED_RDK_VERSION:
        raise RuntimeError(
            f"This entrypoint requires flexivrdk=={REQUIRED_RDK_VERSION}; "
            f"found {installed_version}. RDK 2.x is not supported."
        )
    import flexivrdk
    return flexivrdk


class SerialLeader:
    """Own one UArm serial port and publish calibrated angle offsets."""

    def __init__(self, side, port, rate, stop_event, errors, debug_angles=False):
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("--rate must be finite and positive")
        self.side = side
        self.port = port
        self.rate = rate
        self.stop_event = stop_event
        self.errors = errors
        self.debug_angles = debug_angles
        self.calibration_requested = Event()
        self.calibration_ready = Event()
        self.calibrated = Event()
        self.samples = Queue(maxsize=1)
        self.calibration_candidate = None
        self.zero_angles = None
        self.ser = serial.Serial(port, BAUDRATE, timeout=0.01, write_timeout=0.5)
        self.thread = Thread(target=self._loop, name=f"{side}-uarm-serial", daemon=True)
        try:
            self._init_servos()
        except Exception:
            self.ser.close()
            raise

    def _init_servos(self):
        self.send_command("#000PVER!")
        for index in range(NUM_SERVOS):
            self.send_command("#000PCSK!")
            self.send_command(f"#{index:03d}PULK!")

    def send_command(self, command):
        self.ser.write(command.encode("ascii"))
        time.sleep(0.008)
        return self.ser.read_all().decode("ascii", errors="ignore")

    def read_angles(self):
        angles = []
        for index in range(NUM_SERVOS):
            response = self.send_command(f"#{index:03d}PRAD!")
            match = re.search(rf"#{index:03d}P(\d{{4}})", response)
            if match is None:
                raise ValueError(
                    f"{self.side} UArm servo {index:03d}: invalid response {response!r}"
                )
            pwm = int(match.group(1))
            if not PWM_MIN <= pwm <= PWM_MAX:
                raise ValueError(f"{self.side} servo {index:03d}: PWM out of range: {pwm}")
            angles.append((pwm - PWM_MIN) / (PWM_MAX - PWM_MIN) * ANGLE_RANGE_DEG)
        result = np.asarray(angles, dtype=np.float64)
        if result.shape != (NUM_SERVOS,) or not np.isfinite(result).all():
            raise ValueError(f"{self.side} UArm returned invalid angles")
        return result

    def request_calibration(self):
        if not self.calibrated.is_set():
            self.calibration_requested.set()

    def commit_calibration(self):
        if not self.calibration_ready.is_set() or self.calibration_candidate is None:
            raise RuntimeError(f"{self.side} calibration read is not ready")
        self.zero_angles = self.calibration_candidate.copy()
        self.calibrated.set()

    def latest(self):
        try:
            return self.samples.get_nowait()
        except Empty:
            return None

    def _publish(self, offsets):
        try:
            self.samples.put_nowait(offsets)
        except Exception:
            self.latest()
            self.samples.put_nowait(offsets)

    def _loop(self):
        try:
            while not self.stop_event.is_set():
                tick = time.monotonic()
                if not self.calibrated.is_set():
                    if self.calibration_ready.is_set():
                        self.stop_event.wait(0.02)
                        continue
                    if not self.calibration_requested.wait(0.1):
                        continue
                    self.calibration_requested.clear()
                    if self.stop_event.is_set():
                        break
                    self.calibration_candidate = self.read_angles()
                    self.calibration_ready.set()
                    print(f"[INFO] {self.side}: calibration sample ready")
                else:
                    offsets = np.radians(self.read_angles() - self.zero_angles)
                    if not np.isfinite(offsets).all():
                        raise ValueError(f"{self.side} UArm produced non-finite offsets")
                    self._publish(offsets)
                    if self.debug_angles:
                        values = " ".join(f"{v:+.1f}" for v in np.degrees(offsets))
                        print(f"[ANGLES] {self.side} (deg): {values}")
                self.stop_event.wait(max(0.0, 1.0 / self.rate - (time.monotonic() - tick)))
        except Exception as exc:
            self.errors.put((self.side, exc))
            self.stop_event.set()

    def close(self):
        if self.thread.ident is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                self.ser.cancel_read()
                self.ser.cancel_write()
                self.thread.join(timeout=1.0)
        if self.ser.is_open:
            self.ser.close()


class ArmMapping:
    """Map one leader's relative joint motion and trigger hysteresis."""

    def __init__(self, initial, max_offset=MAX_JOINT_OFFSET,
                 q_min=None, q_max=None):
        self.initial = np.asarray(initial, dtype=np.float64).copy()
        self.q_min = np.asarray(
            RIZON4_Q_MIN if q_min is None else q_min, dtype=np.float64
        ).copy()
        self.q_max = np.asarray(
            RIZON4_Q_MAX if q_max is None else q_max, dtype=np.float64
        ).copy()
        if self.initial.shape != (7,) or not np.isfinite(self.initial).all():
            raise ValueError("Rizon4 baseline must contain seven finite joint positions")
        if (self.q_min.shape != (7,) or self.q_max.shape != (7,)
                or not np.isfinite(self.q_min).all() or not np.isfinite(self.q_max).all()
                or np.any(self.q_min >= self.q_max)
                or np.any(self.initial < self.q_min) or np.any(self.initial > self.q_max)):
            raise ValueError("Rizon4 must report valid absolute joint position limits")
        if not np.isfinite(max_offset) or max_offset <= 0:
            raise ValueError("max joint offset must be finite and positive")
        self.max_offset = float(max_offset)
        self.closed = False
        self.pressed = False

    def convert(self, offsets):
        offsets = np.asarray(offsets, dtype=np.float64)
        if offsets.shape != (8,) or not np.isfinite(offsets).all():
            raise ValueError("Expected eight finite calibrated UArm offsets")
        joint_offsets = np.clip(offsets[:7] * JOINT_SIGNS,
                                -self.max_offset, self.max_offset)
        if not self.pressed and offsets[7] <= TRIGGER_PRESS_ANGLE:
            self.closed = not self.closed
            self.pressed = True
            gripper_edge = True
        elif self.pressed and offsets[7] >= TRIGGER_RELEASE_ANGLE:
            self.pressed = False
            gripper_edge = False
        else:
            gripper_edge = False
        target = np.clip(self.initial + joint_offsets, self.q_min, self.q_max)
        return target, gripper_edge


class DualRealTeleoperator:
    """Coordinate two existing/operational Rizon4 robots and two UArm leaders."""

    def __init__(self, left_robot_sn, right_robot_sn, left_port, right_port,
                 left_gripper_name=None, right_gripper_name=None,
                 rate=50.0, max_joint_offset=MAX_JOINT_OFFSET,
                 max_joint_velocity=DEFAULT_MAX_JOINT_VELOCITY,
                 max_joint_acceleration=DEFAULT_MAX_JOINT_ACCELERATION,
                 grippers=True, init_grippers=False, debug_angles=False, rdk=None):
        if left_robot_sn == right_robot_sn:
            raise ValueError("Left and right robot serial numbers must differ")
        if Path(left_port).resolve() == Path(right_port).resolve():
            raise ValueError("Left and right UArms require different serial ports")
        if grippers and (not left_gripper_name or not right_gripper_name):
            raise ValueError(
                "Both full Flexiv Elements device names are required when "
                "gripper control is enabled"
            )
        self.stop_event = Event()
        self.errors = Queue()
        self.robots = {}
        self.leaders = {}
        self.grippers = {}
        self.tools = {}
        self.gripper_params = {}
        self.mappings = {}
        self.joint_limits = {}
        self.max_vel = {}
        self.control_modes = set()
        self.commanded_grippers = set()
        self.grippers_enabled = grippers
        self.init_grippers = init_grippers
        self.gripper_names = dict(left=left_gripper_name, right=right_gripper_name)
        self.rate = float(rate)
        self.max_joint_offset = max_joint_offset
        self.max_joint_velocity = float(max_joint_velocity)
        self.max_joint_acceleration = float(max_joint_acceleration)
        self.debug_angles = debug_angles
        if not np.isfinite(self.rate) or self.rate <= 0 or self.rate > 100:
            raise ValueError("--rate must be finite and in [0.1, 100] Hz")
        if not np.isfinite(self.max_joint_offset) or self.max_joint_offset <= 0:
            raise ValueError("--max-joint-offset must be finite and positive")
        if (not np.isfinite(self.max_joint_velocity) or self.max_joint_velocity <= 0
                or not np.isfinite(self.max_joint_acceleration)
                or self.max_joint_acceleration <= 0):
            raise ValueError("Joint velocity and acceleration limits must be finite and positive")
        self.max_acc = [self.max_joint_acceleration] * 7
        self.rdk = rdk
        try:
            if self.rdk is None:
                self.rdk = load_flexiv_rdk()
            for side, sn in zip(SIDES, (left_robot_sn, right_robot_sn)):
                self.robots[side] = self.rdk.Robot(sn)
            for side, port in zip(SIDES, (left_port, right_port)):
                self.leaders[side] = SerialLeader(
                    side, port, self.rate, self.stop_event, self.errors, debug_angles
                )
            for side, robot in self.robots.items():
                if robot.fault():
                    raise RuntimeError(f"{side} robot reports a fault; no recovery was attempted")
                if not robot.operational():
                    raise RuntimeError(f"{side} robot is not operational; no enable was attempted")
                if robot.mode() != self.rdk.Mode.IDLE:
                    raise RuntimeError(f"{side} robot must be in IDLE mode before teleoperation")
                info = robot.info()
                if info.DoF != 7:
                    raise ValueError(f"{side} robot must report exactly seven DoF")
                q_min = np.asarray(info.q_min, dtype=np.float64)
                q_max = np.asarray(info.q_max, dtype=np.float64)
                qd_max = np.asarray(info.dq_max, dtype=np.float64)
                if (q_min.shape != (7,) or q_max.shape != (7,) or qd_max.shape != (7,)
                        or not np.isfinite(q_min).all() or not np.isfinite(q_max).all()
                        or not np.isfinite(qd_max).all() or np.any(q_min >= q_max)
                        or np.any(qd_max <= 0)):
                    raise ValueError(f"{side} robot returned invalid joint limits")
                limits = {
                    "q_min": np.maximum(q_min, RIZON4_Q_MIN),
                    "q_max": np.minimum(q_max, RIZON4_Q_MAX),
                    "dq_max": np.minimum(qd_max, RIZON4_QD_MAX),
                }
                if np.any(limits["q_min"] >= limits["q_max"]):
                    raise ValueError(f"{side} RDK and Rizon4 joint limits do not overlap")
                self.joint_limits[side] = limits
                self.max_vel[side] = np.minimum(
                    np.full(7, self.max_joint_velocity), limits["dq_max"]
                ).tolist()
                q = np.asarray(robot.states().q, dtype=np.float64)
                if q.shape != (7,) or not np.isfinite(q).all():
                    raise ValueError(f"{side} robot must report seven finite joint positions")
                if np.any(q < limits["q_min"]) or np.any(q > limits["q_max"]):
                    raise ValueError(f"{side} robot state is outside the Rizon4 hard limits")
        except Exception:
            self.close()
            raise

    def start_serial_threads(self):
        for leader in self.leaders.values():
            leader.thread.start()

    def calibrate_leaders(self):
        for leader in self.leaders.values():
            leader.request_calibration()

    def wait_for_key(self, prompt):
        print(prompt, flush=True)
        while not self.stop_event.is_set():
            self._raise_worker_error()
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                line = sys.stdin.readline()
                if not line or line.strip().lower() in {"q", "quit"}:
                    return False
                if not line.strip():
                    return True
        self._raise_worker_error()
        return False

    def _raise_worker_error(self):
        try:
            side, exc = self.errors.get_nowait()
        except Empty:
            return
        raise RuntimeError(f"{side} UArm serial acquisition failed: {exc}") from exc

    def _wait_for_calibration(self):
        while not self.stop_event.is_set():
            self._raise_worker_error()
            if all(item.calibration_ready.is_set() for item in self.leaders.values()):
                # Commit both references together only after both serial reads succeeded.
                for leader in self.leaders.values():
                    leader.commit_calibration()
                return
            time.sleep(0.02)
        self._raise_worker_error()
        raise RuntimeError("Stopped before both UArms calibrated")

    def _arm_after_confirmation(self):
        self._raise_worker_error()
        for side, robot in self.robots.items():
            if (robot.fault() or not robot.operational()
                    or robot.mode() != self.rdk.Mode.IDLE):
                raise RuntimeError(f"{side} robot became unavailable before start")
            baseline = np.asarray(robot.states().q, dtype=np.float64)
            if baseline.shape != (7,) or not np.isfinite(baseline).all():
                raise ValueError(f"{side} robot returned invalid baseline joint positions")
            limits = self.joint_limits[side]
            self.mappings[side] = ArmMapping(
                baseline,
                max_offset=self.max_joint_offset,
                q_min=limits["q_min"],
                q_max=limits["q_max"],
            )
        # Gripper/tool configuration is deliberately deferred until confirmation.
        if self.grippers_enabled:
            for side, robot in self.robots.items():
                self.grippers[side] = self.rdk.Gripper(robot)
                self.tools[side] = self.rdk.Tool(robot)
                name = self.gripper_names[side]
                self.grippers[side].Enable(name)
                self.tools[side].Switch(name)
                params = self.grippers[side].params()
                values = {attr: float(getattr(params, attr)) for attr in
                          ("min_width", "max_width", "min_vel", "max_vel",
                           "min_force", "max_force")}
                if (not all(np.isfinite(value) for value in values.values())
                        or values["min_width"] >= values["max_width"]
                        or values["min_vel"] <= 0
                        or values["min_vel"] > values["max_vel"]
                        or values["min_force"] <= 0
                        or values["min_force"] > values["max_force"]):
                    raise ValueError(f"{side} gripper returned invalid parameters")
                self.gripper_params[side] = values
                if self.init_grippers:
                    self.commanded_grippers.add(side)
                    self.grippers[side].Init()
                    if not self.wait_for_key(
                            f"{side} GN01 initialization was requested. Wait for it to finish, "
                            "then press Enter; q aborts."):
                        raise RuntimeError("Aborted during gripper initialization")
        for side, robot in self.robots.items():
            robot.SwitchMode(self.rdk.Mode.NRT_JOINT_POSITION)
            self.control_modes.add(side)
            if robot.mode() != self.rdk.Mode.NRT_JOINT_POSITION:
                raise RuntimeError(f"{side} robot did not enter NRT joint-position mode")

    def _command_gripper(self, side):
        gripper = self.grippers[side]
        params = self.gripper_params[side]
        width = params["min_width"] if self.mappings[side].closed else params["max_width"]
        velocity = min(max(GRIPPER_VELOCITY, params["min_vel"]), params["max_vel"])
        force = min(max(GRIPPER_FORCE, params["min_force"]), params["max_force"])
        self.commanded_grippers.add(side)
        gripper.Move(width, velocity, force)

    def run_control(self):
        self._arm_after_confirmation()
        latest = {side: None for side in SIDES}
        period = 1.0 / self.rate
        while not self.stop_event.is_set():
            tick = time.monotonic()
            self._raise_worker_error()
            targets = {}
            for side, leader in self.leaders.items():
                sample = leader.latest()
                if sample is not None:
                    latest[side] = sample
            if all(latest[side] is not None for side in SIDES):
                for side in SIDES:
                    if (self.robots[side].fault() or not self.robots[side].operational()
                            or self.robots[side].mode() != self.rdk.Mode.NRT_JOINT_POSITION):
                        raise RuntimeError(f"{side} robot fault or operational state lost")
                for side in SIDES:
                    target, gripper_edge = self.mappings[side].convert(latest[side])
                    if target.shape != (7,) or not np.isfinite(target).all():
                        raise ValueError(f"{side} computed an invalid joint target")
                    targets[side] = target
                    if gripper_edge and self.grippers_enabled:
                        self._command_gripper(side)
                for side in SIDES:
                    self.robots[side].SendJointPosition(
                        targets[side].tolist(), ZERO_VEL, self.max_vel[side], self.max_acc
                    )
            self.stop_event.wait(max(0.0, period - (time.monotonic() - tick)))

    def run(self):
        try:
            self.start_serial_threads()
            print(
                "Ensure both Rizon4 robots are operational and in IDLE mode."
            )
            print(
                "Release both E-stops and clear faults manually; this program "
                "will not enable or recover robots."
            )
            print("Position both UArms, release both triggers, then hold still.")
            if not self.wait_for_key("Press Enter to calibrate both UArms; q aborts."):
                return
            self.calibrate_leaders()
            self._wait_for_calibration()
            if not self.wait_for_key(
                    "Both UArms are calibrated. Verify the workspace, then press Enter "
                    "to capture live baselines and start; q aborts."):
                return
            self.run_control()
        except KeyboardInterrupt:
            pass
        finally:
            self.close()
        self._raise_worker_error()

    def close(self):
        self.stop_event.set()
        for side, gripper in self.grippers.items():
            if side not in self.commanded_grippers:
                continue
            try:
                gripper.Stop()
            except Exception as exc:
                print(f"[WARN] Best-effort gripper stop failed: {exc}", file=sys.stderr)
        for side, robot in self.robots.items():
            if side not in self.control_modes:
                continue
            try:
                if robot.mode() == self.rdk.Mode.NRT_JOINT_POSITION:
                    robot.Stop()
            except Exception as exc:
                print(f"[WARN] Best-effort robot stop failed: {exc}", file=sys.stderr)
        for leader in self.leaders.values():
            try:
                leader.close()
            except Exception as exc:
                print(f"[WARN] UArm cleanup failed: {exc}", file=sys.stderr)
        self.leaders.clear()
        self.grippers.clear()
        self.tools.clear()
        self.gripper_params.clear()
        self.commanded_grippers.clear()
        self.control_modes.clear()
        self.robots.clear()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-robot-sn", required=True)
    parser.add_argument("--right-robot-sn", required=True)
    parser.add_argument("--left-port", default="/dev/ttyUSB0")
    parser.add_argument("--right-port", default="/dev/ttyUSB1")
    parser.add_argument("--left-gripper-name",
                        help="Exact left device name from Flexiv Elements > Settings > Device")
    parser.add_argument("--right-gripper-name",
                        help="Exact right device name from Flexiv Elements > Settings > Device")
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--max-joint-offset", type=float, default=MAX_JOINT_OFFSET)
    parser.add_argument("--max-joint-velocity", type=float,
                        default=DEFAULT_MAX_JOINT_VELOCITY,
                        help="Conservative NRT speed limit in rad/s, capped at Rizon4 limits")
    parser.add_argument("--max-joint-acceleration", type=float,
                        default=DEFAULT_MAX_JOINT_ACCELERATION,
                        help="NRT acceleration limit in rad/s^2")
    parser.add_argument("--no-grippers", action="store_true",
                        help="Disable default-on GN01 trigger control")
    parser.add_argument("--init-grippers", action="store_true",
                        help="Deliberately call Gripper.Init after start confirmation")
    parser.add_argument("--debug-angles", action="store_true")
    args = parser.parse_args(argv)
    if not args.no_grippers and (not args.left_gripper_name or not args.right_gripper_name):
        parser.error(
            "--left-gripper-name and --right-gripper-name are required unless "
            "--no-grippers is set"
        )
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        DualRealTeleoperator(
            left_robot_sn=args.left_robot_sn,
            right_robot_sn=args.right_robot_sn,
            left_port=args.left_port,
            right_port=args.right_port,
            left_gripper_name=args.left_gripper_name,
            right_gripper_name=args.right_gripper_name,
            rate=args.rate,
            max_joint_offset=args.max_joint_offset,
            max_joint_velocity=args.max_joint_velocity,
            max_joint_acceleration=args.max_joint_acceleration,
            grippers=not args.no_grippers,
            init_grippers=args.init_grippers,
            debug_angles=args.debug_angles,
        ).run()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
